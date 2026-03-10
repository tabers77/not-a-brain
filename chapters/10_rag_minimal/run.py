"""Chapter 10: RAG Minimal

Retrieval-Augmented Generation: instead of changing the model, give it
access to external facts at inference time.  A simple BM25-style
retriever finds relevant facts from a knowledge base and prepends them
to the prompt.

Key finding: RAG dramatically improves factual accuracy on knowledge
questions.  But keyword-based retrieval finds "related" facts for
unanswerable questions too, so the model still hallucinates from
retrieved context.

Usage:
    python chapters/10_rag_minimal/run.py
"""

from __future__ import annotations
from pathlib import Path
import copy
import math
import re
from collections import Counter

import torch

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.models.decoding import decode
from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.evals.harness import (
    AgentInterface, AgentResult, run_eval_suite, save_results,
)
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.tasks.synthetic.knowledge_qa import FACTS
from not_a_brain.utils.training import train, make_dataset
from not_a_brain.utils.visualization import plot_loss_curve, plot_comparison_bar

# ── Config ────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent / "results"
N_TRAIN = 500
N_EVAL = 50
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
PRETRAIN_EPOCHS = 20
SFT_EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-3
SFT_LR = 1e-3
DROPOUT = 0.1
MAX_SEQ_LEN = 96   # Longer to accommodate retrieved context

INST_PREFIX = "INST: "
ANS_MARKER = " ANS: "

# ── Knowledge Base ────────────────────────────────────────────────────

# Build KB from the facts used in KnowledgeQATask
# Each entry: (fact_text, answer, question_text)
KNOWLEDGE_BASE = []
for answer, question_text, _ in FACTS:
    fact_text = f"{answer} is {question_text}"
    KNOWLEDGE_BASE.append({
        "text": fact_text,
        "answer": answer,
        "question": question_text,
    })


# ── BM25-style retriever ─────────────────────────────────────────────

def _tokenize_query(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer for retrieval."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """Compute inverse document frequency for each term."""
    n_docs = len(documents)
    df = Counter()
    for doc_tokens in documents:
        for token in set(doc_tokens):
            df[token] += 1
    return {term: math.log((n_docs + 1) / (count + 1)) + 1
            for term, count in df.items()}


class BM25Retriever:
    """Simple BM25 retriever over a list of text documents."""

    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_tokens = [_tokenize_query(doc) for doc in documents]
        self.idf = _compute_idf(self.doc_tokens)
        self.avg_dl = sum(len(d) for d in self.doc_tokens) / max(len(documents), 1)

    def score(self, query: str, doc_idx: int) -> float:
        """BM25 score for a query against a document."""
        query_tokens = _tokenize_query(query)
        doc_tokens = self.doc_tokens[doc_idx]
        dl = len(doc_tokens)
        tf = Counter(doc_tokens)
        total = 0.0
        for qt in query_tokens:
            if qt not in tf:
                continue
            f = tf[qt]
            idf = self.idf.get(qt, 0.0)
            num = f * (self.k1 + 1)
            denom = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            total += idf * num / denom
        return total

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        """Return top-k (doc_index, score) pairs sorted by relevance."""
        scores = [(i, self.score(query, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── Corpus / sequence helpers ─────────────────────────────────────────

def build_raw_corpus(tasks, n):
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(prompt + answer)
    return corpus


def build_instruction_corpus(tasks, n):
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}")
    return corpus


def build_rag_corpus(tasks, n, retriever, top_k=2):
    """Build SFT corpus with retrieved context prepended."""
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            # Retrieve relevant facts
            results = retriever.retrieve(prompt, top_k=top_k)
            context_parts = []
            for idx, score in results:
                if score > 0:
                    context_parts.append(KNOWLEDGE_BASE[idx]["text"])
            if context_parts:
                context = "CONTEXT: " + ". ".join(context_parts) + ". "
            else:
                context = ""
            corpus.append(
                f"{INST_PREFIX}{context}{prompt}{ANS_MARKER}{answer}")
    return corpus


def prepare_sequences(corpus, tokenizer, max_len=None):
    encoded = []
    for text in corpus:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        if max_len is not None and len(ids) > max_len:
            ids = ids[:max_len]
        encoded.append(ids)
    pad_len = max(len(s) for s in encoded)
    inputs, targets = [], []
    for ids in encoded:
        padded = ids + [tokenizer.pad_id] * (pad_len - len(ids))
        inputs.append(padded[:-1])
        targets.append(padded[1:])
    return (torch.tensor(inputs, dtype=torch.long),
            torch.tensor(targets, dtype=torch.long))


# ── Agents ────────────────────────────────────────────────────────────

class SFTAgent(AgentInterface):
    """Agent with instruction format, no retrieval."""

    def __init__(self, model, tokenizer, model_name="sft", max_gen=20):
        self.model = model
        self.tokenizer = tokenizer
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst = f"{INST_PREFIX}{prompt}{ANS_MARKER}"
        ids = self.tokenizer.encode(inst, add_bos=True)
        out = decode(self.model, ids, max_new_tokens=self.max_gen,
                     temperature=0.0)
        gen = self.tokenizer.decode(out)
        marker = ANS_MARKER.strip()
        answer = gen.split(marker)[-1].strip() if marker in gen else gen.strip()
        for stop in ["\n", "<EOS>", "<PAD>"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]
        return AgentResult(answer=answer.strip(), confidence=0.5)


class RAGAgent(AgentInterface):
    """Agent that retrieves facts from KB before generating."""

    def __init__(self, model, tokenizer, retriever, top_k=2,
                 model_name="rag", max_gen=20):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.top_k = top_k
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        # Retrieve
        results = self.retriever.retrieve(prompt, top_k=self.top_k)
        context_parts = []
        top_score = 0.0
        for idx, score in results:
            if score > 0:
                context_parts.append(KNOWLEDGE_BASE[idx]["text"])
                top_score = max(top_score, score)

        if context_parts:
            context = "CONTEXT: " + ". ".join(context_parts) + ". "
        else:
            context = ""

        inst = f"{INST_PREFIX}{context}{prompt}{ANS_MARKER}"
        ids = self.tokenizer.encode(inst, add_bos=True)
        out = decode(self.model, ids, max_new_tokens=self.max_gen,
                     temperature=0.0)
        gen = self.tokenizer.decode(out)
        marker = ANS_MARKER.strip()
        answer = gen.split(marker)[-1].strip() if marker in gen else gen.strip()
        for stop in ["\n", "<EOS>", "<PAD>"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]
        return AgentResult(
            answer=answer.strip(), confidence=0.5,
            trace=[f"retrieved: {context_parts}", f"top_score: {top_score:.2f}"])


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Chapter 10: RAG Minimal")
    print("=" * 60)
    print()
    print("Instead of changing the model, give it external facts at")
    print("inference time.  A BM25 retriever finds relevant facts from")
    print("a knowledge base and prepends them to the prompt.")
    print()
    print(f"  Knowledge base: {len(KNOWLEDGE_BASE)} facts")
    print(f"  Retriever: BM25 (k1=1.5, b=0.75)")
    print(f"  Top-k retrieval: 2 facts per query")
    print()

    # ------------------------------------------------------------------
    # 1. Build retriever
    # ------------------------------------------------------------------
    kb_texts = [entry["text"] for entry in KNOWLEDGE_BASE]
    retriever = BM25Retriever(kb_texts)

    print("Sample retrievals:")
    sample_queries = [
        "capital of france?",
        "chemical formula of water?",
        "capital of the Moon?",
        "ADD 5 3 =",
    ]
    for q in sample_queries:
        results = retriever.retrieve(q, top_k=2)
        top_facts = [(kb_texts[i], f"{s:.2f}") for i, s in results if s > 0]
        print(f"  Query: '{q}'")
        for fact, score in top_facts:
            print(f"    -> '{fact}' (score={score})")
        if not top_facts:
            print(f"    -> (no relevant facts)")
        print()

    # ------------------------------------------------------------------
    # 2. Build corpora and train
    # ------------------------------------------------------------------
    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
        "unknown": UnknownTask(seed=1),
    }

    print("Building corpora...")
    raw_corpus = build_raw_corpus(tasks, N_TRAIN)
    inst_corpus = build_instruction_corpus(tasks, N_TRAIN)
    rag_corpus = build_rag_corpus(tasks, N_TRAIN, retriever, top_k=2)
    print(f"  Raw:         {len(raw_corpus)} sequences")
    print(f"  Instruction: {len(inst_corpus)} sequences")
    print(f"  RAG:         {len(rag_corpus)} sequences")
    print(f"\n  Example RAG: {rag_corpus[0][:80]}...")

    # Fit tokenizer on all corpora
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + inst_corpus + rag_corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Prepare datasets
    raw_in, raw_tgt = prepare_sequences(raw_corpus, tokenizer, MAX_SEQ_LEN)
    inst_in, inst_tgt = prepare_sequences(inst_corpus, tokenizer, MAX_SEQ_LEN)
    rag_in, rag_tgt = prepare_sequences(rag_corpus, tokenizer, MAX_SEQ_LEN)
    raw_loader = make_dataset(raw_in, raw_tgt, batch_size=BATCH_SIZE)
    inst_loader = make_dataset(inst_in, inst_tgt, batch_size=BATCH_SIZE)
    rag_loader = make_dataset(rag_in, rag_tgt, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------
    # 3. Train base + SFT model (no RAG)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Training: Pre-train + SFT (no RAG)")
    print(f"{'─' * 50}")

    sft_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    print(f"  Parameters: {sft_model.count_parameters():,}")
    pretrain_result = train(sft_model, raw_loader,
                            epochs=PRETRAIN_EPOCHS, lr=LR)
    sft_result = train(sft_model, inst_loader,
                       epochs=SFT_EPOCHS, lr=SFT_LR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        pretrain_result.losses + sft_result.losses,
        title="Chapter 10: SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch10_sft_loss.png"),
    )
    print(f"  Final SFT loss: {sft_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Train RAG-aware model (pre-train + SFT on RAG corpus)
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Training: Pre-train + RAG-SFT (with retrieved context)")
    print(f"{'─' * 50}")

    rag_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    train(rag_model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)
    rag_train_result = train(rag_model, rag_loader,
                             epochs=SFT_EPOCHS, lr=SFT_LR)
    plot_loss_curve(
        rag_train_result.losses,
        title="Chapter 10: RAG-SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch10_rag_loss.png"),
    )
    print(f"  Final RAG-SFT loss: {rag_train_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 5. Benchmark prompts
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Benchmark Prompts: SFT vs RAG")
    print(f"{'─' * 50}")
    print()

    sft_agent = SFTAgent(sft_model, tokenizer, "sft_only")
    rag_agent = RAGAgent(rag_model, tokenizer, retriever, top_k=2,
                         model_name="rag")

    benchmark_prompts = [
        ("ADD 5 3 =", "Computation -- RAG has no arithmetic facts"),
        ("FACT: paris is capital of france. Q: capital of france?",
         "Retrieval -- fact already in prompt + RAG adds more"),
        ("Q: capital of france?",
         "Retrieval WITHOUT context -- RAG must provide the fact"),
        ("Q: What is the capital of the Moon?",
         "Hallucination -- RAG retrieves 'capital of ...' facts"),
    ]

    for prompt, description in benchmark_prompts:
        print(f"  Prompt: '{prompt}'")
        print(f"  Tests:  {description}")
        sft_out = sft_agent.run(prompt)
        rag_out = rag_agent.run(prompt)
        print(f"    SFT:  '{sft_out.answer}'")
        print(f"    RAG:  '{rag_out.answer}'")
        if rag_out.trace:
            print(f"          {rag_out.trace[0]}")
        print()

    # ------------------------------------------------------------------
    # 6. Evaluate
    # ------------------------------------------------------------------
    eval_tasks = {
        "arithmetic": ArithmeticTask(seed=99),
        "copy": CopyTask(seed=99),
        "grammar": GrammarTask(seed=99),
        "knowledge_qa": KnowledgeQATask(seed=99),
        "compositional": CompositionalTask(seed=99),
        "unknown": UnknownTask(seed=99),
    }
    human = HumanAgent()

    print(f"{'─' * 50}")
    print("Evaluation: SFT vs RAG vs Human")
    print(f"{'─' * 50}")

    agents = {
        "sft_only": sft_agent,
        "rag": rag_agent,
        "human": human,
    }

    all_metrics = {}
    for agent_name, agent in agents.items():
        metrics, results = run_eval_suite(
            agent, eval_tasks, n_per_task=N_EVAL)
        all_metrics[agent_name] = metrics
        print(f"\n  {agent_name}:")
        print(f"    Accuracy:           {metrics.accuracy:.1%}")
        print(f"    Hallucination rate: {metrics.hallucination_rate:.1%}")
        print(f"    Per-task:")
        for t, info in metrics.per_task.items():
            print(f"      {t:20s} {info['accuracy']:.1%}")

        save_results(
            results, metrics,
            RESULTS_DIR / f"ch10_{agent_name}.json",
            agent_name=agent_name, chapter="10_rag_minimal")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    task_names = list(eval_tasks.keys())
    scores = {}
    for name, metrics in all_metrics.items():
        scores[name] = [
            metrics.per_task.get(t, {}).get("accuracy", 0.0)
            for t in task_names
        ]

    plot_comparison_bar(
        labels=task_names, scores=scores,
        title="Chapter 10: SFT vs RAG vs Human",
        save_path=str(RESULTS_DIR / "ch10_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch10_comparison.png'}")

    _plot_retrieval_scores(retriever, kb_texts)
    print(f"  Retrieval scores saved to {RESULTS_DIR / 'ch10_retrieval_scores.png'}")

    # ------------------------------------------------------------------
    # 8. Key takeaway
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print()
    print("RAG grounds the model in external evidence:")
    print("  - Knowledge QA improves because the retriever provides")
    print("    the relevant fact even when the prompt lacks it")
    print("  - The model learns to use CONTEXT: prefix as evidence")
    print()
    print("What RAG does NOT fix:")
    print("  - Hallucination on unanswerable questions: BM25 retrieves")
    print("    'capital of france' for 'capital of Moon' (keyword match)")
    print("    and the model treats retrieved facts as valid evidence")
    print("  - Arithmetic, copy, grammar: these tasks don't need a KB")
    print()
    print("The deeper lesson: RAG externalizes knowledge but not judgment.")
    print("The model cannot distinguish 'relevant fact' from 'keyword match'.")
    print("A human would look at 'capital of france' and say 'this doesn't")
    print("answer my question about the Moon.'")
    print()
    print("Human lens: Humans seek evidence, evaluate its relevance,")
    print("and reconcile contradictions.  RAG only does step 1 (seek).")
    print("Steps 2-3 require reasoning the model doesn't have.")
    print()
    print("Next: Chapter 11 (Tools & Function Calls) -- can external")
    print("tools (calculator, lookup) give the model capabilities it")
    print("lacks internally?")


# ── Plotting helpers ──────────────────────────────────────────────────

def _plot_retrieval_scores(retriever, kb_texts) -> None:
    """Heatmap of retrieval scores for benchmark queries vs KB facts."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    queries = [
        "ADD 5 3 =",
        "capital of france?",
        "capital of the Moon?",
        "chemical formula of water?",
        "COPY: abc|",
    ]

    scores = np.zeros((len(queries), len(kb_texts)))
    for i, q in enumerate(queries):
        for j in range(len(kb_texts)):
            scores[i, j] = retriever.score(q, j)

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(scores, cmap="YlOrRd", aspect="auto")
    ax.set_yticks(range(len(queries)))
    ax.set_yticklabels(queries, fontsize=8)
    ax.set_xticks(range(len(kb_texts)))
    ax.set_xticklabels([t[:20] for t in kb_texts], rotation=60,
                        ha="right", fontsize=6)
    ax.set_title("Chapter 10: BM25 Retrieval Scores (Query × KB Fact)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="BM25 Score")
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch10_retrieval_scores.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
