"""Chapter 11: Tools & Function Calling

Give the tiny Transformer external tools — a calculator and a knowledge
lookup — that perform tasks the model cannot do internally.  The model
learns to emit structured CALL:tool(args) markers during generation; the
system intercepts these, executes the tool, and injects RESULT:value
back into the context so generation continues.

Key finding: Tools genuinely extend capabilities (the calculator solves
arithmetic the model struggles with), but the model cannot judge whether
a tool's output actually answers the question.  For "capital of the
Moon?", the lookup tool returns capital-related facts via keyword match,
and the model treats them as valid evidence — same hallucination, now
laundered through a tool.

Usage:
    python chapters/11_tools_and_function_calls/run.py
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

# -- Config ----------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
N_TRAIN = 500
N_EVAL = 50
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
PRETRAIN_EPOCHS = 20
SFT_EPOCHS = 15
BATCH_SIZE = 64
LR = 3e-3
SFT_LR = 1e-3
DROPOUT = 0.1
MAX_SEQ_LEN = 128   # Longer to accommodate tool call + result

INST_PREFIX = "INST: "
ANS_MARKER = " ANS: "
CALL_MARKER = " CALL:"
RESULT_MARKER = " RESULT:"

# -- Knowledge Base --------------------------------------------------------

KNOWLEDGE_BASE = []
for answer, question_text, _ in FACTS:
    fact_text = f"{answer} is {question_text}"
    KNOWLEDGE_BASE.append({
        "text": fact_text,
        "answer": answer,
        "question": question_text,
    })


# -- BM25 Retriever (reused from Ch10) ------------------------------------

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


# -- Tool Definitions ------------------------------------------------------

def tool_calc(args: str) -> str:
    """Calculator tool: parses 'OP a b' and returns the integer result.

    Supports ADD, SUB, MUL.  Returns 'ERROR' on parse failure.
    """
    parts = args.strip().split()
    if len(parts) < 3:
        return "ERROR"
    op = parts[0].upper()
    try:
        a, b = int(parts[1]), int(parts[2])
    except ValueError:
        return "ERROR"
    if op == "ADD":
        return str(a + b)
    elif op == "SUB":
        return str(a - b)
    elif op == "MUL":
        return str(a * b)
    return "ERROR"


def tool_lookup(query: str, retriever: BM25Retriever,
                threshold: float = 0.5) -> str:
    """Lookup tool: BM25 search over KB, return best answer or NOT_FOUND.

    Returns the answer string from the top-scoring KB entry if above
    threshold, otherwise 'NOT_FOUND'.
    """
    results = retriever.retrieve(query, top_k=1)
    if results and results[0][1] > threshold:
        idx = results[0][0]
        return KNOWLEDGE_BASE[idx]["answer"]
    return "NOT_FOUND"


TOOL_REGISTRY = {
    "calc": tool_calc,
    # lookup is bound at runtime with the retriever
}


# -- Corpus / sequence helpers ---------------------------------------------

def build_raw_corpus(tasks, n):
    """Plain text corpus (no instruction format, no tools)."""
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(prompt + answer)
    return corpus


def build_sft_corpus(tasks, n):
    """Instruction-formatted corpus (no tools)."""
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}")
    return corpus


def build_tool_corpus(tasks, n, retriever):
    """Instruction corpus with tool calls embedded in the text.

    - Arithmetic tasks:  CALL:calc(OP a b) RESULT:answer
    - KnowledgeQA tasks: CALL:lookup(question) RESULT:answer
    - Unknown tasks:     CALL:lookup(question) RESULT:NOT_FOUND
    - Other tasks:       no tool call (model answers directly)
    """
    corpus = []
    for task_name, task in tasks.items():
        for prompt, answer in task.training_pairs(n):
            if task_name == "arithmetic":
                # Extract OP, a, b from "ADD 5 3 ="
                match = re.match(r"(ADD|SUB|MUL)\s+(\d+)\s+(\d+)\s*=", prompt)
                if match:
                    calc_args = f"{match.group(1)} {match.group(2)} {match.group(3)}"
                    result = tool_calc(calc_args)
                    text = (f"{INST_PREFIX}{prompt}"
                            f"{CALL_MARKER}calc({calc_args})"
                            f"{RESULT_MARKER}{result}"
                            f"{ANS_MARKER}{answer}")
                else:
                    text = f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}"

            elif task_name == "knowledge_qa":
                # Extract question from "FACT: ... Q: question?"
                q_match = re.search(r"Q:\s*(.+?)$", prompt)
                query = q_match.group(1) if q_match else prompt
                lookup_result = tool_lookup(query, retriever)
                text = (f"{INST_PREFIX}{prompt}"
                        f"{CALL_MARKER}lookup({query})"
                        f"{RESULT_MARKER}{lookup_result}"
                        f"{ANS_MARKER}{answer}")

            elif task_name == "unknown":
                # Unknown questions: lookup returns NOT_FOUND
                q_match = re.search(r"Q:\s*(.+?)$", prompt)
                query = q_match.group(1) if q_match else prompt
                # Force NOT_FOUND for unknown training examples
                text = (f"{INST_PREFIX}{prompt}"
                        f"{CALL_MARKER}lookup({query})"
                        f"{RESULT_MARKER}NOT_FOUND"
                        f"{ANS_MARKER}{answer}")

            else:
                # Copy, grammar, compositional: no tool needed
                text = f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}"

            corpus.append(text)
    return corpus


def prepare_sequences(corpus, tokenizer, max_len=None):
    """Encode corpus into padded input/target tensor pairs."""
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


# -- Agents ----------------------------------------------------------------

class SFTAgent(AgentInterface):
    """Agent with instruction format, no tools."""

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


class ToolAgent(AgentInterface):
    """Agent that intercepts CALL: markers and executes tools.

    Generation loop:
      1. Encode INST: <prompt>
      2. Generate token-by-token
      3. If generated text contains CALL:tool(args), execute the tool
      4. Inject RESULT:value into context and continue generating
      5. Extract answer after ANS: marker
    """

    def __init__(self, model, tokenizer, retriever,
                 model_name="tool", max_gen=40):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def _execute_tool(self, tool_name: str, args: str) -> str:
        """Execute a tool by name and return the result string."""
        tool_name = tool_name.strip().lower()
        if tool_name == "calc":
            return tool_calc(args)
        elif tool_name == "lookup":
            return tool_lookup(args, self.retriever)
        return "UNKNOWN_TOOL"

    def run(self, prompt: str) -> AgentResult:
        inst = f"{INST_PREFIX}{prompt}"
        ids = self.tokenizer.encode(inst, add_bos=True)

        tool_calls = []
        tool_executed = False

        # Generate tokens one at a time, watching for CALL: marker
        for _ in range(self.max_gen):
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x)
            next_id = logits[0, -1, :].argmax().item()
            ids.append(next_id)

            # Check if we've generated a complete tool call
            generated = self.tokenizer.decode(ids)
            if not tool_executed and CALL_MARKER.strip() in generated:
                # Try to parse CALL:tool(args)
                call_match = re.search(
                    r"CALL:(\w+)\(([^)]*)\)", generated)
                if call_match:
                    t_name = call_match.group(1)
                    t_args = call_match.group(2)
                    result = self._execute_tool(t_name, t_args)
                    tool_calls.append(
                        f"CALL:{t_name}({t_args}) -> {result}")

                    # Inject RESULT:value into context
                    result_text = f"{RESULT_MARKER}{result}{ANS_MARKER}"
                    ids = self.tokenizer.encode(
                        generated[:call_match.end()] +
                        f"{RESULT_MARKER}{result}{ANS_MARKER}",
                        add_bos=False)
                    tool_executed = True

            # Check for EOS or ANS marker to stop early
            gen_text = self.tokenizer.decode(ids)
            if "<EOS>" in gen_text:
                break
            marker = ANS_MARKER.strip()
            # Stop if we see ANS: followed by some content
            if marker in gen_text:
                after_ans = gen_text.split(marker)[-1].strip()
                if len(after_ans) >= 1:
                    break

        # Extract answer
        gen_text = self.tokenizer.decode(ids)
        marker = ANS_MARKER.strip()
        answer = gen_text.split(marker)[-1].strip() if marker in gen_text else gen_text.strip()
        for stop in ["\n", "<EOS>", "<PAD>"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]

        trace = tool_calls if tool_calls else ["no tool called"]
        return AgentResult(answer=answer.strip(), confidence=0.5,
                           trace=trace)


class OracleToolAgent(AgentInterface):
    """Agent where the orchestrator (not the model) decides which tool to call.

    This represents the ceiling: perfect tool selection, showing what's
    possible when tools are used correctly.  The model still generates
    the final answer from the augmented prompt.
    """

    def __init__(self, model, tokenizer, retriever,
                 model_name="oracle_tool", max_gen=20):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        tool_calls = []

        # Decide which tool to call based on prompt pattern
        augmented = prompt
        if re.match(r"(ADD|SUB|MUL)\s+\d+\s+\d+\s*=", prompt):
            result = tool_calc(prompt.replace("=", "").strip())
            augmented = f"{prompt} CALL:calc({prompt.strip('= ')}) RESULT:{result}"
            tool_calls.append(f"calc({prompt.strip()}) -> {result}")
        elif "Q:" in prompt or "FACT:" in prompt:
            q_match = re.search(r"Q:\s*(.+?)$", prompt)
            query = q_match.group(1) if q_match else prompt
            result = tool_lookup(query, self.retriever)
            augmented = f"{prompt} CALL:lookup({query}) RESULT:{result}"
            tool_calls.append(f"lookup({query}) -> {result}")

        inst = f"{INST_PREFIX}{augmented}{ANS_MARKER}"
        ids = self.tokenizer.encode(inst, add_bos=True)
        out = decode(self.model, ids, max_new_tokens=self.max_gen,
                     temperature=0.0)
        gen = self.tokenizer.decode(out)
        marker = ANS_MARKER.strip()
        answer = gen.split(marker)[-1].strip() if marker in gen else gen.strip()
        for stop in ["\n", "<EOS>", "<PAD>"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]

        trace = tool_calls if tool_calls else ["no tool needed"]
        return AgentResult(answer=answer.strip(), confidence=0.5,
                           trace=trace)


# -- Main ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Chapter 11: Tools & Function Calling")
    print("=" * 60)
    print()
    print("Give the model external tools that perform tasks it cannot")
    print("do internally.  The model learns to emit CALL:tool(args)")
    print("markers; the system executes the tool and injects the result.")
    print()
    print("Tools:")
    print("  calc  -- arithmetic (ADD, SUB, MUL)")
    print("  lookup -- knowledge base search (BM25)")
    print()

    # ------------------------------------------------------------------
    # 1. Build retriever + tools
    # ------------------------------------------------------------------
    kb_texts = [entry["text"] for entry in KNOWLEDGE_BASE]
    retriever = BM25Retriever(kb_texts)

    print("Tool demos:")
    print(f"  calc('ADD 5 3')     -> {tool_calc('ADD 5 3')}")
    print(f"  calc('MUL 7 8')     -> {tool_calc('MUL 7 8')}")
    print(f"  calc('bad input')   -> {tool_calc('bad input')}")
    print(f"  lookup('capital of france?') -> {tool_lookup('capital of france?', retriever)}")
    print(f"  lookup('capital of the Moon?') -> {tool_lookup('capital of the Moon?', retriever)}")
    print(f"  lookup('zzzzz')     -> {tool_lookup('zzzzz', retriever)}")
    print()

    # ------------------------------------------------------------------
    # 2. Build corpora
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
    sft_corpus = build_sft_corpus(tasks, N_TRAIN)
    tool_corpus = build_tool_corpus(tasks, N_TRAIN, retriever)
    print(f"  Raw:  {len(raw_corpus)} sequences")
    print(f"  SFT:  {len(sft_corpus)} sequences")
    print(f"  Tool: {len(tool_corpus)} sequences")

    # Show sample tool corpus entries
    tool_samples = [s for s in tool_corpus if "CALL:" in s][:3]
    for i, s in enumerate(tool_samples):
        print(f"\n  Tool example {i+1}: {s[:100]}...")
    print()

    # Fit tokenizer on all corpora
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + sft_corpus + tool_corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Prepare datasets
    raw_in, raw_tgt = prepare_sequences(raw_corpus, tokenizer, MAX_SEQ_LEN)
    sft_in, sft_tgt = prepare_sequences(sft_corpus, tokenizer, MAX_SEQ_LEN)
    tool_in, tool_tgt = prepare_sequences(tool_corpus, tokenizer, MAX_SEQ_LEN)
    raw_loader = make_dataset(raw_in, raw_tgt, batch_size=BATCH_SIZE)
    sft_loader = make_dataset(sft_in, sft_tgt, batch_size=BATCH_SIZE)
    tool_loader = make_dataset(tool_in, tool_tgt, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------
    # 3. Train SFT baseline (no tools)
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Pre-train + SFT (no tools)")
    print(f"{'-' * 50}")

    sft_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    print(f"  Parameters: {sft_model.count_parameters():,}")
    pretrain_result = train(sft_model, raw_loader,
                            epochs=PRETRAIN_EPOCHS, lr=LR)
    sft_result = train(sft_model, sft_loader,
                       epochs=SFT_EPOCHS, lr=SFT_LR)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        pretrain_result.losses + sft_result.losses,
        title="Chapter 11: SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch11_sft_loss.png"),
    )
    print(f"  Final SFT loss: {sft_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Train tool-augmented model
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Pre-train + Tool-SFT (with CALL/RESULT markers)")
    print(f"{'-' * 50}")

    tool_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    train(tool_model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)
    tool_train_result = train(tool_model, tool_loader,
                              epochs=SFT_EPOCHS, lr=SFT_LR)
    plot_loss_curve(
        tool_train_result.losses,
        title="Chapter 11: Tool-SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch11_tool_loss.png"),
    )
    print(f"  Final Tool-SFT loss: {tool_train_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 5. Benchmark prompts
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Benchmark Prompts: SFT vs ToolAgent vs OracleToolAgent")
    print(f"{'-' * 50}")
    print()

    sft_agent = SFTAgent(sft_model, tokenizer, "sft_only")
    tool_agent = ToolAgent(tool_model, tokenizer, retriever,
                           model_name="tool_agent")
    oracle_agent = OracleToolAgent(tool_model, tokenizer, retriever,
                                   model_name="oracle_tool")

    benchmark_prompts = [
        ("ADD 5 3 =",
         "Computation -- calc tool should return 8"),
        ("FACT: paris is capital of france. Q: capital of france?",
         "Retrieval -- fact in prompt + lookup confirms"),
        ("Q: capital of france?",
         "Retrieval WITHOUT context -- lookup provides the fact"),
        ("Q: What is the capital of the Moon?",
         "Hallucination -- lookup returns irrelevant capital facts"),
    ]

    for prompt, description in benchmark_prompts:
        print(f"  Prompt: '{prompt}'")
        print(f"  Tests:  {description}")
        sft_out = sft_agent.run(prompt)
        tool_out = tool_agent.run(prompt)
        oracle_out = oracle_agent.run(prompt)
        print(f"    SFT:          '{sft_out.answer}'")
        print(f"    ToolAgent:    '{tool_out.answer}'")
        if tool_out.trace:
            print(f"                  trace: {tool_out.trace}")
        print(f"    OracleTool:   '{oracle_out.answer}'")
        if oracle_out.trace:
            print(f"                  trace: {oracle_out.trace}")
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

    print(f"{'-' * 50}")
    print("Evaluation: SFT vs ToolAgent vs OracleTool vs Human")
    print(f"{'-' * 50}")

    agents = {
        "sft_only": sft_agent,
        "tool_agent": tool_agent,
        "oracle_tool": oracle_agent,
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
            RESULTS_DIR / f"ch11_{agent_name}.json",
            agent_name=agent_name, chapter="11_tools_and_function_calls")

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
        title="Chapter 11: SFT vs Tool Agents vs Human",
        save_path=str(RESULTS_DIR / "ch11_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch11_comparison.png'}")

    _plot_tool_usage(tool_agent, eval_tasks)
    print(f"  Tool usage plot saved to {RESULTS_DIR / 'ch11_tool_usage.png'}")

    # ------------------------------------------------------------------
    # 8. Key takeaway
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print()
    print("Tools genuinely extend the model's capabilities:")
    print("  - Calculator solves arithmetic the model struggles with")
    print("  - Lookup provides facts not in the prompt")
    print("  - The model learns to emit CALL: markers during generation")
    print()
    print("What tools do NOT fix:")
    print("  - Judgment: the model cannot evaluate tool output quality")
    print("  - Abstention: for 'capital of Moon?', lookup returns")
    print("    irrelevant capital facts via keyword match, and the model")
    print("    treats them as valid evidence")
    print("  - Tool selection: the model sometimes calls the wrong tool")
    print("    or doesn't call one when it should")
    print()
    print("The deeper lesson: tools externalize computation but not")
    print("judgment.  A calculator gives perfect answers to well-formed")
    print("questions.  A lookup tool has the same keyword-matching flaw")
    print("as RAG.  And the model has no way to check whether a tool's")
    print("output actually answers the question being asked.")
    print()
    print("Human lens: humans use tools with intent -- they choose which")
    print("tool to use, verify the result makes sense, and try a different")
    print("approach if it doesn't.  The model emits tool calls as learned")
    print("patterns, without understanding what the tool does or whether")
    print("the result is correct.")
    print()
    print("Next: Chapter 12 (Reasoning Scaffolds) -- can explicit")
    print("reasoning steps (chain-of-thought, self-consistency, verify)")
    print("give the model the judgment it lacks?")


# -- Plotting helpers ------------------------------------------------------

def _plot_tool_usage(tool_agent, eval_tasks) -> None:
    """Bar chart showing how often the tool agent calls each tool per task."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    task_names = list(eval_tasks.keys())
    calc_counts = []
    lookup_counts = []
    no_tool_counts = []

    for task_name, task in eval_tasks.items():
        n_calc = 0
        n_lookup = 0
        n_none = 0
        for _ in range(30):
            sample = task.generate()
            result = tool_agent.run(sample.prompt)
            trace = result.trace or []
            trace_str = " ".join(trace)
            if "calc" in trace_str:
                n_calc += 1
            elif "lookup" in trace_str:
                n_lookup += 1
            else:
                n_none += 1
        total = n_calc + n_lookup + n_none
        calc_counts.append(n_calc / total if total > 0 else 0)
        lookup_counts.append(n_lookup / total if total > 0 else 0)
        no_tool_counts.append(n_none / total if total > 0 else 0)

    x = np.arange(len(task_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, calc_counts, width, label="calc", color="#4CAF50")
    ax.bar(x, lookup_counts, width, label="lookup", color="#2196F3")
    ax.bar(x + width, no_tool_counts, width, label="no tool", color="#9E9E9E")

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.set_ylabel("Fraction of prompts")
    ax.set_title("Chapter 11: Tool Usage by Task")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch11_tool_usage.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
