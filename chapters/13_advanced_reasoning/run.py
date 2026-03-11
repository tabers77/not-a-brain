"""Chapter 13: Advanced Reasoning — ReAct, Tree of Thoughts, MCTS

Three modern reasoning algorithms that power today's frontier LLMs:

1. ReAct (Reasoning + Acting): interleave THINK/ACT/OBSERVE steps,
   combining the reasoning of Ch12 with the tool use of Ch11.
   This is how GPT-4 and Claude use tools in practice.

2. Tree of Thoughts (ToT): instead of a single reasoning chain,
   branch into multiple paths, score each, and follow the most
   promising.  Reasoning becomes search over a tree.

3. MCTS + Process Reward Model: the approach behind OpenAI o1/o3
   and DeepSeek-R1.  A separate model scores intermediate reasoning
   steps.  Monte Carlo Tree Search uses these scores to explore
   reasoning paths efficiently — spending more compute on harder
   problems.

Key finding: these algorithms improve accuracy on solvable tasks by
systematically searching through reasoning paths.  But for "capital
of the Moon?", every path in the search space leads to hallucination.
Search can't find an answer that doesn't exist.

Usage:
    python chapters/13_advanced_reasoning/run.py
"""

from __future__ import annotations
from pathlib import Path
import math
import re
import random
from collections import Counter
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.models.decoding import decode, score_sequence
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
MAX_SEQ_LEN = 128

INST_PREFIX = "INST: "
ANS_MARKER = " ANS: "
THINK_MARKER = " THINK: "
ACT_MARKER = " ACT: "
OBSERVE_MARKER = " OBSERVE: "

# -- Knowledge Base & BM25 (reused from Ch10-11) --------------------------

KNOWLEDGE_BASE = []
for answer, question_text, _ in FACTS:
    fact_text = f"{answer} is {question_text}"
    KNOWLEDGE_BASE.append({
        "text": fact_text, "answer": answer, "question": question_text,
    })


def _tokenize_query(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _compute_idf(documents: list[list[str]]) -> dict[str, float]:
    n_docs = len(documents)
    df = Counter()
    for doc_tokens in documents:
        for token in set(doc_tokens):
            df[token] += 1
    return {term: math.log((n_docs + 1) / (count + 1)) + 1
            for term, count in df.items()}


class BM25Retriever:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1, self.b = k1, b
        self.doc_tokens = [_tokenize_query(doc) for doc in documents]
        self.idf = _compute_idf(self.doc_tokens)
        self.avg_dl = sum(len(d) for d in self.doc_tokens) / max(len(documents), 1)

    def score(self, query: str, doc_idx: int) -> float:
        qt = _tokenize_query(query)
        dt = self.doc_tokens[doc_idx]
        tf = Counter(dt)
        dl = len(dt)
        total = 0.0
        for t in qt:
            if t not in tf:
                continue
            f = tf[t]
            idf = self.idf.get(t, 0.0)
            total += idf * f * (self.k1 + 1) / (
                f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl))
        return total

    def retrieve(self, query: str, top_k: int = 3) -> list[tuple[int, float]]:
        scores = [(i, self.score(query, i)) for i in range(len(self.documents))]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# -- Tools (reused from Ch11) ---------------------------------------------

def tool_calc(args: str) -> str:
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
    results = retriever.retrieve(query, top_k=1)
    if results and results[0][1] > threshold:
        return KNOWLEDGE_BASE[results[0][0]]["answer"]
    return "NOT_FOUND"


# -- Corpus helpers --------------------------------------------------------

def build_raw_corpus(tasks, n):
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(prompt + answer)
    return corpus


def _make_cot_reasoning(task_name, prompt, answer):
    if task_name == "arithmetic":
        match = re.match(r"(ADD|SUB|MUL)\s+(\d+)\s+(\d+)\s*=", prompt)
        if match:
            op, a, b = match.group(1), match.group(2), match.group(3)
            sym = {"ADD": "+", "SUB": "-", "MUL": "*"}.get(op, "+")
            return f"op is {op}, {a}{sym}{b}={answer}"
        return f"compute: {answer}"
    elif task_name == "knowledge_qa":
        q_match = re.search(r"Q:\s*(.+?)$", prompt)
        q = q_match.group(1) if q_match else "question"
        return f"fact says {answer} is {q.rstrip('?')}"
    elif task_name == "copy":
        return f"copy input: {answer}"
    elif task_name == "grammar":
        return f"check: {answer}"
    elif task_name == "compositional":
        return f"steps: {answer}"
    elif task_name == "unknown":
        return "no fact given, unanswerable"
    return f"answer: {answer}"


def build_react_corpus(tasks, n, retriever):
    """ReAct corpus: THINK/ACT/OBSERVE interleaved steps."""
    corpus = []
    for task_name, task in tasks.items():
        for prompt, answer in task.training_pairs(n):
            if task_name == "arithmetic":
                match = re.match(r"(ADD|SUB|MUL)\s+(\d+)\s+(\d+)\s*=", prompt)
                if match:
                    calc_args = f"{match.group(1)} {match.group(2)} {match.group(3)}"
                    result = tool_calc(calc_args)
                    text = (f"{INST_PREFIX}{prompt}"
                            f"{THINK_MARKER}need calculator"
                            f"{ACT_MARKER}calc({calc_args})"
                            f"{OBSERVE_MARKER}{result}"
                            f"{THINK_MARKER}result is {result}"
                            f"{ANS_MARKER}{answer}")
                else:
                    text = f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}"
            elif task_name == "knowledge_qa":
                q_match = re.search(r"Q:\s*(.+?)$", prompt)
                query = q_match.group(1) if q_match else prompt
                lr = tool_lookup(query, retriever)
                text = (f"{INST_PREFIX}{prompt}"
                        f"{THINK_MARKER}look up fact"
                        f"{ACT_MARKER}lookup({query})"
                        f"{OBSERVE_MARKER}{lr}"
                        f"{THINK_MARKER}answer is {lr}"
                        f"{ANS_MARKER}{answer}")
            elif task_name == "unknown":
                q_match = re.search(r"Q:\s*(.+?)$", prompt)
                query = q_match.group(1) if q_match else prompt
                text = (f"{INST_PREFIX}{prompt}"
                        f"{THINK_MARKER}look up fact"
                        f"{ACT_MARKER}lookup({query})"
                        f"{OBSERVE_MARKER}NOT_FOUND"
                        f"{THINK_MARKER}not found, unanswerable"
                        f"{ANS_MARKER}{answer}")
            else:
                text = (f"{INST_PREFIX}{prompt}"
                        f"{THINK_MARKER}solve directly"
                        f"{ANS_MARKER}{answer}")
            corpus.append(text)
    return corpus


def build_cot_corpus(tasks, n):
    """CoT corpus for ToT and MCTS (THINK: reasoning ANS: answer)."""
    corpus = []
    for task_name, task in tasks.items():
        for prompt, answer in task.training_pairs(n):
            reasoning = _make_cot_reasoning(task_name, prompt, answer)
            corpus.append(f"{INST_PREFIX}{prompt}"
                          f"{THINK_MARKER}{reasoning}"
                          f"{ANS_MARKER}{answer}")
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


# -- Process Reward Model --------------------------------------------------

class ProcessRewardModel(nn.Module):
    """Tiny reward model that scores reasoning steps.

    Takes a sequence of token IDs (a partial reasoning trace) and
    outputs a scalar score in [0, 1] — how likely this reasoning
    step leads to a correct answer.

    Architecture: shared transformer encoder -> mean pool -> linear -> sigmoid
    """

    def __init__(self, vocab_size, d_model=32, n_heads=2, n_layers=1,
                 d_ff=64, max_seq_len=128, pad_id=0):
        super().__init__()
        self.pad_id = pad_id
        self.encoder = TransformerLM(
            vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
            n_layers=n_layers, d_ff=d_ff, max_seq_len=max_seq_len,
            dropout=0.0, pad_id=pad_id,
        )
        self.score_head = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Sigmoid(),
        )
        # Access the encoder's internal dimension
        self._d_model = d_model

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Score a batch of reasoning sequences.

        Args:
            token_ids: (B, S) token IDs

        Returns:
            (B,) scores in [0, 1]
        """
        # Get hidden states from the encoder (we need pre-projection states)
        # Use the transformer's embedding + blocks directly
        enc = self.encoder
        x = enc.tok_embedding(token_ids) + enc.pos_embedding(
            torch.arange(token_ids.shape[1], device=token_ids.device))
        x = enc.drop(x)
        for block in enc.blocks:
            x = block(x)
        x = enc.ln_f(x)

        # Mean pool over non-pad positions
        mask = (token_ids != self.pad_id).float().unsqueeze(-1)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)

        return self.score_head(pooled).squeeze(-1)

    def score(self, token_ids: list[int]) -> float:
        """Score a single reasoning sequence. Returns float in [0, 1]."""
        self.eval()
        with torch.no_grad():
            x = torch.tensor([token_ids], dtype=torch.long)
            return self.forward(x).item()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_prm_training_data(tasks, n, tokenizer, max_len=64):
    """Build (sequence, label) pairs for PRM training.

    Positive (label=1): correct reasoning traces
    Negative (label=0): wrong reasoning (shuffled answer/reasoning)
    """
    rng = random.Random(42)
    sequences = []
    labels = []

    all_pairs = []
    for task_name, task in tasks.items():
        for prompt, answer in task.training_pairs(n):
            reasoning = _make_cot_reasoning(task_name, prompt, answer)
            all_pairs.append((task_name, prompt, answer, reasoning))

    for task_name, prompt, answer, reasoning in all_pairs:
        # Positive: correct reasoning
        text = f"{INST_PREFIX}{prompt}{THINK_MARKER}{reasoning}{ANS_MARKER}{answer}"
        ids = tokenizer.encode(text, add_bos=True)
        if len(ids) > max_len:
            ids = ids[:max_len]
        sequences.append(ids)
        labels.append(1.0)

        # Negative: wrong answer with reasoning
        other = rng.choice(all_pairs)
        wrong_answer = other[2]
        if wrong_answer == answer:
            wrong_answer = "WRONG"
        wrong_text = (f"{INST_PREFIX}{prompt}"
                      f"{THINK_MARKER}{other[3]}"
                      f"{ANS_MARKER}{wrong_answer}")
        wrong_ids = tokenizer.encode(wrong_text, add_bos=True)
        if len(wrong_ids) > max_len:
            wrong_ids = wrong_ids[:max_len]
        sequences.append(wrong_ids)
        labels.append(0.0)

    # Pad sequences
    pad_len = max(len(s) for s in sequences)
    padded = []
    for ids in sequences:
        padded.append(ids + [tokenizer.pad_id] * (pad_len - len(ids)))

    return (torch.tensor(padded, dtype=torch.long),
            torch.tensor(labels, dtype=torch.float))


def train_prm(prm, sequences, labels, epochs=10, lr=1e-3, batch_size=32):
    """Train the PRM with binary cross-entropy loss."""
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(sequences, labels)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(prm.parameters(), lr=lr)
    loss_fn = nn.BCELoss()

    prm.train()
    losses = []
    for epoch in range(epochs):
        epoch_loss = 0.0
        n = 0
        for batch_seqs, batch_labels in loader:
            scores = prm(batch_seqs)
            loss = loss_fn(scores, batch_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n += 1
        avg = epoch_loss / max(n, 1)
        losses.append(avg)
        print(f"  PRM Epoch {epoch+1}/{epochs} -- loss: {avg:.4f}")
    prm.eval()
    return losses


# -- Answer extraction helper ----------------------------------------------

def _extract_answer(gen_text):
    marker = ANS_MARKER.strip()
    answer = gen_text.split(marker)[-1].strip() if marker in gen_text else gen_text.strip()
    for stop in ["\n", "<EOS>", "<PAD>", THINK_MARKER.strip(), ACT_MARKER.strip()]:
        if stop in answer:
            answer = answer[:answer.index(stop)]
    return answer.strip()


# -- Agents ----------------------------------------------------------------

class SFTAgent(AgentInterface):
    """Baseline: instruction format, no reasoning."""

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
        return AgentResult(answer=_extract_answer(gen), confidence=0.5)


class ReActAgent(AgentInterface):
    """ReAct: interleave THINK/ACT/OBSERVE steps with tool execution.

    The model generates THINK: and ACT: markers.  When ACT: is detected,
    the system executes the tool, injects OBSERVE: result, and lets the
    model continue generating until ANS:.

    This is how GPT-4 and Claude use tools in practice.
    """

    def __init__(self, model, tokenizer, retriever,
                 model_name="react", max_gen=60):
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def _execute_action(self, action_text: str) -> str:
        calc_match = re.search(r"calc\(([^)]*)\)", action_text)
        if calc_match:
            return tool_calc(calc_match.group(1))
        lookup_match = re.search(r"lookup\(([^)]*)\)", action_text)
        if lookup_match:
            return tool_lookup(lookup_match.group(1), self.retriever)
        return "UNKNOWN_ACTION"

    def run(self, prompt: str) -> AgentResult:
        inst = f"{INST_PREFIX}{prompt}{THINK_MARKER}"
        ids = self.tokenizer.encode(inst, add_bos=True)
        trace = []
        actions_executed = 0
        max_actions = 2

        for _ in range(self.max_gen):
            x = torch.tensor([ids], dtype=torch.long)
            with torch.no_grad():
                logits = self.model(x)
            next_id = logits[0, -1, :].argmax().item()
            ids.append(next_id)
            gen = self.tokenizer.decode(ids)

            # Check for ACT: marker — execute tool
            if (actions_executed < max_actions
                    and ACT_MARKER.strip() in gen):
                act_match = re.search(
                    r"ACT:\s*(\w+\([^)]*\))", gen)
                if act_match:
                    action = act_match.group(1)
                    result = self._execute_action(action)
                    trace.append(f"ACT: {action} -> {result}")
                    actions_executed += 1
                    # Inject OBSERVE: result and continue
                    injected = (gen[:act_match.end()] +
                                f"{OBSERVE_MARKER}{result}{THINK_MARKER}")
                    ids = self.tokenizer.encode(injected, add_bos=False)

            # Stop conditions
            if "<EOS>" in gen:
                break
            ans_m = ANS_MARKER.strip()
            if ans_m in gen:
                after = gen.split(ans_m)[-1].strip()
                if len(after) >= 1:
                    break

        gen = self.tokenizer.decode(ids)
        answer = _extract_answer(gen)
        if not trace:
            trace = ["no action executed"]
        return AgentResult(answer=answer, confidence=0.5, trace=trace)


class ToTAgent(AgentInterface):
    """Tree of Thoughts: branch into multiple reasoning paths, pick best.

    Algorithm:
      1. Generate N different first reasoning steps (temperature sampling)
      2. Score each path using model log-probability
      3. Expand the top-k paths to completion
      4. Return the answer from the highest-scoring complete path

    This treats reasoning as search over a tree of possible thought
    sequences, using the model's own confidence as a heuristic.
    """

    def __init__(self, model, tokenizer, n_branches=4, top_k=2,
                 model_name="tot", max_gen=60):
        self.model = model
        self.tokenizer = tokenizer
        self.n_branches = n_branches
        self.top_k = top_k
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst = f"{INST_PREFIX}{prompt}{THINK_MARKER}"
        base_ids = self.tokenizer.encode(inst, add_bos=True)

        # Step 1: Generate N reasoning branches
        branches = []
        for _ in range(self.n_branches):
            out = decode(self.model, list(base_ids),
                         max_new_tokens=self.max_gen,
                         temperature=0.8)
            gen = self.tokenizer.decode(out)
            answer = _extract_answer(gen)
            # Score using model log-probability
            score = score_sequence(self.model, out)
            branches.append((answer, score, gen))

        # Step 2: Sort by score, pick top-k
        branches.sort(key=lambda x: x[1], reverse=True)
        top_branches = branches[:self.top_k]

        # Step 3: Re-score top branches with greedy completion
        best_answer = ""
        best_score = float("-inf")
        all_answers = []
        for answer, branch_score, gen in top_branches:
            all_answers.append((answer, branch_score))
            if branch_score > best_score and answer:
                best_score = branch_score
                best_answer = answer

        # Fallback: if no good answer, use majority vote across all branches
        if not best_answer:
            all_ans = [a for a, _, _ in branches if a]
            if all_ans:
                best_answer = Counter(all_ans).most_common(1)[0][0]

        trace = [
            f"branches: {[(a, f'{s:.2f}') for a, s in all_answers]}",
            f"all: {[a for a, _, _ in branches]}",
            f"winner: '{best_answer}' (score={best_score:.2f})",
        ]
        return AgentResult(answer=best_answer, confidence=0.5, trace=trace)


class MCTSAgent(AgentInterface):
    """MCTS + Process Reward Model: search reasoning paths with a trained judge.

    This is the approach behind OpenAI o1/o3 and DeepSeek-R1.

    Algorithm:
      1. Start with the prompt as root node
      2. For each iteration:
         a. SELECT: pick the most promising node (UCB1 score)
         b. EXPAND: generate a new reasoning step from that node
         c. EVALUATE: PRM scores the expanded path
         d. BACKPROPAGATE: update ancestor scores
      3. Return the answer from the highest-scoring leaf

    The PRM (Process Reward Model) is a separately trained model that
    scores intermediate reasoning steps — not just final answers.
    """

    def __init__(self, model, tokenizer, prm, n_iterations=8,
                 model_name="mcts", max_gen=60):
        self.model = model
        self.tokenizer = tokenizer
        self.prm = prm
        self.n_iterations = n_iterations
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst = f"{INST_PREFIX}{prompt}{THINK_MARKER}"
        base_ids = self.tokenizer.encode(inst, add_bos=True)

        # MCTS tree: list of (ids, score, visit_count, parent_idx)
        nodes = [(list(base_ids), 0.0, 1, -1)]
        leaf_results = []

        for iteration in range(self.n_iterations):
            # SELECT: UCB1 — balance exploration vs exploitation
            best_idx = 0
            best_ucb = float("-inf")
            total_visits = sum(n[2] for n in nodes)
            for i, (ids, score, visits, _) in enumerate(nodes):
                exploit = score / max(visits, 1)
                explore = math.sqrt(2 * math.log(total_visits + 1) / max(visits, 1))
                ucb = exploit + explore
                if ucb > best_ucb:
                    best_ucb = ucb
                    best_idx = i

            parent_ids = nodes[best_idx][0]
            # Truncate to fit model's max_seq_len minus room for generation
            max_ctx = MAX_SEQ_LEN - 30
            if len(parent_ids) > max_ctx:
                parent_ids = parent_ids[:max_ctx]

            # EXPAND: generate a new reasoning step
            out = decode(self.model, list(parent_ids),
                         max_new_tokens=min(30, self.max_gen),
                         temperature=0.7)
            gen = self.tokenizer.decode(out)

            # EVALUATE: PRM scores the path
            prm_ids = out[:MAX_SEQ_LEN]
            prm_score = self.prm.score(prm_ids)

            # Add node to tree
            nodes.append((out, prm_score, 1, best_idx))

            # BACKPROPAGATE: update parent scores
            idx = best_idx
            while idx >= 0:
                ids_n, score_n, visits_n, parent_n = nodes[idx]
                nodes[idx] = (ids_n, score_n + prm_score, visits_n + 1, parent_n)
                idx = parent_n

            # Track leaf answers
            answer = _extract_answer(gen)
            if answer:
                leaf_results.append((answer, prm_score))

        # Return highest PRM-scored answer
        if leaf_results:
            leaf_results.sort(key=lambda x: x[1], reverse=True)
            best_answer = leaf_results[0][0]
        else:
            # Fallback: greedy
            out = decode(self.model, list(base_ids),
                         max_new_tokens=self.max_gen, temperature=0.0)
            best_answer = _extract_answer(self.tokenizer.decode(out))

        trace = [
            f"iterations: {self.n_iterations}",
            f"nodes explored: {len(nodes)}",
            f"leaf answers: {[(a, f'{s:.2f}') for a, s in leaf_results[:5]]}",
            f"winner: '{best_answer}'"
                + (f" (prm={leaf_results[0][1]:.2f})" if leaf_results else ""),
        ]
        return AgentResult(answer=best_answer, confidence=0.5, trace=trace)


# -- Main ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Chapter 13: Advanced Reasoning")
    print("  ReAct, Tree of Thoughts, MCTS + Process Reward Model")
    print("=" * 60)
    print()
    print("Three algorithms that power frontier LLMs:")
    print("  1. ReAct -- reasoning interleaved with tool use")
    print("  2. Tree of Thoughts -- branching search over reasoning")
    print("  3. MCTS + PRM -- guided tree search with a trained judge")
    print()

    # ------------------------------------------------------------------
    # 1. Setup tools
    # ------------------------------------------------------------------
    kb_texts = [entry["text"] for entry in KNOWLEDGE_BASE]
    retriever = BM25Retriever(kb_texts)

    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
        "unknown": UnknownTask(seed=1),
    }

    # ------------------------------------------------------------------
    # 2. Build corpora
    # ------------------------------------------------------------------
    print("Building corpora...")
    raw_corpus = build_raw_corpus(tasks, N_TRAIN)
    react_corpus = build_react_corpus(tasks, N_TRAIN, retriever)
    cot_corpus = build_cot_corpus(tasks, N_TRAIN)
    print(f"  Raw:   {len(raw_corpus)} sequences")
    print(f"  ReAct: {len(react_corpus)} sequences")
    print(f"  CoT:   {len(cot_corpus)} sequences")

    react_samples = [s for s in react_corpus if "ACT:" in s][:2]
    for i, s in enumerate(react_samples):
        print(f"\n  ReAct example {i+1}: {s[:120]}...")
    print()

    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + react_corpus + cot_corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    raw_in, raw_tgt = prepare_sequences(raw_corpus, tokenizer, MAX_SEQ_LEN)
    react_in, react_tgt = prepare_sequences(react_corpus, tokenizer, MAX_SEQ_LEN)
    cot_in, cot_tgt = prepare_sequences(cot_corpus, tokenizer, MAX_SEQ_LEN)
    raw_loader = make_dataset(raw_in, raw_tgt, batch_size=BATCH_SIZE)
    react_loader = make_dataset(react_in, react_tgt, batch_size=BATCH_SIZE)
    cot_loader = make_dataset(cot_in, cot_tgt, batch_size=BATCH_SIZE)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 3. Train ReAct model
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Pre-train + ReAct-SFT (THINK/ACT/OBSERVE)")
    print(f"{'-' * 50}")

    react_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    print(f"  Parameters: {react_model.count_parameters():,}")
    train(react_model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)
    react_result = train(react_model, react_loader,
                         epochs=SFT_EPOCHS, lr=SFT_LR)
    plot_loss_curve(
        react_result.losses,
        title="Chapter 13: ReAct-SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch13_react_loss.png"),
    )
    print(f"  Final ReAct loss: {react_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Train CoT model (shared by ToT and MCTS)
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Pre-train + CoT-SFT (for ToT and MCTS)")
    print(f"{'-' * 50}")

    cot_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    train(cot_model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)
    cot_result = train(cot_model, cot_loader, epochs=SFT_EPOCHS, lr=SFT_LR)
    plot_loss_curve(
        cot_result.losses,
        title="Chapter 13: CoT-SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch13_cot_loss.png"),
    )
    print(f"  Final CoT loss: {cot_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 5. Train Process Reward Model
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Process Reward Model (PRM)")
    print(f"{'-' * 50}")

    prm = ProcessRewardModel(
        vocab_size=tokenizer.vocab_size, d_model=32, n_heads=2,
        n_layers=1, d_ff=64, max_seq_len=MAX_SEQ_LEN,
        pad_id=tokenizer.pad_id,
    )
    print(f"  PRM parameters: {prm.count_parameters():,}")

    prm_seqs, prm_labels = build_prm_training_data(
        tasks, 200, tokenizer, max_len=MAX_SEQ_LEN)
    print(f"  PRM training samples: {len(prm_labels)}"
          f" ({prm_labels.sum().int()} positive, "
          f"{(1 - prm_labels).sum().int()} negative)")
    prm_losses = train_prm(prm, prm_seqs, prm_labels, epochs=10, lr=1e-3)

    # Quick PRM sanity check
    good_text = "INST: ADD 5 3 = THINK: op is ADD, 5+3=8 ANS: 8"
    bad_text = "INST: ADD 5 3 = THINK: op is ADD, 5+3=8 ANS: fish"
    good_ids = tokenizer.encode(good_text, add_bos=True)[:MAX_SEQ_LEN]
    bad_ids = tokenizer.encode(bad_text, add_bos=True)[:MAX_SEQ_LEN]
    print(f"\n  PRM sanity check:")
    print(f"    Good reasoning score: {prm.score(good_ids):.3f}")
    print(f"    Bad reasoning score:  {prm.score(bad_ids):.3f}")

    # ------------------------------------------------------------------
    # 6. Benchmark prompts
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Benchmark Prompts: ReAct vs ToT vs MCTS")
    print(f"{'-' * 50}")
    print()

    sft_agent = SFTAgent(cot_model, tokenizer, "sft_baseline")
    react_agent = ReActAgent(react_model, tokenizer, retriever,
                             model_name="react")
    tot_agent = ToTAgent(cot_model, tokenizer, n_branches=4, top_k=2,
                         model_name="tot")
    mcts_agent = MCTSAgent(cot_model, tokenizer, prm, n_iterations=8,
                           model_name="mcts")

    benchmark_prompts = [
        ("ADD 5 3 =",
         "Computation -- ReAct uses calc, ToT/MCTS search for best path"),
        ("FACT: paris is capital of france. Q: capital of france?",
         "Retrieval -- ReAct uses lookup, others reason from context"),
        ("Q: What is the capital of the Moon?",
         "Hallucination -- every search path leads to a wrong answer"),
    ]

    for prompt, description in benchmark_prompts:
        print(f"  Prompt: '{prompt}'")
        print(f"  Tests:  {description}")
        sft_out = sft_agent.run(prompt)
        react_out = react_agent.run(prompt)
        tot_out = tot_agent.run(prompt)
        mcts_out = mcts_agent.run(prompt)
        print(f"    SFT:   '{sft_out.answer}'")
        print(f"    ReAct: '{react_out.answer}'")
        if react_out.trace:
            for t in react_out.trace[:2]:
                print(f"           {t[:80]}")
        print(f"    ToT:   '{tot_out.answer}'")
        if tot_out.trace:
            for t in tot_out.trace[:2]:
                print(f"           {t[:80]}")
        print(f"    MCTS:  '{mcts_out.answer}'")
        if mcts_out.trace:
            for t in mcts_out.trace[:2]:
                print(f"           {t[:80]}")
        print()

    # ------------------------------------------------------------------
    # 7. Evaluate
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
    print("Evaluation: SFT vs ReAct vs ToT vs MCTS vs Human")
    print(f"{'-' * 50}")

    agents = {
        "sft_baseline": sft_agent,
        "react": react_agent,
        "tot": tot_agent,
        "mcts": mcts_agent,
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
            RESULTS_DIR / f"ch13_{agent_name}.json",
            agent_name=agent_name, chapter="13_advanced_reasoning")

    # ------------------------------------------------------------------
    # 8. Plots
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
        title="Chapter 13: Advanced Reasoning vs Human",
        save_path=str(RESULTS_DIR / "ch13_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch13_comparison.png'}")

    _plot_search_comparison(tot_agent, mcts_agent, eval_tasks)
    print(f"  Search comparison saved to {RESULTS_DIR / 'ch13_search_depth.png'}")

    # ------------------------------------------------------------------
    # 9. Key takeaway
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("THE FINAL PUNCHLINE")
    print("=" * 60)
    print()
    print("After 13 chapters and every major technique in modern AI:")
    print()
    print("  Ch01-05: Architecture evolution   -> hallucinates")
    print("  Ch06:    Scaling                   -> hallucinates")
    print("  Ch07:    Instruction tuning        -> hallucinates")
    print("  Ch08:    Preference alignment      -> hallucinates")
    print("  Ch09:    Decoding strategies        -> hallucinates")
    print("  Ch10:    RAG                       -> hallucinates")
    print("  Ch11:    Tool use                  -> hallucinates")
    print("  Ch12:    Basic reasoning (CoT)     -> hallucinates")
    print("  Ch13:    Advanced reasoning         -> hallucinates")
    print("          (ReAct, ToT, MCTS+PRM)")
    print()
    print("ReAct combines reasoning with tool use -- but the lookup")
    print("tool returns irrelevant facts via keyword match.")
    print()
    print("Tree of Thoughts searches multiple reasoning paths --")
    print("but every path in the search space leads to hallucination.")
    print("You can't find an answer that doesn't exist by searching")
    print("harder.")
    print()
    print("MCTS + PRM uses a trained judge to guide search -- but the")
    print("PRM was trained on the same data with the same blind spots.")
    print("It scores 'tokyo' and 'paris' as plausible answers to")
    print("'capital of Moon?' because they ARE plausible-looking tokens.")
    print("The PRM has no world model to know the Moon has no capital.")
    print()
    print("The thesis holds: similar outputs != same mechanism.")
    print("That's not a brain.")


# -- Plotting helpers ------------------------------------------------------

def _plot_search_comparison(tot_agent, mcts_agent, eval_tasks) -> None:
    """Compare search-based agents: how many paths explored vs accuracy."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    task_names = list(eval_tasks.keys())
    tot_acc = []
    mcts_acc = []

    for task_name, task in eval_tasks.items():
        tot_correct = 0
        mcts_correct = 0
        n = 20
        for _ in range(n):
            sample = task.generate()
            tot_r = tot_agent.run(sample.prompt)
            mcts_r = mcts_agent.run(sample.prompt)
            grade_tot = task.grade(tot_r.answer, sample)
            grade_mcts = task.grade(mcts_r.answer, sample)
            tot_correct += grade_tot.correct
            mcts_correct += grade_mcts.correct
        tot_acc.append(tot_correct / n)
        mcts_acc.append(mcts_correct / n)

    x = np.arange(len(task_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width/2, tot_acc, width, label="ToT (branch & score)",
           color="#FF9800")
    ax.bar(x + width/2, mcts_acc, width, label="MCTS + PRM (guided search)",
           color="#9C27B0")

    ax.set_xticks(x)
    ax.set_xticklabels(task_names, rotation=30, ha="right")
    ax.set_ylabel("Accuracy")
    ax.set_title("Chapter 13: Tree Search vs MCTS+PRM")
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch13_search_depth.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
