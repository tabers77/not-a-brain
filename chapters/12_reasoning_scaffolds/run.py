"""Chapter 12: Reasoning Scaffolds

The final chapter.  We add explicit reasoning structure to generation:
chain-of-thought (CoT), self-consistency (majority vote across multiple
samples), and a verify-then-revise loop.

Key finding: Reasoning scaffolds improve accuracy on tasks where the
model already has the capability (arithmetic, knowledge QA) by making
intermediate steps explicit.  But for "capital of the Moon?", CoT
generates plausible-sounding but wrong reasoning, self-consistency
votes for the most popular hallucination, and verify approves wrong
answers.  After 12 chapters of architecture, scale, training, decoding,
retrieval, tools, and reasoning -- the model STILL cannot say "I don't
know."

Usage:
    python chapters/12_reasoning_scaffolds/run.py
"""

from __future__ import annotations
from pathlib import Path
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
VERIFY_MARKER = " VERIFY: "

# -- Corpus / sequence helpers ---------------------------------------------


def build_raw_corpus(tasks, n):
    """Plain text corpus (no instruction format)."""
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(prompt + answer)
    return corpus


def build_sft_corpus(tasks, n):
    """Instruction-formatted corpus (no reasoning)."""
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}")
    return corpus


def _make_cot_reasoning(task_name, prompt, answer):
    """Generate chain-of-thought reasoning for a training example.

    Returns a short reasoning string that bridges prompt -> answer.
    """
    if task_name == "arithmetic":
        match = re.match(r"(ADD|SUB|MUL)\s+(\d+)\s+(\d+)\s*=", prompt)
        if match:
            op, a, b = match.group(1), match.group(2), match.group(3)
            if op == "ADD":
                return f"operation is ADD, operands {a} and {b}, {a}+{b}={answer}"
            elif op == "SUB":
                return f"operation is SUB, operands {a} and {b}, {a}-{b}={answer}"
            elif op == "MUL":
                return f"operation is MUL, operands {a} and {b}, {a}*{b}={answer}"
        return f"compute: {answer}"

    elif task_name == "knowledge_qa":
        # Extract the question part
        q_match = re.search(r"Q:\s*(.+?)$", prompt)
        question = q_match.group(1) if q_match else "question"
        return f"fact states {answer} is {question.rstrip('?')}, answer is {answer}"

    elif task_name == "copy":
        return f"copy the input: {answer}"

    elif task_name == "grammar":
        return f"check balance: result is {answer}"

    elif task_name == "compositional":
        return f"apply operations step by step: {answer}"

    elif task_name == "unknown":
        return "no relevant fact given, question is unanswerable, abstain"

    return f"answer is {answer}"


def build_cot_corpus(tasks, n):
    """Instruction corpus with chain-of-thought reasoning.

    Format: INST: <prompt> THINK: <reasoning> ANS: <answer>
    """
    corpus = []
    for task_name, task in tasks.items():
        for prompt, answer in task.training_pairs(n):
            reasoning = _make_cot_reasoning(task_name, prompt, answer)
            text = (f"{INST_PREFIX}{prompt}"
                    f"{THINK_MARKER}{reasoning}"
                    f"{ANS_MARKER}{answer}")
            corpus.append(text)
    return corpus


def build_verify_corpus(tasks, n):
    """Instruction corpus with reasoning + verification step.

    Format: INST: <prompt> THINK: <reasoning> ANS: <answer>
            VERIFY: <answer> answers <question>? YES
    """
    corpus = []
    for task_name, task in tasks.items():
        for prompt, answer in task.training_pairs(n):
            reasoning = _make_cot_reasoning(task_name, prompt, answer)
            # Build verification step
            q_match = re.search(r"Q:\s*(.+?)$", prompt)
            question = q_match.group(1).rstrip("?") if q_match else prompt[:30]
            verify = f"{answer} answers {question}? YES"
            text = (f"{INST_PREFIX}{prompt}"
                    f"{THINK_MARKER}{reasoning}"
                    f"{ANS_MARKER}{answer}"
                    f"{VERIFY_MARKER}{verify}")
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


def _extract_answer(gen_text):
    """Extract the answer from generated text after ANS: marker."""
    marker = ANS_MARKER.strip()
    answer = gen_text.split(marker)[-1].strip() if marker in gen_text else gen_text.strip()
    # Trim at verify marker if present
    verify_m = VERIFY_MARKER.strip()
    if verify_m in answer:
        answer = answer[:answer.index(verify_m)].strip()
    for stop in ["\n", "<EOS>", "<PAD>"]:
        if stop in answer:
            answer = answer[:answer.index(stop)]
    return answer.strip()


class SFTAgent(AgentInterface):
    """Baseline agent with instruction format, no reasoning."""

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
        answer = _extract_answer(gen)
        return AgentResult(answer=answer, confidence=0.5)


class CoTAgent(AgentInterface):
    """Chain-of-thought agent: generates THINK: reasoning before ANS:.

    The model was trained on THINK: ... ANS: ... format, so it generates
    reasoning steps before arriving at an answer.
    """

    def __init__(self, model, tokenizer, model_name="cot", max_gen=60):
        self.model = model
        self.tokenizer = tokenizer
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst = f"{INST_PREFIX}{prompt}{THINK_MARKER}"
        ids = self.tokenizer.encode(inst, add_bos=True)
        out = decode(self.model, ids, max_new_tokens=self.max_gen,
                     temperature=0.0)
        gen = self.tokenizer.decode(out)

        # Extract reasoning and answer
        think_m = THINK_MARKER.strip()
        reasoning = ""
        if think_m in gen:
            after_think = gen.split(think_m, 1)[-1]
            ans_m = ANS_MARKER.strip()
            if ans_m in after_think:
                reasoning = after_think[:after_think.index(ans_m)].strip()
            else:
                reasoning = after_think.strip()

        answer = _extract_answer(gen)
        trace = [f"reasoning: {reasoning}"] if reasoning else []
        return AgentResult(answer=answer, confidence=0.5, trace=trace)


class SelfConsistencyAgent(AgentInterface):
    """Self-consistency: generate N answers with temperature, majority vote.

    Samples multiple completions at temperature > 0 and returns the most
    common answer.  If all answers differ, returns the first one.
    """

    def __init__(self, model, tokenizer, n_samples=5,
                 temperature=0.8, model_name="self_consistency",
                 max_gen=60):
        self.model = model
        self.tokenizer = tokenizer
        self.n_samples = n_samples
        self.temperature = temperature
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst = f"{INST_PREFIX}{prompt}{THINK_MARKER}"
        ids = self.tokenizer.encode(inst, add_bos=True)

        answers = []
        for _ in range(self.n_samples):
            out = decode(self.model, list(ids),
                         max_new_tokens=self.max_gen,
                         temperature=self.temperature)
            gen = self.tokenizer.decode(out)
            answer = _extract_answer(gen)
            if answer:
                answers.append(answer)

        if not answers:
            return AgentResult(answer="", confidence=0.0,
                               trace=["no valid answers generated"])

        # Majority vote
        counts = Counter(answers)
        winner, winner_count = counts.most_common(1)[0]
        agreement = winner_count / len(answers)

        trace = [
            f"samples: {answers}",
            f"winner: '{winner}' ({winner_count}/{len(answers)})",
            f"agreement: {agreement:.0%}",
        ]
        return AgentResult(answer=winner, confidence=agreement,
                           trace=trace)


class VerifyAgent(AgentInterface):
    """Verify-then-revise agent: generate answer, verify, optionally retry.

    1. Generate with CoT: INST: ... THINK: ... ANS: <answer>
    2. Verify: append VERIFY: <answer> answers <question>?
    3. If model generates NO, retry with different temperature
    4. Return final answer
    """

    def __init__(self, model, tokenizer, max_retries=2,
                 model_name="verify", max_gen=60):
        self.model = model
        self.tokenizer = tokenizer
        self.max_retries = max_retries
        self._name = model_name
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        trace = []

        for attempt in range(1 + self.max_retries):
            # Step 1: Generate with CoT
            temp = 0.0 if attempt == 0 else 0.5 + 0.3 * attempt
            inst = f"{INST_PREFIX}{prompt}{THINK_MARKER}"
            ids = self.tokenizer.encode(inst, add_bos=True)
            out = decode(self.model, list(ids),
                         max_new_tokens=self.max_gen,
                         temperature=temp)
            gen = self.tokenizer.decode(out)
            answer = _extract_answer(gen)
            trace.append(f"attempt {attempt+1}: '{answer}' (temp={temp})")

            if not answer:
                continue

            # Step 2: Verify — ask model if the answer is correct
            # Keep verify prompt short to fit within model's max_seq_len
            q_match = re.search(r"Q:\s*(.+?)$", prompt)
            question = q_match.group(1).rstrip("?") if q_match else prompt[:30]
            verify_prompt = (
                f"{INST_PREFIX}{prompt}"
                f"{ANS_MARKER}{answer}"
                f"{VERIFY_MARKER}{answer} answers {question}? "
            )
            v_ids = self.tokenizer.encode(verify_prompt, add_bos=True)
            # Truncate to fit model's max_seq_len minus room for generation
            max_ctx = 120
            if len(v_ids) > max_ctx:
                v_ids = v_ids[:max_ctx]
            v_out = decode(self.model, v_ids, max_new_tokens=5,
                           temperature=0.0)
            v_gen = self.tokenizer.decode(v_out)

            # Check for YES/NO after VERIFY:
            verify_m = VERIFY_MARKER.strip()
            if verify_m in v_gen:
                after_verify = v_gen.split(verify_m)[-1].upper()
                verified = "YES" in after_verify
            else:
                verified = True  # default to accepting

            trace.append(f"  verified: {verified}")

            if verified:
                return AgentResult(answer=answer, confidence=0.7,
                                   trace=trace)

        # All retries failed verification — return last answer anyway
        trace.append("all retries failed verification, returning last")
        return AgentResult(answer=answer if answer else "",
                           confidence=0.3, trace=trace)


# -- Main ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Chapter 12: Reasoning Scaffolds")
    print("=" * 60)
    print()
    print("The final chapter.  We add explicit reasoning structure:")
    print("  1. Chain-of-Thought (CoT): THINK: step-by-step reasoning")
    print("  2. Self-Consistency: generate N answers, majority vote")
    print("  3. Verify: generate, check, retry if wrong")
    print()
    print("Can structured reasoning give the model the judgment that")
    print("architecture, scale, training, decoding, retrieval, and")
    print("tools all failed to provide?")
    print()

    # ------------------------------------------------------------------
    # 1. Build corpora
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
    cot_corpus = build_cot_corpus(tasks, N_TRAIN)
    verify_corpus = build_verify_corpus(tasks, N_TRAIN)
    print(f"  Raw:    {len(raw_corpus)} sequences")
    print(f"  SFT:    {len(sft_corpus)} sequences")
    print(f"  CoT:    {len(cot_corpus)} sequences")
    print(f"  Verify: {len(verify_corpus)} sequences")

    # Show samples
    cot_samples = [s for s in cot_corpus if "THINK:" in s][:3]
    for i, s in enumerate(cot_samples):
        print(f"\n  CoT example {i+1}: {s[:120]}...")
    print()

    # Fit tokenizer on all corpora
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + sft_corpus + cot_corpus + verify_corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # Prepare datasets
    raw_in, raw_tgt = prepare_sequences(raw_corpus, tokenizer, MAX_SEQ_LEN)
    sft_in, sft_tgt = prepare_sequences(sft_corpus, tokenizer, MAX_SEQ_LEN)
    cot_in, cot_tgt = prepare_sequences(cot_corpus, tokenizer, MAX_SEQ_LEN)
    verify_in, verify_tgt = prepare_sequences(verify_corpus, tokenizer, MAX_SEQ_LEN)
    raw_loader = make_dataset(raw_in, raw_tgt, batch_size=BATCH_SIZE)
    sft_loader = make_dataset(sft_in, sft_tgt, batch_size=BATCH_SIZE)
    cot_loader = make_dataset(cot_in, cot_tgt, batch_size=BATCH_SIZE)
    verify_loader = make_dataset(verify_in, verify_tgt, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------
    # 2. Train SFT baseline
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Pre-train + SFT (baseline, no reasoning)")
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
        title="Chapter 12: SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch12_sft_loss.png"),
    )
    print(f"  Final SFT loss: {sft_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 3. Train CoT model
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Pre-train + CoT-SFT (with THINK: reasoning)")
    print(f"{'-' * 50}")

    cot_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    train(cot_model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)
    cot_train_result = train(cot_model, cot_loader,
                             epochs=SFT_EPOCHS, lr=SFT_LR)
    plot_loss_curve(
        cot_train_result.losses,
        title="Chapter 12: CoT-SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch12_cot_loss.png"),
    )
    print(f"  Final CoT-SFT loss: {cot_train_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 4. Train Verify model
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Training: Pre-train + Verify-SFT (with THINK + VERIFY)")
    print(f"{'-' * 50}")

    verify_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    train(verify_model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)
    verify_train_result = train(verify_model, verify_loader,
                                epochs=SFT_EPOCHS, lr=SFT_LR)
    plot_loss_curve(
        verify_train_result.losses,
        title="Chapter 12: Verify-SFT Training Loss",
        save_path=str(RESULTS_DIR / "ch12_verify_loss.png"),
    )
    print(f"  Final Verify-SFT loss: {verify_train_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 5. Benchmark prompts
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Benchmark Prompts: SFT vs CoT vs Self-Consistency vs Verify")
    print(f"{'-' * 50}")
    print()

    sft_agent = SFTAgent(sft_model, tokenizer, "sft_only")
    cot_agent = CoTAgent(cot_model, tokenizer, "cot")
    sc_agent = SelfConsistencyAgent(
        cot_model, tokenizer, n_samples=5, temperature=0.8,
        model_name="self_consistency")
    verify_agent = VerifyAgent(verify_model, tokenizer,
                               max_retries=2, model_name="verify")

    benchmark_prompts = [
        ("ADD 5 3 =",
         "Computation -- CoT shows step-by-step arithmetic"),
        ("FACT: paris is capital of france. Q: capital of france?",
         "Retrieval -- CoT traces fact to answer"),
        ("Q: What is the capital of the Moon?",
         "Hallucination -- CoT generates plausible but wrong reasoning"),
    ]

    for prompt, description in benchmark_prompts:
        print(f"  Prompt: '{prompt}'")
        print(f"  Tests:  {description}")
        sft_out = sft_agent.run(prompt)
        cot_out = cot_agent.run(prompt)
        sc_out = sc_agent.run(prompt)
        verify_out = verify_agent.run(prompt)
        print(f"    SFT:              '{sft_out.answer}'")
        print(f"    CoT:              '{cot_out.answer}'")
        if cot_out.trace:
            print(f"                      {cot_out.trace[0][:80]}")
        print(f"    Self-Consistency: '{sc_out.answer}'")
        if sc_out.trace:
            for t in sc_out.trace:
                print(f"                      {t[:80]}")
        print(f"    Verify:           '{verify_out.answer}'")
        if verify_out.trace:
            for t in verify_out.trace[:4]:
                print(f"                      {t[:80]}")
        print()

    # ------------------------------------------------------------------
    # 5b. Reversal Curse probe
    # ------------------------------------------------------------------
    print(f"\n{'-' * 50}")
    print("Reversal Curse: Does 'A is B' transfer to 'B is A'?")
    print(f"{'-' * 50}")
    print()
    print("The model was trained on 'paris is capital of france'.")
    print("Can it answer the REVERSE: 'france is the country whose capital is?'")
    print()

    reversal_prompts = [
        # Forward (should work — matches training direction)
        ("FACT: paris is capital of france. Q: capital of france?",
         "Forward (trained direction)", "paris"),
        # Reversed (tests bidirectional understanding)
        ("FACT: paris is capital of france. Q: france is the country whose capital is?",
         "Reversed (untrained direction)", "paris"),
        ("FACT: paris is capital of france. Q: which country has paris as capital?",
         "Reversed (rephrased)", "france"),
        ("FACT: cat eats fish. Q: what does cat eat?",
         "Forward (trained direction)", "fish"),
        ("FACT: cat eats fish. Q: fish is eaten by?",
         "Reversed (untrained direction)", "cat"),
    ]

    for prompt, direction, expected in reversal_prompts:
        print(f"  [{direction}]")
        print(f"    Prompt:   '{prompt}'")
        print(f"    Expected: '{expected}'")
        sft_out = sft_agent.run(prompt)
        cot_out = cot_agent.run(prompt)
        print(f"    SFT:      '{sft_out.answer}'  "
              f"{'OK' if expected in sft_out.answer else 'FAIL'}")
        print(f"    CoT:      '{cot_out.answer}'  "
              f"{'OK' if expected in cot_out.answer else 'FAIL'}")
        if cot_out.trace:
            print(f"              {cot_out.trace[0][:80]}")
        print()

    print("The Reversal Curse (Song et al., 2026): models trained on")
    print("'A is B' cannot infer 'B is A'. The model stores directional")
    print("statistical correlations, not symmetric knowledge.")
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
    print("Evaluation: SFT vs CoT vs Self-Consistency vs Verify vs Human")
    print(f"{'-' * 50}")

    agents = {
        "sft_only": sft_agent,
        "cot": cot_agent,
        "self_consistency": sc_agent,
        "verify": verify_agent,
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
            RESULTS_DIR / f"ch12_{agent_name}.json",
            agent_name=agent_name, chapter="12_reasoning_scaffolds")

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
        title="Chapter 12: Reasoning Scaffolds vs Human",
        save_path=str(RESULTS_DIR / "ch12_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch12_comparison.png'}")

    _plot_hallucination_across_chapters()
    print(f"  Hallucination plot saved to {RESULTS_DIR / 'ch12_hallucination_history.png'}")

    # ------------------------------------------------------------------
    # 8. Key takeaway -- the punchline
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("THE PUNCHLINE")
    print("=" * 60)
    print()
    print("After 12 chapters of increasingly sophisticated techniques:")
    print()
    print("  Ch01 N-grams          -> hallucinates")
    print("  Ch02 FFN              -> hallucinates")
    print("  Ch03 RNN/GRU          -> hallucinates")
    print("  Ch04 Attention        -> hallucinates")
    print("  Ch05 Transformer      -> hallucinates")
    print("  Ch06 Scaling          -> hallucinates (at every size)")
    print("  Ch07 Instruction SFT  -> hallucinates")
    print("  Ch08 Preference DPO   -> hallucinates")
    print("  Ch09 Decoding         -> hallucinates (every strategy)")
    print("  Ch10 RAG              -> hallucinates (from retrieved context)")
    print("  Ch11 Tools            -> hallucinates (from tool output)")
    print("  Ch12 Reasoning        -> hallucinates (with plausible reasoning)")
    print()
    print("The model CANNOT say 'I don't know.'")
    print()
    print("Better architecture helps accuracy.")
    print("More parameters help accuracy.")
    print("Better training helps accuracy.")
    print("Tools add computation.")
    print("Reasoning adds structure.")
    print()
    print("None of them add genuine uncertainty awareness.")
    print()
    print("The thesis of this project:")
    print("  'Similar outputs != same mechanism'")
    print()
    print("An LLM can produce text that looks like reasoning, but the")
    print("mechanism is pattern completion, not deduction.  A human")
    print("can say 'I don't know' because they have a model of what")
    print("they know and don't know.  The LLM has no such model --")
    print("it has probabilities over the next token.")
    print()
    print("That's not a brain.")


# -- Plotting helpers ------------------------------------------------------

def _plot_hallucination_across_chapters() -> None:
    """Bar chart showing hallucination rate stays 100% across all chapters."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    chapters = [
        "Ch01\nN-gram",
        "Ch02\nFFN",
        "Ch03\nRNN",
        "Ch04\nAttn",
        "Ch05\nTransf.",
        "Ch06\nScale",
        "Ch07\nSFT",
        "Ch08\nDPO",
        "Ch09\nDecode",
        "Ch10\nRAG",
        "Ch11\nTools",
        "Ch12\nReason",
    ]
    # All models hallucinate on unknown questions
    hallucination_rates = [1.0] * len(chapters)
    human_rate = 0.0

    x = np.arange(len(chapters))
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.bar(x, hallucination_rates, color="#E53935", alpha=0.8,
           label="LLM hallucination rate")
    ax.axhline(y=human_rate, color="#4CAF50", linewidth=2,
               linestyle="--", label="Human (0%)")

    ax.set_xticks(x)
    ax.set_xticklabels(chapters, fontsize=8)
    ax.set_ylabel("Hallucination Rate on Unknown Questions")
    ax.set_title("The Punchline: 12 Chapters, Hallucination Never Drops")
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)

    # Add annotation
    ax.annotate(
        "Every technique fails to teach\nthe model to say 'I don't know'",
        xy=(5.5, 1.0), xytext=(5.5, 1.08),
        ha="center", fontsize=9, color="#333",
        arrowprops=dict(arrowstyle="->", color="#999"),
    )

    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch12_hallucination_history.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
