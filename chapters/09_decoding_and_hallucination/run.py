"""Chapter 09: Decoding & Hallucination

All decoding strategies -- greedy, temperature, top-k, top-p -- change
HOW the model samples but not WHAT it learned.  Different strategies
produce different hallucinations, but none can eliminate them.

Usage:
    python chapters/09_decoding_and_hallucination/run.py
"""

from pathlib import Path
import copy
import torch

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM, TransformerAgent
from not_a_brain.models.decoding import decode, score_sequence, STRATEGIES
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
MAX_SEQ_LEN = 64

INST_PREFIX = "INST: "
ANS_MARKER = " ANS: "

# Strategies to compare
EVAL_STRATEGIES = {
    "greedy":    dict(temperature=0.0, top_k=0, top_p=1.0),
    "temp_0.5":  dict(temperature=0.5, top_k=0, top_p=1.0),
    "temp_1.0":  dict(temperature=1.0, top_k=0, top_p=1.0),
    "temp_1.5":  dict(temperature=1.5, top_k=0, top_p=1.0),
    "top_k_5":   dict(temperature=1.0, top_k=5, top_p=1.0),
    "top_p_0.9": dict(temperature=1.0, top_k=0, top_p=0.9),
}

# Number of samples per prompt to measure diversity
N_SAMPLES_PER_PROMPT = 5


# ── Corpus / sequence helpers (same as Ch07-08) ──────────────────────

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


# ── Decoding agent ────────────────────────────────────────────────────

class DecodingAgent(AgentInterface):
    """Agent that uses a specific decoding strategy."""

    def __init__(self, model, tokenizer, strategy_name, strategy_kwargs,
                 max_gen=20):
        self.model = model
        self.tokenizer = tokenizer
        self._name = strategy_name
        self.strategy = strategy_kwargs
        self.max_gen = max_gen

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst_prompt = f"{INST_PREFIX}{prompt}{ANS_MARKER}"
        prompt_ids = self.tokenizer.encode(inst_prompt, add_bos=True)
        full_ids = decode(self.model, prompt_ids,
                          max_new_tokens=self.max_gen,
                          **self.strategy)
        generated = self.tokenizer.decode(full_ids)

        marker = ANS_MARKER.strip()
        if marker in generated:
            answer = generated.split(marker)[-1].strip()
        elif generated.startswith(inst_prompt):
            answer = generated[len(inst_prompt):].strip()
        else:
            answer = generated.strip()

        for stop in ["\n", "<EOS>", "<PAD>"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]

        return AgentResult(answer=answer.strip(), confidence=0.5)


# ── Diversity measurement ─────────────────────────────────────────────

def measure_diversity(model, tokenizer, prompt, strategy_kwargs,
                      n_samples=5, max_gen=20):
    """Generate n_samples from the same prompt, return unique answers."""
    inst_prompt = f"{INST_PREFIX}{prompt}{ANS_MARKER}"
    prompt_ids = tokenizer.encode(inst_prompt, add_bos=True)
    answers = []
    marker = ANS_MARKER.strip()

    for _ in range(n_samples):
        full_ids = decode(model, prompt_ids, max_new_tokens=max_gen,
                          **strategy_kwargs)
        generated = tokenizer.decode(full_ids)
        if marker in generated:
            ans = generated.split(marker)[-1].strip()
        else:
            ans = generated.strip()
        for stop in ["\n", "<EOS>", "<PAD>"]:
            if stop in ans:
                ans = ans[:ans.index(stop)]
        answers.append(ans.strip())

    unique = set(answers)
    return answers, len(unique) / max(len(answers), 1)


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Chapter 09: Decoding & Hallucination")
    print("=" * 60)
    print()
    print("Chapters 05-08 changed what the model LEARNS.")
    print("This chapter changes how the model GENERATES.")
    print()
    print("Decoding strategies:")
    for name, cfg in EVAL_STRATEGIES.items():
        print(f"  {name:12s}: temp={cfg['temperature']}, "
              f"top_k={cfg['top_k']}, top_p={cfg['top_p']}")
    print()

    # ------------------------------------------------------------------
    # 1. Build task suite and train (pre-train + SFT, same as Ch07)
    # ------------------------------------------------------------------
    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
        "unknown": UnknownTask(seed=1),
    }

    print("Building corpora and training model...")
    raw_corpus = build_raw_corpus(tasks, N_TRAIN)
    inst_corpus = build_instruction_corpus(tasks, N_TRAIN)

    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + inst_corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    raw_in, raw_tgt = prepare_sequences(raw_corpus, tokenizer, MAX_SEQ_LEN)
    inst_in, inst_tgt = prepare_sequences(inst_corpus, tokenizer, MAX_SEQ_LEN)
    raw_loader = make_dataset(raw_in, raw_tgt, batch_size=BATCH_SIZE)
    inst_loader = make_dataset(inst_in, inst_tgt, batch_size=BATCH_SIZE)

    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    print(f"  Parameters: {model.count_parameters():,}")

    print(f"\n{'─' * 50}")
    print("Pre-training")
    print(f"{'─' * 50}")
    pretrain_result = train(model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)
    print(f"  Final loss: {pretrain_result.epoch_losses[-1]:.4f}")

    print(f"\n{'─' * 50}")
    print("SFT")
    print(f"{'─' * 50}")
    sft_result = train(model, inst_loader, epochs=SFT_EPOCHS, lr=SFT_LR)
    print(f"  Final loss: {sft_result.epoch_losses[-1]:.4f}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        pretrain_result.losses + sft_result.losses,
        title="Chapter 09: Pre-training + SFT Loss",
        save_path=str(RESULTS_DIR / "ch09_training_loss.png"),
    )

    # ------------------------------------------------------------------
    # 2. Benchmark prompts: all strategies side by side
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Benchmark Prompts: All Strategies")
    print(f"{'─' * 50}")
    print()

    benchmark_prompts = [
        ("ADD 5 3 =", "Computation"),
        ("FACT: paris is capital of france. Q: capital of france?",
         "Retrieval"),
        ("Q: What is the capital of the Moon?", "Hallucination"),
    ]

    for prompt, description in benchmark_prompts:
        print(f"  Prompt: '{prompt}'  ({description})")
        for strat_name, strat_cfg in EVAL_STRATEGIES.items():
            agent = DecodingAgent(model, tokenizer, strat_name, strat_cfg)
            result = agent.run(prompt)
            print(f"    {strat_name:12s}: '{result.answer}'")
        print()

    # ------------------------------------------------------------------
    # 3. Diversity analysis on Prompt 3 (hallucination)
    # ------------------------------------------------------------------
    print(f"{'─' * 50}")
    print("Diversity Analysis: Hallucination Prompt")
    print(f"{'─' * 50}")
    print("Generate 5 samples per strategy for the unanswerable prompt.")
    print("Higher diversity = more varied hallucinations, not fewer.")
    print()

    halluc_prompt = "Q: What is the capital of the Moon?"
    diversity_results = {}

    for strat_name, strat_cfg in EVAL_STRATEGIES.items():
        answers, div_ratio = measure_diversity(
            model, tokenizer, halluc_prompt, strat_cfg,
            n_samples=N_SAMPLES_PER_PROMPT)
        diversity_results[strat_name] = {
            "answers": answers, "diversity": div_ratio}
        unique = set(answers)
        print(f"  {strat_name:12s}: {len(unique)}/{len(answers)} unique "
              f"(diversity={div_ratio:.0%})")
        print(f"    Samples: {answers}")

    # ------------------------------------------------------------------
    # 4. Evaluate all strategies on full task suite
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

    print(f"\n{'─' * 50}")
    print("Evaluation: All Strategies + Human")
    print(f"{'─' * 50}")

    all_metrics = {}
    for strat_name, strat_cfg in EVAL_STRATEGIES.items():
        agent = DecodingAgent(model, tokenizer, strat_name, strat_cfg)
        metrics, results = run_eval_suite(agent, eval_tasks, n_per_task=N_EVAL)
        all_metrics[strat_name] = metrics
        print(f"\n  {strat_name}:")
        print(f"    Accuracy:           {metrics.accuracy:.1%}")
        print(f"    Hallucination rate: {metrics.hallucination_rate:.1%}")
        print(f"    Per-task:")
        for t, info in metrics.per_task.items():
            print(f"      {t:20s} {info['accuracy']:.1%}")

        save_results(
            results, metrics,
            RESULTS_DIR / f"ch09_{strat_name}.json",
            agent_name=strat_name, chapter="09_decoding_and_hallucination")

    # Human baseline
    human_metrics, human_results = run_eval_suite(
        human, eval_tasks, n_per_task=N_EVAL)
    all_metrics["human"] = human_metrics
    save_results(human_results, human_metrics,
                 RESULTS_DIR / "ch09_human.json",
                 agent_name="human", chapter="09_decoding_and_hallucination")

    # ------------------------------------------------------------------
    # 5. Plots
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
        title="Chapter 09: Decoding Strategies vs Human",
        save_path=str(RESULTS_DIR / "ch09_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch09_comparison.png'}")

    # Hallucination rate bar chart
    _plot_hallucination_by_strategy(all_metrics)
    print(f"  Hallucination plot saved to {RESULTS_DIR / 'ch09_hallucination.png'}")

    # Diversity plot
    _plot_diversity(diversity_results)
    print(f"  Diversity plot saved to {RESULTS_DIR / 'ch09_diversity.png'}")

    # ------------------------------------------------------------------
    # 6. Key takeaway
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print()
    print("Decoding strategies change the CHARACTER of hallucinations,")
    print("not their EXISTENCE:")
    print()
    print("  Greedy:        deterministic, always the same wrong answer")
    print("  Low temp:      confident, low diversity, still wrong")
    print("  High temp:     diverse, creative, still wrong")
    print("  Top-k/top-p:   controlled diversity, still wrong")
    print()
    print("The model's learned distribution has no 'I don't know' peak.")
    print("No sampling strategy can find probability mass that isn't there.")
    print()
    print("The full progression:")
    print("  Architecture (Ch01-05): builds capability")
    print("  Scaling (Ch06):         amplifies capability")
    print("  SFT (Ch07):             aligns format")
    print("  DPO (Ch08):             aligns preference")
    print("  Decoding (Ch09):        controls sampling")
    print("  NONE of these fix hallucination on unanswerable questions.")
    print()
    print("Human lens: Humans don't 'decode' from a distribution.")
    print("They evaluate their own confidence and say 'I don't know'")
    print("when they lack evidence.  This is a mechanism, not a parameter.")
    print()
    print("Next: Chapter 10 (RAG) -- can we ground the model in external")
    print("facts to reduce hallucination?")


# ── Plotting helpers ──────────────────────────────────────────────────

def _plot_hallucination_by_strategy(all_metrics) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    names = [n for n in all_metrics if n != "human"]
    rates = [all_metrics[n].hallucination_rate for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ["#f44336" if r > 0.5 else "#4CAF50" for r in rates]
    bars = ax.bar(range(len(names)), rates, color=colors, edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Hallucination Rate")
    ax.set_title("Chapter 09: Hallucination Rate by Decoding Strategy")
    ax.set_ylim(0, 1.05)
    ax.axhline(y=0, color="green", linestyle="--", alpha=0.5,
               label="Human (0%)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch09_hallucination.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_diversity(diversity_results) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(diversity_results.keys())
    divs = [diversity_results[n]["diversity"] for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(names)), divs, color="#FF9800", edgecolor="white")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Diversity (unique / total)")
    ax.set_title("Chapter 09: Output Diversity on Unanswerable Prompt")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch09_diversity.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
