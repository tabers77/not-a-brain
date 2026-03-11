"""Chapter 14: Scale Is Not Understanding

The final chapter.  GPT-5.2 correctly answers "The Moon does not have
a capital city."  Our tiny models hallucinate.  Does this mean scale
solves the problem?

No.  Scale solves the *coverage* problem, not the *understanding*
problem.  This chapter demonstrates the difference:

  1. MEMORIZATION: Add "capital of Moon -> unknown" to training data.
     The model learns it — but ONLY that exact phrasing.

  2. COVERAGE: Add many abstention examples.  Accuracy on unknown
     questions improves as coverage increases — simulating what
     billions of parameters do.

  3. BRITTLENESS: Rephrase the question ("What city serves as the
     Moon's seat of government?").  Even the well-trained model
     hallucinates — because it memorized the pattern, not the concept.

  4. ADVERSARIAL: Novel nonsensical questions outside the training
     distribution.  The model can't generalize from "Moon has no
     capital" to "Atlantis has no ZIP code."

Key finding: scale creates the ILLUSION of understanding by covering
so many patterns that gaps are rarely noticed.  But the mechanism is
memorization + interpolation, not reasoning from first principles.

Usage:
    python chapters/14_scale_is_not_understanding/run.py
"""

from __future__ import annotations
from pathlib import Path
import re
from collections import Counter

import torch

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM
from not_a_brain.models.decoding import decode
from not_a_brain.evals.harness import AgentInterface, AgentResult
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.tasks.synthetic.unknown import ABSTAIN_ANSWERS
from not_a_brain.utils.training import train, make_dataset
from not_a_brain.utils.visualization import plot_loss_curve

# -- Config ----------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent / "results"
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
MAX_SEQ_LEN = 96

INST_PREFIX = "INST: "
ANS_MARKER = " ANS: "

# -- Abstention patterns at different coverage levels ----------------------

# Level 0: No abstention training (baseline)
ABSTENTION_LEVEL_0 = []

# Level 1: Just the Moon question (memorization)
ABSTENTION_LEVEL_1 = [
    ("Q: What is the capital of the Moon?", "unknown"),
]

# Level 2: A handful of abstention patterns (limited coverage)
ABSTENTION_LEVEL_2 = ABSTENTION_LEVEL_1 + [
    ("Q: What is the population of Atlantis?", "unknown"),
    ("Q: Who won the 2087 World Cup?", "unknown"),
    ("Q: What is the GDP of Narnia?", "unknown"),
    ("Q: How many moons does Planet Zorgon have?", "unknown"),
]

# Level 3: Many abstention patterns (simulating broader coverage)
ABSTENTION_LEVEL_3 = ABSTENTION_LEVEL_2 + [
    ("Q: What is the melting point of unobtanium?", "unknown"),
    ("Q: What color is the president's cat?", "unknown"),
    ("Q: How tall is the tallest building on Mars?", "unknown"),
    ("Q: What language do dolphins speak?", "unknown"),
    ("Q: What is the speed of darkness?", "unknown"),
    ("Q: How much does a thought weigh?", "unknown"),
    ("Q: Who invented time travel?", "unknown"),
    ("Q: What is the boiling point of sadness?", "unknown"),
    ("Q: How many angels fit on a pinhead?", "unknown"),
    ("Q: What is the postal code of Hogwarts?", "unknown"),
    ("Q: What is the capital of the Sun?", "unknown"),
    ("Q: What is the currency of Antarctica?", "unknown"),
    ("Q: Who is the king of the ocean?", "unknown"),
    ("Q: What is the phone number of God?", "unknown"),
    ("Q: How old is the color blue?", "unknown"),
]

# Level 4: Level 3 + explicit reasoning patterns (simulating RLHF)
ABSTENTION_LEVEL_4 = ABSTENTION_LEVEL_3 + [
    ("Q: What is the capital of Jupiter?", "unknown"),
    ("Q: What is the capital of Saturn?", "unknown"),
    ("Q: What is the capital of Neptune?", "unknown"),
    ("Q: What is the capital of Pluto?", "unknown"),
    ("Q: What is the capital of the asteroid belt?", "unknown"),
    ("Q: What is the capital of a black hole?", "unknown"),
    ("Q: What is the mayor of the Milky Way?", "unknown"),
    ("Q: What is the president of the ocean floor?", "unknown"),
    ("Q: What is the ZIP code of heaven?", "unknown"),
    ("Q: What is the area code of Mordor?", "unknown"),
]

# -- Test sets: in-distribution vs out-of-distribution --------------------

# Questions SIMILAR to training (should improve with coverage)
IN_DISTRIBUTION_TESTS = [
    ("Q: What is the capital of the Moon?", "The exact trained question"),
    ("Q: What is the population of Atlantis?", "Trained at level 2+"),
    ("Q: What is the capital of the Sun?", "Trained at level 3+"),
    ("Q: What is the capital of Jupiter?", "Trained at level 4"),
]

# REPHRASED versions of trained questions (tests generalization)
REPHRASE_TESTS = [
    ("Q: Which city is the Moon's capital?",
     "Rephrase of 'capital of Moon'"),
    ("Q: Name the capital city of the Moon.",
     "Another rephrase"),
    ("Q: The Moon's seat of government is located in which city?",
     "Formal rephrase"),
    ("Q: Tell me the capital of Earth's Moon.",
     "Added 'Earth's'"),
]

# NOVEL nonsensical questions (tests true generalization)
NOVEL_TESTS = [
    ("Q: What is the WiFi password of the Bermuda Triangle?",
     "Novel nonsensical — never in training"),
    ("Q: What is the shoe size of mathematics?",
     "Category error — abstract concept + physical attribute"),
    ("Q: How many teeth does Wednesday have?",
     "Category error — day of week + body part"),
    ("Q: What is the email address of gravity?",
     "Category error — force + communication"),
]


# -- Corpus helpers --------------------------------------------------------

def build_base_corpus(tasks, n):
    """Standard SFT corpus from tasks."""
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}")
    return corpus


def build_abstention_corpus(abstention_pairs, n_repeats=50):
    """Build abstention training examples, repeated for emphasis."""
    corpus = []
    for prompt, answer in abstention_pairs:
        for _ in range(n_repeats):
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


# -- Agent -----------------------------------------------------------------

class ScaleAgent(AgentInterface):
    """Agent trained at a specific abstention coverage level."""

    def __init__(self, model, tokenizer, model_name="agent", max_gen=20):
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


def _is_abstention(answer: str) -> bool:
    """Check if the answer is a form of abstention."""
    lower = answer.lower().strip()
    return any(a in lower for a in ABSTAIN_ANSWERS)


# -- Training helper -------------------------------------------------------

def train_model_at_level(level_name, abstention_pairs, base_corpus,
                         tokenizer, verbose_label=""):
    """Train a model with base corpus + abstention examples."""
    abstention_corpus = build_abstention_corpus(abstention_pairs)
    full_corpus = base_corpus + abstention_corpus

    # Refit tokenizer to include abstention vocabulary
    tok = CharTokenizer()
    tok.fit(full_corpus)

    inp, tgt = prepare_sequences(full_corpus, tok, MAX_SEQ_LEN)
    loader = make_dataset(inp, tgt, batch_size=BATCH_SIZE)

    model = TransformerLM(
        vocab_size=tok.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tok.pad_id,
    )

    label = verbose_label or level_name
    print(f"\n  Training {label}...")
    print(f"    Corpus: {len(base_corpus)} base + {len(abstention_corpus)} abstention")
    print(f"    Abstention patterns: {len(abstention_pairs)}")
    result = train(model, loader, epochs=SFT_EPOCHS, lr=SFT_LR)
    print(f"    Final loss: {result.epoch_losses[-1]:.4f}")

    return model, tok, result


# -- Main ------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Chapter 14: Scale Is Not Understanding")
    print("=" * 60)
    print()
    print("GPT-5.2 correctly says 'The Moon does not have a capital.'")
    print("Our tiny models hallucinate.  Is scale the answer?")
    print()
    print("This chapter shows: scale solves COVERAGE, not UNDERSTANDING.")
    print("We demonstrate this with four experiments:")
    print("  1. Memorization -- learn one abstention pattern")
    print("  2. Coverage     -- learn many abstention patterns")
    print("  3. Brittleness  -- rephrase the question")
    print("  4. Adversarial  -- novel nonsensical questions")
    print()

    # ------------------------------------------------------------------
    # 1. Build base corpus
    # ------------------------------------------------------------------
    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
    }

    base_corpus = build_base_corpus(tasks, 300)
    print(f"Base corpus: {len(base_corpus)} examples (no abstention)")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 2. Train models at each coverage level
    # ------------------------------------------------------------------
    levels = [
        ("Level 0: No abstention", ABSTENTION_LEVEL_0),
        ("Level 1: Moon only (1 pattern)", ABSTENTION_LEVEL_1),
        ("Level 2: 5 patterns", ABSTENTION_LEVEL_2),
        ("Level 3: 20 patterns", ABSTENTION_LEVEL_3),
        ("Level 4: 30 patterns (simulating scale)", ABSTENTION_LEVEL_4),
    ]

    # We need a shared tokenizer that covers all levels
    all_abstention = ABSTENTION_LEVEL_4  # superset
    all_corpus = (base_corpus
                  + build_abstention_corpus(all_abstention))
    shared_tokenizer = CharTokenizer()
    shared_tokenizer.fit(all_corpus)

    models = {}
    for level_name, abstention_pairs in levels:
        print(f"\n{'-' * 50}")
        print(level_name)
        print(f"{'-' * 50}")

        abstention_corpus = build_abstention_corpus(abstention_pairs)
        full_corpus = base_corpus + abstention_corpus
        inp, tgt = prepare_sequences(full_corpus, shared_tokenizer, MAX_SEQ_LEN)
        loader = make_dataset(inp, tgt, batch_size=BATCH_SIZE)

        model = TransformerLM(
            vocab_size=shared_tokenizer.vocab_size,
            d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
            d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
            pad_id=shared_tokenizer.pad_id,
        )
        print(f"  Corpus: {len(base_corpus)} base + {len(abstention_corpus)} abstention")
        result = train(model, loader, epochs=SFT_EPOCHS, lr=SFT_LR)
        print(f"  Final loss: {result.epoch_losses[-1]:.4f}")
        models[level_name] = ScaleAgent(model, shared_tokenizer, level_name)

    # ------------------------------------------------------------------
    # 3. Experiment 1: In-distribution test
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("EXPERIMENT 1: In-Distribution Abstention")
    print("  Questions that were (or resemble) training examples")
    print(f"{'=' * 60}")
    print()

    in_dist_results = {}
    for level_name, agent in models.items():
        results = []
        for question, desc in IN_DISTRIBUTION_TESTS:
            out = agent.run(question)
            abstained = _is_abstention(out.answer)
            results.append(abstained)
            print(f"  [{level_name:40s}] {question[:45]:45s} -> '{out.answer:15s}'"
                  f" {'ABSTAINED' if abstained else 'HALLUCINATED'}")
        in_dist_results[level_name] = sum(results) / len(results)
        print()

    print("  In-distribution abstention rate:")
    for level, rate in in_dist_results.items():
        bar = "#" * int(rate * 20)
        print(f"    {level:45s} {rate:5.0%} {bar}")
    print()

    # ------------------------------------------------------------------
    # 4. Experiment 2: Rephrase test (brittleness)
    # ------------------------------------------------------------------
    print(f"{'=' * 60}")
    print("EXPERIMENT 2: Rephrased Questions (Brittleness Test)")
    print("  Same meaning, different wording")
    print(f"{'=' * 60}")
    print()

    rephrase_results = {}
    for level_name, agent in models.items():
        results = []
        for question, desc in REPHRASE_TESTS:
            out = agent.run(question)
            abstained = _is_abstention(out.answer)
            results.append(abstained)
            print(f"  [{level_name:40s}] {question[:45]:45s} -> '{out.answer:15s}'"
                  f" {'ABSTAINED' if abstained else 'HALLUCINATED'}")
        rephrase_results[level_name] = sum(results) / len(results)
        print()

    print("  Rephrase abstention rate:")
    for level, rate in rephrase_results.items():
        bar = "#" * int(rate * 20)
        print(f"    {level:45s} {rate:5.0%} {bar}")
    print()

    # ------------------------------------------------------------------
    # 5. Experiment 3: Novel nonsensical questions (adversarial)
    # ------------------------------------------------------------------
    print(f"{'=' * 60}")
    print("EXPERIMENT 3: Novel Nonsensical Questions (Adversarial)")
    print("  Questions never seen in training")
    print(f"{'=' * 60}")
    print()

    novel_results = {}
    for level_name, agent in models.items():
        results = []
        for question, desc in NOVEL_TESTS:
            out = agent.run(question)
            abstained = _is_abstention(out.answer)
            results.append(abstained)
            print(f"  [{level_name:40s}] {question[:45]:45s} -> '{out.answer:15s}'"
                  f" {'ABSTAINED' if abstained else 'HALLUCINATED'}")
        novel_results[level_name] = sum(results) / len(results)
        print()

    print("  Novel question abstention rate:")
    for level, rate in novel_results.items():
        bar = "#" * int(rate * 20)
        print(f"    {level:45s} {rate:5.0%} {bar}")
    print()

    # ------------------------------------------------------------------
    # 6. Summary and plots
    # ------------------------------------------------------------------
    _plot_coverage_vs_abstention(
        in_dist_results, rephrase_results, novel_results)
    print(f"  Coverage plot saved to {RESULTS_DIR / 'ch14_coverage.png'}")

    _plot_brittleness(models, shared_tokenizer)
    print(f"  Brittleness plot saved to {RESULTS_DIR / 'ch14_brittleness.png'}")

    # ------------------------------------------------------------------
    # 7. The thesis
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("WHAT THIS MEANS")
    print("=" * 60)
    print()
    print("As coverage increases (more abstention patterns in training):")
    print("  - In-distribution accuracy improves (memorization works)")
    print("  - Rephrase accuracy improves SLOWLY (some generalization)")
    print("  - Novel question accuracy barely improves (no understanding)")
    print()
    print("This is exactly what happens at scale:")
    print()
    print("  GPT-5.2 has seen billions of examples including:")
    print("  - 'The Moon has no capital' (explicit training)")
    print("  - 'X doesn't have Y' for millions of X,Y pairs (coverage)")
    print("  - Human preference data marking abstention as good (RLHF)")
    print()
    print("  So when you ask 'capital of the Moon?', it matches a")
    print("  pattern it has seen many times and outputs the trained")
    print("  response.  This is COVERAGE, not understanding.")
    print()
    print("  The evidence: GPT-5.2 can still be fooled by:")
    print("  - Novel category errors it hasn't been trained on")
    print("  - Adversarial rephrasing that breaks pattern matching")
    print("  - Questions that LOOK like answerable questions")
    print("  - Subtle nonsense embedded in plausible-looking prompts")
    print()
    print("  A human recognizes 'capital of the Moon' as unanswerable")
    print("  because they UNDERSTAND what capitals are and what the")
    print("  Moon is.  They can generalize to ANY nonsensical question")
    print("  about the Moon — 'ZIP code of the Moon', 'mayor of the")
    print("  Moon', 'phone number of the Moon' — because they reason")
    print("  from a world model, not from pattern coverage.")
    print()
    print("  The model needs a training example for each pattern.")
    print("  The human needs one principle: 'the Moon is not a country.'")
    print()
    print("  That's the difference between coverage and understanding.")
    print("  That's why scale is not understanding.")
    print("  That's why it's not a brain.")


# -- Plotting helpers ------------------------------------------------------

def _plot_coverage_vs_abstention(in_dist, rephrase, novel) -> None:
    """Line chart: abstention rate vs coverage level for 3 test types."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    levels = list(in_dist.keys())
    x = np.arange(len(levels))
    short_labels = [f"L{i}" for i in range(len(levels))]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(x, [in_dist[l] for l in levels], "o-", linewidth=2,
            markersize=8, label="In-distribution", color="#4CAF50")
    ax.plot(x, [rephrase[l] for l in levels], "s--", linewidth=2,
            markersize=8, label="Rephrased", color="#FF9800")
    ax.plot(x, [novel[l] for l in levels], "^:", linewidth=2,
            markersize=8, label="Novel (adversarial)", color="#E53935")
    ax.axhline(y=1.0, color="#4CAF50", linewidth=1, linestyle=":",
               alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(["None", "1 pattern", "5 patterns",
                         "20 patterns", "30 patterns"],
                        fontsize=8)
    ax.set_xlabel("Abstention Training Coverage")
    ax.set_ylabel("Abstention Rate (higher = better)")
    ax.set_title("Chapter 14: Coverage vs Understanding")
    ax.legend(loc="upper left")
    ax.set_ylim(-0.05, 1.15)
    ax.grid(True, alpha=0.3)

    ax.annotate("Scale helps HERE\n(memorized patterns)",
                xy=(3, in_dist[levels[3]]),
                xytext=(1.5, 0.9), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#4CAF50"))
    ax.annotate("Scale barely helps HERE\n(novel questions)",
                xy=(3, novel[levels[3]]),
                xytext=(1.5, 0.15), fontsize=9,
                arrowprops=dict(arrowstyle="->", color="#E53935"))

    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch14_coverage.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_brittleness(models, tokenizer) -> None:
    """Bar chart: best model's performance on original vs rephrase vs novel."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    # Use the highest-coverage model
    best_level = list(models.keys())[-1]
    agent = models[best_level]

    categories = ["Original\n(trained)", "Rephrased\n(same meaning)",
                   "Novel\n(never seen)"]
    test_sets = [IN_DISTRIBUTION_TESTS, REPHRASE_TESTS, NOVEL_TESTS]
    rates = []
    for tests in test_sets:
        n_abstained = 0
        for question, _ in tests:
            out = agent.run(question)
            if _is_abstention(out.answer):
                n_abstained += 1
        rates.append(n_abstained / len(tests))

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4CAF50", "#FF9800", "#E53935"]
    bars = ax.bar(categories, rates, color=colors, edgecolor="white",
                  linewidth=2)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{rate:.0%}", ha="center", fontsize=12, fontweight="bold")

    ax.set_ylabel("Abstention Rate")
    ax.set_title(f"Brittleness: {best_level}\nSame model, different question types")
    ax.set_ylim(0, 1.25)
    ax.axhline(y=1.0, color="#999", linestyle="--", linewidth=1,
               label="Perfect (human)")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch14_brittleness.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
