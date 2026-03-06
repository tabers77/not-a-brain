"""Chapter 02: Feed-Forward Language Model (MLP)

First neural language model. Fixed-window MLP: embed the last W characters,
pass through hidden layers, predict the next character.

Shows: learned embeddings generalize better than counting, but fixed window
still limits long-range understanding.

Usage:
    python chapters/02_ffn_lm/run.py
"""

from pathlib import Path
import torch

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.ffn_lm import FFNLM, FFNAgent
from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.evals.harness import run_eval_suite, save_results
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.utils.training import train, make_dataset
from not_a_brain.utils.visualization import plot_loss_curve, plot_comparison_bar

RESULTS_DIR = Path(__file__).parent / "results"
N_TRAIN = 500
N_EVAL = 50
CONTEXT_WINDOW = 8
D_EMBED = 16
D_HIDDEN = 64
EPOCHS = 15
BATCH_SIZE = 64
LR = 3e-3


def build_training_corpus(tasks: dict, n: int) -> list[str]:
    """Build a training corpus from task prompt+answer pairs."""
    corpus = []
    for task in tasks.values():
        pairs = task.training_pairs(n)
        for prompt, answer in pairs:
            corpus.append(prompt + answer)
    return corpus


def prepare_sequences(corpus: list[str], tokenizer: CharTokenizer,
                      max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert corpus into padded (input, target) sequence pairs.

    Input is the full sequence (model handles windowing internally).
    Target is the sequence shifted by one position.
    """
    encoded = []
    for text in corpus:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        encoded.append(ids)

    if max_len is None:
        max_len = max(len(s) for s in encoded)

    # Pad and create input/target pairs (shift by 1)
    all_inputs = []
    all_targets = []
    for ids in encoded:
        # Pad to max_len
        padded = ids + [tokenizer.pad_id] * (max_len - len(ids))
        all_inputs.append(padded[:-1])   # everything except last
        all_targets.append(padded[1:])   # everything except first

    inputs = torch.tensor(all_inputs, dtype=torch.long)
    targets = torch.tensor(all_targets, dtype=torch.long)
    return inputs, targets


def main():
    print("=" * 60)
    print("Chapter 02: Feed-Forward Language Model (MLP)")
    print("=" * 60)
    print()
    print("A fixed-window MLP: embed the last W characters, concatenate,")
    print("pass through a hidden layer, and predict the next character.")
    print()
    print(f"  Context window (W): {CONTEXT_WINDOW}")
    print(f"  Embedding dim:      {D_EMBED}")
    print(f"  Hidden dim:         {D_HIDDEN}")
    print()
    print("Unlike n-grams, the MLP learns embeddings — similar characters")
    print("get similar representations, enabling generalization within the window.")
    print("See chapter.md for full formulas and explanations.")

    # 1. Build task suite
    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
        "unknown": UnknownTask(seed=1),
    }

    # 2. Build training corpus
    print("\nBuilding training corpus...")
    corpus = build_training_corpus(tasks, N_TRAIN)
    print(f"  {len(corpus)} training sequences")

    # 3. Fit tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # 4. Prepare training data
    print(f"\nPreparing training sequences (W={CONTEXT_WINDOW})...")
    inputs, targets = prepare_sequences(corpus, tokenizer)
    print(f"  {len(inputs)} training sequences")
    print(f"  Input shape:  {inputs.shape}  (batch, seq_len)")
    print(f"  Target shape: {targets.shape}  (batch, seq_len)")

    # 5. Build model
    model = FFNLM(
        vocab_size=tokenizer.vocab_size,
        context_window=CONTEXT_WINDOW,
        d_embed=D_EMBED,
        d_hidden=D_HIDDEN,
        pad_id=tokenizer.pad_id,
    )
    print(f"\nModel: FFNLM")
    print(f"  Parameters: {model.count_parameters():,}")

    # 6. Train
    print(f"\nTraining for {EPOCHS} epochs...")
    train_loader = make_dataset(inputs, targets, batch_size=BATCH_SIZE)
    result = train(model, train_loader, epochs=EPOCHS, lr=LR)

    # Save loss curve
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        result.losses,
        title="Chapter 02: FFN LM Training Loss",
        save_path=str(RESULTS_DIR / "ch02_loss_curve.png"),
    )
    print(f"\nLoss curve saved to {RESULTS_DIR / 'ch02_loss_curve.png'}")

    # 7. Show sample generations
    print("\n" + "-" * 50)
    print("Sample Generations")
    print("-" * 50)
    print(f"The MLP sees {CONTEXT_WINDOW} characters of context.")
    print("Compare with n-grams: embeddings help, but the window is still fixed.")
    print()

    ffn_agent = FFNAgent(model, tokenizer, "ffn_lm", max_gen=20)

    sample_prompts = [
        ("ADD 5 3 =", "Needs 5+3=8. Window captures '5 3 =' but model must learn addition."),
        ("COPY: abc|", "Needs 'abc'. If 'abc|' fits in window, model has a chance."),
        ("CHECK: ( )", "Bracket matching. Local pattern, should improve over n-grams."),
    ]
    for prompt, explanation in sample_prompts:
        result_gen = ffn_agent.run(prompt)
        print(f"  Prompt: '{prompt}'")
        print(f"  Why it's hard: {explanation}")
        print(f"    FFN LM: '{result_gen.answer}'")
        print()

    # 8. Evaluate all agents
    eval_tasks = {
        "arithmetic": ArithmeticTask(seed=99),
        "copy": CopyTask(seed=99),
        "grammar": GrammarTask(seed=99),
        "knowledge_qa": KnowledgeQATask(seed=99),
        "compositional": CompositionalTask(seed=99),
        "unknown": UnknownTask(seed=99),
    }

    human = HumanAgent()

    print("-" * 50)
    print("Evaluation")
    print("-" * 50)
    print("Running FFN LM and human agent on 50 samples per task.")
    print()

    agents = {
        "ffn_lm": ffn_agent,
        "human_agent": human,
    }

    all_metrics = {}
    for agent_name, agent in agents.items():
        metrics, results = run_eval_suite(agent, eval_tasks, n_per_task=N_EVAL)
        all_metrics[agent_name] = metrics
        print(f"  {agent_name}:")
        print(f"    Accuracy:           {metrics.accuracy:.1%}")
        print(f"    Hallucination rate: {metrics.hallucination_rate:.1%}")
        print(f"    Abstention rate:    {metrics.abstention_rate:.1%}")
        print(f"    Per-task:")
        for t, info in metrics.per_task.items():
            print(f"      {t:20s} {info['accuracy']:.1%}")
        print()

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        save_results(results, metrics,
                     RESULTS_DIR / f"ch02_{agent_name}.json",
                     agent_name=agent_name, chapter="02_ffn_lm")

    # 9. Plot comparison
    task_names = list(eval_tasks.keys())
    scores = {}
    for agent_name, metrics in all_metrics.items():
        scores[agent_name] = [
            metrics.per_task.get(t, {}).get("accuracy", 0.0)
            for t in task_names
        ]

    plot_comparison_bar(
        labels=task_names,
        scores=scores,
        title="Chapter 02: FFN LM vs Human Agent",
        save_path=str(RESULTS_DIR / "ch02_comparison.png"),
    )
    print(f"Plot saved to {RESULTS_DIR / 'ch02_comparison.png'}")

    # 10. Key takeaway
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print(f"The FFN LM ({model.count_parameters():,} parameters) learns from gradients")
    print(f"instead of counting. With a {CONTEXT_WINDOW}-character window, it can:")
    print("  - Learn embeddings that group similar characters")
    print("  - Capture patterns within its fixed window")
    print("  - Outperform n-grams on tasks with short-range structure")
    print()
    print("But it still fails at:")
    print("  - Long-range dependencies (prompt info beyond the window)")
    print("  - Variable-length reasoning (fixed window = fixed capacity)")
    print("  - Abstention (always generates, never says 'I don't know')")
    print()
    print("Human lens: Humans generalize rules; MLPs memorize windows.")
    print("A human who learns 5+3=8 can compute 50+30. The MLP can only")
    print("learn patterns it has seen within its W-character view.")


if __name__ == "__main__":
    main()
