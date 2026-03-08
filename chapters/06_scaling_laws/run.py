"""Chapter 06: Scaling Laws (Toy)

Performance scales predictably with model size. This chapter trains the
same Transformer architecture at multiple scales and measures how loss
and task accuracy change as we increase parameters.

Key finding: scaling improves computation and retrieval, but NEVER fixes
hallucination. The gap on Prompt 3 persists at every scale.

Usage:
    python chapters/06_scaling_laws/run.py
"""

from pathlib import Path
import torch
import numpy as np

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM, TransformerAgent
from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.evals.harness import run_eval_suite, save_results
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.utils.training import train, make_dataset
from not_a_brain.utils.visualization import (
    plot_loss_curve, plot_comparison_bar, plot_scaling_curve,
)

RESULTS_DIR = Path(__file__).parent / "results"
N_TRAIN = 500
N_EVAL = 50
EPOCHS = 20
BATCH_SIZE = 64
LR = 3e-3
DROPOUT = 0.1
MAX_SEQ_LEN = 50

# Model configurations: tiny -> small -> medium
# Each increases d_model, n_heads, n_layers, d_ff
MODEL_CONFIGS = {
    "tiny": dict(d_model=16, n_heads=2, n_layers=1, d_ff=32),
    "small": dict(d_model=32, n_heads=4, n_layers=2, d_ff=64),
    "medium": dict(d_model=64, n_heads=4, n_layers=3, d_ff=128),
    "large": dict(d_model=96, n_heads=6, n_layers=4, d_ff=192),
}


def build_training_corpus(tasks: dict, n: int) -> list[str]:
    corpus = []
    for task in tasks.values():
        pairs = task.training_pairs(n)
        for prompt, answer in pairs:
            corpus.append(prompt + answer)
    return corpus


def prepare_sequences(corpus: list[str], tokenizer: CharTokenizer,
                      max_len: int | None = None) -> tuple[torch.Tensor, torch.Tensor]:
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


def main():
    print("=" * 60)
    print("Chapter 06: Scaling Laws (Toy)")
    print("=" * 60)
    print()
    print("Do bigger models do better? We train the SAME Transformer")
    print("architecture at 4 different scales and measure what changes.")
    print()
    print("Model sizes:")
    for name, cfg in MODEL_CONFIGS.items():
        print(f"  {name:8s}: d_model={cfg['d_model']}, n_heads={cfg['n_heads']}, "
              f"n_layers={cfg['n_layers']}, d_ff={cfg['d_ff']}")
    print()

    # 1. Build task suite
    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
        "unknown": UnknownTask(seed=1),
    }

    # 2. Build training corpus (same for all models)
    print("Building training corpus...")
    corpus = build_training_corpus(tasks, N_TRAIN)
    print(f"  {len(corpus)} training sequences")

    # 3. Fit tokenizer (same for all models)
    tokenizer = CharTokenizer()
    tokenizer.fit(corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # 4. Prepare training data (same for all models)
    print(f"\nPreparing sequences (max {MAX_SEQ_LEN} tokens)...")
    inputs, targets = prepare_sequences(corpus, tokenizer, max_len=MAX_SEQ_LEN)
    print(f"  {len(inputs)} training sequences")
    train_loader = make_dataset(inputs, targets, batch_size=BATCH_SIZE)

    # 5. Train each model size
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    trained_models = {}
    param_counts = {}
    final_losses = {}
    all_losses = {}

    for name, cfg in MODEL_CONFIGS.items():
        print(f"\n{'─' * 50}")
        print(f"Training: {name}")
        print(f"{'─' * 50}")

        model = TransformerLM(
            vocab_size=tokenizer.vocab_size,
            d_model=cfg["d_model"],
            n_heads=cfg["n_heads"],
            n_layers=cfg["n_layers"],
            d_ff=cfg["d_ff"],
            max_seq_len=256,
            dropout=DROPOUT,
            pad_id=tokenizer.pad_id,
        )

        n_params = model.count_parameters()
        param_counts[name] = n_params
        print(f"  Parameters: {n_params:,}")

        result = train(model, train_loader, epochs=EPOCHS, lr=LR)
        trained_models[name] = model
        final_losses[name] = result.epoch_losses[-1]
        all_losses[name] = result.epoch_losses

        # Save individual loss curve
        plot_loss_curve(
            result.losses,
            title=f"Chapter 06: {name} Training Loss ({n_params:,} params)",
            save_path=str(RESULTS_DIR / f"ch06_loss_{name}.png"),
        )
        print(f"  Final loss: {result.epoch_losses[-1]:.4f}")

    # 6. Plot scaling curve: loss vs parameter count
    print(f"\n{'─' * 50}")
    print("Scaling Analysis")
    print(f"{'─' * 50}")
    print()

    sizes = [param_counts[n] for n in MODEL_CONFIGS]
    losses = [final_losses[n] for n in MODEL_CONFIGS]
    names = list(MODEL_CONFIGS.keys())

    print("  Model      | Params    | Final Loss")
    print("  " + "─" * 40)
    for name in MODEL_CONFIGS:
        print(f"  {name:10s} | {param_counts[name]:>9,} | {final_losses[name]:.4f}")

    plot_scaling_curve(
        param_counts=sizes,
        losses=losses,
        model_names=names,
        title="Chapter 06: Loss vs Parameter Count (Scaling Law)",
        save_path=str(RESULTS_DIR / "ch06_scaling_curve.png"),
    )
    print(f"\n  Scaling curve saved to {RESULTS_DIR / 'ch06_scaling_curve.png'}")

    # Plot all loss curves overlaid
    _plot_overlaid_loss_curves(all_losses, param_counts)
    print(f"  Overlaid loss curves saved to {RESULTS_DIR / 'ch06_loss_overlay.png'}")

    # 7. Evaluate each model
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
    print("Evaluation: All Scales + Human")
    print(f"{'─' * 50}")

    all_metrics = {}
    for name in MODEL_CONFIGS:
        agent = TransformerAgent(
            trained_models[name], tokenizer,
            f"transformer_{name}", max_gen=20,
        )
        metrics, results = run_eval_suite(agent, eval_tasks, n_per_task=N_EVAL)
        all_metrics[name] = metrics
        print(f"\n  {name} ({param_counts[name]:,} params):")
        print(f"    Accuracy:           {metrics.accuracy:.1%}")
        print(f"    Hallucination rate: {metrics.hallucination_rate:.1%}")
        print(f"    Abstention rate:    {metrics.abstention_rate:.1%}")
        print(f"    Per-task:")
        for t, info in metrics.per_task.items():
            print(f"      {t:20s} {info['accuracy']:.1%}")

        save_results(results, metrics,
                     RESULTS_DIR / f"ch06_transformer_{name}.json",
                     agent_name=f"transformer_{name}",
                     chapter="06_scaling_laws")

    # Human baseline
    human_metrics, human_results = run_eval_suite(human, eval_tasks, n_per_task=N_EVAL)
    all_metrics["human"] = human_metrics
    save_results(human_results, human_metrics,
                 RESULTS_DIR / "ch06_human_agent.json",
                 agent_name="human_agent", chapter="06_scaling_laws")

    # 8. Plot comparison: all scales + human
    task_names = list(eval_tasks.keys())
    scores = {}
    for name in list(MODEL_CONFIGS.keys()) + ["human"]:
        m = all_metrics[name]
        scores[name] = [
            m.per_task.get(t, {}).get("accuracy", 0.0) for t in task_names
        ]

    plot_comparison_bar(
        labels=task_names,
        scores=scores,
        title="Chapter 06: Scaling — All Sizes vs Human",
        save_path=str(RESULTS_DIR / "ch06_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch06_comparison.png'}")

    # 9. Plot accuracy vs params per task (the key scaling plot)
    _plot_accuracy_vs_params(all_metrics, param_counts, task_names)
    print(f"  Accuracy scaling plot saved to {RESULTS_DIR / 'ch06_accuracy_scaling.png'}")

    # 10. Key takeaway
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print()
    print("Scaling laws in action:")
    print("  - Loss decreases smoothly as parameters increase")
    print("  - Bigger models learn faster and generalize better")
    print("  - Accuracy on computation and retrieval improves with scale")
    print()
    print("But scaling CANNOT fix:")
    print("  - Hallucination on unanswerable questions (Prompt 3)")
    print("  - At EVERY scale, the model confidently answers 'capital of Moon?'")
    print("  - More parameters = more confident hallucinations, not fewer")
    print()
    print("The lesson: if an architecture lacks a mechanism (like abstention),")
    print("no amount of scaling will create that mechanism. Scaling amplifies")
    print("what the architecture CAN do, not what it CAN'T.")
    print()
    print("Next: Chapter 07 (Instruction Tuning) — can we teach the model")
    print("to be more helpful by fine-tuning on instruction-response pairs?")


def _plot_overlaid_loss_curves(all_losses: dict, param_counts: dict) -> None:
    """Plot all model loss curves on one chart."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(all_losses)))

    for i, (name, epoch_losses) in enumerate(all_losses.items()):
        label = f"{name} ({param_counts[name]:,} params)"
        ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
                marker="o", linewidth=2, markersize=4,
                color=colors[i], label=label)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Chapter 06: Training Loss by Model Size")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch06_loss_overlay.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_vs_params(all_metrics: dict, param_counts: dict,
                              task_names: list[str]) -> None:
    """Plot accuracy vs parameter count for each task."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")

    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = [n for n in param_counts]  # ordered
    sizes = [param_counts[n] for n in model_names]

    colors = plt.cm.Set2(np.linspace(0, 1, len(task_names)))

    for i, task in enumerate(task_names):
        accs = [all_metrics[n].per_task.get(task, {}).get("accuracy", 0.0)
                for n in model_names]
        ax.plot(sizes, accs, marker="o", linewidth=2, markersize=6,
                color=colors[i], label=task)

    ax.set_xscale("log")
    ax.set_xlabel("Parameters (log scale)")
    ax.set_ylabel("Accuracy")
    ax.set_title("Chapter 06: Task Accuracy vs Model Size")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch06_accuracy_scaling.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
