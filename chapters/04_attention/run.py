"""Chapter 04: Attention Mechanism

The key insight that leads to the Transformer: instead of compressing
everything into a fixed state, let the model look back at any position
in the input directly.

Shows: attention heatmaps reveal what the model "looks at", copy task
improves because the model can attend to the characters it needs to reproduce.

Usage:
    python chapters/04_attention/run.py
"""

from pathlib import Path
import torch
import numpy as np

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.layers import AttentionLM, AttentionAgent
from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.evals.harness import run_eval_suite, save_results
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.utils.training import train, make_dataset
from not_a_brain.utils.visualization import (
    plot_loss_curve, plot_comparison_bar, plot_attention_heatmap,
)

RESULTS_DIR = Path(__file__).parent / "results"
N_TRAIN = 500
N_EVAL = 50
D_MODEL = 32
N_HEADS = 4
EPOCHS = 15
BATCH_SIZE = 64
LR = 3e-3
MAX_SEQ_LEN = 50


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


def visualize_attention(model: AttentionLM, tokenizer: CharTokenizer,
                        text: str, save_path: str, title: str) -> None:
    """Run a forward pass and save attention heatmaps for each head."""
    ids = tokenizer.encode(text, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        model(x)
    weights = model.get_attention_weights()  # (1, n_heads, S, S)
    if weights is None:
        return

    tokens = ["<BOS>"] + list(text)
    n_heads = weights.shape[1]

    # Plot average across heads
    avg_weights = weights[0].mean(dim=0).cpu().numpy()
    plot_attention_heatmap(
        avg_weights, tokens,
        title=f"{title} (avg across {n_heads} heads)",
        save_path=save_path,
    )

    # Plot each head separately
    base = Path(save_path)
    for h in range(n_heads):
        head_weights = weights[0, h].cpu().numpy()
        head_path = str(base.parent / f"{base.stem}_head{h}{base.suffix}")
        plot_attention_heatmap(
            head_weights, tokens,
            title=f"{title} — Head {h}",
            save_path=head_path,
        )


def main():
    print("=" * 60)
    print("Chapter 04: Attention Mechanism")
    print("=" * 60)
    print()
    print("The key idea: instead of compressing everything into a fixed")
    print("hidden state (like RNNs), let the model LOOK BACK at any")
    print("position in the input.")
    print()
    print("  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V")
    print()
    print(f"  d_model:  {D_MODEL}")
    print(f"  n_heads:  {N_HEADS}")
    print(f"  d_k/head: {D_MODEL // N_HEADS}")
    print()
    print("Each head learns different attention patterns — one might")
    print("focus on recent tokens, another on the '|' delimiter, etc.")
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
    print(f"\nPreparing sequences (max {MAX_SEQ_LEN} tokens)...")
    inputs, targets = prepare_sequences(corpus, tokenizer, max_len=MAX_SEQ_LEN)
    print(f"  {len(inputs)} training sequences")
    print(f"  Sequence length: {inputs.shape[1]}")

    train_loader = make_dataset(inputs, targets, batch_size=BATCH_SIZE)

    # 5. Train attention LM
    model = AttentionLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        max_seq_len=256,
        pad_id=tokenizer.pad_id,
    )
    print(f"\nModel: AttentionLM ({model.count_parameters():,} params)")
    print("Training...")
    result = train(model, train_loader, epochs=EPOCHS, lr=LR)

    # Save loss curve
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        result.losses,
        title="Chapter 04: Attention LM Training Loss",
        save_path=str(RESULTS_DIR / "ch04_loss_curve.png"),
    )
    print(f"\nLoss curve saved to {RESULTS_DIR / 'ch04_loss_curve.png'}")
    print(f"  Final loss: {result.epoch_losses[-1]:.4f}")

    # 6. Visualize attention on key examples
    print("\n" + "-" * 50)
    print("Attention Heatmaps")
    print("-" * 50)
    print("Saving attention heatmaps — these show WHERE the model looks")
    print("when predicting each character.")
    print()

    heatmap_examples = [
        ("COPY: hi|", "Copy task — does it attend to 'h' and 'i'?"),
        ("ADD 5 3 =", "Arithmetic — does it attend to the numbers?"),
        ("CHECK: ( )", "Grammar — does it attend to matching brackets?"),
    ]
    for text, description in heatmap_examples:
        safe_name = text.replace(" ", "_").replace(":", "").replace("|", "").replace("(", "").replace(")", "")
        save_path = str(RESULTS_DIR / f"ch04_attn_{safe_name}.png")
        visualize_attention(model, tokenizer, text, save_path,
                            title=f"Attention: '{text}'")
        print(f"  '{text}' -> {save_path}")
        print(f"    {description}")

    # 7. Sample generations
    print("\n" + "-" * 50)
    print("Sample Generations")
    print("-" * 50)
    print("Unlike the RNN, the attention model can look back at any")
    print("position. Watch for improvements on copy and knowledge QA.")
    print()

    attn_agent = AttentionAgent(model, tokenizer, "attention_lm", max_gen=20)

    sample_prompts = [
        ("ADD 5 3 =", "Can attend directly to '5' and '3', not just compressed state."),
        ("COPY: abc|", "Can attend to 'a', 'b', 'c' individually through the '|'."),
        ("CHECK: ( ( ) )", "Can match opening and closing brackets across distance."),
    ]
    for prompt, explanation in sample_prompts:
        result_gen = attn_agent.run(prompt)
        print(f"  Prompt: '{prompt}'")
        print(f"  Why attention helps: {explanation}")
        print(f"    Attention LM: '{result_gen.answer}'")
        print()

    # 8. Evaluate
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
    print("Running attention LM and human agent on 50 samples per task.")
    print()

    agents = {
        "attention_lm": attn_agent,
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

        save_results(results, metrics,
                     RESULTS_DIR / f"ch04_{agent_name}.json",
                     agent_name=agent_name, chapter="04_attention")

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
        title="Chapter 04: Attention LM vs Human Agent",
        save_path=str(RESULTS_DIR / "ch04_comparison.png"),
    )
    print(f"Plot saved to {RESULTS_DIR / 'ch04_comparison.png'}")

    # 10. Key takeaway
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print(f"The Attention LM ({model.count_parameters():,} params) can look back")
    print("at any position in the input — no fixed window, no lossy compression.")
    print()
    print("What attention enables:")
    print("  - Direct access to any input position (no compression)")
    print("  - Multiple heads learn different patterns simultaneously")
    print("  - Attention heatmaps show WHAT the model focuses on")
    print()
    print("What's still missing:")
    print("  - No feed-forward layers (can't compute complex functions)")
    print("  - No layer stacking (can't build compositional features)")
    print("  - No residual connections (deep networks would be hard to train)")
    print("  - Still can't abstain")
    print()
    print("Human lens: Humans attend guided by GOALS — 'I need the number")
    print("after ADD'. Attention is content-based SIMILARITY — positions")
    print("attend to each other if their embeddings align. Same mechanism,")
    print("different driver.")


if __name__ == "__main__":
    main()
