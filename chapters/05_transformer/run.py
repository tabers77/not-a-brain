"""Chapter 05: Transformer

The full architecture: multi-head attention + feed-forward layers + residual
connections + layer normalization, stacked into multiple layers.

This is the architecture behind GPT, BERT, and every modern LLM.

What's new vs Chapter 04 (attention-only):
    - Feed-forward layers add computation (attention can only retrieve/blend)
    - Residual connections enable depth without vanishing gradients
    - Layer normalization stabilizes training
    - Multiple layers enable compositional reasoning

Usage:
    python chapters/05_transformer/run.py
"""

from pathlib import Path
import torch

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
    plot_loss_curve, plot_comparison_bar, plot_attention_heatmap,
)

RESULTS_DIR = Path(__file__).parent / "results"
N_TRAIN = 500
N_EVAL = 50
D_MODEL = 64
N_HEADS = 4
N_LAYERS = 2
D_FF = 128
EPOCHS = 20
BATCH_SIZE = 64
LR = 3e-3
DROPOUT = 0.1
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


def visualize_attention(model: TransformerLM, tokenizer: CharTokenizer,
                        text: str, save_path: str, title: str,
                        layer: int = -1) -> None:
    """Run a forward pass and save attention heatmaps for the specified layer."""
    ids = tokenizer.encode(text, add_bos=True)
    x = torch.tensor([ids], dtype=torch.long)
    model.eval()
    with torch.no_grad():
        model(x)
    weights = model.get_attention_weights(layer=layer)
    if weights is None:
        return

    tokens = ["<BOS>"] + list(text)
    n_heads = weights.shape[1]

    # Average across heads
    avg_weights = weights[0].mean(dim=0).cpu().numpy()
    plot_attention_heatmap(
        avg_weights, tokens,
        title=f"{title} (avg across {n_heads} heads)",
        save_path=save_path,
    )

    # Per-head heatmaps
    base = Path(save_path)
    for h in range(n_heads):
        head_weights = weights[0, h].cpu().numpy()
        head_path = str(base.parent / f"{base.stem}_head{h}{base.suffix}")
        plot_attention_heatmap(
            head_weights, tokens,
            title=f"{title} - Head {h}",
            save_path=head_path,
        )


def main():
    print("=" * 60)
    print("Chapter 05: Transformer")
    print("=" * 60)
    print()
    print("The full architecture: attention + feed-forward + residuals + layer norm.")
    print("This is the architecture behind GPT and every modern LLM.")
    print()
    print(f"  d_model:  {D_MODEL}")
    print(f"  n_heads:  {N_HEADS}")
    print(f"  n_layers: {N_LAYERS}")
    print(f"  d_ff:     {D_FF}")
    print(f"  dropout:  {DROPOUT}")
    print()
    print("What's new vs attention-only (Chapter 04):")
    print("  - Feed-forward layers: can COMPUTE, not just retrieve")
    print("  - Residual connections: x + sublayer(x) preserves the original signal")
    print("  - Layer norm: stabilizes training of deep networks")
    print("  - Multiple layers: compositional feature building")

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

    # 5. Train transformer
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=256,
        dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    print(f"\nModel: TransformerLM ({model.count_parameters():,} params)")
    print(f"  {N_LAYERS} layers x (MHA + FFN + residual + layer norm)")
    print("Training...")
    result = train(model, train_loader, epochs=EPOCHS, lr=LR)

    # Save loss curve
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        result.losses,
        title="Chapter 05: Transformer Training Loss",
        save_path=str(RESULTS_DIR / "ch05_loss_curve.png"),
    )
    print(f"\nLoss curve saved to {RESULTS_DIR / 'ch05_loss_curve.png'}")
    print(f"  Final loss: {result.epoch_losses[-1]:.4f}")

    # 6. Visualize attention from different layers
    print("\n" + "-" * 50)
    print("Attention Heatmaps (Layer Comparison)")
    print("-" * 50)
    print("The Transformer has multiple layers. Earlier layers attend to")
    print("local/surface patterns; later layers attend to semantic patterns.")
    print()

    heatmap_examples = [
        ("COPY: hi|", "Copy task"),
        ("ADD 5 3 =", "Arithmetic task"),
    ]

    for text, description in heatmap_examples:
        safe_name = text.replace(" ", "_").replace(":", "").replace("|", "").replace("(", "").replace(")", "")
        for layer_idx in range(N_LAYERS):
            save_path = str(RESULTS_DIR / f"ch05_attn_L{layer_idx}_{safe_name}.png")
            visualize_attention(model, tokenizer, text, save_path,
                                title=f"{description} (Layer {layer_idx})",
                                layer=layer_idx)
            print(f"  '{text}' Layer {layer_idx} -> {save_path}")

    # 7. Sample generations
    print("\n" + "-" * 50)
    print("Sample Generations")
    print("-" * 50)
    print("The Transformer can both retrieve (attention) and compute (FFN).")
    print("Watch for improvements on arithmetic and compositional tasks.")
    print()

    transformer_agent = TransformerAgent(model, tokenizer, "transformer_lm", max_gen=20)

    sample_prompts = [
        ("ADD 5 3 =", "FFN can now compute the sum, not just retrieve the operands."),
        ("COPY: abc|", "Attention retrieves; FFN + residual preserve the signal."),
        ("CHECK: ( ( ) )", "Multiple layers can match nested brackets step by step."),
        ("COMPOSE: ADD 2 3 THEN MUL 4 =", "Needs multi-step reasoning across layers."),
    ]
    for prompt, explanation in sample_prompts:
        result_gen = transformer_agent.run(prompt)
        print(f"  Prompt: '{prompt}'")
        print(f"  Why Transformer helps: {explanation}")
        print(f"    Output: '{result_gen.answer}'")
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
    print("Running Transformer and human agent on 50 samples per task.")
    print()

    agents = {
        "transformer_lm": transformer_agent,
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
                     RESULTS_DIR / f"ch05_{agent_name}.json",
                     agent_name=agent_name, chapter="05_transformer")

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
        title="Chapter 05: Transformer vs Human Agent",
        save_path=str(RESULTS_DIR / "ch05_comparison.png"),
    )
    print(f"Plot saved to {RESULTS_DIR / 'ch05_comparison.png'}")

    # 10. Key takeaway
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print(f"The Transformer ({model.count_parameters():,} params) combines:")
    print("  - Attention: look at any position (Chapter 04)")
    print("  - Feed-forward: compute on what you found (NEW)")
    print("  - Residual connections: preserve signal through depth (NEW)")
    print("  - Layer norm: stable training (NEW)")
    print("  - Multiple layers: compose features hierarchically (NEW)")
    print()
    print("What improves:")
    print("  - Arithmetic: FFN can compute sums, not just retrieve operands")
    print("  - Compositional: multi-layer processing enables multi-step reasoning")
    print("  - Copy/Grammar: residuals preserve signal, layer norm stabilizes")
    print()
    print("What's still missing:")
    print("  - Still can't abstain (no uncertainty mechanism)")
    print("  - Tiny model on tiny data — real LLMs need scale")
    print("  - No pretraining on diverse data — only sees our task formats")
    print()
    print("Human lens: Humans reason in 'layers' too — recognizing words,")
    print("parsing structure, understanding meaning, then reasoning about it.")
    print("The Transformer does this with literal stacked layers: early layers")
    print("capture surface patterns, later layers compose them into answers.")
    print("But each 'layer' is learned end-to-end, not designed by hand.")


if __name__ == "__main__":
    main()
