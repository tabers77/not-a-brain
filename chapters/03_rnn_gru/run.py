"""Chapter 03: Recurrent Language Models (RNN & GRU)

First models with "memory". Recurrent models process one token at a time,
maintaining a hidden state that summarizes the full history.

Shows: variable-length processing, GRU > RNN, but long-range still degrades.
Human lens: Human working memory is structured; RNN state is compressed soup.

Usage:
    python chapters/03_rnn_gru/run.py
"""

from pathlib import Path
import torch

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.rnn_lm import RNNLM, GRULM, RNNAgent
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
D_EMBED = 16
D_HIDDEN = 64
EPOCHS = 15
BATCH_SIZE = 64
LR = 3e-3
MAX_SEQ_LEN = 50  # constraint: keep sequences short for CPU


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
    """Convert corpus into padded (input, target) sequence pairs."""
    encoded = []
    for text in corpus:
        ids = tokenizer.encode(text, add_bos=True, add_eos=True)
        # Truncate to max sequence length
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
    print("Chapter 03: Recurrent Language Models (RNN & GRU)")
    print("=" * 60)
    print()
    print("Recurrent models process one token at a time, maintaining a")
    print("hidden state h_t that (in theory) captures the full history.")
    print()
    print("  RNN: h_t = tanh(W_ih * e_t + W_hh * h_{t-1})")
    print("  GRU: h_t = gated update (keeps useful info, forgets noise)")
    print()
    print(f"  Embedding dim: {D_EMBED}")
    print(f"  Hidden dim:    {D_HIDDEN}")
    print(f"  Max seq len:   {MAX_SEQ_LEN}")
    print()
    print("Key question: can a compressed state vector remember what matters?")
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

    # 5. Train vanilla RNN
    rnn_model = RNNLM(
        vocab_size=tokenizer.vocab_size,
        d_embed=D_EMBED,
        d_hidden=D_HIDDEN,
        pad_id=tokenizer.pad_id,
    )
    print(f"\n--- Vanilla RNN ({rnn_model.count_parameters():,} params) ---")
    print("Training...")
    rnn_result = train(rnn_model, train_loader, epochs=EPOCHS, lr=LR)

    # 6. Train GRU
    gru_model = GRULM(
        vocab_size=tokenizer.vocab_size,
        d_embed=D_EMBED,
        d_hidden=D_HIDDEN,
        pad_id=tokenizer.pad_id,
    )
    print(f"\n--- GRU ({gru_model.count_parameters():,} params) ---")
    print("Training...")
    gru_result = train(gru_model, train_loader, epochs=EPOCHS, lr=LR)

    # Save loss curves
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        rnn_result.losses,
        title="Chapter 03: Vanilla RNN Training Loss",
        save_path=str(RESULTS_DIR / "ch03_rnn_loss.png"),
    )
    plot_loss_curve(
        gru_result.losses,
        title="Chapter 03: GRU Training Loss",
        save_path=str(RESULTS_DIR / "ch03_gru_loss.png"),
    )
    print(f"\nLoss curves saved to {RESULTS_DIR}")

    # 7. Compare final training losses
    print(f"\n  RNN final loss: {rnn_result.epoch_losses[-1]:.4f}")
    print(f"  GRU final loss: {gru_result.epoch_losses[-1]:.4f}")

    # 8. Sample generations
    print("\n" + "-" * 50)
    print("Sample Generations")
    print("-" * 50)
    print("Unlike FFN, the RNN/GRU has no fixed window — it reads the")
    print("full prompt. But does the hidden state retain the right info?")
    print()

    rnn_agent = RNNAgent(rnn_model, tokenizer, "rnn", max_gen=20)
    gru_agent = RNNAgent(gru_model, tokenizer, "gru", max_gen=20)

    sample_prompts = [
        ("ADD 5 3 =", "Needs 5+3=8. RNN has seen all tokens, but must compress them."),
        ("COPY: abc|", "Needs 'abc'. Hidden state must remember the exact characters."),
        ("CHECK: ( ( ) )", "Nested brackets. Needs to track depth — natural for a stack, hard for a vector."),
    ]
    for prompt, explanation in sample_prompts:
        rnn_out = rnn_agent.run(prompt)
        gru_out = gru_agent.run(prompt)
        print(f"  Prompt: '{prompt}'")
        print(f"  Why it's hard: {explanation}")
        print(f"    RNN: '{rnn_out.answer}'")
        print(f"    GRU: '{gru_out.answer}'")
        print()

    # 9. Evaluate all agents
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
    print("Running RNN, GRU, and human agent on 50 samples per task.")
    print()

    agents = {
        "rnn": rnn_agent,
        "gru": gru_agent,
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
                     RESULTS_DIR / f"ch03_{agent_name}.json",
                     agent_name=agent_name, chapter="03_rnn_gru")

    # 10. Plot comparison
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
        title="Chapter 03: RNN vs GRU vs Human Agent",
        save_path=str(RESULTS_DIR / "ch03_comparison.png"),
    )
    print(f"Plot saved to {RESULTS_DIR / 'ch03_comparison.png'}")

    # 11. Key takeaway
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print("Recurrent models remove the fixed window — they 'read' the full")
    print("prompt and maintain a hidden state. But:")
    print()
    print(f"  RNN  ({rnn_model.count_parameters():,} params): state = compressed soup")
    print(f"  GRU  ({gru_model.count_parameters():,} params): gated state = slightly better soup")
    print()
    print("The GRU's gates help information persist, but the hidden state")
    print("is still a fixed-size vector that must compress EVERYTHING.")
    print()
    print("What they still can't do:")
    print("  - Precise computation (arithmetic)")
    print("  - Selective attention (look back at specific tokens)")
    print("  - Abstention (say 'I don't know')")
    print()
    print("Human lens: Human working memory is STRUCTURED — distinct slots")
    print("for different pieces of information. The RNN state is UNSTRUCTURED")
    print("— everything mixed into one vector. Humans can also look back at")
    print("the prompt; RNNs can't.")


if __name__ == "__main__":
    main()
