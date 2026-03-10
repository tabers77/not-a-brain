"""Chapter 07: Instruction Tuning (Toy SFT)

Pre-train a Transformer on raw task text, then fine-tune on
instruction-formatted data. Compare base vs SFT model.

Key finding: SFT makes outputs cleaner and more task-aligned, but
still can't abstain on unanswerable questions. Format != capability.

Usage:
    python chapters/07_instruction_tuning/run.py
"""

from pathlib import Path
import copy
import torch

from not_a_brain.models.tokenizer import CharTokenizer
from not_a_brain.models.transformer import TransformerLM, TransformerAgent
from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.evals.harness import (
    AgentInterface, AgentResult, run_eval_suite, save_results,
)
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)
from not_a_brain.utils.training import train, make_dataset, generate
from not_a_brain.utils.visualization import plot_loss_curve, plot_comparison_bar

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
SFT_LR = 1e-3  # Lower LR for fine-tuning
DROPOUT = 0.1
MAX_SEQ_LEN = 64  # Slightly longer for instruction markers

INST_PREFIX = "INST: "
ANS_MARKER = " ANS: "


def build_raw_corpus(tasks: dict, n: int) -> list[str]:
    """Pre-training corpus: raw prompt + answer concatenation."""
    corpus = []
    for task in tasks.values():
        pairs = task.training_pairs(n)
        for prompt, answer in pairs:
            corpus.append(prompt + answer)
    return corpus


def build_instruction_corpus(tasks: dict, n: int) -> list[str]:
    """SFT corpus: instruction-formatted prompt + answer."""
    corpus = []
    for task in tasks.values():
        pairs = task.training_pairs(n)
        for prompt, answer in pairs:
            corpus.append(f"{INST_PREFIX}{prompt}{ANS_MARKER}{answer}")
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


class SFTAgent(AgentInterface):
    """Agent that wraps prompts with instruction format before generating."""

    def __init__(self, model: TransformerLM, tokenizer: CharTokenizer,
                 model_name: str = "sft_model", max_gen: int = 30,
                 temperature: float = 0.0):
        self.model = model
        self.tokenizer = tokenizer
        self._name = model_name
        self.max_gen = max_gen
        self.temperature = temperature

    @property
    def name(self) -> str:
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst_prompt = f"{INST_PREFIX}{prompt}{ANS_MARKER}"
        prompt_ids = self.tokenizer.encode(inst_prompt, add_bos=True)
        full_ids = generate(self.model, prompt_ids,
                            max_new_tokens=self.max_gen,
                            temperature=self.temperature)
        generated = self.tokenizer.decode(full_ids)

        # Extract answer after the last ANS marker
        marker = ANS_MARKER.strip()
        if marker in generated:
            parts = generated.split(marker)
            answer = parts[-1].strip()
        elif generated.startswith(inst_prompt):
            answer = generated[len(inst_prompt):].strip()
        else:
            answer = generated.strip()

        # Stop at newline or special tokens
        for stop in ["\n", "<EOS>", "<PAD>"]:
            if stop in answer:
                answer = answer[:answer.index(stop)]

        return AgentResult(answer=answer.strip(), confidence=0.5)


def main() -> None:
    print("=" * 60)
    print("Chapter 07: Instruction Tuning (Toy SFT)")
    print("=" * 60)
    print()
    print("Can we make a model more 'helpful' by fine-tuning on")
    print("instruction-formatted data? We pre-train a Transformer on")
    print("raw task text, then fine-tune (SFT) on the same tasks")
    print("wrapped in an instruction format.")
    print()
    print("Instruction format:")
    print('  Raw:  "ADD 5 3 =8"')
    print('  SFT:  "INST: ADD 5 3 = ANS: 8"')
    print()
    print(f"  d_model:       {D_MODEL}")
    print(f"  n_heads:       {N_HEADS}")
    print(f"  n_layers:      {N_LAYERS}")
    print(f"  d_ff:          {D_FF}")
    print(f"  pretrain:      {PRETRAIN_EPOCHS} epochs @ lr={LR}")
    print(f"  SFT:           {SFT_EPOCHS} epochs @ lr={SFT_LR}")

    # ------------------------------------------------------------------
    # 1. Build task suite
    # ------------------------------------------------------------------
    tasks = {
        "arithmetic": ArithmeticTask(seed=1),
        "copy": CopyTask(seed=1),
        "grammar": GrammarTask(seed=1),
        "knowledge_qa": KnowledgeQATask(seed=1),
        "compositional": CompositionalTask(seed=1),
        "unknown": UnknownTask(seed=1),
    }

    # ------------------------------------------------------------------
    # 2. Build both corpora
    # ------------------------------------------------------------------
    print("\nBuilding corpora...")
    raw_corpus = build_raw_corpus(tasks, N_TRAIN)
    inst_corpus = build_instruction_corpus(tasks, N_TRAIN)
    print(f"  Raw corpus:         {len(raw_corpus)} sequences")
    print(f"  Instruction corpus: {len(inst_corpus)} sequences")
    print(f"\n  Example raw:  {raw_corpus[0][:60]}")
    print(f"  Example inst: {inst_corpus[0][:60]}")

    # ------------------------------------------------------------------
    # 3. Fit tokenizer on combined vocabulary
    # ------------------------------------------------------------------
    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + inst_corpus)
    print(f"\n  Vocab size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # 4. Prepare both datasets
    # ------------------------------------------------------------------
    print(f"\nPreparing sequences (max {MAX_SEQ_LEN} tokens)...")
    raw_inputs, raw_targets = prepare_sequences(
        raw_corpus, tokenizer, max_len=MAX_SEQ_LEN)
    inst_inputs, inst_targets = prepare_sequences(
        inst_corpus, tokenizer, max_len=MAX_SEQ_LEN)

    raw_loader = make_dataset(raw_inputs, raw_targets, batch_size=BATCH_SIZE)
    inst_loader = make_dataset(inst_inputs, inst_targets, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------
    # 5. Phase 1: Pre-train base model on raw text
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Phase 1: Pre-training (raw text)")
    print(f"{'─' * 50}")

    base_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=256,
        dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    print(f"  Parameters: {base_model.count_parameters():,}")
    pretrain_result = train(base_model, raw_loader, epochs=PRETRAIN_EPOCHS, lr=LR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        pretrain_result.losses,
        title="Chapter 07: Pre-training Loss (Raw Text)",
        save_path=str(RESULTS_DIR / "ch07_pretrain_loss.png"),
    )
    print(f"  Final pre-training loss: {pretrain_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 6. Phase 2: Fine-tune (SFT) a copy of the base model
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Phase 2: Supervised Fine-Tuning (Instruction Format)")
    print(f"{'─' * 50}")
    print("Starting from the pre-trained weights, fine-tune on")
    print("instruction-formatted data with a lower learning rate.")
    print()

    sft_model = copy.deepcopy(base_model)
    sft_result = train(sft_model, inst_loader, epochs=SFT_EPOCHS, lr=SFT_LR)

    plot_loss_curve(
        sft_result.losses,
        title="Chapter 07: SFT Loss (Instruction Format)",
        save_path=str(RESULTS_DIR / "ch07_sft_loss.png"),
    )
    print(f"  Final SFT loss: {sft_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 7. Control: train from scratch on instruction data
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Control: Training from scratch on instruction data")
    print(f"{'─' * 50}")
    print("What if we skip pre-training and train directly on instructions?")
    print()

    scratch_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        d_ff=D_FF,
        max_seq_len=256,
        dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    scratch_result = train(
        scratch_model, inst_loader,
        epochs=PRETRAIN_EPOCHS + SFT_EPOCHS, lr=LR,
    )

    plot_loss_curve(
        scratch_result.losses,
        title="Chapter 07: From-Scratch on Instructions",
        save_path=str(RESULTS_DIR / "ch07_scratch_loss.png"),
    )
    print(f"  Final loss: {scratch_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 8. Sample generations: benchmark prompts
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Sample Generations: Base vs SFT")
    print(f"{'─' * 50}")
    print()

    base_agent = TransformerAgent(
        base_model, tokenizer, "base_model", max_gen=20)
    sft_agent = SFTAgent(
        sft_model, tokenizer, "sft_model", max_gen=20)
    scratch_agent = SFTAgent(
        scratch_model, tokenizer, "scratch_sft", max_gen=20)

    benchmark_prompts = [
        ("ADD 5 3 =",
         "Computation — can the model compute the sum?"),
        ("FACT: paris is capital of france. Q: capital of france?",
         "Retrieval — can the model find the answer in context?"),
        ("Q: What is the capital of the Moon?",
         "Hallucination — does the model abstain?"),
    ]

    for prompt, description in benchmark_prompts:
        print(f"  Prompt: '{prompt}'")
        print(f"  Tests:  {description}")

        base_out = base_agent.run(prompt)
        sft_out = sft_agent.run(prompt)
        scratch_out = scratch_agent.run(prompt)

        print(f"    Base model:   '{base_out.answer}'")
        print(f"    SFT model:    '{sft_out.answer}'")
        print(f"    Scratch+inst: '{scratch_out.answer}'")
        print()

    # ------------------------------------------------------------------
    # 9. Evaluate all models
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

    print(f"{'─' * 50}")
    print("Evaluation: Base vs SFT vs Scratch vs Human")
    print(f"{'─' * 50}")

    agents = {
        "base_model": base_agent,
        "sft_model": sft_agent,
        "scratch_sft": scratch_agent,
        "human_agent": human,
    }

    all_metrics = {}
    for agent_name, agent in agents.items():
        metrics, results = run_eval_suite(
            agent, eval_tasks, n_per_task=N_EVAL)
        all_metrics[agent_name] = metrics
        print(f"\n  {agent_name}:")
        print(f"    Accuracy:           {metrics.accuracy:.1%}")
        print(f"    Hallucination rate: {metrics.hallucination_rate:.1%}")
        print(f"    Abstention rate:    {metrics.abstention_rate:.1%}")
        print(f"    Per-task:")
        for t, info in metrics.per_task.items():
            print(f"      {t:20s} {info['accuracy']:.1%}")

        save_results(
            results, metrics,
            RESULTS_DIR / f"ch07_{agent_name}.json",
            agent_name=agent_name, chapter="07_instruction_tuning")

    # ------------------------------------------------------------------
    # 10. Plots
    # ------------------------------------------------------------------
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
        title="Chapter 07: Base vs SFT vs From-Scratch vs Human",
        save_path=str(RESULTS_DIR / "ch07_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch07_comparison.png'}")

    _plot_loss_comparison(pretrain_result, sft_result, scratch_result)
    print(f"  Loss comparison saved to {RESULTS_DIR / 'ch07_loss_comparison.png'}")

    # ------------------------------------------------------------------
    # 11. Key takeaway
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print()
    print("Instruction tuning (SFT) does three things:")
    print("  1. FORMAT ALIGNMENT — the model learns to output answers in the")
    print("     expected format (cleaner, less noisy)")
    print("  2. TASK COMPLIANCE — the model gets better at following the")
    print("     'instruction -> response' pattern")
    print("  3. TRANSFER — pre-trained knowledge makes SFT faster and more")
    print("     effective than training from scratch")
    print()
    print("What SFT does NOT do:")
    print("  - Add new capabilities the base model didn't have")
    print("  - Fix hallucination on unanswerable questions")
    print("  - Create genuine understanding or reasoning")
    print()
    print("SFT is like teaching someone to write in a specific format:")
    print("it doesn't make them smarter, but it makes their existing")
    print("knowledge more accessible and useful.")
    print()
    print("Human lens: When humans learn to 'follow instructions', they")
    print("use goal-directed behavior — parsing intent, planning steps,")
    print("checking results. SFT models learn surface-level format")
    print("patterns, not genuine instruction-following.")
    print()
    print("Next: Chapter 08 (Preference / RLHF) — can we teach the model")
    print("what humans PREFER, not just what format to use?")


def _plot_loss_comparison(pretrain_result, sft_result, scratch_result) -> None:
    """Plot all three training loss curves together."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    # Pre-training losses
    ax.plot(range(1, len(pretrain_result.epoch_losses) + 1),
            pretrain_result.epoch_losses,
            marker="o", linewidth=2, markersize=4,
            color="#2196F3", label="Pre-training (raw text)")

    # SFT losses (offset x-axis to show continuation)
    sft_start = len(pretrain_result.epoch_losses)
    sft_x = range(sft_start + 1,
                   sft_start + len(sft_result.epoch_losses) + 1)
    ax.plot(sft_x, sft_result.epoch_losses,
            marker="s", linewidth=2, markersize=4,
            color="#4CAF50", label="SFT (instruction format)")

    # Scratch losses
    ax.plot(range(1, len(scratch_result.epoch_losses) + 1),
            scratch_result.epoch_losses,
            marker="^", linewidth=2, markersize=4,
            color="#FF9800", label="From-scratch (instructions only)")

    # Mark the pre-train -> SFT transition
    ax.axvline(x=sft_start + 0.5, color="gray", linestyle="--", alpha=0.5,
               label="Pre-train -> SFT transition")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Chapter 07: Training Loss Comparison")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch07_loss_comparison.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
