"""Chapter 08: Preference Optimization (Toy DPO)

After SFT (Chapter 07) aligns format, DPO aligns *preference*: which
response does a human prefer?  We build preference pairs (chosen vs
rejected), then shift the SFT model toward preferred outputs using
Direct Preference Optimization.

Key finding: DPO can shift style and reduce some errors, but the model
still hallucinates on unanswerable questions.  Preference != understanding.

Usage:
    python chapters/08_preference_and_rlhf/run.py
"""

from pathlib import Path
import copy
import random

import torch
import torch.nn.functional as F

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
DPO_EPOCHS = 10
BATCH_SIZE = 64
LR = 3e-3
SFT_LR = 1e-3
DPO_LR = 5e-4   # Even lower for preference tuning
DROPOUT = 0.1
MAX_SEQ_LEN = 64
DPO_BETA = 0.1  # KL penalty strength

INST_PREFIX = "INST: "
ANS_MARKER = " ANS: "

# ── Corpus builders ───────────────────────────────────────────────────


def build_raw_corpus(tasks: dict, n: int) -> list[str]:
    """Pre-training corpus: raw prompt + answer."""
    corpus = []
    for task in tasks.values():
        for prompt, answer in task.training_pairs(n):
            corpus.append(prompt + answer)
    return corpus


def build_instruction_corpus(tasks: dict, n: int) -> list[str]:
    """SFT corpus: INST: prompt ANS: answer."""
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


# ── Preference pair generation ────────────────────────────────────────

def _wrong_answer_arithmetic(sample, rng):
    """Plausible wrong answer: off by a small amount or an operand."""
    correct = int(sample.expected)
    choices = [
        str(sample.metadata["a"]),           # just echo an operand
        str(sample.metadata["b"]),
        str(correct + rng.choice([-1, 1, 2, -2])),  # close but wrong
        str(sample.metadata["a"] + sample.metadata["b"] + 1),
    ]
    bad = rng.choice([c for c in choices if c != sample.expected] or ["0"])
    return bad


def _wrong_answer_copy(sample, rng):
    """Scramble the sequence."""
    seq = list(sample.expected)
    rng.shuffle(seq)
    bad = "".join(seq)
    return bad if bad != sample.expected else bad + "x"


def _wrong_answer_grammar(sample, rng):
    """Flip valid/invalid."""
    return "invalid" if sample.expected == "valid" else "valid"


def _wrong_answer_knowledge(sample, rng):
    """Use a plausible but wrong entity."""
    distractors = ["tokyo", "berlin", "london", "rome", "O2", "mars", "7"]
    choices = [d for d in distractors if d != sample.expected]
    return rng.choice(choices)


def _wrong_answer_unknown(sample, rng):
    """Any confident answer (hallucination) is the rejected one.
    For unknown tasks, chosen='unknown', rejected=hallucinated answer."""
    return rng.choice(["paris", "earth", "42", "yes", "tokyo", "mars"])


def _wrong_answer_compositional(sample, rng):
    """Reverse or garble the expected output."""
    bad = sample.expected[::-1]
    return bad if bad != sample.expected else sample.expected + "x"


WRONG_GENERATORS = {
    "arithmetic": _wrong_answer_arithmetic,
    "copy": _wrong_answer_copy,
    "grammar": _wrong_answer_grammar,
    "knowledge_qa": _wrong_answer_knowledge,
    "unknown": _wrong_answer_unknown,
    "compositional": _wrong_answer_compositional,
}


def build_preference_pairs(tasks: dict, n: int,
                           seed: int = 42) -> list[tuple[str, str, str]]:
    """Build (prompt, chosen, rejected) triples in instruction format."""
    rng = random.Random(seed)
    pairs = []
    for task_name, task in tasks.items():
        gen_wrong = WRONG_GENERATORS[task_name]
        for _ in range(n):
            sample = task.generate()
            chosen = sample.expected
            rejected = gen_wrong(sample, rng)
            # Format as instruction text
            chosen_text = f"{INST_PREFIX}{sample.prompt}{ANS_MARKER}{chosen}"
            rejected_text = f"{INST_PREFIX}{sample.prompt}{ANS_MARKER}{rejected}"
            pairs.append((sample.prompt, chosen_text, rejected_text))
    return pairs


# ── DPO implementation ────────────────────────────────────────────────

def _sequence_logprobs(model, token_ids, pad_id):
    """Compute sum of log-probs for each sequence in a batch.

    Args:
        token_ids: (B, L) full sequence including prompt + response
        pad_id: token id for padding (masked out of loss)

    Returns:
        (B,) tensor of sequence log-probabilities
    """
    inputs = token_ids[:, :-1]
    targets = token_ids[:, 1:]
    logits = model(inputs)                           # (B, L-1, V)
    log_probs = F.log_softmax(logits, dim=-1)        # (B, L-1, V)
    token_lps = log_probs.gather(
        2, targets.unsqueeze(-1)).squeeze(-1)         # (B, L-1)

    # Mask padding
    mask = (targets != pad_id).float()
    return (token_lps * mask).sum(dim=-1)             # (B,)


def dpo_loss(policy_model, ref_model, chosen_ids, rejected_ids,
             pad_id, beta=0.1):
    """DPO loss for one batch of preference pairs.

    L_DPO = -log sigma(beta * (log pi(y_w|x) - log pi_ref(y_w|x))
                      - beta * (log pi(y_l|x) - log pi_ref(y_l|x)))
    """
    # Policy log-probs
    pi_chosen = _sequence_logprobs(policy_model, chosen_ids, pad_id)
    pi_rejected = _sequence_logprobs(policy_model, rejected_ids, pad_id)

    # Reference log-probs (no grad)
    with torch.no_grad():
        ref_chosen = _sequence_logprobs(ref_model, chosen_ids, pad_id)
        ref_rejected = _sequence_logprobs(ref_model, rejected_ids, pad_id)

    chosen_reward = beta * (pi_chosen - ref_chosen)
    rejected_reward = beta * (pi_rejected - ref_rejected)
    loss = -F.logsigmoid(chosen_reward - rejected_reward)
    return loss.mean()


def _encode_preference_batch(pairs, tokenizer, max_len):
    """Encode preference pairs into padded tensors."""
    chosen_enc, rejected_enc = [], []
    for _, chosen_text, rejected_text in pairs:
        c_ids = tokenizer.encode(chosen_text, add_bos=True, add_eos=True)
        r_ids = tokenizer.encode(rejected_text, add_bos=True, add_eos=True)
        if max_len:
            c_ids = c_ids[:max_len]
            r_ids = r_ids[:max_len]
        chosen_enc.append(c_ids)
        rejected_enc.append(r_ids)

    pad_len = max(max(len(s) for s in chosen_enc),
                  max(len(s) for s in rejected_enc))
    pad_id = tokenizer.pad_id

    def _pad(seqs):
        return torch.tensor(
            [s + [pad_id] * (pad_len - len(s)) for s in seqs],
            dtype=torch.long)

    return _pad(chosen_enc), _pad(rejected_enc)


def train_dpo(policy_model, ref_model, pref_pairs, tokenizer,
              epochs=10, lr=5e-4, beta=0.1, batch_size=32,
              max_len=64, verbose=True):
    """Train with DPO on preference pairs."""
    chosen_ids, rejected_ids = _encode_preference_batch(
        pref_pairs, tokenizer, max_len)
    pad_id = tokenizer.pad_id
    n = len(pref_pairs)

    policy_model.train()
    ref_model.eval()
    optimizer = torch.optim.Adam(policy_model.parameters(), lr=lr)
    epoch_losses = []

    for epoch in range(epochs):
        # Shuffle
        perm = torch.randperm(n)
        epoch_loss = 0.0
        n_batches = 0

        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            c_batch = chosen_ids[idx]
            r_batch = rejected_ids[idx]

            loss = dpo_loss(policy_model, ref_model,
                            c_batch, r_batch, pad_id, beta=beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg = epoch_loss / max(n_batches, 1)
        epoch_losses.append(avg)
        if verbose:
            print(f"  Epoch {epoch + 1}/{epochs} — DPO loss: {avg:.4f}")

    policy_model.eval()
    return epoch_losses


# ── SFT Agent (reused from Ch07) ─────────────────────────────────────

class SFTAgent(AgentInterface):
    """Agent that wraps prompts with instruction format."""

    def __init__(self, model, tokenizer, model_name="sft_model",
                 max_gen=30, temperature=0.0):
        self.model = model
        self.tokenizer = tokenizer
        self._name = model_name
        self.max_gen = max_gen
        self.temperature = temperature

    @property
    def name(self):
        return self._name

    def run(self, prompt: str) -> AgentResult:
        inst_prompt = f"{INST_PREFIX}{prompt}{ANS_MARKER}"
        prompt_ids = self.tokenizer.encode(inst_prompt, add_bos=True)
        full_ids = generate(self.model, prompt_ids,
                            max_new_tokens=self.max_gen,
                            temperature=self.temperature)
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


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("Chapter 08: Preference Optimization (Toy DPO)")
    print("=" * 60)
    print()
    print("SFT (Chapter 07) aligned format.  Now we align *preference*:")
    print("which response does a human prefer?  DPO shifts the model")
    print("toward chosen responses and away from rejected ones.")
    print()
    print("Pipeline: pre-train -> SFT -> DPO")
    print()
    print(f"  d_model:     {D_MODEL}")
    print(f"  n_heads:     {N_HEADS}")
    print(f"  n_layers:    {N_LAYERS}")
    print(f"  d_ff:        {D_FF}")
    print(f"  pretrain:    {PRETRAIN_EPOCHS} epochs @ lr={LR}")
    print(f"  SFT:         {SFT_EPOCHS} epochs @ lr={SFT_LR}")
    print(f"  DPO:         {DPO_EPOCHS} epochs @ lr={DPO_LR}, beta={DPO_BETA}")

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
    # 2. Build corpora
    # ------------------------------------------------------------------
    print("\nBuilding corpora...")
    raw_corpus = build_raw_corpus(tasks, N_TRAIN)
    inst_corpus = build_instruction_corpus(tasks, N_TRAIN)
    print(f"  Raw corpus:         {len(raw_corpus)} sequences")
    print(f"  Instruction corpus: {len(inst_corpus)} sequences")

    tokenizer = CharTokenizer()
    tokenizer.fit(raw_corpus + inst_corpus)
    print(f"  Vocab size: {tokenizer.vocab_size}")

    # ------------------------------------------------------------------
    # 3. Prepare datasets
    # ------------------------------------------------------------------
    print(f"\nPreparing sequences (max {MAX_SEQ_LEN} tokens)...")
    raw_inputs, raw_targets = prepare_sequences(
        raw_corpus, tokenizer, max_len=MAX_SEQ_LEN)
    inst_inputs, inst_targets = prepare_sequences(
        inst_corpus, tokenizer, max_len=MAX_SEQ_LEN)
    raw_loader = make_dataset(raw_inputs, raw_targets, batch_size=BATCH_SIZE)
    inst_loader = make_dataset(inst_inputs, inst_targets, batch_size=BATCH_SIZE)

    # ------------------------------------------------------------------
    # 4. Phase 1: Pre-train
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Phase 1: Pre-training (raw text)")
    print(f"{'─' * 50}")

    base_model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ff=D_FF, max_seq_len=256, dropout=DROPOUT,
        pad_id=tokenizer.pad_id,
    )
    print(f"  Parameters: {base_model.count_parameters():,}")
    pretrain_result = train(base_model, raw_loader,
                            epochs=PRETRAIN_EPOCHS, lr=LR)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_loss_curve(
        pretrain_result.losses,
        title="Chapter 08: Pre-training Loss",
        save_path=str(RESULTS_DIR / "ch08_pretrain_loss.png"),
    )
    print(f"  Final loss: {pretrain_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 5. Phase 2: SFT
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Phase 2: Supervised Fine-Tuning")
    print(f"{'─' * 50}")

    sft_model = copy.deepcopy(base_model)
    sft_result = train(sft_model, inst_loader,
                       epochs=SFT_EPOCHS, lr=SFT_LR)
    plot_loss_curve(
        sft_result.losses,
        title="Chapter 08: SFT Loss",
        save_path=str(RESULTS_DIR / "ch08_sft_loss.png"),
    )
    print(f"  Final SFT loss: {sft_result.epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 6. Build preference pairs
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Building preference pairs")
    print(f"{'─' * 50}")

    pref_tasks = {
        "arithmetic": ArithmeticTask(seed=7),
        "copy": CopyTask(seed=7),
        "grammar": GrammarTask(seed=7),
        "knowledge_qa": KnowledgeQATask(seed=7),
        "compositional": CompositionalTask(seed=7),
        "unknown": UnknownTask(seed=7),
    }
    pref_pairs = build_preference_pairs(pref_tasks, n=200)
    print(f"  {len(pref_pairs)} preference pairs")

    # Show examples
    for i in [0, 200, 1000]:
        if i < len(pref_pairs):
            prompt, chosen, rejected = pref_pairs[i]
            print(f"\n  Pair {i}:")
            print(f"    Prompt:   {prompt[:50]}")
            print(f"    Chosen:   ...{chosen[-30:]}")
            print(f"    Rejected: ...{rejected[-30:]}")

    # ------------------------------------------------------------------
    # 7. Phase 3: DPO
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Phase 3: Direct Preference Optimization (DPO)")
    print(f"{'─' * 50}")
    print("Starting from SFT weights.  The reference model is a frozen")
    print("copy of the SFT model.  DPO pushes the policy toward chosen")
    print("responses and away from rejected ones.")
    print()

    ref_model = copy.deepcopy(sft_model)   # frozen reference
    dpo_model = copy.deepcopy(sft_model)   # trainable policy

    dpo_epoch_losses = train_dpo(
        dpo_model, ref_model, pref_pairs, tokenizer,
        epochs=DPO_EPOCHS, lr=DPO_LR, beta=DPO_BETA,
        batch_size=BATCH_SIZE, max_len=MAX_SEQ_LEN,
    )

    _plot_dpo_loss(dpo_epoch_losses)
    print(f"  Final DPO loss: {dpo_epoch_losses[-1]:.4f}")

    # ------------------------------------------------------------------
    # 8. Sample generations: benchmark prompts
    # ------------------------------------------------------------------
    print(f"\n{'─' * 50}")
    print("Sample Generations: SFT vs DPO")
    print(f"{'─' * 50}")
    print()

    sft_agent = SFTAgent(sft_model, tokenizer, "sft_model", max_gen=20)
    dpo_agent = SFTAgent(dpo_model, tokenizer, "dpo_model", max_gen=20)

    benchmark_prompts = [
        ("ADD 5 3 =",
         "Computation -- does DPO help?"),
        ("FACT: paris is capital of france. Q: capital of france?",
         "Retrieval -- does DPO help?"),
        ("Q: What is the capital of the Moon?",
         "Hallucination -- can DPO teach abstention?"),
    ]

    for prompt, description in benchmark_prompts:
        print(f"  Prompt: '{prompt}'")
        print(f"  Tests:  {description}")
        sft_out = sft_agent.run(prompt)
        dpo_out = dpo_agent.run(prompt)
        print(f"    SFT model: '{sft_out.answer}'")
        print(f"    DPO model: '{dpo_out.answer}'")
        print()

    # ------------------------------------------------------------------
    # 9. Evaluate
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
    print("Evaluation: SFT vs DPO vs Human")
    print(f"{'─' * 50}")

    agents = {
        "sft_model": sft_agent,
        "dpo_model": dpo_agent,
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
            RESULTS_DIR / f"ch08_{agent_name}.json",
            agent_name=agent_name, chapter="08_preference_and_rlhf")

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
        title="Chapter 08: SFT vs DPO vs Human",
        save_path=str(RESULTS_DIR / "ch08_comparison.png"),
    )
    print(f"\n  Comparison plot saved to {RESULTS_DIR / 'ch08_comparison.png'}")

    _plot_training_overview(pretrain_result, sft_result, dpo_epoch_losses)
    print(f"  Training overview saved to {RESULTS_DIR / 'ch08_training_overview.png'}")

    # ------------------------------------------------------------------
    # 11. Key takeaway
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY")
    print("=" * 60)
    print()
    print("DPO shifts the model's outputs toward human-preferred responses:")
    print("  1. PREFERENCE ALIGNMENT -- the model learns which response a")
    print("     human would pick from a pair (chosen vs rejected)")
    print("  2. REWARD IMPLICIT IN POLICY -- DPO folds the reward model")
    print("     directly into the policy, avoiding a separate reward step")
    print("  3. KL REGULARIZATION -- beta controls how far the policy can")
    print("     drift from the SFT reference (prevents mode collapse)")
    print()
    print("What DPO does NOT do:")
    print("  - Add capabilities the SFT model didn't have")
    print("  - Guarantee correct answers (it prefers, not proves)")
    print("  - Fix hallucination (preference for 'unknown' is fragile)")
    print()
    print("The progression so far:")
    print("  Pre-training: learns capabilities (Ch05)")
    print("  Scaling:      amplifies capabilities (Ch06)")
    print("  SFT:          aligns format (Ch07)")
    print("  DPO:          aligns preference (Ch08)")
    print("  None of these create genuine understanding or abstention.")
    print()
    print("Human lens: Human preferences are grounded in values, goals,")
    print("and experience.  DPO learns statistical correlations between")
    print("prompts and preferred responses -- pattern matching, not values.")
    print()
    print("Next: Chapter 09 (Decoding & Hallucination) -- can we reduce")
    print("hallucination by changing HOW the model generates, not WHAT")
    print("it learned?")


# ── Plotting helpers ──────────────────────────────────────────────────

def _plot_dpo_loss(epoch_losses) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(epoch_losses) + 1), epoch_losses,
            marker="o", linewidth=2, markersize=4, color="#9C27B0")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("DPO Loss")
    ax.set_title("Chapter 08: DPO Training Loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch08_dpo_loss.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_training_overview(pretrain_result, sft_result,
                            dpo_epoch_losses) -> None:
    """All three phases on one chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))

    # Pre-training
    pt = pretrain_result.epoch_losses
    ax.plot(range(1, len(pt) + 1), pt,
            marker="o", linewidth=2, markersize=4,
            color="#2196F3", label="Pre-training (raw text)")

    # SFT
    sft_start = len(pt)
    sft = sft_result.epoch_losses
    ax.plot(range(sft_start + 1, sft_start + len(sft) + 1), sft,
            marker="s", linewidth=2, markersize=4,
            color="#4CAF50", label="SFT (instruction format)")

    # DPO (different y-scale, use twin axis)
    ax2 = ax.twinx()
    dpo_start = sft_start + len(sft)
    ax2.plot(range(dpo_start + 1, dpo_start + len(dpo_epoch_losses) + 1),
             dpo_epoch_losses,
             marker="^", linewidth=2, markersize=4,
             color="#9C27B0", label="DPO (preference)")
    ax2.set_ylabel("DPO Loss", color="#9C27B0")
    ax2.tick_params(axis="y", labelcolor="#9C27B0")

    # Phase dividers
    ax.axvline(x=sft_start + 0.5, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=dpo_start + 0.5, color="gray", linestyle="--", alpha=0.4)
    ax.text(sft_start / 2, ax.get_ylim()[1] * 0.95, "Pre-train",
            ha="center", fontsize=8, color="gray")
    ax.text(sft_start + len(sft) / 2, ax.get_ylim()[1] * 0.95, "SFT",
            ha="center", fontsize=8, color="gray")
    ax.text(dpo_start + len(dpo_epoch_losses) / 2,
            ax.get_ylim()[1] * 0.95, "DPO",
            ha="center", fontsize=8, color="gray")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Chapter 08: Full Training Pipeline")
    ax.legend(loc="upper left", fontsize=8)
    ax2.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(RESULTS_DIR / "ch08_training_overview.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
