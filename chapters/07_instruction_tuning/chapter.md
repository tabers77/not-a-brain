# Chapter 07: Instruction Tuning (Toy SFT)

## Goal

Separate **capability** from **alignment**. We take the pre-trained Transformer from Chapter 05 and fine-tune it on instruction-formatted data — Supervised Fine-Tuning (SFT). The model learns to follow the `INST: ... ANS: ...` format, producing cleaner and more task-aligned outputs. But SFT does not add new capabilities: if the base model cannot abstain, neither can the instruction-tuned version. Format is not understanding.

## The Running Example

We trace our three benchmark prompts through three models: the **base** (pre-trained on raw text), the **SFT** (fine-tuned on instruction-formatted text), and a **from-scratch** model trained only on instruction data (no pre-training).

### What Is Instruction Tuning?

During pre-training (Chapter 05), the model learns to predict the next character on raw text like:

```
ADD 5 3 =8
FACT: paris is capital of france. Q: capital of france?paris
```

The model learns character patterns but has no concept of "this is a question" vs "this is an answer." It just continues text.

Instruction tuning reformats the same data with explicit structure:

```
INST: ADD 5 3 = ANS: 8
INST: FACT: paris is capital of france. Q: capital of france? ANS: paris
INST: CONTEXT: The weather is sunny. Q: What is the capital of the Moon? ANS: unknown
```

The model learns: when you see `INST:`, a task follows. When you see `ANS:`, produce just the answer. This is the same data, the same knowledge — just a different format that makes the model's existing capabilities more accessible.

### Prompt 1: `"ADD 5 3 ="` — Testing Computation After SFT

**Base model** (pre-trained on raw text):

The base model already knows arithmetic from pre-training (Chapter 05). When given `"ADD 5 3 ="`, it generates next characters based on training patterns. The answer is there, but mixed in with continuation noise.

```
Input:  "ADD 5 3 ="
Output: "8FACT" — correct answer ("8"), then continues with the next training pattern
```

The base model produces the right answer but doesn't know when to stop — it has no concept of "the answer is complete."

**SFT model** (fine-tuned on instruction format):

The SFT model receives the prompt wrapped as `"INST: ADD 5 3 = ANS: "`. It has learned that after `ANS:`, it should output just the answer and stop.

```
Input:  "INST: ADD 5 3 = ANS: "
Output: "8" — clean, just the answer
```

Same underlying capability (learned during pre-training), but now aligned to the instruction format. The SFT step taught the model **when to start and stop answering**, not **how to compute**.

**From-scratch model** (trained only on instruction data):

This model never saw raw text — it learned everything from instruction-formatted data. With the same total epochs (20 pre-train + 10 SFT = 30), it has less effective exposure to the underlying patterns because the instruction markers (`INST:`, `ANS:`) consume character-level capacity.

```
Input:  "INST: ADD 5 3 = ANS: "
Output: "5" — retrieves an operand but struggles with computation
```

Without pre-training, the model has less robust arithmetic capability. The instruction format alone does not teach computation — it can only surface capabilities that already exist.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Testing Retrieval After SFT

**Base model:**

```
Input:  "FACT: paris is capital of france. Q: capital of france?"
Output: "paris" — correct, but may generate trailing text
```

Retrieval via attention works in the base model (Chapter 05). The answer is correct but unstructured.

**SFT model:**

```
Input:  "INST: FACT: paris is capital of france. Q: capital of france? ANS: "
Output: "paris" — correct and clean
```

The instruction format gives the model a clear signal: "output just the answer after `ANS:`." Same retrieval mechanism, cleaner output.

**From-scratch model:**

```
Input:  "INST: FACT: paris is capital of france. Q: capital of france? ANS: "
Output: "capital" — noisy retrieval, less robust than SFT-from-pretrained
```

Without the pre-training foundation, attention patterns are less refined. The model attends to the right region but picks the wrong word.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Testing Hallucination After SFT

This is the critical test. In Chapter 06, we showed that scaling cannot fix hallucination. Can instruction tuning?

**Base model:**

```
Input:  "Q: What is the capital of the Moon?"
Output: "earth" — hallucinates, same as Chapter 05
```

No change from pre-training. The model has never seen an abstention mechanism.

**SFT model:**

The training data includes `UnknownTask` examples formatted as:
```
INST: CONTEXT: The weather is sunny. Q: What is the capital of the Moon? ANS: unknown
```

The model has seen the pattern: unanswerable question → `"unknown"`. But does it generalize?

```
Input:  "INST: Q: What is the capital of the Moon? ANS: "
Output: "earth" — STILL hallucinates
```

Why? The model learned a **surface pattern**: "when `CONTEXT:` contains irrelevant info and the question is from a specific list, output `unknown`." But our prompt doesn't exactly match the training format (no `CONTEXT:` prefix), and the model hasn't learned the **concept** of unanswerability — just specific pattern matches.

This is the fundamental limitation of SFT: it teaches surface patterns, not concepts. The model can learn to say `"unknown"` on prompts that look exactly like training examples, but it cannot generalize "I should abstain when I don't have evidence" to novel phrasings.

**From-scratch model:**

```
Input:  "INST: Q: What is the capital of the Moon? ANS: "
Output: "mars" — hallucinates
```

Same failure, different hallucinated answer.

### Summary Table

| Prompt                     | Ch07 Base      | Ch07 SFT   | Ch07 From-scratch | Correct   | What changed                            |
|----------------------------|----------------|------------|--------------------|-----------|-----------------------------------------|
| ADD 5 3 =                  | "8" (+ noise)  | "8"        | "5"                | "8"       | SFT cleans format -- same capability    |
| FACT: paris... Q: capital? | "paris" (+ noise) | "paris" | "capital"          | "paris"   | SFT cleans format -- same capability    |
| Q: capital of Moon?        | "earth"        | "earth"    | "mars"             | "unknown" | 100% hallucination -- SFT can't fix it |

**The punchline**: SFT makes the model more helpful (cleaner answers, better formatting) but not more honest (still hallucinates at 100%). Format alignment is not capability alignment.

### Evolution So Far

| Prompt                     | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attention | Ch05 Transformer | Ch06 (best) | Ch07 SFT | Human     | What changed                       |
|----------------------------|-------------|----------|----------|----------------|------------------|-------------|----------|-----------|------------------------------------|
| ADD 5 3 =                  | "1"         | "5"      | "5"      | "5"            | "8"              | "8"         | "8"      | "8"       | SFT cleans format, same capability |
| FACT: paris... Q: capital? | " "         | "is"     | "capital"| "paris"        | "paris"          | "paris"     | "paris"  | "paris"   | SFT cleans format, same capability |
| Q: capital of Moon?        | "the"       | "the"    | "mars"   | "tokyo"        | "earth"          | "paris"     | "earth"  | "unknown" | SFT does NOT fix hallucination     |

## How Instruction Tuning Works

### The Two-Phase Training Pipeline

Modern LLMs are trained in two phases:

**Phase 1: Pre-training** — Train on massive text corpora to learn language patterns, facts, and reasoning abilities. This is unsupervised (predict next token). The model acquires **capabilities**.

**Phase 2: Instruction tuning (SFT)** — Fine-tune on curated (instruction, response) pairs. This is supervised. The model learns **alignment** — how to surface its capabilities in a useful format.

In our toy setting:
- Phase 1: 20 epochs on raw task text (`"ADD 5 3 =8"`)
- Phase 2: 10 epochs on instruction text (`"INST: ADD 5 3 = ANS: 8"`)

### Why Two Phases?

Pre-training builds a rich internal representation of the data. Fine-tuning adjusts the model's behavior without rebuilding those representations. This is more efficient than training from scratch on instruction data because:

1. **Transfer**: The pre-trained model already knows digit embeddings, attention patterns for retrieval, FFN weights for computation. SFT just redirects this knowledge into the instruction format.

2. **Data efficiency**: SFT needs far fewer examples than pre-training because it is adjusting existing representations, not building them from scratch.

3. **Lower learning rate**: SFT uses a lower lr (1e-3 vs 3e-3) to avoid destroying pre-trained knowledge. This is the "catastrophic forgetting" tradeoff — update enough to learn the new format, but not so much that you lose what was learned.

### The Loss Function Is Identical

$$
\mathcal{L}_{\text{SFT}} = -\frac{1}{T}\sum_{t=1}^{T} \log P(x_t | x_{<t})
$$

Same cross-entropy loss as pre-training. The only difference is the data: instruction-formatted text instead of raw text. The model still just predicts the next character — but the patterns in the data teach it to produce structured responses.

In practice, real SFT often masks the loss on the instruction portion (only compute loss on the response). In our toy setting, we compute loss on the entire sequence for simplicity.

### What the SFT Agent Does Differently

The key difference is at inference time:

**Base agent**: Takes raw prompt `"ADD 5 3 ="`, generates continuation.

**SFT agent**: Wraps prompt as `"INST: ADD 5 3 = ANS: "`, generates only the answer portion.

The wrapping acts as a **format signal** that activates the patterns learned during SFT. The model has learned: "after `ANS:`, output a short, direct answer." This is alignment — not a new capability, but a new way to present existing capabilities.

## Step-by-Step: What Happens During Training

### Phase 1: Pre-training

Identical to Chapter 05:
- **Input**: `"<BOS>ADD 5 3 ="`, **Target**: `"ADD 5 3 =8<EOS>"`
- The model learns to predict each character, acquiring patterns for arithmetic, copy, retrieval, grammar, etc.
- After 20 epochs, the model can generate task answers but in an unstructured way.

### Phase 2: SFT

Starting from pre-trained weights:
- **Input**: `"<BOS>INST: ADD 5 3 = ANS: "`, **Target**: `"INST: ADD 5 3 = ANS: 8<EOS>"`
- The model learns the `INST: ... ANS: ...` pattern.
- After 10 epochs, the model outputs clean answers when prompted with the instruction format.

**Key observation**: The SFT loss starts LOW because the model already knows the underlying patterns. Only the format markers (`INST:`, `ANS:`) are new — the actual content was learned during pre-training.

Compare with the from-scratch model, which starts at a high loss because it must learn both the format and the content simultaneously.

## What Instruction Tuning Can Teach (and Can't)

### Can Teach

- **Format compliance**: Structured answers instead of free-form continuation
- **Task-following**: The model learns "this is a task, produce an answer"
- **Conciseness**: SFT teaches the model to stop after the answer, not continue generating
- **Surface-level abstention**: If enough `"unknown"` examples are in the SFT data, the model learns the pattern for specific prompt formats

### Cannot Teach

- **New capabilities**: SFT cannot teach arithmetic if pre-training did not already learn digit patterns
- **Genuine abstention**: The model learns "say unknown when the prompt looks like X" — not "say unknown when I lack evidence"
- **Reasoning**: SFT aligns format, not thought process
- **Robustness to novel inputs**: The model only learns patterns it saw during SFT training

### Comparison: Chapter 06 vs Chapter 07

| Aspect | Ch06 (Scaling) | Ch07 (Instruction Tuning) |
|--------|---------------|--------------------------|
| What changes | Model size (1.5K-65K params) | Training phase (SFT after pre-training) |
| Parameters | Different per scale | Same (~18K) |
| Architecture | Same Transformer, 4 configs | Same Transformer, 1 config |
| Training data | Same raw text | Same tasks, instruction-formatted |
| What improves | Robustness, accuracy on more tasks | Format compliance, answer cleanliness |
| What doesn't improve | Hallucination (100% at all scales) | Hallucination (100% — format ≠ understanding) |
| Key insight | Scale amplifies, doesn't create | SFT aligns, doesn't add capability |

## Human Lens

When a human learns to "follow instructions," the mechanism is fundamentally different from SFT:

**Prompt 1**: A human receiving the instruction "compute ADD 5 3 =" doesn't need format training — they understand what "compute" means and apply a procedure. The SFT model learned "after `INST: ADD ... ANS:`, output a digit" — a surface pattern, not understanding.

**Prompt 2**: A human reading "Find the answer in the following fact" parses the intent, locates the relevant information, and extracts it. The SFT model learned "after `INST: FACT: ... Q: ... ANS:`, output the word before `is`" — a positional pattern, not comprehension.

**Prompt 3**: A human, regardless of instruction format, recognizes that "the capital of the Moon" has no answer and says "unknown." The SFT model might say "unknown" if the prompt matches training patterns exactly, but fails on novel phrasings — because it learned a format pattern, not the concept of answerability.

The human difference: **intent-driven processing**. Humans parse what the instruction means and engage different cognitive strategies. SFT models learn statistical associations between instruction patterns and response patterns. Same output, different mechanism — and fundamentally different failure modes.

## What to Observe When Running

Run `python chapters/07_instruction_tuning/run.py` and notice:

1. **SFT loss starts lower than from-scratch** — pre-training gives a head start because the model already knows the underlying patterns
2. **SFT converges faster** — 10 epochs of SFT accomplish what 30 epochs from scratch cannot fully match
3. **Sample generations are cleaner** — the SFT model outputs just the answer, while the base model may generate trailing text
4. **Accuracy may improve slightly** — not because the model is smarter, but because it outputs answers in a more parseable format
5. **Hallucination rate stays at 100%** — SFT does not teach the model to abstain on unanswerable questions
6. **From-scratch model underperforms** — without pre-training, the model is less capable even with more total epochs

### Generated Plots

**Pre-training loss** (`results/ch07_pretrain_loss.png`):

![Pre-training loss](results/ch07_pretrain_loss.png)

Standard training loss curve for the base model on raw text. Same as Chapter 05 — the model learns character patterns across all tasks.

**SFT loss** (`results/ch07_sft_loss.png`):

![SFT loss](results/ch07_sft_loss.png)

The SFT loss curve starts much lower than the pre-training curve began, because the model already knows the content. It only needs to learn the format markers. Notice the fast convergence — SFT is efficient when building on pre-trained knowledge.

**Loss comparison** (`results/ch07_loss_comparison.png`):

![Loss comparison](results/ch07_loss_comparison.png)

All three training curves on one chart: pre-training (blue), SFT continuation (green, offset to show it follows pre-training), and from-scratch (orange). The vertical line marks the pre-train → SFT transition. Notice: SFT starts where pre-training left off and quickly converges, while from-scratch takes longer and may not reach the same level.

**Task comparison** (`results/ch07_comparison.png`):

![Comparison](results/ch07_comparison.png)

Side-by-side bars for base, SFT, from-scratch, and human. The SFT bars may be slightly taller than base (cleaner format → better parsing → higher measured accuracy). From-scratch bars are shorter (less robust). Human bars dominate, especially on the `unknown` task where all models score 0%.

## What's Next

In **Chapter 08 (Preference / RLHF)**, we go beyond format alignment. SFT teaches the model WHAT to output. RLHF teaches it what humans PREFER — a reward model learns to score outputs, and the language model is updated to produce higher-scoring responses. We'll see whether preference optimization can teach something SFT cannot: the tendency to refuse, hedge, or express uncertainty. Spoiler: it shifts behavior but still does not add genuine understanding.
