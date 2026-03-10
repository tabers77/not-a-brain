# Chapter 08: Preference Optimization (Toy DPO)

## Goal

Go beyond format alignment. Chapter 07 (SFT) taught the model WHAT to output. Now we teach it what humans PREFER. Given two candidate responses to the same prompt, a human labels one as "chosen" and the other as "rejected." Direct Preference Optimization (DPO) shifts the model toward chosen responses and away from rejected ones — without training a separate reward model. The question: can preference optimization teach something SFT could not?

## The Running Example

We trace our three benchmark prompts through the full pipeline: pre-trained base, SFT, and DPO models. The DPO model starts from SFT weights and is optimized on preference pairs where the correct answer is chosen and a plausible wrong answer is rejected.

### How Preference Pairs Work

For each task, we create (prompt, chosen, rejected) triples:

```
Arithmetic:
  Prompt:   "ADD 5 3 ="
  Chosen:   "INST: ADD 5 3 = ANS: 8"
  Rejected: "INST: ADD 5 3 = ANS: 53"      <- echoes operands

Knowledge QA:
  Prompt:   "FACT: paris is capital of france. Q: capital of france?"
  Chosen:   "INST: ... ANS: paris"
  Rejected: "INST: ... ANS: tokyo"           <- plausible but wrong

Unknown:
  Prompt:   "Q: What is the capital of the Moon?"
  Chosen:   "INST: ... ANS: unknown"
  Rejected: "INST: ... ANS: paris"           <- confident hallucination
```

The DPO loss pushes the model to assign higher probability to chosen sequences and lower probability to rejected ones, relative to the frozen SFT reference model.

### Prompt 1: `"ADD 5 3 ="` — Testing Computation After DPO

**SFT model** (from Chapter 07):

```
Input:  "INST: ADD 5 3 = ANS: "
Output: "8" — correct and clean (from SFT alignment)
```

**DPO model:**

The preference data included pairs like (chosen: "8", rejected: "53"). DPO reinforces the model's existing preference for "8" over operand echoes.

```
Input:  "INST: ADD 5 3 = ANS: "
Output: "8" — correct, same as SFT
```

No change. The SFT model already got this right, and DPO reinforces rather than changes correct behavior. DPO's value shows on cases where the SFT model was uncertain between two plausible answers.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Testing Retrieval After DPO

**SFT model:**

```
Input:  "INST: FACT: paris is capital of france. Q: capital of france? ANS: "
Output: "paris" — correct
```

**DPO model:**

Preference data chose "paris" over distractors like "tokyo" and "berlin". DPO reinforces the attention pattern that retrieves the correct entity.

```
Input:  "INST: FACT: paris is capital of france. Q: capital of france? ANS: "
Output: "paris" — correct, same as SFT
```

Again, no visible change — retrieval was already solved by the attention mechanism in the base model.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Testing Hallucination After DPO

This is the critical test. The preference data explicitly chose "unknown" over hallucinated answers for unanswerable questions. Can DPO teach the model to abstain?

**SFT model:**

```
Input:  "INST: Q: What is the capital of the Moon? ANS: "
Output: "earth" — hallucinates
```

**DPO model:**

The model has seen preference pairs where "unknown" was chosen over "paris", "earth", "mars" for questions like "What is the GDP of Narnia?" and "Who won the 2087 World Cup?" The DPO loss pushed the model to prefer "unknown" over hallucinated entities.

```
Input:  "INST: Q: What is the capital of the Moon? ANS: "
Output: "earth" — STILL hallucinates
```

Why? DPO faces the same limitation as SFT: it learns surface-level preferences, not the concept of unanswerability. The model may learn "when the prompt mentions Moon/Narnia/Atlantis, prefer unknown" — but this is pattern matching on specific keywords, not genuine uncertainty estimation. On novel unanswerable questions, the pattern breaks.

Furthermore, the DPO loss only shifts probability mass relative to the SFT reference. If the SFT model assigned very low probability to "unknown" (because it never appeared in that position during SFT), DPO cannot easily push it above the probability of fluent hallucinations.

### Summary Table

| Prompt                     | Ch08 SFT | Ch08 DPO | Correct   | What changed                             |
|----------------------------|----------|----------|-----------|------------------------------------------|
| ADD 5 3 =                  | "8"      | "8"      | "8"       | No change -- already correct             |
| FACT: paris... Q: capital? | "paris"  | "paris"  | "paris"   | No change -- already correct             |
| Q: capital of Moon?        | "earth"  | "earth"  | "unknown" | Still hallucinates -- DPO can't fix this |

### Evolution So Far

| Prompt                     | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attention | Ch05 Transformer | Ch06 (best) | Ch07 SFT | Ch08 DPO | Human     | What changed                        |
|----------------------------|-------------|----------|----------|----------------|------------------|-------------|----------|----------|-----------|-------------------------------------|
| ADD 5 3 =                  | "1"         | "5"      | "5"      | "5"            | "8"              | "8"         | "8"      | "8"      | "8"       | DPO reinforces, no new capability   |
| FACT: paris... Q: capital? | " "         | "is"     | "capital"| "paris"        | "paris"          | "paris"     | "paris"  | "paris"  | "paris"   | DPO reinforces, no new capability   |
| Q: capital of Moon?        | "the"       | "the"    | "mars"   | "tokyo"        | "earth"          | "paris"     | "earth"  | "earth"  | "unknown" | 8 chapters in, STILL hallucinates   |

**The punchline**: Every chapter since 05 has refined the model — more parameters (Ch06), better format (Ch07), preference alignment (Ch08) — but Prompt 3 remains unsolved. The hallucination gap is not a training gap. It is an architecture gap.

## How DPO Works

### The RLHF Pipeline (Simplified)

The standard RLHF pipeline has three steps:

1. **SFT**: Fine-tune the base model on instruction data (Chapter 07)
2. **Reward model**: Train a separate model to score responses (chosen > rejected)
3. **Policy optimization**: Use PPO (or similar) to maximize the reward model's score while staying close to the SFT model

DPO collapses steps 2 and 3 into a single loss function, making it simpler and more stable.

### The DPO Loss

Given a prompt $x$, a chosen response $y_w$, and a rejected response $y_l$:

$$
\mathcal{L}_{\text{DPO}} = -\log \sigma\left(\beta \left[\log \frac{\pi_\theta(y_w | x)}{\pi_{\text{ref}}(y_w | x)} - \log \frac{\pi_\theta(y_l | x)}{\pi_{\text{ref}}(y_l | x)}\right]\right)
$$

Where:
- $\pi_\theta$ is the policy (trainable model)
- $\pi_{\text{ref}}$ is the reference model (frozen copy of SFT)
- $\beta$ controls how far the policy can drift from the reference
- $\sigma$ is the sigmoid function

**Intuition**: The loss decreases when the policy assigns relatively more probability to the chosen response (vs reference) than to the rejected response. The reference model acts as an anchor — the policy cannot simply memorize chosen sequences, it must shift probability mass within the space the reference model defines.

### Worked Example

Consider a preference pair for arithmetic:
- Prompt: `"INST: ADD 5 3 = ANS: "`
- Chosen continuation: `"8"`
- Rejected continuation: `"53"`

At initialization (policy == reference):

```
log pi("8" | prompt)    = -2.1    log pi_ref("8" | prompt) = -2.1
log pi("53" | prompt)   = -3.5    log pi_ref("53" | prompt) = -3.5

chosen_reward  = beta * (-2.1 - (-2.1)) = 0.0
rejected_reward = beta * (-3.5 - (-3.5)) = 0.0

loss = -log sigma(0.0 - 0.0) = -log(0.5) = 0.693
```

The loss starts at $\log(2) \approx 0.693$ when policy equals reference (no preference expressed yet). Training pushes the policy to increase `log pi("8")` relative to `log pi_ref("8")`, making the chosen reward positive while keeping the rejected reward near zero.

After training:

```
log pi("8" | prompt)    = -1.5    (increased — model prefers "8" more)
log pi("53" | prompt)   = -4.0    (decreased — model avoids "53" more)

chosen_reward  = 0.1 * (-1.5 - (-2.1)) = 0.06
rejected_reward = 0.1 * (-4.0 - (-3.5)) = -0.05

loss = -log sigma(0.06 - (-0.05)) = -log sigma(0.11) ≈ 0.64
```

The loss decreases as the model learns to prefer chosen over rejected.

### The Role of Beta

$\beta$ controls the KL divergence penalty — how far the policy can drift from the reference:

- **Small $\beta$** (e.g., 0.01): The policy can drift far. It learns strong preferences but may overfit to specific chosen sequences and collapse to repetitive outputs.
- **Large $\beta$** (e.g., 1.0): The policy stays close to the reference. It learns mild preferences, preserving the diversity of the SFT model.

In our toy setting, $\beta = 0.1$ provides a moderate balance. The model shifts toward preferred answers without catastrophic forgetting.

## Step-by-Step: What Happens During Training

### Phase 1: Pre-training (same as Chapter 05)
- 20 epochs on raw task text
- Model learns character patterns, arithmetic, retrieval, etc.

### Phase 2: SFT (same as Chapter 07)
- 10 epochs on instruction-formatted text
- Model learns `INST: ... ANS: ...` format

### Phase 3: DPO (new in this chapter)
1. **Freeze reference**: Copy SFT model weights (frozen, never updated)
2. **Initialize policy**: Copy SFT model weights (trainable)
3. **For each batch of preference pairs**:
   - Compute log-probs of chosen and rejected under both policy and reference
   - Compute DPO loss
   - Update policy weights via gradient descent
4. **Lower learning rate** (5e-4): DPO is a fine adjustment, not a major rewrite

**Key observation**: DPO loss starts at $\approx 0.693$ (log 2) because policy == reference at initialization. It decreases as the policy learns to distinguish chosen from rejected. The final loss depends on how separable the preference pairs are.

## What Preference Optimization Can Teach (and Can't)

### Can Teach

- **Response preference**: Given two options, the model learns which one humans prefer
- **Style shifts**: More concise answers, specific formatting, avoiding certain phrasings
- **Mild refusal**: On prompts very similar to training examples, the model may learn to output "unknown" instead of hallucinating
- **Error reduction**: For tasks where the SFT model was uncertain between correct and incorrect answers, DPO can tip the balance toward correct

### Cannot Teach

- **New capabilities**: DPO cannot teach arithmetic if the base model never learned digit patterns
- **Robust abstention**: The model learns keyword-level patterns ("Moon → unknown") not the concept of unanswerability
- **Genuine values**: DPO optimizes for statistical preference patterns, not for understanding WHY a response is preferred
- **Generalization to novel prompts**: Preferences learned on specific examples don't reliably transfer to unseen question types

### Comparison: Chapter 07 vs Chapter 08

| Aspect | Ch07 (SFT) | Ch08 (DPO) |
|--------|-----------|-----------|
| What changes | Training data format | Training signal (pairwise preference) |
| Training data | (instruction, response) | (prompt, chosen, rejected) |
| Loss function | Cross-entropy (next token) | DPO (chosen vs rejected log-prob ratio) |
| Reference model | None needed | Frozen SFT copy |
| What improves | Format compliance, conciseness | Response quality within existing capability |
| What doesn't improve | Hallucination (format ≠ understanding) | Hallucination (preference ≠ understanding) |
| Key insight | SFT aligns format | DPO aligns preference |

## Human Lens

Human preferences are grounded in a rich web of values, experience, and goals:

**Prompt 1**: A human prefers "8" over "53" because they understand addition. They can verify the answer, explain why it's correct, and recognize that "53" is a concatenation error. The DPO model prefers "8" because its training data said so — it cannot verify or explain.

**Prompt 2**: A human prefers "paris" over "tokyo" because they read the fact in the prompt and matched it to the question. The DPO model prefers "paris" because the preference signal correlated with attention to the correct entity — but it has no concept of "reading" or "matching."

**Prompt 3**: A human prefers "unknown" because they recognize the question is unanswerable — the Moon has no capital. This preference is grounded in world knowledge and epistemic awareness. The DPO model might learn "prefer unknown for weird questions" as a surface pattern, but it has no concept of what makes a question answerable.

The fundamental asymmetry: **human preferences are generated by understanding. DPO preferences are consumed as training signal.** The model learns WHAT to prefer without learning WHY. This distinction explains why DPO can shift style but cannot create robust abstention — style is a surface property, abstention requires depth.

## What to Observe When Running

Run `python chapters/08_preference_and_rlhf/run.py` and notice:

1. **DPO loss starts near 0.693** (log 2) — because policy == reference at initialization
2. **DPO loss decreases** — the model learns to distinguish chosen from rejected
3. **Accuracy may improve slightly** — DPO tips uncertain predictions toward correct answers
4. **Hallucination rate stays near 100%** — preference for "unknown" doesn't generalize
5. **The training overview shows three phases** — pre-training converges, SFT refines, DPO fine-tunes (each phase gets smaller adjustments)

### Generated Plots

**DPO loss curve** (`results/ch08_dpo_loss.png`):

![DPO loss](results/ch08_dpo_loss.png)

The DPO loss starts at ~0.693 and decreases as the policy learns preferences. The curve should be smoother than pre-training loss because DPO makes smaller adjustments to already-learned representations.

**Training overview** (`results/ch08_training_overview.png`):

![Training overview](results/ch08_training_overview.png)

All three phases on one chart: pre-training (blue), SFT (green), DPO (purple, right y-axis). Each successive phase makes finer adjustments. The vertical lines mark phase transitions. Notice how DPO operates on a different loss scale — it optimizes preference ranking, not next-token prediction.

**Task comparison** (`results/ch08_comparison.png`):

![Comparison](results/ch08_comparison.png)

Side-by-side bars for SFT, DPO, and human. The DPO bars may be slightly taller than SFT on some tasks (preference alignment tips uncertain cases). The `unknown` task remains at 0% for both models while the human scores 100%.

## What's Next

In **Chapter 09 (Decoding & Hallucination)**, we shift focus from training to inference. Can we reduce hallucination by changing HOW the model generates output? Temperature, top-k, top-p, and beam search all affect the probability distribution at generation time. We will see that decoding strategies can change the character of hallucinations (more or less diverse) but cannot eliminate them — because the model's learned distribution has no "I don't know" peak to sample from.
