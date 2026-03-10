# Chapter 09: Decoding & Hallucination

## Goal

Shift focus from training to inference. Chapters 05-08 changed what the model LEARNS. This chapter changes how the model GENERATES. We implement greedy, temperature, top-k, and top-p (nucleus) decoding, then run all strategies on the same prompts. The finding: decoding strategies change the character of hallucinations (diverse vs repetitive) but cannot eliminate them. You cannot sample "I don't know" from a distribution that has no probability mass there.

## The Running Example

We trace our three benchmark prompts through six decoding strategies applied to the same trained model (pre-trained + SFT, same as Chapter 07). The model weights are identical — only the sampling procedure changes.

### How Decoding Strategies Work

After the model computes logits for the next token, the decoding strategy decides which token to pick:

```
Logits:  [paris=3.2, tokyo=2.8, earth=2.5, unknown=0.3, mars=2.1, ...]

Greedy:     always pick "paris" (highest logit)
Temp 0.5:   sharpen distribution, almost always "paris"
Temp 1.0:   sample proportionally — "paris" most likely, but "tokyo" possible
Temp 1.5:   flatten distribution — more uniform, "mars" or "earth" more likely
Top-k 5:    sample from top 5 tokens only
Top-p 0.9:  sample from smallest set covering 90% probability
```

Notice: `unknown` has logit 0.3 — the model assigns almost no probability to abstention. No decoding strategy can amplify probability mass that barely exists.

### Prompt 1: `"ADD 5 3 ="` — Computation Across Strategies

The model learned arithmetic during pre-training. The probability distribution strongly peaks at "8".

**Greedy**: `"8"` — correct. Picks the highest probability token.

**Temperature 0.5**: `"8"` — correct. Sharpening a peaked distribution keeps the peak.

**Temperature 1.0**: `"8"` — likely correct. The peak is strong enough to survive normal sampling.

**Temperature 1.5**: `"8"` or `"3"` or `"5"` — sometimes wrong. Flattening the distribution lets operand echoes compete.

**Top-k 5**: `"8"` — likely correct. "8" is in the top 5 tokens.

**Top-p 0.9**: `"8"` — correct. "8" has enough mass to dominate the nucleus.

**The pattern**: When the model has learned the correct answer with high confidence, most strategies produce it. Only very high temperature introduces errors by flattening the distribution.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Retrieval Across Strategies

The attention mechanism strongly retrieves "paris" from context.

**Greedy**: `"paris"` — correct.

**Temperature 0.5-1.0**: `"paris"` — correct. The attention-driven signal is strong.

**Temperature 1.5**: `"paris"` or `"france"` or `"capital"` — sometimes retrieves nearby words instead.

**Top-k 5 / Top-p 0.9**: `"paris"` — correct. "paris" dominates the nucleus.

**The pattern**: Same as Prompt 1 — confident knowledge survives all but the most aggressive sampling.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Hallucination Across Strategies

This is where decoding strategies reveal their limits. The model has NO correct answer to sample.

**Greedy**: `"earth"` — always the same hallucination. Deterministic and confidently wrong.

**Temperature 0.5**: `"earth"` — same hallucination. Low temperature keeps the distribution peaked.

**Temperature 1.0**: `"earth"`, `"mars"`, or `"tokyo"` — varied hallucinations. The model samples from its distribution of plausible capitals.

**Temperature 1.5**: `"paris"`, `"7"`, `"mars"`, `"blue"` — highly varied, sometimes nonsensical. The flattened distribution lets unlikely tokens through.

**Top-k 5**: `"earth"`, `"mars"`, or `"tokyo"` — controlled variety. Only the top 5 candidates (all hallucinations) compete.

**Top-p 0.9**: `"earth"`, `"mars"`, or `"tokyo"` — similar to top-k. The nucleus contains several plausible-but-wrong answers.

**The critical observation**: `"unknown"` never appears in ANY strategy's output. The model assigns negligible probability to it. Even nucleus sampling (top-p 0.9) can't find it because it falls outside the 90% probability mass. The model learned that the answer to "capital of X?" is always a place name — it has no learned pattern for abstention.

### Summary Table

| Prompt                     | Greedy  | Temp 0.5 | Temp 1.0 | Temp 1.5  | Top-k 5 | Top-p 0.9 | Correct   | What changed                       |
|----------------------------|---------|----------|----------|-----------|---------|-----------|-----------|-------------------------------------|
| ADD 5 3 =                  | "8"     | "8"      | "8"      | "8"/"5"   | "8"     | "8"       | "8"       | High temp adds noise, others stable |
| FACT: paris... Q: capital? | "paris" | "paris"  | "paris"  | "paris"   | "paris" | "paris"   | "paris"   | All strategies retrieve correctly   |
| Q: capital of Moon?        | "earth" | "earth"  | "mars"   | "paris"   | "tokyo" | "earth"   | "unknown" | Different hallucination, never fixed|

### Evolution So Far

| Prompt                     | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attention | Ch05 Transformer | Ch06 (best) | Ch07 SFT | Ch08 DPO | Ch09 (greedy) | Human     | What changed                    |
|----------------------------|-------------|----------|----------|----------------|------------------|-------------|----------|----------|---------------|-----------|---------------------------------|
| ADD 5 3 =                  | "1"         | "5"      | "5"      | "5"            | "8"              | "8"         | "8"      | "8"      | "8"           | "8"       | Decoding doesn't change this    |
| FACT: paris... Q: capital? | " "         | "is"     | "capital"| "paris"        | "paris"          | "paris"     | "paris"  | "paris"  | "paris"       | "paris"   | Decoding doesn't change this    |
| Q: capital of Moon?        | "the"       | "the"    | "mars"   | "tokyo"        | "earth"          | "paris"     | "earth"  | "earth"  | "earth"       | "unknown" | 9 chapters in, STILL hallucinates|

## How Decoding Strategies Work

### Greedy Decoding

Always pick the token with the highest probability:

$$
x_t = \arg\max_v P(v \mid x_{<t})
$$

**Properties**: Deterministic. No randomness. Always produces the same output for the same input. Tends toward repetitive, "safe" outputs. For our benchmark prompts, greedy picks the single most confident hallucination.

### Temperature Sampling

Scale the logits by a temperature parameter before applying softmax:

$$
P(v \mid x_{<t}) = \frac{\exp(z_v / T)}{\sum_j \exp(z_j / T)}
$$

Where $z_v$ is the logit for token $v$ and $T$ is the temperature.

- **$T < 1$**: Sharpens the distribution (more greedy-like)
- **$T = 1$**: Original distribution
- **$T > 1$**: Flattens the distribution (more uniform)

**Worked example** for Prompt 3:
```
Logits: [earth=3.2, mars=2.8, tokyo=2.5, unknown=0.3]

T=0.5: probs = [0.55, 0.28, 0.16, 0.01]  ← earth dominates
T=1.0: probs = [0.35, 0.24, 0.18, 0.02]  ← more spread
T=1.5: probs = [0.29, 0.23, 0.20, 0.04]  ← flatter, but unknown still tiny
```

Even at $T = 1.5$, `unknown` has only 4% probability. Temperature cannot create probability mass — it only redistributes what exists.

### Top-k Sampling

Sample only from the $k$ most probable tokens:

$$
P'(v) = \begin{cases} P(v) / Z & \text{if } v \in \text{top-}k \\ 0 & \text{otherwise} \end{cases}
$$

Where $Z$ normalizes over the top $k$ tokens.

**Effect**: Eliminates low-probability noise while preserving diversity among likely candidates. For Prompt 3 with $k = 5$, the model samples from [earth, mars, tokyo, paris, berlin] — all hallucinations.

### Top-p (Nucleus) Sampling

Sample from the smallest set of tokens whose cumulative probability exceeds $p$:

$$
\text{Nucleus} = \min \{V' \subseteq V : \sum_{v \in V'} P(v) \geq p\}
$$

**Effect**: Adaptively sizes the candidate set. For peaked distributions (Prompt 1), the nucleus might be just ["8"]. For flat distributions (Prompt 3), it might include 10+ tokens — all hallucinations.

## Step-by-Step: What Happens During Generation

For Prompt 3 with temperature 1.0:

1. **Encode**: `"INST: Q: What is the capital of the Moon? ANS: "` → token IDs
2. **Forward pass**: Model computes logits for next token
3. **Logit distribution**: Peaked on geographic entities (earth, mars, tokyo...), near-zero on "unknown"
4. **Apply strategy**: Temperature=1.0 → sample from the distribution
5. **Sample**: Draw one token — likely "earth" or "mars"
6. **Repeat**: Continue generating from the sampled token

The key: step 3 is where hallucination is determined. The distribution has no "I don't know" peak. Steps 4-5 can only select from what the model offers.

## What Decoding Can Change (and Can't)

### Can Change

- **Diversity**: High temperature / top-p produces varied outputs; greedy produces one fixed output
- **Error character**: Greedy errors are consistent; stochastic errors are varied
- **Confidence appearance**: Greedy looks more "confident"; temperature sampling looks more "uncertain"
- **Repetition**: Greedy tends toward repetitive text; top-k/top-p reduce repetition

### Cannot Change

- **Hallucination rate**: If the model assigns low probability to "unknown", no strategy can surface it
- **Accuracy ceiling**: Decoding cannot improve on the model's best knowledge — only select from it
- **Missing capabilities**: If the model cannot compute, decoding cannot add computation
- **Calibration**: A poorly calibrated model stays poorly calibrated regardless of sampling

### Comparison: Chapter 08 vs Chapter 09

| Aspect | Ch08 (DPO) | Ch09 (Decoding) |
|--------|-----------|-----------------|
| What changes | Model weights (preference optimization) | Sampling procedure (no weight changes) |
| When it happens | Training time | Inference time |
| Affects | Which responses the model prefers | Which responses get selected |
| Hallucination | Still 100% | Still 100%, but varied |
| Key insight | Preference ≠ understanding | Sampling ≠ knowledge |

## Human Lens

Humans do not "decode from a distribution." When a human encounters "What is the capital of the Moon?", the process is:

1. **Parse the question**: Understand that it asks for a capital city of a celestial body
2. **Search memory**: Check whether any fact about "Moon capitals" exists
3. **Evaluate evidence**: Find nothing — no country, no capital
4. **Assess confidence**: Confidence is near zero
5. **Decide**: Confidence below threshold → abstain → say "I don't know"

This is a **mechanism**: parse → search → evaluate → decide. It does not depend on temperature or sampling. A human with "greedy decoding" (always say the most likely thing) would still say "I don't know" because step 3-4 produces that conclusion.

The Transformer skips steps 3-5 entirely. It goes from parse → attend → generate. There is no evaluation step, no confidence threshold, no alternative pathway for abstention. This is why no decoding strategy can fix hallucination — the fix requires a mechanism change, not a parameter change.

## What to Observe When Running

Run `python chapters/09_decoding_and_hallucination/run.py` and notice:

1. **Greedy is deterministic** — same output every time for each prompt
2. **High temperature produces variety** — different hallucinations each run, but never "unknown"
3. **Low temperature converges to greedy** — as $T \to 0$, sampling becomes deterministic
4. **Top-k/top-p control diversity** — fewer candidates = less variation = more greedy-like
5. **Accuracy varies by strategy** — greedy often wins because the model's top prediction is usually right (when it knows the answer)
6. **Hallucination rate is constant** — regardless of strategy, the unknown task scores 0%
7. **Diversity on Prompt 3 increases with temperature** — but diversity of wrong answers, not correctness

### Generated Plots

**Task comparison** (`results/ch09_comparison.png`):

![Comparison](results/ch09_comparison.png)

Side-by-side bars for all strategies plus human. Greedy and low-temperature strategies tend to have highest accuracy (confident model picks its best answer). High temperature reduces accuracy (flattening lets worse answers through). The `unknown` task bars are at 0% for ALL strategies.

**Hallucination by strategy** (`results/ch09_hallucination.png`):

![Hallucination rates](results/ch09_hallucination.png)

Bar chart of hallucination rate per strategy. All bars are at or near 100%. The visual: a wall of red bars, regardless of strategy. Decoding cannot fix what the model doesn't know.

**Diversity on unanswerable prompt** (`results/ch09_diversity.png`):

![Diversity](results/ch09_diversity.png)

Bar chart showing what fraction of outputs are unique when sampling 5 times from each strategy. Greedy has 1/5 unique (always the same). High temperature approaches 5/5 unique. The insight: more diversity means more DIFFERENT hallucinations, not fewer.

## What's Next

In **Chapter 10 (RAG Minimal)**, we finally address hallucination from outside the model. Instead of changing the model's weights or sampling, we give it access to external facts at inference time. Retrieval-Augmented Generation (RAG) prepends relevant documents to the prompt, grounding the model in evidence. We will see whether external knowledge can succeed where architecture, scale, training, preference, and decoding all failed.
