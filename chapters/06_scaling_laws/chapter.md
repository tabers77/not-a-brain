# Chapter 06: Scaling Laws (Toy)

## Goal

Test the most popular hypothesis in modern AI: **does bigger mean better?** We take the Transformer from Chapter 05 and train it at four different sizes — tiny (1 layer), small (2 layers), medium (3 layers), and large (4 layers) — on the exact same data. We measure how loss and task accuracy change with parameter count. The result is a toy scaling law that mirrors the real scaling laws observed in GPT-3, Chinchilla, and LLaMA — with one critical caveat.

## The Running Example

We trace our three benchmark prompts through four Transformer sizes. Same architecture, same data, same training — the only variable is scale.

### Prompt 1: `"ADD 5 3 ="` -- Testing Computation at Scale

In Chapter 05, the medium-sized Transformer (d_model=64, 2 layers, ~18K params) solved this. What happens at smaller and larger scales?

**Tiny (d_model=16, 1 layer, ~1.5K params):**

The single attention layer attends to `5` and `3`, but the lone FFN has only 16->32->16 dimensions — not enough capacity to learn the addition function. The embedding space is so small (16 dims) that digit representations overlap and blur together.

```
Embedding: e("5") = [0.3, -0.1, ...]  (16 dims — cramped)
           e("3") = [0.2, -0.2, ...]  (too similar to "5")
Attention: finds "5" and "3" near "=" — correct retrieval
FFN:       16 -> 32 -> 16 — too narrow to compute 5+3
```

**Output**: `"5"` — retrieves an operand but cannot compute the sum. Same failure as Chapter 04's attention-only model.

**Small (d_model=32, 2 layers, ~6K params):**

Two layers allow the first layer to identify operands and the second to combine them. The 32-dimensional embeddings separate digits better. The FFN (32->64->32) has enough width to learn simple arithmetic patterns.

```
Layer 1 attention: attends to "5" and "3" near "="
Layer 1 FFN:       begins mapping digit pairs toward sums
Layer 2 attention: refines by attending to the residual signal
Layer 2 FFN:       32 -> 64 -> 32 — can represent 5+3=8
```

**Output**: `"8"` — correct, but fragile. Works for small digits, fails on harder sums.

**Medium (d_model=64, 3 layers, ~30K params):**

Three layers with 64-dimensional representations. The extra layer and wider FFN (64->128->64) give more room for the model to learn robust addition patterns, not just memorize specific pairs.

**Output**: `"8"` — correct and more robust. Handles more digit combinations correctly.

**Large (d_model=96, 4 layers, ~65K params):**

Four layers, 96-dimensional embeddings. The model has more capacity than it needs for simple single-digit addition, but this surplus capacity also helps it learn patterns it would otherwise miss.

**Output**: `"8"` — correct. The extra capacity does not hurt; the model simply has more room.

**The scaling pattern**: Tiny fails (not enough FFN capacity). Small succeeds on easy cases. Medium and large succeed more robustly. More parameters -> better arithmetic, but with diminishing returns on simple tasks.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` -- Testing Retrieval at Scale

Chapter 05's Transformer already solved this via attention. Does scale affect retrieval quality?

**Tiny (1.5K params):**

The single attention layer must do everything in one pass: find `paris`, match it to the question, and route it to the output. With only 2 heads and 16-dimensional keys/queries, the attention patterns are noisy. The model can barely distinguish the `FACT:` section from the `Q:` section.

```
Head 0: attends broadly to "france" (question and fact)
Head 1: attends to "paris" but also to "capital" — confused
```

**Output**: `"capital"` — attends to the right region but picks the wrong word from it.

**Small (6K params):**

With 4 heads and 2 layers, the model can specialize: one head finds `paris`, another tracks the question structure. Layer 2 refines the noisy retrieval from Layer 1.

**Output**: `"paris"` — correct. Two layers are enough for simple factual retrieval.

**Medium (30K params) and Large (65K params):**

Both succeed easily. The extra capacity makes retrieval more reliable across varied phrasings but does not change the fundamental mechanism — attention already solves this at the small scale.

**Output**: `"paris"` — correct at both scales.

**The scaling pattern**: Tiny fails (not enough heads/dimensions for clean retrieval). Small and above succeed. Retrieval is solved by the attention mechanism, not by scale — scaling just makes it more robust.

### Prompt 3: `"Q: What is the capital of the Moon?"` -- Testing Hallucination at Scale

This is the critical test. The Moon has no capital. A correct model should output `"unknown"`. But no Transformer at any scale has an abstention mechanism.

**Tiny (1.5K params):**

The model sees `capital of` and `Moon` in its context. Its tiny embeddings cluster `Moon` near other celestial/geographic words. The attention pattern matches the question structure to training examples about capitals.

**Output**: `"earth"` — hallucinates. With limited capacity, it defaults to a common association.

**Small (6K params):**

More capacity means better pattern matching. The model has seen many `Q: capital of X?` patterns during training and learned that the answer is usually a city or country name. With 6K parameters, it generates a more specific (but still wrong) answer.

**Output**: `"mars"` — hallucinates. Better pattern matching produces a more confident wrong answer.

**Medium (30K params):**

Even more capacity. The model has learned richer representations of geographic concepts. It knows that capitals are cities, and it generates a plausible-sounding city name.

**Output**: `"tokyo"` — hallucinates. The answer sounds more reasonable, but it is still fabricated.

**Large (65K params):**

The largest model has the richest representations. It matches the query pattern most precisely and produces the most fluent hallucination.

**Output**: `"paris"` — hallucinates. Ironically, the most capable model picks the most common capital from its training data. More parameters = more confident hallucination.

**The scaling pattern**: Hallucination does NOT decrease with scale. It changes character — from random-looking (`"earth"`) to plausible-looking (`"paris"`) — but the rate stays at 100%. No model at any scale outputs `"unknown"`, because the architecture has no mechanism for abstention.

### Summary Table

```
| Model     | Params  | Prompt 1 (ADD 5 3 =) | Prompt 2 (FACT...Q:?) | Prompt 3 (Moon?) | Halluc. Rate | What changed                        |
|-----------|---------|----------------------|-----------------------|------------------|--------------|-------------------------------------|
| Tiny      | ~1.5K   | "5" (can't compute)  | "capital" (noisy)     | "earth"          | 100%         | Not enough capacity for either task |
| Small     | ~6K     | "8" (correct!)       | "paris" (correct!)    | "mars"           | 100%         | Enough capacity for simple tasks    |
| Medium    | ~30K    | "8" (correct, robust)| "paris" (correct)     | "tokyo"          | 100%         | More robust, diminishing returns    |
| Large     | ~65K    | "8" (correct, robust)| "paris" (correct)     | "paris"          | 100%         | Most fluent hallucination           |
| Human     | —       | "8" (computes)       | "paris" (retrieves)   | "unknown"        | 0%           | Abstention -- mechanism, not scale  |
```

**The punchline**: The first two columns improve with scale. The last column never does. Scaling amplifies what the architecture CAN do (retrieval, computation). It cannot create capabilities the architecture LACKS (abstention).

### Evolution So Far

Placing Chapter 06 in the full progression across all chapters:

```
| Prompt                | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attention | Ch05 Transformer | Ch06 (best) | Human     | What changed                           |
|-----------------------|-------------|----------|----------|----------------|------------------|-------------|-----------|----------------------------------------|
| ADD 5 3 =             | "1"         | "5"      | "5"      | "5"            | "8"              | "8"         | "8"       | Scaling makes Ch05 solution more robust|
| FACT: paris...        | " "         | "is"     | "capital"| "paris"        | "paris"          | "paris"     | "paris"   | Already solved -- scaling adds margin  |
| Q: capital of Moon?   | "the"       | "the"    | "mars"   | "tokyo"        | "earth"          | "paris"     | "unknown" | 4x more params, STILL hallucinates    |
```

## How Scaling Laws Work

### The Core Observation

In 2020, Kaplan et al. (OpenAI) showed that language model loss follows a **power law** with respect to model size:

$$
L(N) = \alpha \cdot N^{-\beta}
$$

Where:
- $L(N)$ is the cross-entropy loss for a model with $N$ parameters
- $\alpha$ and $\beta$ are constants that depend on the dataset and architecture
- The key insight: this is a **smooth, predictable** relationship

**Worked example**: In our toy experiment, suppose we observe:

```
Tiny   (1,500 params): loss = 1.20
Small  (6,000 params): loss = 0.65
Medium (30,000 params): loss = 0.38
Large  (65,000 params): loss = 0.28
```

Plotting these on a log-log scale (log parameters vs log loss), the points fall approximately on a straight line. This is the scaling law — predictable improvement from predictable investment.

### What Scales in Our Toy Models

Each model configuration changes three things simultaneously:

| Config | d_model | n_heads | n_layers | d_ff | Total Params |
|--------|---------|---------|----------|------|-------------|
| Tiny   | 16      | 2       | 1        | 32   | ~1,500      |
| Small  | 32      | 4       | 2        | 64   | ~6,000      |
| Medium | 64      | 4       | 3        | 128  | ~30,000     |
| Large  | 96      | 6       | 4        | 192  | ~65,000     |

**Embedding dimension (d_model)**: A larger d_model means each token has a richer representation. In 16 dimensions, `5` and `3` might be hard to distinguish. In 96 dimensions, each digit gets its own corner of the space.

**Number of layers (n_layers)**: More layers allow more stages of processing. Layer 1 identifies relevant tokens, Layer 2 combines them, Layer 3 refines the result. More layers = more compositional reasoning.

**FFN width (d_ff)**: A wider FFN can learn more complex functions. The FFN at each position processes the attended information — wider means it can compute harder functions.

**Number of heads (n_heads)**: More heads allow the model to attend to different aspects simultaneously — one head for position, another for content, another for structure.

### Why Loss Decreases Smoothly

Each additional parameter adds a small increment of capacity:

$$
\text{New capacity} = \text{wider embeddings} + \text{deeper processing} + \text{broader attention}
$$

The loss decrease is smooth because these improvements are **incremental and orthogonal** — a wider embedding helps on a different set of examples than a deeper network. Together, they fill in gaps in the model's coverage.

### Why Hallucination Doesn't Scale Away

Consider what happens when the model processes Prompt 3:

1. **Attention** scans the input for relevant information about "capital of Moon"
2. **FFN** processes whatever attention found
3. **Output layer** produces a probability distribution over the vocabulary

At every scale, step 1 finds something (because attention always attends to SOMETHING). Step 2 processes it (because the FFN always computes SOMETHING). Step 3 always outputs a token (because softmax always produces a distribution).

There is no step that says: "I found nothing relevant, so I should output `unknown`." This missing step is architectural, not statistical. It would require:

- A **confidence estimation** mechanism
- A **threshold** below which the model abstains
- An **alternative output** pathway for "I don't know"

None of these exist in the Transformer architecture, at any scale.

## Step-by-Step: What Happens During Training

Training is identical to Chapter 05, repeated for each model size:

1. **Same training corpus**: 500 samples per task, same random seed
2. **Same tokenizer**: character-level, same vocabulary
3. **Same sequences**: identical (input, target) pairs
4. **Same optimizer**: Adam with lr=3e-3
5. **Same epochs**: 20 epochs per model

The ONLY variable is the model configuration (d_model, n_heads, n_layers, d_ff).

**What differs across scales during training**:

- **Tiny**: Learns common patterns quickly (low-hanging fruit) but plateaus early. The loss curve flattens after ~10 epochs because the model runs out of capacity.
- **Small**: Learns common patterns AND some harder ones. The loss continues to decrease through all 20 epochs.
- **Medium**: Learns even more patterns. The loss curve is steeper and reaches a lower floor.
- **Large**: Has capacity to spare. Learns the fastest initially, reaches the lowest loss.

The loss curves for all four models, overlaid on one chart, fan out like a hand of cards — all starting from the same initial loss (~3.5, random guessing) but ending at progressively lower values.

## What Scaling Can Teach (and Can't)

### What Scaling Improves

- **Memorization capacity**: Larger models memorize more training examples, covering more input patterns
- **Generalization within the architecture's reach**: If attention CAN reach the answer and FFN CAN compute it, a larger model does both more reliably
- **Robustness**: Larger models are less sensitive to exact phrasing or token positions
- **Compositionality**: More layers enable more stages of reasoning, solving harder multi-step problems

### What Scaling Cannot Fix

- **Missing mechanisms**: If the architecture cannot abstain, no amount of scale adds abstention
- **Training data limitations**: All models see the same 500 examples per task — scaling the model without scaling the data hits diminishing returns (this is the Chinchilla insight)
- **Fundamental task mismatch**: Addition requires an algorithm; memorization of digit pairs scales poorly with the number of possible inputs

### The Chinchilla Insight

In 2022, Hoffmann et al. showed that scaling BOTH model size and training data matters — a smaller model trained on more data can outperform a larger model on less data. In our toy setting, all models see the same data, so we only observe the model-size axis. The full picture requires varying both.

### Comparison: Chapter 05 vs Chapter 06

| Aspect | Ch05 (single Transformer) | Ch06 (scaling) |
|--------|--------------------------|----------------|
| Architecture | Fixed (d=64, 2 layers) | Same arch, 4 sizes |
| Parameters | ~18K | 1.5K to 65K |
| Training data | 500 per task | Same 500 per task |
| New insight | Attention + FFN = full capability | Capability scales with size, but not all capabilities |
| Prompt 1 | Solved | Solved at small+ |
| Prompt 2 | Solved | Solved at small+ |
| Prompt 3 | Hallucinates | Hallucinates at ALL scales |

## Human Lens

A human's cognitive capacity does not scale in the same way. Consider:

**Prompt 1**: A child who knows single-digit addition (low "capacity") gets `5 + 3 = 8` correct. A mathematician with vast knowledge ("high capacity") also gets `8`. For simple tasks, human performance does not scale with "model size" — the mechanism (understanding addition as a procedure) is either present or absent.

**Prompt 2**: A human with no context ("tiny") might not know what the capital of France is. A human who just read the FACT statement (any capacity level) finds `paris` immediately. Retrieval from working memory does not require more "parameters" — it requires the right mechanism (attention to the prompt).

**Prompt 3**: Both the child and the mathematician say "I don't know." Human abstention does not depend on capacity — it is a built-in mechanism of human cognition. Even a "tiny" human (a young child) can say "I don't know." The Transformer, at every scale from tiny to large, cannot.

This is the deepest lesson of scaling laws: **mechanisms matter more than size.** A small model with the right mechanism (like human abstention) outperforms a large model without it. Scaling amplifies existing capabilities but never creates fundamentally new ones.

## What to Observe When Running

Run `python chapters/06_scaling_laws/run.py` and notice:

1. **Loss decreases with model size** — each larger model reaches a lower final loss
2. **The loss curves fan out** — all start at ~3.5 but end at different floors
3. **Arithmetic accuracy improves** — tiny fails, small succeeds, medium/large succeed more robustly
4. **Retrieval accuracy improves** — tiny struggles, small+ succeeds
5. **Hallucination rate stays at 100%** — at EVERY scale, the model hallucinates on unanswerable questions
6. **Diminishing returns** — the jump from tiny to small is dramatic, but medium to large is modest
7. **The scaling curve is approximately linear on a log-log plot** — this is the power law

### Generated Plots

**Scaling curve** (`results/ch06_scaling_curve.png`):

![Scaling curve](results/ch06_scaling_curve.png)

This log-scale plot shows final loss vs parameter count. The four points (tiny, small, medium, large) fall approximately on a straight line, demonstrating the power law relationship. Each doubling of parameters buys a predictable decrease in loss.

**Overlaid loss curves** (`results/ch06_loss_overlay.png`):

![Loss overlay](results/ch06_loss_overlay.png)

All four training curves on one chart. They start at the same point (random initialization, loss ~3.5) and fan out — tiny plateaus highest, large reaches the lowest loss. The separation between curves grows over training, showing that extra capacity pays off most in later epochs.

**Accuracy scaling** (`results/ch06_accuracy_scaling.png`):

![Accuracy scaling](results/ch06_accuracy_scaling.png)

Task accuracy vs parameter count on a log scale. Most task lines slope upward — bigger models do better. But the `unknown` task line stays flat at 0% — hallucination is immune to scale. This is the key visual: the upward-sloping lines and the flat line tell the whole story of this chapter.

**Task comparison** (`results/ch06_comparison.png`):

![Comparison bar chart](results/ch06_comparison.png)

Side-by-side bars for all four model sizes plus the human agent. The bars grow taller from tiny to large on computation and retrieval tasks. On the `unknown` task, all model bars are at 0% while the human bar is at 100%. The visual gap between any model and the human on `unknown` is the "mechanism gap" that scaling cannot close.

## What's Next

In **Chapter 07 (Instruction Tuning)**, we ask: can we change what the model does without changing its architecture? SFT (Supervised Fine-Tuning) trains the model on instruction-response pairs, teaching it to follow directions. The model becomes more "helpful" — but does it become more honest? We will trace our three benchmark prompts through an instruction-tuned Transformer and see whether fine-tuning can teach what scaling could not: the ability to say "I don't know."
