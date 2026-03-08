# Chapter 04: Attention Mechanism

## Goal

Give the model the ability to **look back** at any position in the input, instead of relying on a compressed hidden state (RNN) or a fixed window (FFN). This is the single most important idea in modern language models.

## The Running Example

We trace the same three benchmark prompts through the Attention LM (d_model=32, 4 heads). This is where things get exciting — one of the three prompts is finally solved.

### Prompt 1: `"ADD 5 3 ="` -- expected `"8"` (computation)

Unlike the RNN, which must squeeze the entire sequence into a single state vector, attention lets every position directly query ALL previous positions. At position `"="` (position 9), the model computes attention scores against every earlier token:

```
tokens:    <BOS>  A    D    D    ' '  5    ' '  3    ' '  =
positions:  0     1    2    3    4    5    6    7    8    9

scores at "=": q_9 . k_0, q_9 . k_1, ... q_9 . k_9
```

After softmax, the model focuses on the operands — positions 5 (`"5"`) and 7 (`"3"`):

```
weights = [0.02, 0.05, 0.03, 0.03, 0.02, 0.35, 0.05, 0.32, 0.05, 0.08]
           <BOS>  A    D    D    ' '  5    ' '  3    ' '  =
                                      ^^^^            ^^^^
                                      focuses here
```

The model RETRIEVES the operands perfectly. But here is the problem: the output is a weighted average of value vectors:

```
output = 0.35 * value("5") + 0.32 * value("3") + 0.05 * value("A") + ...
```

A weighted average of the embeddings of `"5"` and `"3"` does not give `"8"`. Averaging is blending, not computing. The model needs a nonlinear transformation (a feed-forward layer) to turn "I found 5 and 3 and the operation is ADD" into "the answer is 8." Attention alone has no such layer.

**Output: `"5"`** — retrieves an operand, cannot compute. WRONG.

Attention solved the WHERE problem (finding the operands). But computing 5+3=8 requires a feed-forward network, which this model does not have. Contrast with Chapter 03: the RNN could not even find the operands (compressed soup). Attention finds them easily — but finding is not computing.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` -- expected `"paris"` (retrieval)

This is the breakthrough prompt. The full prompt is roughly 53 characters. The answer `"paris"` sits at positions 6-11. The question mark `"?"` is at approximately position 52.

Consider what the previous models faced:
- **Chapter 02 (FFN)**: with an 8-character context window, the model could only see `"france?"` at generation time. The word `"paris"` was 40+ characters back — completely invisible.
- **Chapter 03 (GRU)**: the model read the entire sequence, but `"paris"` was 40+ steps in the past. After passing through dozens of state compressions, the specific characters p-a-r-i-s had degraded into an unrecoverable blur.

With attention, position `"?"` can directly attend to ANY previous position. Distance does not matter.

At the final position `"?"`, the attention weights look like this:

```
Position:  ... 6    7    8    9    10   11  ...  48   49   50   51   52
Token:     ... p    a    r    i    s    ' ' ...  c    e    ?
Weight:    ... 0.14 0.13 0.12 0.11 0.12 0.01 ... 0.03 0.02 0.06
                ^^^^^^^^^^^^^^^^^^^^^^^^^^
                model looks right at the answer
```

High weights on positions 6-10 (`"p"`, `"a"`, `"r"`, `"i"`, `"s"`) — the model looks right at the answer. Low weights on everything else.

Multi-head attention makes this even richer. Different heads capture different patterns:
- **Head 0**: attends to `"paris"` (the answer entity)
- **Head 1**: attends to `"capital"` in the fact (matching the question word)
- **Head 2**: attends to nearby tokens (local context)
- **Head 3**: attends to `"france"` (context matching between fact and question)

**Output: `"paris"`** — CORRECT. First model to solve this prompt.

Position 6 is just as accessible as position 50. The retrieval problem that defeated both the FFN and the RNN is trivially solved by attention.

### Prompt 3: `"Q: What is the capital of the Moon?"` -- expected `"unknown"` (abstention)

Attention processes the entire question. At generation time, the model attends to `"capital"` and searches for associations. In training data, `"capital"` co-occurs with city names — Paris, Tokyo, London. The query at the final position finds high-scoring keys from training examples about capitals.

**Output: `"tokyo"`** — HALLUCINATION.

Attention made retrieval perfect, but hallucination is worse in a subtle way. The FFN and RNN produced nonsensical outputs (`"the"`, `"mars"`) that were obviously wrong. Attention retrieves a wrong but plausible-looking answer — a real capital city, just not the right one. There is no mechanism for "I don't have this knowledge." Attention always retrieves the most similar thing from what it has seen, even when the correct answer is to abstain.

### Summary Table

```
| Prompt                 | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attention | Correct   | What changed                          |
|------------------------|-------------|----------|----------|----------------|-----------|---------------------------------------|
| ADD 5 3 =              | "1"         | "5"      | "5"      | "5"            | "8"       | Attends to operands, but no FFN       |
| FACT: paris... Q: ...? | " "         | "is"     | "capital"| "paris" (!)    | "paris"   | SOLVED -- attention reaches "paris"   |
| Q: capital of Moon?    | "the"       | "the"    | "mars"   | "tokyo"        | "unknown" | Still hallucinates -- no abstention   |
```

The pattern: each chapter solves a harder subproblem. Bigrams had no context. The FFN had a window. The RNN had a compressing state. Attention has direct access — and that is enough for retrieval. Computation and abstention remain unsolved.

## Scaled Dot-Product Attention

Given input embeddings $\mathbf{X} \in \mathbb{R}^{S \times d}$ (sequence of $S$ tokens, each with dimension $d$):

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
$$

Where:
- $Q = X \mathbf{W}_Q$ — **queries**: "what am I looking for?"
- $K = X \mathbf{W}_K$ — **keys**: "what do I contain?"
- $V = X \mathbf{W}_V$ — **values**: "what information do I provide?"
- $d_k$ — dimension of keys (the $\sqrt{d_k}$ prevents dot products from getting too large)

### Worked Example: Attention on `"ADD 5 3 ="`

Let's trace the full computation when the model processes the arithmetic prompt. The sequence has 10 tokens: `<BOS>`, `A`, `D`, `D`, `' '`, `5`, `' '`, `3`, `' '`, `=`.

At position 9 (`"="`), the model needs to decide what to generate next. It computes:

```
q_9 = W_Q @ embedding("=")           <- "I'm at the equals sign, what do I need?"

k_0 = W_K @ embedding(<BOS>)         <- "I'm the start token"
k_1 = W_K @ embedding("A")           <- "I'm an A"
k_2 = W_K @ embedding("D")           <- "I'm a D"
k_3 = W_K @ embedding("D")           <- "I'm a D"
k_4 = W_K @ embedding(" ")           <- "I'm a space"
k_5 = W_K @ embedding("5")           <- "I'm a 5"
k_6 = W_K @ embedding(" ")           <- "I'm a space"
k_7 = W_K @ embedding("3")           <- "I'm a 3"
k_8 = W_K @ embedding(" ")           <- "I'm a space"
k_9 = W_K @ embedding("=")           <- "I'm the equals sign"

scores = [q_9 . k_0, q_9 . k_1, ..., q_9 . k_9] / sqrt(d_k)
```

If the model has learned that after `"="` it needs the operands, then q_9 will be similar to k_5 (`"5"`) and k_7 (`"3"`) — the dot products will be high for those positions:

```
scores (before softmax): [-0.2, 0.4, 0.1, 0.1, -0.1, 2.6, 0.3, 2.4, 0.2, 0.5]
                          <BOS>  A    D    D    ' '   5    ' '  3    ' '  =

weights (after softmax):  [0.02, 0.05, 0.03, 0.03, 0.02, 0.35, 0.05, 0.32, 0.05, 0.08]
                           ^--- barely looks at these ---^  ^-- focuses here --^
```

The output is a weighted sum of values: mostly the value vectors of `"5"` and `"3"`. Attention successfully retrieves the operands — but a weighted average of value vectors cannot perform addition. The model outputs `"5"` (the highest-weighted operand), not `"8"`.

### Why Divide by $\sqrt{d_k}$?

Without the scaling factor, dot products grow with dimension. For $d_k = 64$, the dot products can easily reach values like 50-100, making softmax extremely peaked (one position gets ~100% weight, all others ~0%). This makes gradients tiny and learning slow.

**Worked example**: Two random vectors of dimension 64, each with entries ~N(0,1). Their dot product has variance $\approx 64$, so values of 8-10 are common. After softmax, $e^{10} / (e^{10} + e^{0}) \approx 0.9999$ — essentially hard attention. Dividing by $\sqrt{64} = 8$ brings the variance back to ~1, giving softmax a useful gradient.

## The Causal Mask

For language modeling, position $t$ must not attend to positions $t+1, t+2, \ldots$ (it cannot see the future). We enforce this with a **causal mask**: set scores to $-\infty$ for all future positions before the softmax.

Consider the first five tokens of `"ADD 5 3 ="`:

```
Score matrix (before masking):         After masking:
  <BOS>  A    D    D   ' '              <BOS>  A     D     D    ' '
<BOS> 2.1  0.3  0.1  0.5  0.2       <BOS> 2.1  -inf  -inf  -inf  -inf
A     0.4  1.8  0.6  0.2  0.1       A     0.4  1.8   -inf  -inf  -inf
D     0.1  0.3  1.5  0.7  0.4       D     0.1  0.3   1.5   -inf  -inf
D     0.2  0.1  0.4  2.0  0.3       D     0.2  0.1   0.4   2.0   -inf
' '   0.3  0.5  0.2  0.1  1.7       ' '   0.3  0.5   0.2   0.1   1.7
```

After softmax, $-\infty$ becomes 0 weight. Position `D` (position 2) can only attend to `<BOS>`, `A`, and itself — never to the second `D` or `' '`. This ensures the model generates tokens autoregressively: each prediction depends only on what came before.

## Multi-Head Attention

A single attention head can only learn one pattern (e.g., "attend to the nearest digit"). Multi-head attention runs multiple heads in parallel, each with its own $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$, then combines their outputs:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O
$$

$$
\text{head}_i = \text{Attention}(X \mathbf{W}_{Q_i}, X \mathbf{W}_{K_i}, X \mathbf{W}_{V_i})
$$

With $H$ heads and model dimension $d_{model}$, each head operates in dimension $d_k = d_{model} / H$.

### Head Specialization on `"FACT: paris is capital of france. Q: capital of france?"`

With $d_{model} = 32$ and 4 heads, each head has $d_k = 8$. On the retrieval prompt, different heads learn to extract different relationships at the final position `"?"`:

**Head 0 — Entity retrieval**: attends strongly to positions 6-10 (`"paris"`). This head learns that after a question, the answer entity in the preceding fact is what matters most. Weights peak at `"p"` (0.18) and decay across `"a"`, `"r"`, `"i"`, `"s"`.

**Head 1 — Keyword matching**: attends to `"capital"` in the fact (positions 18-24) and `"capital"` in the question (positions 43-49). This head detects that the same word appears in both the fact and the question — a strong signal that this fact is relevant.

**Head 2 — Local context**: attends to the most recent tokens (`"france"`, `"?"`). This provides the model with information about what was just asked, ensuring the output is contextually appropriate.

**Head 3 — Context bridging**: attends to `"france"` in the fact (positions 28-33). This head connects the country mentioned in the question to the country in the fact, confirming that the fact is about the right entity.

The output projection $\mathbf{W}_O$ combines all four heads. The result: position `"?"` has a rich representation that encodes both the answer (`"paris"` from Head 0) and the confidence that this is the right fact (Heads 1 and 3 confirm the keyword and context match).

## Positional Encoding

Attention treats the input as a **set** — it has no notion of position. We add positional embeddings so the model knows that `"a"` at position 7 (inside `"paris"`) is different from `"a"` at position 22 (inside `"capital"`):

$$
\mathbf{x}_t = \text{Embedding}(c_t) + \text{PosEmbed}(t)
$$

Without this, the model cannot distinguish `"FACT: paris is capital"` from `"FACT: capital is paris"` — it would see the same set of embeddings in both cases. For retrieval tasks like Prompt 2, positional encoding is critical: the model must learn that the answer comes right after `"FACT:"`, not anywhere else in the sentence.

## Step-by-Step: What Happens During Training

1. **Build corpus**: same task prompt+answer pairs as before
2. **Embed + add positions**: each token gets embedding + position encoding
3. **Multi-head attention**: every position attends to all previous positions (causal mask), extracting relevant information
4. **Output projection**: attended representation -> logits over vocabulary
5. **Loss**: cross-entropy, same as RNN/FFN
6. **Backprop**: gradients flow directly through the attention weights — no vanishing gradient problem through recurrence!

### Input/Target Alignment for the Benchmark Prompts

**Prompt 1** — `"ADD 5 3 =8"` (with BOS/EOS):

```
Input:   <BOS>  A   D   D   ' '  5   ' '  3   ' '  =   8
Target:    A    D   D   ' '  5   ' '  3   ' '  =   8  <EOS>
```

The critical position: input `"="` must predict target `"8"`. The attention at this position looks back at `"5"` and `"3"`, but the weighted average of their value vectors does not produce `"8"`.

**Prompt 2** — `"FACT: paris is capital of france. Q: capital of france?paris"`:

```
Input:   <BOS>  F  A  C  T  :  ' '  p  a  r  i  s  ...  ?  p  a  r  i  s
Target:    F    A  C  T  :  ' '  p  a  r  i  s  ' ' ...  p  a  r  i  s  <EOS>
```

The critical position: input `"?"` must predict target `"p"`. Attention at this position reaches all the way back to position 7 and finds `"p"` — the start of the answer. This is the first model where the gradient for this prediction does not degrade over distance.

**Key difference from RNN training**: the gradient from the loss at position 52 flows directly to position 7 through the attention weights, not through 45 matrix multiplications. This is why attention-based models learn long-range dependencies much more easily.

## What Attention Can Learn (and Can't)

### Can Learn

- **Direct retrieval**: look up specific tokens anywhere in the input
- **Multiple patterns**: different heads capture different relationships
- **Long-range dependencies**: no vanishing gradient — position 1 is just as accessible as position 50

### Cannot Learn (with attention alone)

- **Complex computation**: attention is a weighted average — it can retrieve and blend, but cannot compute "5 + 3 = 8" (that needs feed-forward layers)
- **Compositional reasoning**: one layer of attention can find relevant tokens but cannot chain multiple reasoning steps
- **Abstention**: still produces output for every input — no "I don't know" mechanism

### Improvement Over All Previous Models

| Aspect | Ch01 Bigram | Ch02 FFN | Ch03 RNN/GRU | Ch04 Attention |
|--------|-------------|----------|--------------|----------------|
| Context | 1 character | Fixed window ($W$ chars) | Entire history (compressed) | Entire history (direct access) |
| Long-range retrieval | None | Blind beyond $W$ | Degrades with distance | Equal access to all positions |
| Gradient flow | N/A (no backprop) | Through window only | Through recurrence chain | Direct through attention weights |
| Parallelism | N/A | All positions in parallel | Sequential (slow) | All positions in parallel (fast) |
| Computation | None | Nonlinear layers | Nonlinear state update | Weighted average only |
| Interpretability | Lookup table | Hidden layers opaque | Hidden state opaque | Attention weights show what model focuses on |

## Human Lens

When humans read `"FACT: paris is capital of france. Q: capital of france?"`, they do not compress it into soup (like the RNN) or only see the last 8 characters (like the FFN). They scan and retrieve — just like attention. You read the question, recognize "capital of france" as the key phrase, scan back to the fact, and find "paris." Same behavior as the attention model on Prompt 2.

But the mechanism is completely different. Human attention is **goal-directed**: "I need the capital, so I look for it." Machine attention is **similarity-based**: queries and keys with similar embeddings get high scores. A human understands WHY they are looking for "paris." The attention model just finds that the query vector at position `"?"` happens to align with the key vectors at positions 6-10. Same behavior, different mechanism.

For Prompt 3 (`"Q: What is the capital of the Moon?"`), the difference is stark. A human recognizes "the Moon has no capital" through world knowledge and abstains — "I don't know" or "there is no capital." The attention model has no mechanism for "I don't have this knowledge." It always retrieves the most similar thing from training. The query for "capital" finds keys associated with real capitals, and the model confidently produces "tokyo." Worse, this answer looks more plausible than the RNN's "mars" or the FFN's "the" — attention makes hallucination harder to detect by producing fluent, wrong answers.

Humans also have **goal-directed abstention**: they can recognize when a question has no answer. Every model so far lacks this. Attention makes retrieval human-like, but the failure mode shifts from "obviously wrong" to "subtly wrong."

## What to Observe When Running

Run `python chapters/04_attention/run.py` and notice:

1. **Loss drops faster** than RNN — no vanishing gradient through recurrence
2. **Retrieval tasks improve dramatically** — attention can directly retrieve characters from anywhere in the input
3. **Attention heatmaps are interpretable** — you can literally see what the model looks at
4. **Different heads attend differently** — check the per-head heatmaps
5. **Arithmetic is still hard** — attention can retrieve numbers but cannot compute sums
6. **Still 0% abstention** — no uncertainty mechanism

### Generated Plots

**Training loss curve** (`results/ch04_loss_curve.png`):

![Loss curve](results/ch04_loss_curve.png)

Compare with the RNN loss curves from Chapter 03. Attention-based training should converge to a lower loss, reflecting better access to the full sequence context through direct attention rather than compressed state.

**Attention heatmaps** (`results/ch04_attn_*.png`):

The heatmaps are the star of this chapter. Each heatmap shows which positions each output attends to.

![Attention on ADD task](results/ch04_attn_ADD_5_3_=.png)

In the arithmetic heatmap, look at what the `"="` position attends to. It should focus on the digits `"5"` and `"3"` — confirming that attention solves the retrieval problem. But the model still outputs `"5"` instead of `"8"`, because retrieval is not computation.

![Attention on COPY task](results/ch04_attn_COPY_hi.png)

In the copy task heatmap, look for bright cells connecting the output positions (after `"|"`) to the input characters (`"h"`, `"i"`). If the model has learned the copy pattern, you will see focused attention from post-`"|"` positions back to the content characters.

**Task comparison** (`results/ch04_comparison.png`):

![Comparison bar chart](results/ch04_comparison.png)

This chart shows how attention-based direct access to the input compares against the human agent. The attention LM should show its biggest gains on tasks that require looking back at specific positions (copy, knowledge QA). Tasks requiring computation (arithmetic) or structured reasoning remain difficult — attention can retrieve the right inputs but cannot process them.

## What's Next

In **Chapter 05 (Transformer)**, we combine multi-head attention with feed-forward layers, residual connections, and layer normalization, stacked into multiple layers. The feed-forward layers add the computation ability that attention alone lacks — so Prompt 1 (`"ADD 5 3 ="`) may finally be solved. The Transformer is the architecture behind GPT, BERT, and every modern LLM.
