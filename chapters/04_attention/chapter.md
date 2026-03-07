# Chapter 04: Attention Mechanism

## Goal

Give the model the ability to **look back** at any position in the input, instead of relying on a compressed hidden state (RNN) or a fixed window (FFN). This is the single most important idea in modern language models.

## The Core Idea

In an RNN, the model must compress the entire input into a fixed-size vector $\mathbf{h}_t$. By the time it needs to generate an answer, information about early tokens is degraded or lost.

Attention solves this: at every generation step, the model **directly queries** all input positions and retrieves the most relevant information.

Think of it like this:
- **RNN**: "I memorized the whole prompt into this one vector. Let me try to reconstruct what I need."
- **Attention**: "Let me look back at the prompt and find the specific parts I need right now."

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

### Worked Example: Attention on `"COPY: hi|"`

Let's trace what happens when the model must predict the character after `"|"`. The model has 8 tokens: `<BOS>`, `C`, `O`, `P`, `Y`, `:`, `h`, `i`, `|`.

At position 8 (`"|"`), the model computes:

```
q_8 = W_Q @ embedding("|")     <- "I'm at the pipe, what do I need?"

k_0 = W_K @ embedding(<BOS>)   <- "I'm the start token"
k_1 = W_K @ embedding("C")     <- "I'm a C"
k_2 = W_K @ embedding("O")     <- "I'm an O"
...
k_6 = W_K @ embedding("h")     <- "I'm an h"
k_7 = W_K @ embedding("i")     <- "I'm an i"

scores = [q_8 . k_0, q_8 . k_1, ..., q_8 . k_7] / sqrt(d_k)
```

If the model has learned that after `"|"`, it needs to copy the characters, then q_8 will be similar to k_6 (`"h"`) and k_7 (`"i"`) — the dot products will be high for those positions:

```
scores (before softmax): [0.1, 0.2, 0.1, 0.1, 0.1, 0.3, 2.8, 2.5]
                          <BOS>  C    O    P    Y    :    h    i

weights (after softmax):  [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.42, 0.38]
                           ^barely looks at these^              ^focuses here^
```

The output is a weighted sum of values: mostly the value vectors of `"h"` and `"i"`. This directly provides the information needed to produce `"h"` as the first output character.

**This is impossible for RNNs** — they have no mechanism to "look back" at positions 6 and 7. They only have whatever survived compression into $\mathbf{h}_8$.

### Why Divide by $\sqrt{d_k}$?

Without the scaling factor, dot products grow with dimension. For $d_k = 64$, the dot products can easily reach values like 50-100, making softmax extremely peaked (one position gets ~100% weight, all others ~0%). This makes gradients tiny and learning slow.

**Worked example**: Two random vectors of dimension 64, each with entries ~N(0,1). Their dot product has variance $\approx 64$, so values of 8-10 are common. After softmax, $e^{10} / (e^{10} + e^{0}) \approx 0.9999$ — essentially hard attention. Dividing by $\sqrt{64} = 8$ brings the variance back to ~1, giving softmax a useful gradient.

## The Causal Mask

For language modeling, position $t$ must not attend to positions $t+1, t+2, \ldots$ (it can't see the future). We enforce this with a **causal mask**: set scores to $-\infty$ for all future positions before the softmax.

```
Score matrix (before masking):        After masking:
  <BOS>  C  O  P  Y                     <BOS>  C    O    P    Y
<BOS> 2.1 0.3 0.1 0.5 0.2            <BOS> 2.1  -inf -inf -inf -inf
C     0.4 1.8 0.6 0.2 0.1            C     0.4  1.8  -inf -inf -inf
O     0.1 0.3 1.5 0.7 0.4            O     0.1  0.3  1.5  -inf -inf
P     0.2 0.1 0.4 2.0 0.3            P     0.2  0.1  0.4  2.0  -inf
Y     0.3 0.5 0.2 0.1 1.7            Y     0.3  0.5  0.2  0.1  1.7
```

After softmax, $-\infty$ becomes 0 weight. Position `O` can only attend to `<BOS>`, `C`, and `O` itself — never to `P` or `Y`.

## Multi-Head Attention

A single attention head can only learn one pattern (e.g., "attend to the nearest digit"). Multi-head attention runs multiple heads in parallel, each with its own $\mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V$, then combines their outputs:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) \mathbf{W}_O
$$

$$
\text{head}_i = \text{Attention}(X \mathbf{W}_{Q_i}, X \mathbf{W}_{K_i}, X \mathbf{W}_{V_i})
$$

With $H$ heads and model dimension $d_{model}$, each head operates in dimension $d_k = d_{model} / H$.

**Worked example**: With $d_{model} = 32$ and 4 heads, each head has $d_k = 8$. On the prompt `"ADD 5 3 ="`:
- **Head 0** might learn: "always attend to the most recent token" (useful for local patterns)
- **Head 1** might learn: "attend to digits" (5 and 3 have similar embeddings)
- **Head 2** might learn: "attend to the operation token" (ADD, SUB, MUL)
- **Head 3** might learn: "attend to the delimiter =" (signals "now produce the answer")

Each head extracts different information from the same input. The output projection $\mathbf{W}_O$ combines them.

## Positional Encoding

Attention treats the input as a **set** — it has no notion of position. We add positional embeddings so the model knows that `"h"` at position 6 is different from `"h"` at position 2:

$$
\mathbf{x}_t = \text{Embedding}(c_t) + \text{PosEmbed}(t)
$$

Without this, the model can't distinguish `"COPY: hi|"` from `"COPY: ih|"` — it would see the same set of embeddings in both cases.

## Step-by-Step: What Happens During Training

1. **Build corpus**: same task prompt+answer pairs as before
2. **Embed + add positions**: each token gets embedding + position encoding
3. **Multi-head attention**: every position attends to all previous positions (causal mask), extracting relevant information
4. **Output projection**: attended representation → logits over vocabulary
5. **Loss**: cross-entropy, same as RNN/FFN
6. **Backprop**: gradients flow directly through the attention weights — no vanishing gradient problem through recurrence!

**Key difference from RNN training**: the gradient from the loss at position 30 flows directly to position 1 through the attention weights, not through 29 matrix multiplications. This is why attention-based models learn long-range dependencies much more easily.

## What Attention Can Learn (and Can't)

### Can Learn

- **Direct retrieval**: look up specific tokens anywhere in the input
- **Multiple patterns**: different heads capture different relationships
- **Long-range dependencies**: no vanishing gradient — position 1 is just as accessible as position 29

### Cannot Learn (with attention alone)

- **Complex computation**: attention is a weighted average — it can retrieve and blend, but can't compute "5 + 3 = 8" (that needs feed-forward layers)
- **Compositional reasoning**: one layer of attention can find relevant tokens but can't chain multiple reasoning steps
- **Abstention**: still produces output for every input

### Improvement Over RNN/GRU

| Aspect | RNN/GRU | Attention LM |
|--------|---------|-------------|
| Access to history | Through compressed state | Direct to any position |
| Long-range | Degrades with distance | Equal access to all positions |
| Gradient flow | Through recurrence chain | Direct through attention weights |
| Parallelism | Sequential (slow) | All positions in parallel (fast) |
| Interpretability | Hidden state is opaque | Attention weights show what model focuses on |

## Human Lens

Humans also attend selectively — when you read `"ADD 5 3 ="`, you focus on `5`, `3`, and `ADD` to compute the answer. But there's a crucial difference:

- **Human attention** is guided by **goals**: "I need the numbers and the operation"
- **Machine attention** is guided by **content similarity**: queries and keys with similar embeddings get high scores

A human reading `"COPY: hi|"` thinks: "I need to copy the text between `:` and `|`." They attend to `h` and `i` because they *understand the task structure*.

The attention model attends to `h` and `i` because their key vectors happen to align with the query vector at position `|` — not because it understands what "copy" means. **Same behavior, different mechanism.**

Also, humans can attend to things they haven't seen yet (looking ahead in a sentence, anticipating structure). Causal attention is strictly backward-looking.

## What to Observe When Running

Run `python chapters/04_attention/run.py` and notice:

1. **Loss drops faster** than RNN — no vanishing gradient through recurrence
2. **Copy task may improve** — attention can directly retrieve the characters to copy
3. **Attention heatmaps are interpretable** — you can literally see what the model looks at
4. **Different heads attend differently** — check the per-head heatmaps
5. **Arithmetic is still hard** — attention can retrieve numbers but can't compute sums
6. **Still 0% abstention** — no uncertainty mechanism

### Generated Plots

**Training loss curve** (`results/ch04_loss_curve.png`):

![Loss curve](results/ch04_loss_curve.png)

Compare with the RNN loss curves from Chapter 03. Attention-based training should converge to a lower loss, reflecting better access to the full sequence context through direct attention rather than compressed state.

**Attention heatmaps** (`results/ch04_attn_*.png`):

The heatmaps are the star of this chapter. Each heatmap shows which positions each output attends to.

![Attention on COPY task](results/ch04_attn_COPY_hi.png)

In the copy task heatmap, look for bright cells connecting the output positions (after `|`) to the input characters (`h`, `i`). If the model has learned the copy pattern, you'll see diagonal or focused attention from post-`|` positions back to the content characters.

![Attention on ADD task](results/ch04_attn_ADD_5_3_=.png)

In the arithmetic heatmap, look at what the `=` position attends to. Does it focus on the digits `5` and `3`? Even if the model can't compute the sum, it might learn to attend to the operands.

**Task comparison** (`results/ch04_comparison.png`):

![Comparison bar chart](results/ch04_comparison.png)

This chart shows how attention-based direct access to the input compares against the human agent. The attention LM should show its biggest gains on tasks that require looking back at specific positions (copy, knowledge QA). Tasks requiring computation (arithmetic) or structured reasoning (compositional) remain difficult — attention can retrieve the right inputs but can't process them.

## What's Next

In **Chapter 05 (Transformer)**, we assemble the full architecture: multi-head attention + feed-forward layers + residual connections + layer norm, stacked into multiple layers. The feed-forward layers add the computation ability that attention alone lacks. Residual connections enable depth. This is the architecture behind GPT, BERT, and every modern LLM.
