# Chapter 03: Recurrent Language Models (RNN & GRU)

## Goal

Remove the fixed context window. Recurrent models process one token at a time, maintaining a hidden state that theoretically captures the entire history. The question: does a compressed state vector actually remember what matters?

## How Recurrent Models Work

### Vanilla RNN

At each time step $t$, the RNN combines the current input with the previous hidden state:

$$
\mathbf{h}_t = \tanh(\mathbf{W}_{ih} \mathbf{e}_t + \mathbf{W}_{hh} \mathbf{h}_{t-1} + \mathbf{b})
$$

$$
\text{logits}_t = \mathbf{W}_{out} \mathbf{h}_t
$$

Where:
- $\mathbf{e}_t = \text{Embedding}(c_t)$ is the current character's embedding
- $\mathbf{h}_t \in \mathbb{R}^d$ is the hidden state — a fixed-size vector that "summarizes" everything seen so far
- $\mathbf{W}_{ih} \in \mathbb{R}^{d \times e}$ maps input to hidden space
- $\mathbf{W}_{hh} \in \mathbb{R}^{d \times d}$ maps previous state to current state

The key insight: **the hidden state is the model's memory**. But it's a fixed-size vector that must compress the entire history. Information from early tokens gets overwritten by later ones.

**Worked example — tracing the RNN on `"COPY: hi|"`**: Let's walk through what happens step by step with a tiny hidden state of dimension 3 (real model uses 64):

```
t=0: input="C"  h_0 = tanh(W_ih * e("C") + W_hh * [0,0,0])
                 h_0 = [0.3, -0.1, 0.5]     <- state knows we saw "C"

t=1: input="O"  h_1 = tanh(W_ih * e("O") + W_hh * [0.3,-0.1,0.5])
                 h_1 = [0.1, 0.4, -0.2]     <- state mixed "C" and "O" together

t=2: input="P"  h_2 = [0.6, 0.2, 0.1]      <- "COP" compressed into 3 numbers
t=3: input="Y"  h_3 = [-0.1, 0.7, 0.3]     <- "COPY" in 3 numbers
t=4: input=":"  h_4 = [0.4, 0.5, -0.4]      <- model hopefully recognizes "COPY:"
t=5: input=" "  h_5 = [0.2, 0.3, -0.1]
t=6: input="h"  h_6 = [0.8, -0.3, 0.6]     <- "h" enters the state
t=7: input="i"  h_7 = [0.1, 0.9, -0.2]     <- "h" is now mixed with "i"
t=8: input="|"  h_8 = [-0.3, 0.4, 0.7]     <- the "|" signal arrives
```

Now the model must generate `"hi"` from h_8 = [-0.3, 0.4, 0.7]. But can these 3 numbers reconstruct the exact characters `h` and `i`? The information from t=6 and t=7 has been compressed and mixed with everything else. With only 3 dimensions, the answer is probably no. With 64 dimensions (as in our actual model), it sometimes works — but it's always a lossy compression.

### The Vanishing Gradient Problem

During backpropagation through time, gradients flow through the chain:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
$$

Each factor $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$ involves $\mathbf{W}_{hh}$ and the tanh derivative. Over many steps, this product either **vanishes** (gradients → 0, can't learn long-range) or **explodes** (gradients → ∞, training destabilizes).

**Worked example**: Imagine a 30-token sequence where the first token determines the answer (like `"ADD"` at position 0 telling us to add, not subtract). The gradient from the loss at position 30 must travel back through 30 matrix multiplications. If each step shrinks the gradient by 0.9x, after 30 steps: $0.9^{30} = 0.04$ — the gradient reaching position 0 is 25x weaker than at position 30. The model barely learns that the first token matters. With a factor of 0.8x per step: $0.8^{30} = 0.001$ — essentially zero.

This is why vanilla RNNs struggle with sequences longer than ~10-20 tokens.

### GRU (Gated Recurrent Unit)

The GRU adds **gates** that control information flow, mitigating the vanishing gradient:

**Update gate** — how much of the old state to keep:
$$
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{e}_t])
$$

**Reset gate** — how much of the old state to expose to the candidate:
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{e}_t])
$$

**Candidate state** — new information from current input:
$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{e}_t])
$$

**Final state** — interpolation between old and new:
$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

The key: when $\mathbf{z}_t \approx 0$, the state passes through unchanged (gradient flows freely). This creates a "highway" for information to travel across many time steps.

**Worked example — how the GRU gate helps on `"COPY: hi|"`**: When the GRU reads the irrelevant characters `:` and ` ` (positions 4-5), it can learn to set the update gate $\mathbf{z}_t \approx 0$, meaning: "nothing important here, keep the old state unchanged." The hidden state carrying `"COPY"` passes through undamaged. Then when it reads `"h"` at position 6, the gate opens ($\mathbf{z}_t \approx 1$): "this is important, update the state." And when `"|"` arrives, the gate can selectively keep the `"hi"` information while adding the "now generate" signal.

The vanilla RNN has no such control — every single character equally overwrites the state through the same tanh transform, so `":"` and `" "` degrade the `"COPY"` information just as much as `"h"` adds to it.

### GRU vs RNN: Parameter Count

For hidden size $d$ and embedding size $e$:
- **RNN**: $d(e + d + 1)$ recurrent params (one transform)
- **GRU**: $3 \times d(e + d + 1)$ recurrent params (three gates/transforms)

The GRU uses ~3x more parameters for the same hidden size, but the gating makes training much more stable.

## Step-by-Step: What Happens During Training

1. **Build corpus**: same task prompt+answer pairs as Chapter 02
2. **Create sequences**: full sequences padded to the same length, shifted by one for input/target
3. **Forward pass**: feed each sequence through the RNN/GRU one token at a time, collecting logits at each position
4. **Loss**: cross-entropy between predicted and actual next character at every position
5. **Backprop through time**: gradients flow backward through the entire sequence — this is where vanishing/exploding gradients matter

**Worked example**: Consider the training sequence `"ADD 5 3 =8"` (with BOS/EOS added: `<BOS>ADD 5 3 =8<EOS>`).

```
Input:   <BOS>  A   D   D     5     3     =   8
Target:    A    D   D       5     3     =   8  <EOS>
```

At each position, the model predicts the next character and we compare against the target:
- Position 0: model sees `<BOS>`, must predict `A`. Gets loss $-\log P(\text{A})$.
- Position 7: model sees `=` (having read `<BOS>ADD 5 3 =`), must predict `8`. This is the hard part — the model's hidden state must still contain enough about `5` and `3` to produce `8`.
- Total loss = average of all position losses.

The gradient for the loss at position 7 flows back through positions 6, 5, 4... all the way to 0. At each step it passes through $\mathbf{W}_{hh}$. By the time it reaches the `A` at position 1, the gradient may be tiny (vanishing) — so the model barely learns that the `A` (indicating "add") matters for the answer at position 7.

## Improvement Over FFN LM

| Aspect | FFN LM | RNN/GRU |
|--------|--------|---------|
| Context | Fixed window ($W$ chars) | Entire history (in theory) |
| Memory | None — window slides | Hidden state persists |
| Long-range | Blind beyond $W$ | Possible but degrades |
| Parameters | $O(W \cdot d \cdot h)$ | $O(d^2)$ — independent of sequence length |
| Training | Stable | RNN: brittle. GRU: more stable |

## What RNNs Can Learn (and Can't)

### Can Learn

- **Variable-length patterns**: no fixed window — can process any length
- **Short-range dependencies**: recent tokens strongly influence the hidden state
- **Sequential structure**: the model "reads" left to right, like humans do

### Cannot Learn (Reliably)

- **Long-range dependencies**: the hidden state compresses information — details from 20+ tokens ago are often lost
- **Precise copying**: the state must encode the exact characters to copy, but it's a lossy compression
- **Abstention**: still always generates — no "I don't know" mechanism
- **Parallel patterns**: RNNs process sequentially — they can't attend to two distant parts simultaneously

## Human Lens

Human working memory is **structured**: we maintain distinct slots for different pieces of information (the operation, the numbers, the intermediate result). An RNN's hidden state is **compressed soup** — everything mixed into one vector.

Consider the prompt `"ADD 35 27 ="`:
- **Human**: parses "ADD", stores 35 and 27 as separate numbers, applies addition algorithm step by step, tracks carries
- **RNN**: reads character by character, updating a single vector. The information about "35" is entangled with "ADD" and "27" in an unstructured way

Humans also have **explicit retrieval**: they can look back at the prompt. RNNs can't — they only have whatever survived the compression into $\mathbf{h}_t$.

The GRU's gates are a step toward structured memory (deciding what to keep and what to forget), but they operate on the entire state at once — there's no concept of "slot 1 holds the first number, slot 2 holds the second."

## What to Observe When Running

Run `python chapters/03_rnn_gru/run.py` and notice:

1. **GRU beats vanilla RNN** — gating helps information persist
2. **Grammar gets traction** (~32-38%) — bracket patterns have local structure the state can track
3. **Copy and arithmetic stay at 0%** — precise memory and computation are too much for compressed state
4. **Training loss curves** — GRU reaches lower final loss than RNN
5. **Still 0% abstention** — no mechanism to say "I don't know"
6. **Parameter count stays small** — RNN ~12k, GRU ~22k, well under 100k

### Generated Plots

**Training loss curves** (`results/ch03_rnn_loss.png` and `results/ch03_gru_loss.png`):

![RNN loss curve](results/ch03_rnn_loss.png)
![GRU loss curve](results/ch03_gru_loss.png)

Compare the two loss curves side by side. Both drop from ~2.8-3.0 (random) but the GRU reaches a lower final loss (~0.41 vs ~0.53). This gap comes from gating: the GRU can selectively preserve and update information, leading to better next-character predictions across the sequence. The RNN curve may also show more noise, reflecting the vanishing gradient making optimization harder.

**Task comparison** (`results/ch03_comparison.png`):

![Comparison bar chart](results/ch03_comparison.png)

This chart reveals something interesting: despite having access to the full sequence history (unlike the FFN LM's 8-character window), the RNN and GRU perform worse overall than the FFN on knowledge QA. Why? The FFN memorized specific short answer patterns within its window, while the RNN/GRU must compress the entire sequence into a single vector — and that compression loses the specific details needed for exact-match answers. Grammar is the one bright spot (~32-38%) because bracket validity depends on local structure the hidden state can track. The human agent remains at 100%, highlighting that "seeing the full sequence" is necessary but not sufficient — you also need structured memory and reasoning.

## What's Next

In **Chapter 04 (Attention)**, we add the ability to look back at any part of the input directly, instead of relying on a compressed state. This is the key insight that leads to the Transformer.
