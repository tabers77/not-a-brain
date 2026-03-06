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

### The Vanishing Gradient Problem

During backpropagation through time, gradients flow through the chain:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
$$

Each factor $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$ involves $\mathbf{W}_{hh}$ and the tanh derivative. Over many steps, this product either **vanishes** (gradients → 0, can't learn long-range) or **explodes** (gradients → ∞, training destabilizes).

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
2. **Both beat FFN LM** on tasks where prompt info is beyond the FFN's window
3. **Copy task improves** — the hidden state can (sometimes) carry the characters to copy
4. **Arithmetic stays hard** — requires precise computation, not just pattern memory
5. **Training loss curves** — RNN loss may be noisier than GRU
6. **Still 0% abstention** — no mechanism to say "I don't know"
7. **Parameter count stays small** — well under 100k, CPU-friendly

## What's Next

In **Chapter 04 (Attention)**, we add the ability to look back at any part of the input directly, instead of relying on a compressed state. This is the key insight that leads to the Transformer.
