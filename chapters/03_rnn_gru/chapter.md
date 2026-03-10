# Chapter 03: Recurrent Language Models (RNN and GRU)

## Goal

Remove the fixed context window. Recurrent models process one token at a time, maintaining a hidden state that theoretically captures the entire history. The question: does a compressed state vector actually remember what matters?

## The Running Example

We trace three benchmark prompts through both the vanilla RNN and the GRU. These same prompts appear in every chapter, so you can track exactly how each architecture improves (or fails to improve) on the same problems.

### Prompt 1: `"ADD 5 3 ="` -- expected `"8"` (tests computation)

**RNN (d_hidden=64)**. The RNN reads the prompt character by character, updating its hidden state at each step. Here is the trace with a small hidden state (showing the first few dimensions of a 64-dimensional vector):

```
t=0: input="A"  h_0 = tanh(W_ih * e("A") + W_hh * [0,0,...])
                 h_0 = [0.3, -0.1, 0.5, ...]       <- "saw an A"

t=1: input="D"  h_1 = tanh(W_ih * e("D") + W_hh * h_0)
                 h_1 = [0.1, 0.4, -0.2, ...]       <- "AD" compressed

t=2: input="D"  h_2 = [-0.2, 0.6, 0.1, ...]       <- "ADD" compressed

t=3: input=" "  h_3 = [0.4, 0.3, -0.3, ...]       <- space passes through

t=4: input="5"  h_4 = [0.8, -0.2, 0.7, ...]       <- "5" enters the state

t=5: input=" "  h_5 = [0.5, 0.1, -0.1, ...]       <- space begins overwriting "5"

t=6: input="3"  h_6 = [0.7, 0.5, 0.3, ...]        <- "3" enters, but "5" is being overwritten

t=7: input=" "  h_7 = [0.3, 0.2, -0.4, ...]       <- another space degrades both operands

t=8: input="="  h_8 = [-0.1, 0.6, 0.4, ...]       <- must predict from this compressed vector
```

Now the model must produce `"8"` from h_8 = [-0.1, 0.6, 0.4, ...]. This single vector must encode: "this is an ADD operation, the first operand was 5, the second operand was 3." That is three separate pieces of information tangled into 64 numbers. The information is there but entangled -- the model cannot separate "5" from "ADD" from "3" because every character was mixed into the same vector through the same tanh transform.

RNN output: `"53"` (mashes the operands together) -- WRONG.

**GRU improvement**. The GRU's update gate can learn to protect the "5" in h_4 when the space at t=5 arrives, by setting z_5 close to 0 ("nothing important, keep the old state"). This means "5" survives longer in the state. But even with better preservation, the GRU still cannot compute 5 + 3 = 8. Addition requires an algorithm, not memory.

GRU output: `"5"` (preserves one operand but cannot compute) -- WRONG.

The RNN can SEE both operands (unlike the FFN's fixed window), but it compresses them into soup. The GRU keeps the soup fresher, but soup is still soup.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` -- expected `"paris"` (tests retrieval)

**RNN (d_hidden=64)**. The word "paris" enters the hidden state at approximately positions 6 through 11. Then 40+ more characters pass through, each one transforming h via tanh(W_hh * h + W_ih * e). By position ~50 (the `?`), the information about "paris" has been through roughly 40 matrix multiplications.

Vanishing gradient makes this concrete: the gradient for "paris" at position 6 is approximately $0.9^{40} \approx 0.015$ of its original strength by the time it reaches the loss at the end. The state "remembers" something vague about capitals and france, but the specific word "paris" has degraded into noise.

RNN output: `"france"` (confused -- retrieves the wrong part of the fact) -- WRONG.

**GRU**. The gates help: the update gate can protect important information through the irrelevant middle tokens ("is", "capital", "of"). But "paris" is still 40 steps back, and even with gating, the signal decays. The GRU latches onto a different fragment of the compressed state.

GRU output: `"capital"` -- WRONG.

This is the fundamental RNN limitation: information degrades with distance. The model read "paris" and could have stored it, but 40 steps of sequential compression destroyed the specifics.

### Prompt 3: `"Q: What is the capital of the Moon?"` -- expected `"unknown"` (tests hallucination/abstention)

**RNN**. The model processes the entire question into a final hidden state h. That state encodes something like "question about a capital of something." At generation time, the model picks the most likely next character based on training patterns -- and training patterns always have an answer. There is no mechanism for the model to say "I have never seen a fact about the Moon's capital, so I should abstain."

RNN output: `"earth"` -- HALLUCINATION.

**GRU**. Better memory does not help here. The GRU faithfully preserves the question through its gates, but at generation time, it faces the same problem: the softmax always produces a distribution, and the model always picks a token. A different fragment of the compressed state activates a different wrong answer.

GRU output: `"mars"` -- HALLUCINATION (different but equally wrong).

The problem is not remembering -- it is knowing when you DON'T know. No amount of gating fixes this. The model needs a fundamentally different mechanism to abstain.

### Summary Table

Comparing all architectures so far on the three benchmark prompts:

| Prompt                     | Ch01 Bigram | Ch02 FFN | Ch03 RNN | Ch03 GRU | Correct   | What changed                        |
|----------------------------|-------------|----------|----------|----------|-----------|-------------------------------------|
| ADD 5 3 =                  | "1"         | "5"      | "53"     | "5"      | "8"       | Sees all tokens, but compresses     |
| FACT: paris... Q: capital? | " "         | "is"     | "france" | "capital"| "paris"   | No window limit, but state decays   |
| Q: capital of Moon?        | "the"       | "the"    | "earth"  | "mars"   | "unknown" | Still hallucinates -- no abstention |

Each chapter adds capability but reveals a new bottleneck. The bigram was blind beyond one character. The FFN was blind beyond its window. The RNN can see everything but compresses it into soup. The GRU keeps the soup fresher but still cannot compute or abstain.

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
- $\mathbf{h}_t \in \mathbb{R}^d$ is the hidden state -- a fixed-size vector that "summarizes" everything seen so far
- $\mathbf{W}_{ih} \in \mathbb{R}^{d \times e}$ maps input to hidden space
- $\mathbf{W}_{hh} \in \mathbb{R}^{d \times d}$ maps previous state to current state

The key insight: **the hidden state is the model's memory**. But it is a fixed-size vector that must compress the entire history. Information from early tokens gets overwritten by later ones.

**Secondary worked example -- tracing the RNN on `"COPY: hi|"`**: Here is what happens step by step with a tiny hidden state of dimension 3 (the real model uses 64):

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

Now the model must generate `"hi"` from h_8 = [-0.3, 0.4, 0.7]. But can these 3 numbers reconstruct the exact characters `h` and `i`? The information from t=6 and t=7 has been compressed and mixed with everything else. With only 3 dimensions, the answer is probably no. With 64 dimensions (as in our actual model), it sometimes works -- but it is always a lossy compression.

### The Vanishing Gradient Problem

During backpropagation through time, gradients flow through the chain:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{h}_1} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_T} \prod_{t=2}^{T} \frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}
$$

Each factor $\frac{\partial \mathbf{h}_t}{\partial \mathbf{h}_{t-1}}$ involves $\mathbf{W}_{hh}$ and the tanh derivative. Over many steps, this product either **vanishes** (gradients approach 0, cannot learn long-range) or **explodes** (gradients approach infinity, training destabilizes).

**Worked example with the FACT prompt**: Consider `"FACT: paris is capital of france. Q: capital of france?"` -- roughly 50 characters. The answer depends on "paris" at position 6, but the loss is computed after position 50. The gradient from the loss must travel back through 44 matrix multiplications. If each step shrinks the gradient by 0.9x, after 44 steps: $0.9^{44} = 0.012$ -- the gradient reaching "paris" is about 80x weaker than at the end. The model barely learns that "paris" matters for the answer.

With a factor of 0.8x per step: $0.8^{44} = 0.0004$ -- essentially zero. This is why vanilla RNNs struggle with sequences longer than roughly 10-20 tokens. By the time we reach Prompt 2's length, the gradient has all but vanished.

### GRU (Gated Recurrent Unit)

The GRU adds **gates** that control information flow, mitigating the vanishing gradient:

**Update gate** -- how much of the old state to keep:
$$
\mathbf{z}_t = \sigma(\mathbf{W}_z [\mathbf{h}_{t-1}, \mathbf{e}_t])
$$

**Reset gate** -- how much of the old state to expose to the candidate:
$$
\mathbf{r}_t = \sigma(\mathbf{W}_r [\mathbf{h}_{t-1}, \mathbf{e}_t])
$$

**Candidate state** -- new information from current input:
$$
\tilde{\mathbf{h}}_t = \tanh(\mathbf{W} [\mathbf{r}_t \odot \mathbf{h}_{t-1}, \mathbf{e}_t])
$$

**Final state** -- interpolation between old and new:
$$
\mathbf{h}_t = (1 - \mathbf{z}_t) \odot \mathbf{h}_{t-1} + \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
$$

The key: when $\mathbf{z}_t \approx 0$, the state passes through unchanged (gradient flows freely). This creates a "highway" for information to travel across many time steps.

**Worked example -- how the GRU gate helps on the ADD prompt**: When the GRU reads `"ADD 5 3 ="`, the update gate at t=5 (the space after "5") can learn to set $\mathbf{z}_5 \approx 0$: "nothing important here, keep the old state unchanged." The dimensions of h that encode "5" pass through undamaged. Then when "3" arrives at t=6, the gate opens ($\mathbf{z}_6 \approx 1$) to incorporate the second operand. This selective gating is why the GRU can at least preserve "5" in its output, while the vanilla RNN mashes both operands into noise.

But for the FACT prompt, even with gating, "paris" must survive 40 steps. The gate would need to stay near zero for every one of those intervening characters -- possible in principle, but hard to learn, and any leakage compounds over distance.

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
5. **Backprop through time**: gradients flow backward through the entire sequence -- this is where vanishing/exploding gradients matter

**Worked example**: Consider the training sequence `"ADD 5 3 =8"` (with BOS/EOS added: `<BOS>ADD 5 3 =8<EOS>`).

```
Input:   <BOS>  A   D   D     5     3     =   8
Target:    A    D   D       5     3     =   8  <EOS>
```

At each position, the model predicts the next character and we compare against the target:
- Position 0: model sees `<BOS>`, must predict `A`. Gets loss $-\log P(\text{A})$.
- Position 7: model sees `=` (having read `<BOS>ADD 5 3 =`), must predict `8`. This is the hard part -- the model's hidden state must still contain enough about `5` and `3` to produce `8`.
- Total loss = average of all position losses.

The gradient for the loss at position 7 flows back through positions 6, 5, 4... all the way to 0. At each step it passes through $\mathbf{W}_{hh}$. By the time it reaches the `A` at position 1, the gradient may be tiny (vanishing) -- so the model barely learns that the `A` (indicating "add") matters for the answer at position 7.

Now consider the FACT prompt during training: `<BOS>FACT: paris is capital of france. Q: capital of france?paris<EOS>`. The loss at the position where the model must predict `p` (the start of "paris") depends on information from 40+ positions back. The gradient must traverse every one of those positions. Even with the GRU's gating, this is a long chain -- and the model struggles to connect the answer "paris" to the fact stated far upstream.

## What RNNs Can Learn (and Can't)

### Can Learn

- **Variable-length patterns**: no fixed window -- can process any length
- **Short-range dependencies**: recent tokens strongly influence the hidden state
- **Sequential structure**: the model "reads" left to right, like humans do

### Cannot Learn (Reliably)

- **Long-range dependencies**: the hidden state compresses information -- details from 20+ tokens ago are often lost
- **Precise copying**: the state must encode the exact characters to copy, but it is a lossy compression
- **Abstention**: still always generates -- no "I don't know" mechanism
- **Parallel patterns**: RNNs process sequentially -- they cannot attend to two distant parts simultaneously

### Improvement Over FFN LM

| Aspect | FFN LM | RNN/GRU |
|--------|--------|---------|
| Context | Fixed window ($W$ chars) | Entire history (in theory) |
| Memory | None -- window slides | Hidden state persists |
| Long-range | Blind beyond $W$ | Possible but degrades |
| Parameters | $O(W \cdot d \cdot h)$ | $O(d^2)$ -- independent of sequence length |
| Training | Stable | RNN: brittle. GRU: more stable |

## Human Lens

Consider how a human handles each of the three benchmark prompts, then compare with the RNN.

**Prompt 1: `"ADD 5 3 ="`**. Human working memory is structured: they store "ADD", "5", and "3" as three separate items in distinct slots. They retrieve the addition algorithm, apply it to the operands, and produce "8". The RNN mixes everything into one vector. There are no slots -- "ADD" and "5" and "3" are entangled in the same 64 numbers, and the model must somehow extract a computation from that soup.

**Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"`**. A human reads the fact, stores "paris = capital of france" as a structured association, then when asked the question, scans their memory and retrieves "paris" directly. They do not compress the entire sentence into a fixed-size vector -- they can go back and look. The RNN cannot look back. It has only whatever survived 40 steps of sequential compression. The specific word "paris" has degraded into a vague sense of "something about capitals."

**Prompt 3: `"Q: What is the capital of the Moon?"`**. A human can introspect: "I have no knowledge about a capital of the Moon. This question has no answer." They abstain. The RNN has no introspective mechanism -- it processes the question into a hidden state and then generates whatever the softmax distribution favors, producing confident-sounding nonsense like "earth."

The GRU's gates are a step toward structured memory (deciding what to keep and what to forget), but they operate on the entire state at once -- there is no concept of "slot 1 holds the first number, slot 2 holds the second." And no amount of gating creates the ability to say "I don't know."

## What to Observe When Running

Run `python chapters/03_rnn_gru/run.py` and notice:

1. **GRU beats vanilla RNN** -- gating helps information persist
2. **Grammar gets traction** (~32-38%) -- bracket patterns have local structure the state can track
3. **Copy and arithmetic stay at 0%** -- precise memory and computation are too much for compressed state
4. **Training loss curves** -- GRU reaches lower final loss than RNN
5. **Still 0% abstention** -- no mechanism to say "I don't know"
6. **Parameter count stays small** -- RNN ~12k, GRU ~22k, well under 100k

### Generated Plots

**Training loss curves** (`results/ch03_rnn_loss.png` and `results/ch03_gru_loss.png`):

![RNN loss curve](results/ch03_rnn_loss.png)
![GRU loss curve](results/ch03_gru_loss.png)

Compare the two loss curves side by side. Both drop from ~2.8-3.0 (random) but the GRU reaches a lower final loss (~0.41 vs ~0.53). This gap comes from gating: the GRU can selectively preserve and update information, leading to better next-character predictions across the sequence. The RNN curve may also show more noise, reflecting the vanishing gradient making optimization harder.

**Task comparison** (`results/ch03_comparison.png`):

![Comparison bar chart](results/ch03_comparison.png)

This chart reveals something interesting: despite having access to the full sequence history (unlike the FFN LM's 8-character window), the RNN and GRU perform worse overall than the FFN on knowledge QA. Why? The FFN memorized specific short answer patterns within its window, while the RNN/GRU must compress the entire sequence into a single vector -- and that compression loses the specific details needed for exact-match answers. Grammar is the one bright spot (~32-38%) because bracket validity depends on local structure the hidden state can track. The human agent remains at 100%, highlighting that "seeing the full sequence" is necessary but not sufficient -- you also need structured memory and reasoning.

## What's Next

In **Chapter 04 (Attention)**, we add the ability to look back at any part of the input directly, instead of relying on a compressed state. This is the key insight that leads to the Transformer. The FACT retrieval prompt -- where "paris" must survive 40 steps of compression -- will finally be within reach: attention lets the model point back to position 6 and read "paris" directly, no matter how far away it is.
