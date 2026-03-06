# Chapter 02: Feed-Forward Language Model (MLP)

## Goal

Replace counting with learning. A fixed-window MLP language model takes the previous $W$ characters, embeds them, and passes them through a neural network to predict the next character. This is the first model in our progression that uses gradients.

## How It Works

### Architecture

The model conditions on a **fixed context window** of $W$ characters:

$$
P(c_t \mid c_{t-W}, \dots, c_{t-1}) = \text{softmax}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot [\mathbf{e}_{t-W}; \dots; \mathbf{e}_{t-1}] + \mathbf{b}_1) + \mathbf{b}_2)
$$

Where:
- $\mathbf{e}_i = \text{Embedding}(c_i)$ maps each character to a dense vector of dimension $d$
- $[\cdot; \cdot]$ denotes concatenation — the input to the MLP is all $W$ embeddings flattened into one vector of dimension $W \times d$
- $\mathbf{W}_1 \in \mathbb{R}^{h \times Wd}$ and $\mathbf{W}_2 \in \mathbb{R}^{V \times h}$ are learned weight matrices
- $V$ is the vocabulary size

**Worked example**: Consider predicting the next character after `"ADD 5 3 ="` with a window $W=8$ and embedding dim $d=4$.

The model sees the last 8 characters: `A`, `D`, `D`, ` `, `5`, ` `, `3`, ` `, `=` — wait, that's 9 characters. With $W=8$, the window is `D 5 3 =` (the last 8). Each gets a 4-dimensional embedding:

```
e("D") = [0.2, -0.1, 0.8, 0.3]
e(" ") = [0.0,  0.1, 0.0, 0.0]
e("5") = [0.9,  0.7, 0.1, 0.4]    <- digits cluster together in embedding space
e(" ") = [0.0,  0.1, 0.0, 0.0]
e("3") = [0.8,  0.6, 0.2, 0.3]    <- notice: similar to "5"!
e(" ") = [0.0,  0.1, 0.0, 0.0]
e("=") = [0.5,  0.0, 0.9, 0.1]
```

These 8 vectors are concatenated into one input vector of size 8 x 4 = 32, then fed through the MLP. The MLP outputs a probability for each character in the vocabulary. If it learned that "5" and "3" together with "=" should produce "8", it outputs a high probability for "8".

But notice: the `A` at the start (which tells us this is addition, not subtraction) has already fallen off the left side of the window. The model is blind to it.

### Embedding Layer

Each character $c$ gets mapped to a learned vector:

$$
\mathbf{e}_c = \mathbf{E}[c] \quad \text{where } \mathbf{E} \in \mathbb{R}^{V \times d}
$$

Unlike n-grams which treat each character as an isolated symbol, embeddings place similar characters (like digits `0`-`9`) in nearby regions of the vector space. This is the first hint of **generalization**.

**Why this matters — a concrete comparison**: In the bigram model, `5` and `7` are completely unrelated symbols. If the model learned that `=5` is a valid ending, that tells it nothing about `=7`. But with embeddings, `5` and `7` get similar vectors (because they appear in similar contexts — after `ADD`, before `=`, etc.). So learning about one digit partially transfers to others. The model doesn't have to memorize every single combination from scratch.

### Training: Cross-Entropy Loss

The model is trained to minimize the cross-entropy between its predicted distribution and the true next character:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(c_i \mid c_{i-W}, \dots, c_{i-1})
$$

This is optimized with gradient descent (Adam optimizer). Unlike n-grams which just count, the MLP **adjusts its internal representations** to make better predictions.

**Worked example**: On the first training step, the model sees the window `"DD 5 3 ="` and predicts the next character. Suppose it assigns P(`8`) = 0.02 (nearly random). The true answer is `8`, so the loss is $-\log(0.02) = 3.9$ — very high. Gradients flow back through the network, nudging the weights so that next time it sees a similar pattern, P(`8`) will be higher. After many such updates across the whole corpus, the loss drops from ~3.5 to ~0.34 (see the loss curve plot below).

### Generation

Same as n-grams — autoregressive, one character at a time:

1. Encode the prompt as character IDs
2. Feed the last $W$ characters through the network
3. Take the argmax (greedy) of the output distribution
4. Append the predicted character
5. Slide the window forward and repeat

## Step-by-Step: What Happens During Training

1. **Build training corpus**: same as Chapter 01 — prompt+answer pairs from all tasks
2. **Fit tokenizer**: character-level, same as before
3. **Prepare sequences**: for each training string, create (context, target) pairs:
   - Given `"ADD 5 3 =8"` with window $W=8$:
   - Input: `"ADD 5 3 "`, Target: `=`
   - Input: `"DD 5 3 ="`, Target: `8`
4. **Train with SGD**: forward pass, compute loss, backpropagate, update weights
5. **Loss decreases** over epochs as the network learns patterns

## What the MLP Can Learn (and Can't)

### Can Learn

- **Local patterns within the window**: if the answer is determined by the last $W$ characters, the MLP can learn it
- **Distributed representations**: digits that appear in similar contexts get similar embeddings
- **Non-linear combinations**: unlike n-grams which just count, the MLP can learn interactions between positions in the window

### Cannot Learn

- **Long-range dependencies**: if the answer depends on characters more than $W$ positions back, the MLP is blind to them
- **Variable-length patterns**: the window is fixed — it can't adapt to shorter or longer prompts
- **Abstention**: like n-grams, it always produces an output — no "I don't know"

### Improvement Over N-grams

| Aspect | N-gram | FFN LM |
|--------|--------|--------|
| Context | 1-2 characters | $W$ characters (configurable) |
| Representation | One-hot counting | Learned embeddings |
| Generalization | None — exact match only | Within-window pattern learning |
| Parameters | $O(V^n)$ counts | $O(W \cdot d \cdot h + h \cdot V)$ weights |
| Training | Counting (instant) | Gradient descent (minutes) |

## Human Lens

Humans don't process language through a fixed sliding window. They:

1. **Parse the entire prompt** regardless of length — working memory holds the full context
2. **Generalize rules** — knowing `5+3=8` means they can compute `50+30=80` without retraining
3. **Apply algorithms** — addition is a procedure, not a pattern-matching operation
4. **Adapt context dynamically** — they attend to what's relevant, not a fixed window

The MLP is a "memorizer of windows" — it can learn associations within its fixed view but can't reason about what it sees. If you increase the window, it gets better on tasks that fit in that window, but the parameter count explodes.

## What to Observe When Running

Run `python chapters/02_ffn_lm/run.py` and notice:

1. **Loss curve decreases** — the model is learning (unlike n-grams which just counted)
2. **Copy task improves** — with a large enough window, the model can memorize short copy patterns
3. **Arithmetic stays low** — the answer depends on understanding numbers, not just local patterns
4. **Still 100% hallucination on unknowns** — no abstention mechanism
5. **Improvement over trigram** — the MLP's learned embeddings capture more than raw co-occurrence counts
6. **Effect of window size** — compare $W=4$ vs $W=8$ to see how context matters

### Generated Plots

**Training loss curve** (`results/ch02_loss_curve.png`):

![Loss curve](results/ch02_loss_curve.png)

This is the first time we see a loss curve in this project — n-grams had no training loop. The curve drops steeply from ~3.5 (random guessing over ~79 characters) down to ~0.34 over 15 epochs. The steep initial drop means the model quickly learns the most common patterns (like "after `=`, produce a digit"). The slower tail means it's refining harder patterns within its window.

**Task comparison** (`results/ch02_comparison.png`):

![Comparison bar chart](results/ch02_comparison.png)

Compare this with the Chapter 01 plot: the FFN LM shows its first real progress on knowledge QA (36%), where short answers sometimes fit within the 8-character window. Arithmetic and copy remain at 0% — the relevant information (operands, characters to copy) falls outside the window. The human agent still dominates at 100% across the board.

## What's Next

In **Chapter 03 (RNN/GRU)**, we remove the fixed window entirely. Recurrent models process one character at a time but maintain a hidden state that theoretically captures the entire history. The question: does a compressed state vector actually remember what matters?
