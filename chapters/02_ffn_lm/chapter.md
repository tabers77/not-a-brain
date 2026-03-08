# Chapter 02: Feed-Forward Language Model (MLP)

## Goal

Replace counting with learning. A fixed-window MLP language model takes the previous $W$ characters, embeds them, and passes them through a neural network to predict the next character. This is the first model in our progression that uses gradients.

## The Running Example

We trace three benchmark prompts through the FFN language model with window $W=8$ and embedding dimension $d=4$. These same three prompts will follow us through every chapter, so we can measure exactly what each architectural upgrade buys us.

### Prompt 1: `"ADD 5 3 ="` -- Testing Computation

The full prompt is 10 characters. With $W=8$, the model sees only the last 8: `D 5 3 =` (characters `D`, ` `, `5`, ` `, `3`, ` `, `=`). The `A` that tells us "this is addition" has fallen off the left edge.

What the model sees:
- The operands `5` and `3` -- yes, they are inside the window
- The `=` sign signaling "produce the answer now" -- yes
- The operation type `ADD` -- partially. It sees `D` but not `A`

The model embeds these 8 characters into a 32-dimensional vector (8 chars x 4 dims) and pushes it through the MLP. If during training the model saw many examples of `5`, `3`, and `=` appearing together with the answer `8`, it might memorize this specific combination. But an MLP cannot execute an addition algorithm -- it can only recall patterns. If the exact combination `D 5 3 =` appeared in training, maybe it gets lucky. If not, it guesses.

**Output**: `"5"` (retrieves an operand it can see, but cannot compute the sum) -- WRONG.

Compare with Chapter 01: the bigram saw only `=` and picked a random digit. Now the FFN sees `5`, `3`, AND `=`. Progress -- but seeing is not computing.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` -- Testing Retrieval

The full prompt is 53 characters. With $W=8$, the model sees only the last 8: `france?"`. The answer `paris` sits at positions 6-11 -- roughly 40 characters outside the window. The model literally cannot see it.

What the model sees:
- `france?"` -- the tail end of the question
- Nothing about `paris`, nothing about `FACT:`, nothing about the stated relationship

The model knows it needs to produce something after `france?"`. From training, common continuations of text ending in `france?` might include words like `is`, `the`, or other frequent fragments.

**Output**: `"is"` (generates a common continuation of `france?`) -- WRONG.

This is the FFN's fundamental limit: the answer is RIGHT THERE in the prompt, but outside the window. The model is as blind to `paris` as the bigram was.

### Prompt 3: `"Q: What is the capital of the Moon?"` -- Testing Hallucination

The full prompt is 38 characters. With $W=8$, the model sees: ` Moon?"` or similar. There is no correct factual answer -- the Moon has no capital. A good model should abstain or say `unknown`.

The model sees a question-like ending and generates a common continuation -- some frequent word or character.

**Output**: `"the"` (a high-frequency continuation) -- HALLUCINATION.

Same as n-grams: no abstention mechanism exists. The FFN improved on n-grams for short-range patterns, but hallucination behavior is unchanged.

### Summary Table

```
| Prompt                | Ch01 Bigram         | Ch02 FFN (W=8)           | Correct   | What changed                              |
|-----------------------|---------------------|--------------------------|-----------|-------------------------------------------|
| ADD 5 3 =             | "1" (sees only "=") | "5" (sees operands,      | "8"       | Wider context -- sees more, but still     |
|                       |                     |  can't compute)          |           |  can't compute                            |
| FACT: paris...        | " " (sees only "?") | "is" (sees "france?",    | "paris"   | Wider context -- but answer is still      |
|                       |                     |  not "paris")            |           |  too far back                             |
| Q: capital of Moon?   | "the"               | "the"                    | "unknown" | No change -- still 100% hallucination     |
```

## How It Works

### Architecture

The model conditions on a **fixed context window** of $W$ characters:

$$
P(c_t \mid c_{t-W}, \dots, c_{t-1}) = \text{softmax}(\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot [\mathbf{e}_{t-W}; \dots; \mathbf{e}_{t-1}] + \mathbf{b}_1) + \mathbf{b}_2)
$$

Where:
- $\mathbf{e}_i = \text{Embedding}(c_i)$ maps each character to a dense vector of dimension $d$
- $[\cdot; \cdot]$ denotes concatenation -- the input to the MLP is all $W$ embeddings flattened into one vector of dimension $W \times d$
- $\mathbf{W}_1 \in \mathbb{R}^{h \times Wd}$ and $\mathbf{W}_2 \in \mathbb{R}^{V \times h}$ are learned weight matrices
- $V$ is the vocabulary size

**Worked example with Prompt 1**: Consider predicting the next character after `"ADD 5 3 ="` with $W=8$ and $d=4$. The window captures `D 5 3 =` (the last 8 characters). Each gets a 4-dimensional embedding:

```
e("D") = [0.2, -0.1, 0.8, 0.3]
e(" ") = [0.0,  0.1, 0.0, 0.0]
e("5") = [0.9,  0.7, 0.1, 0.4]    <- digits cluster together in embedding space
e(" ") = [0.0,  0.1, 0.0, 0.0]
e("3") = [0.8,  0.6, 0.2, 0.3]    <- notice: similar to "5"!
e(" ") = [0.0,  0.1, 0.0, 0.0]
e("=") = [0.5,  0.0, 0.9, 0.1]
```

These 8 vectors are concatenated into one input vector of size $8 \times 4 = 32$, then fed through the MLP. The hidden layer applies $\text{ReLU}(\mathbf{W}_1 \cdot \mathbf{x} + \mathbf{b}_1)$, producing a vector that mixes information from all 8 positions. The output layer projects this to a probability over every character in the vocabulary.

The MLP can detect that `5` and `3` co-occur with `=`, and if it memorized this combination, it might output `8`. But the `A` that tells us "this is addition, not subtraction" fell off the window. Without knowing the operation, even a perfect memorizer is guessing.

**Worked example with Prompt 2**: For `"FACT: paris is capital of france. Q: capital of france?"`, the window captures `france?"`. The 8 embeddings are concatenated into a 32-dim vector. The MLP processes this vector, but no amount of non-linear transformation can recover `paris` from `france?"` -- the information simply is not in the input. The model produces whatever continuation it learned for question-ending patterns.

### Embedding Layer

Each character $c$ gets mapped to a learned vector:

$$
\mathbf{e}_c = \mathbf{E}[c] \quad \text{where } \mathbf{E} \in \mathbb{R}^{V \times d}
$$

Unlike n-grams which treat each character as an isolated symbol, embeddings place similar characters (like digits `0`-`9`) in nearby regions of the vector space. This is the first hint of **generalization**.

**Why this matters -- a concrete comparison with the benchmark prompts**: In the bigram model, `5` and `7` are completely unrelated symbols. If the model learned that `=5` is a valid ending, that tells it nothing about `=7`. But with embeddings, `5` and `7` get similar vectors (because they appear in similar contexts -- after digits, before `=`, inside arithmetic prompts like `"ADD 5 3 ="`). So learning about one digit partially transfers to others. When the model sees `3` in Prompt 1, its embedding is close to `5`'s embedding -- both are digits. The model does not have to memorize every single combination from scratch.

For Prompt 2, the embedding of `?` learns to signal "end of question" across many training examples. But embedding similarity cannot bridge a 40-character gap -- it only helps with characters that are already inside the window.

### Training: Cross-Entropy Loss

The model is trained to minimize the cross-entropy between its predicted distribution and the true next character:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log P_\theta(c_i \mid c_{i-W}, \dots, c_{i-1})
$$

This is optimized with gradient descent (Adam optimizer). Unlike n-grams which just count, the MLP **adjusts its internal representations** to make better predictions.

**Worked example**: On the first training step, the model sees the window `"D 5 3 ="` from Prompt 1 and predicts the next character. Suppose it assigns P(`8`) = 0.02 (nearly random). The true answer is `8`, so the loss is $-\log(0.02) = 3.9$ -- very high. Gradients flow back through the network, nudging the weights so that next time it sees a similar pattern, P(`8`) will be higher. After many such updates across the whole corpus, the loss drops from ~3.5 to ~0.34 (see the loss curve plot below).

For Prompt 2, training examples with `"france?"` followed by various characters give contradictory signals -- sometimes the next character is `p` (from `paris`), sometimes something else entirely. The model averages over these signals, which is why it defaults to a generic continuation like `is`.

### Generation

Same as n-grams -- autoregressive, one character at a time:

1. Encode the prompt as character IDs
2. Feed the last $W$ characters through the network
3. Take the argmax (greedy) of the output distribution
4. Append the predicted character
5. Slide the window forward and repeat

## Step-by-Step: What Happens During Training

1. **Build training corpus**: same as Chapter 01 -- prompt+answer pairs from all tasks
2. **Fit tokenizer**: character-level, same as before
3. **Prepare sequences**: for each training string, create (context, target) pairs:
   - Given `"ADD 5 3 =8"` with window $W=8$:
   - Input: `"ADD 5 3 "`, Target: `=`
   - Input: `"DD 5 3 ="`, Target: `8`
   - Given `"FACT: paris is capital of france. Q: capital of france?paris"` with window $W=8$:
   - Input: `"france?p"`, Target: `a` -- the model sees `p` but not why `p` is correct
   - Input: `"rance?pa"`, Target: `r` -- each step, it sees the answer emerging but not its source
4. **Train with SGD**: forward pass, compute loss, backpropagate, update weights
5. **Loss decreases** over epochs as the network learns patterns

## What the MLP Can Learn (and Can't)

### Can Learn

- **Local patterns within the window**: if the answer is determined by the last $W$ characters, the MLP can learn it
- **Distributed representations**: digits that appear in similar contexts get similar embeddings
- **Non-linear combinations**: unlike n-grams which just count, the MLP can learn interactions between positions in the window

### Cannot Learn

- **Long-range dependencies**: if the answer depends on characters more than $W$ positions back, the MLP is blind to them (Prompt 2 demonstrates this -- `paris` is 40+ characters away)
- **Variable-length patterns**: the window is fixed -- it cannot adapt to shorter or longer prompts
- **Computation**: the MLP memorizes input-output mappings, it does not execute algorithms like addition (Prompt 1 demonstrates this)
- **Abstention**: like n-grams, it always produces an output -- no "I don't know" (Prompt 3 demonstrates this)

### Improvement Over N-grams

| Aspect | N-gram | FFN LM |
|--------|--------|--------|
| Context | 1-2 characters | $W$ characters (configurable) |
| Representation | One-hot counting | Learned embeddings |
| Generalization | None -- exact match only | Within-window pattern learning |
| Parameters | $O(V^n)$ counts | $O(W \cdot d \cdot h + h \cdot V)$ weights |
| Training | Counting (instant) | Gradient descent (minutes) |

## Human Lens

A human given `"ADD 5 3 ="` reads the FULL prompt regardless of length. There is no fixed window -- working memory holds the entire context. The human sees `ADD`, recognizes it as addition, retrieves the operands `5` and `3`, executes a mental addition procedure, and writes `8`. The FFN sees `D 5 3 =`, has no concept of "addition as a procedure," and can only succeed if it memorized this exact window during training.

For `"FACT: paris is capital of france. Q: capital of france?"`, the human scans the prompt and finds `paris` stated as the answer at the beginning. It does not matter that `paris` is 40 characters away from the question -- human attention jumps to the relevant part. The FFN is locked into its 8-character window and cannot reach back.

For `"Q: What is the capital of the Moon?"`, the human recognizes the question is unanswerable and abstains: "unknown" or "there is no capital of the Moon." The FFN has no mechanism for uncertainty. It always produces output, regardless of whether the question makes sense.

The FFN is a "memorizer of windows" -- it can learn associations within its fixed view, but reasoning about what it sees is beyond its architecture. Increasing the window helps on tasks that fit within it, but the parameter count explodes as $W$ grows, and the core limitations (no computation, no abstention) remain.

## What to Observe When Running

Run `python chapters/02_ffn_lm/run.py` and notice:

1. **Loss curve decreases** -- the model is learning (unlike n-grams which just counted)
2. **Copy task improves** -- with a large enough window, the model can memorize short copy patterns
3. **Arithmetic stays low** -- the answer depends on understanding numbers, not just local patterns
4. **Still 100% hallucination on unknowns** -- no abstention mechanism
5. **Improvement over trigram** -- the MLP's learned embeddings capture more than raw co-occurrence counts
6. **Effect of window size** -- compare $W=4$ vs $W=8$ to see how context matters

### Generated Plots

**Training loss curve** (`results/ch02_loss_curve.png`):

![Loss curve](results/ch02_loss_curve.png)

This is the first time we see a loss curve in this project -- n-grams had no training loop. The curve drops steeply from ~3.5 (random guessing over ~79 characters) down to ~0.34 over 15 epochs. The steep initial drop means the model quickly learns the most common patterns (like "after `=`, produce a digit"). The slower tail means it is refining harder patterns within its window.

**Task comparison** (`results/ch02_comparison.png`):

![Comparison bar chart](results/ch02_comparison.png)

Compare this with the Chapter 01 plot: the FFN LM shows its first real progress on knowledge QA (36%), where short answers sometimes fit within the 8-character window. Arithmetic and copy remain at 0% -- the relevant information (operands, characters to copy) falls outside the window. The human agent still dominates at 100% across the board.

## What's Next

In **Chapter 03 (RNN/GRU)**, we remove the fixed window entirely. Recurrent models process one character at a time but maintain a hidden state that theoretically captures the entire history. The question: does a compressed state vector actually remember what matters? We will trace our three benchmark prompts through the recurrent architecture and see whether `paris` -- 40 characters back -- finally becomes reachable.
