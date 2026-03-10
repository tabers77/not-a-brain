# Chapter 05: Transformer

## Goal

Assemble the full Transformer: multi-head attention + feed-forward layers + residual connections + layer normalization, stacked into multiple layers. This is the architecture behind GPT, BERT, and every modern LLM.

Chapter 04 solved WHERE to look (attention). Chapter 05 adds WHAT TO DO with what you found (feed-forward networks). That is the full recipe: retrieval + computation.

## The Running Example

We trace all 3 benchmark prompts through the Transformer with $d_{model}=64$, $n_{heads}=4$, $n_{layers}=2$, $d_{ff}=128$. This is the most detailed trace in the book because this is the most important chapter -- the Transformer is the architecture that powers everything that follows.

---

### Prompt 1: `"ADD 5 3 ="` -- THE BREAKTHROUGH

Expected output: `"8"`

This is the prompt that every previous architecture failed on. The bigram model guessed `"1"`. The FFN guessed `"5"`. The GRU guessed `"5"`. The attention model guessed `"5"`. All failures, because none of them could both retrieve the operands AND compute on them. The Transformer can.

**Embedding**

Each token gets a token embedding plus a positional embedding, both 64-dimensional vectors:

```
tokens:    <BOS>  A     D     D    ' '    5    ' '    3    ' '    =
positions:   0    1     2     3     4     5     6     7     8     9

For position 9 ("="):
  tok_embed("=") = [0.5, 0.0, 0.9, 0.1, ...]        (64 values)
  pos_embed(9)   = [0.1, -0.2, 0.0, 0.3, ...]        (64 values)
  x_9            = [0.6, -0.2, 0.9, 0.4, ...]         (sum of both)

For position 5 ("5"):
  tok_embed("5") = [0.9, 0.7, 0.1, 0.4, ...]
  pos_embed(5)   = [0.0, 0.1, -0.1, 0.2, ...]
  x_5            = [0.9, 0.8, 0.0, 0.6, ...]

For position 7 ("3"):
  tok_embed("3") = [0.4, 0.6, 0.3, 0.8, ...]
  pos_embed(7)   = [0.2, -0.1, 0.1, 0.0, ...]
  x_7            = [0.6, 0.5, 0.4, 0.8, ...]
```

At this point, each position is an island. No position knows about any other position. The `=` sign does not yet know that `5` and `3` exist.

**Layer 0 -- Attention: retrieve the operands**

First, LayerNorm stabilizes the values:

```
ln_x_9 = LayerNorm(x_9)
  mean of x_9 = 0.425,  std = 0.41
  ln_x_9 = (x_9 - 0.425) / 0.41 = [0.43, -1.52, 1.16, -0.06, ...]
```

Then multi-head attention. Each of 4 heads ($d_k=16$ each) independently decides where to look. Position `=` computes queries and matches them against keys at all previous positions:

```
Head 0 (learns: "attend to the operation token"):
  q_9 . k_1 = 3.1   (A -- part of "ADD")
  q_9 . k_2 = 0.4   (D)
  q_9 . k_5 = 0.2   (5)
  q_9 . k_7 = 0.1   (3)
  After softmax / sqrt(16):
  weights = [0.01, 0.82, 0.05, 0.04, 0.01, 0.03, 0.01, 0.02, 0.01, 0.00]
                   ^A is the focus -- this head identifies the operation

Head 1 (learns: "attend to the digit operands"):
  q_9 . k_5 = 2.8   (5 -- first operand)
  q_9 . k_7 = 2.5   (3 -- second operand)
  After softmax:
  weights = [0.01, 0.02, 0.01, 0.01, 0.01, 0.42, 0.01, 0.38, 0.01, 0.12]
                                               ^5          ^3

Head 2 (learns: "attend to nearby context"):
  Focuses on positions 7, 8, 9 (recent tokens)

Head 3 (learns: "attend to the delimiter"):
  Focuses on the "=" itself and spaces
```

Each head retrieves different information. Their outputs are concatenated ($4 \times 16 = 64$) and projected:

```
attn_out_9 = W_O * concat(head0_out, head1_out, head2_out, head3_out)
           = [0.2, 0.5, -0.3, 0.1, ...]    (64 values)
```

This is the same mechanism from Chapter 04. So far, nothing new.

**Layer 0 -- Residual: preserve the original signal**

```
x_9 = x_9 + attn_out_9
    = [0.6, -0.2, 0.9, 0.4, ...] + [0.2, 0.5, -0.3, 0.1, ...]
    = [0.8,  0.3, 0.6, 0.5, ...]
```

The original embedding of `=` is preserved, now enriched with information about `ADD`, `5`, and `3`. Nothing has been lost.

**Layer 0 -- FFN: THIS IS NEW. This is where computation happens.**

This is the component that was missing from every previous chapter. The feed-forward network takes the blended representation (which now contains "I see ADD, 5, and 3") and COMPUTES on it.

```
ln_x_9 = LayerNorm(x_9)    = [0.51, -0.32, 0.18, 0.08, ...]

Step 1 -- Expand: 64 dimensions -> 128 dimensions
  h_raw = ln_x_9 @ W1 + b1 = [1.2, -0.5, 0.8, -2.1, 0.3, ...]  (128 dims)

Step 2 -- GELU activation: the nonlinearity that enables computation
  h = GELU(h_raw) = [1.1, 0.0, 0.7, 0.0, 0.2, ...]              (128 dims)

  GELU zeros out some dimensions and activates others.
  The surviving neurons form a sparse "code" for "ADD(5,3)".

  Neuron 17 fires strongly: it has learned "digit + digit in ADD context"
  Neuron 42 fires strongly: it has learned to activate for sums near 8
  Neuron 91 is zeroed out: it responds to SUB operations, not ADD
  Neuron 105 is zeroed out: it responds to large numbers, not small ones

  This is the key insight: GELU is nonlinear. A weighted average (which is
  all attention can do) is linear -- it cannot learn "5 + 3 = 8" because
  that function is not a weighted average of the embeddings of 5 and 3.
  But GELU(x W1 + b1) CAN learn this, because it can represent arbitrary
  functions through its expand-activate-compress pattern.

Step 3 -- Compress: 128 dimensions -> 64 dimensions
  ffn_out = h @ W2 + b2 = [0.1, 0.0, 0.3, -0.1, ...]             (64 dims)
```

The FFN is applied independently to each position -- positions only interact through attention. Think of it as: attention gathers information across positions, then the FFN processes that information within each position.

**Layer 0 -- Residual again: computation result added to signal**

```
x_9 = x_9 + ffn_out
    = [0.8, 0.3, 0.6, 0.5, ...] + [0.1, 0.0, 0.3, -0.1, ...]
    = [0.9, 0.3, 0.9, 0.4, ...]
```

After Layer 0, position `=` encodes: "I am at an equals sign, in an ADD task, with operands 5 and 3, and the FFN has started processing what that means."

**Layer 1 -- Attention: attend to PROCESSED outputs**

Layer 1 attends over the outputs of Layer 0 -- not the raw embeddings. Every position is now richer because each one has already been processed by Layer 0's attention + FFN:

```
Layer 0 output at position 5: encodes "the digit 5 in an ADD context"
Layer 0 output at position 7: encodes "the digit 3 in an ADD context"
Layer 0 output at position 1: encodes "this is an ADD operation"

Layer 1 attention at position 9 ("="):
  Now attending to PROCESSED representations, not raw embeddings.
  The features are richer, so attention can be more precise.
  Head 1 now sees "5-in-addition-context" rather than just "the digit 5".
```

**Layer 1 -- FFN: refine the computation**

The second FFN refines the representation further. Two rounds of "retrieve then compute" have been enough:

```
x_9 = [1.2, -0.5, 0.7, 0.9, ...]     (after Layer 1)
```

This vector now encodes "the answer is 8."

**Output: LayerNorm -> Linear -> logits**

```
ln_x_9 = LayerNorm(x_9)
logits  = ln_x_9 @ W_output    (64 -> vocab_size)

logits = [..., -2.1, -1.8, -0.5, 0.3, 1.1, 0.2, -0.9, -1.5, 4.7, -2.0, ...]
                                                                     ^"8"
         (indices:  0     1     2    3    4    5     6     7    8     9  ...)

After softmax: P("8") = 0.89,  P("5") = 0.03,  P("3") = 0.02, ...
```

**Output: `"8"` -- CORRECT. First model to solve computation.**

Chapter 04 solved WHERE to look (attention found `5` and `3`). Chapter 05 adds WHAT TO DO with what you found (FFN computes `5 + 3 = 8`). That is the full recipe: retrieval + computation.

---

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"`

Expected output: `"paris"`

**Attention retrieves "paris" -- same mechanism as Chapter 04**

The attention heads at the final position (`?`) attend back to the fact statement. Head 1 matches "capital of france" in the question to "capital of france" in the fact, and retrieves the associated value -- the representation at the position of "paris".

**Residual connections preserve the signal through both layers**

This is where residual connections matter most. The "paris" signal must survive through two attention sublayers and two FFN sublayers -- four transformations in total. Without residuals, each transformation could overwrite the signal. With residuals:

```
x_final = embedding("?")
        + Layer0_attention_output      ("paris" retrieved from the fact)
        + Layer0_ffn_output            (processing the retrieval)
        + Layer1_attention_output      (refined retrieval)
        + Layer1_ffn_output            (final processing)
```

The "paris" signal, once retrieved by attention, is added to the residual stream and preserved through every subsequent layer.

**Output: `"paris"` -- CORRECT (same as Ch04, but more robust thanks to residuals)**

Attention already solved this in Chapter 04. The Transformer makes it more reliable -- residual connections ensure the retrieved information survives through multiple layers. In a deeper model (say, 12 or 24 layers), this robustness becomes critical. Without residuals, the signal would degrade with every layer. With residuals, it is carried intact.

---

### Prompt 3: `"Q: What is the capital of the Moon?"`

Expected output: `"unknown"`

**Attention scans the input**

The attention heads at the final position scan back through the tokens. They find "capital" and "Moon". Head 1 matches "capital" against training examples it has seen -- facts about Paris being the capital of France, Berlin being the capital of Germany, Tokyo being the capital of Japan. The attention weights distribute across these stored patterns in the model's parameters.

**FFN COMPUTES on this -- but computes the WRONG thing**

The FFN takes the attention output -- a blend of capital-related representations -- and processes it. It has learned the pattern "when you see 'capital of X', output a city name." It has no mechanism to check whether X actually has a capital. It produces a plausible city.

**Output: `"earth"` -- STILL HALLUCINATION**

The Transformer is the most powerful architecture we have built so far, and it STILL hallucinates. Better retrieval + better computation makes accuracy better on answerable questions, but NEVER makes hallucination better on unanswerable questions. The model has no mechanism to distinguish "I found a good answer" from "I fabricated something plausible." It runs forward through its layers and produces output -- always. It cannot pause and say "I do not know." That requires an explicit uncertainty or abstention mechanism that does not exist in this architecture.

---

### Summary Table

This is the climax of the project so far. Five architectures, three prompts, one clear pattern:

| Prompt                     | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attention | Ch05 Transformer | Human     | What changed                        |
|----------------------------|-------------|----------|----------|----------------|------------------|-----------|-------------------------------------|
| ADD 5 3 =                  | "1"         | "5"      | "5"      | "5"            | "8"              | "8"       | SOLVED -- attention + FFN computes  |
| FACT: paris... Q: capital? | " "         | "is"     | "capital"| "paris"        | "paris"          | "paris"   | Still solved (from Ch04)            |
| Q: capital of Moon?        | "the"       | "the"    | "mars"   | "tokyo"        | "earth"          | "unknown" | Still hallucinates -- no abstention |

Two out of three. The model can now retrieve AND compute. But it still cannot abstain -- it will always hallucinate on unanswerable questions.

The gap between the Transformer and the human on Prompt 3 is not a matter of scale. A 175-billion-parameter GPT-3 has the same problem. It is an architectural gap. The Transformer has no introspection, no mechanism to evaluate its own confidence, no way to distinguish knowledge from fabrication. Scaling up gives it more knowledge and better patterns, but it does not give it the ability to say "I do not know."

## How the Transformer Works

### The Full Architecture

A decoder-only Transformer (GPT-style) stacks $N$ identical blocks. Each block has two sub-layers:

```
x = x + MultiHeadAttention(LayerNorm(x))     <- retrieve relevant information
x = x + FeedForward(LayerNorm(x))            <- compute on what you found
```

The full model:

```
Token Embedding + Positional Embedding
-> Dropout
-> TransformerBlock 1
-> TransformerBlock 2
-> ...
-> TransformerBlock N
-> LayerNorm
-> Linear (to vocab_size)
```

### Feed-Forward Network

$$
\text{FFN}(x) = \text{GELU}(x \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2
$$

Where $\mathbf{W}_1 \in \mathbb{R}^{d_{model} \times d_{ff}}$ expands to a wider dimension, and $\mathbf{W}_2 \in \mathbb{R}^{d_{ff} \times d_{model}}$ projects back.

The FFN is the computation engine of the Transformer. The `ADD 5 3` trace above is the primary example: after attention blends the operand information into the `=` position, the FFN expands to 128 dimensions, applies GELU (which zeros out irrelevant neurons and activates relevant ones), and compresses back to 64 dimensions. The expand-activate-compress pattern lets the network learn arbitrary functions -- including "5 + 3 = 8."

Why does expansion help? In 64 dimensions, the representation is crowded -- it must encode the token identity, position, operation type, and operand values all at once. Expanding to 128 dimensions gives the network room to disentangle these features. Some neurons can specialize for "addition of small numbers," others for "specific digit sums," others for "operation type detection." After processing, the result is compressed back to 64 dimensions for the next layer.

### Residual Connections

$$
\text{output} = x + \text{sublayer}(x)
$$

The `FACT: paris` trace above is the primary example for residuals. The "paris" signal, once retrieved by attention in Layer 0, is added to the residual stream. Every subsequent sublayer (Layer 0 FFN, Layer 1 attention, Layer 1 FFN) adds to this stream rather than replacing it. The original signal survives intact through the entire network.

Without residuals, each layer completely replaces the representation:

```
Without residuals:
  x = embedding("?")                           = [0.5, 0.0, 0.9, 0.1, ...]
  x = attention(x)                             = [0.2, 0.5, -0.3, 0.1, ...]
      ^ the original embedding is GONE -- replaced entirely

With residuals:
  x = embedding("?")                           = [0.5, 0.0, 0.9, 0.1, ...]
  x = x + attention(x)                         = [0.7, 0.5, 0.6, 0.2, ...]
      ^ original embedding is still there, enriched with attention output
```

Residuals also help gradients flow during training:

```
No residuals:  grad = grad * dFFN1 * dAttn1 * dFFN0 * dAttn0   (may vanish)
With residuals: grad = grad * (I + dFFN1) * (I + dAttn1) * ... (identity ensures flow)
```

### Layer Normalization

$$
\text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma + \epsilon} + \beta
$$

LayerNorm centers and scales each position's vector before it enters a sublayer. This prevents activation values from drifting to extreme ranges as they pass through multiple layers.

```
Before LayerNorm:
  x_9 = [0.8, 0.3, 0.6, 0.5, 12.1, -8.3, 0.1, 0.4, ...]
  mean = 0.81,  std = 4.2
  Some values are huge (12.1, -8.3) -- these would saturate softmax and GELU.

After LayerNorm:
  x_9 = gamma * (x_9 - 0.81) / 4.2 + beta
      = [0.0, -0.12, -0.05, -0.07, 2.69, -2.17, -0.17, -0.10, ...]
  Values are now centered and scaled. The GELU and softmax that follow
  will operate in their well-behaved range.
```

We use **pre-norm** (GPT-2 style): normalize before each sublayer, not after.

```
Pre-norm:   x = x + sublayer(LayerNorm(x))    <- we use this
Post-norm:  x = LayerNorm(x + sublayer(x))    <- original Transformer paper
```

Pre-norm is more stable because the residual path carries un-normalized values, so gradients flow freely through the skip connection.

### Weight Tying

The token embedding matrix and the output projection share the same weights:

```
Embedding:  token_id -> 64-dim vector     (matrix E, shape: vocab_size x 64)
Output:     64-dim vector -> logits        (matrix W, shape: vocab_size x 64)

With weight tying: W = E  (same matrix!)
```

Token `8` has embedding $e_8 = [0.3, 0.9, -0.1, ...]$. To produce the logit for token `8`, the model computes $\text{dot}(\text{hidden\_state}, e_8)$. If the hidden state is "close" to the embedding of `8` in the 64-dim space, the logit is high. This creates a clean symmetry: the model "understands" tokens and "produces" tokens in the same vector space.

## Step-by-Step: What Happens During Training

1. **Build corpus**: same task prompt+answer pairs as all previous chapters
   - `"ADD 5 3 =8"`, `"FACT: paris is capital of france. Q: capital of france?paris"`, etc.

2. **Fit tokenizer**: character-level, same as before

3. **Prepare sequences**: pad to equal length, shift by one for input/target:
   ```
   Input:   <BOS>  A   D   D   ' '  5  ' '  3  ' '  =   8
   Target:    A    D   D  ' '   5  ' '  3  ' '  =    8  <EOS>
   ```

   For the benchmark prompts:
   ```
   Prompt 1: input ends with "=", target at that position is "8"
   Prompt 2: input ends with "?", target at that position is "p" (start of "paris")
   Prompt 3: input ends with "?", target at that position is "u" (start of "unknown")
   ```

4. **Forward pass**: the full Transformer processes all positions in parallel (unlike the sequential RNN). At each position, it produces logits over the vocabulary.

5. **Loss**: cross-entropy at every position:
   - Position 0: model sees `<BOS>`, must predict `A`. Loss = $-\log P(\text{A})$.
   - Position 9: model sees everything up to `=`, must predict `8`. This is the hard part -- the model must attend back to `5` and `3`, then compute the sum.
   - Total loss = average across all positions.

6. **Backprop**: gradients flow through three paths simultaneously:
   - **Through attention weights**: directly from position 9 to positions 5 and 7 (where `5` and `3` are). No vanishing gradient -- the connection is direct.
   - **Through residual connections**: direct skip path from the output layer to the embedding layer. No 30 sequential matrix multiplications like the RNN.
   - **Through the FFN**: gradients tell the FFN weights how to turn "I see 5 and 3 in an ADD context" into "output 8."

**Key difference from all previous chapters**: The RNN (Chapter 03) had to pass gradients through 9 sequential steps to connect position 9 to position 0. The attention-only model (Chapter 04) had direct attention paths but no FFN to process them. The Transformer has both: direct attention paths AND computation at every layer.

## What the Transformer Can Learn (and Can't)

### Can Learn

- **Retrieval + computation in combination**: attention finds the operands, FFN computes the result -- this is why `ADD 5 3 = 8` finally works
- **Multi-step reasoning**: Layer 0 retrieves raw information, Layer 1 reasons about it
- **Long-range dependencies**: same as Chapter 04 -- direct attention to any position
- **Stable deep processing**: residuals + layer norm allow stacking multiple layers without gradient issues

### Cannot Learn

- **Abstention**: the model always generates an output -- there is no "I do not know" mechanism. This is NOT a scale problem. GPT-3 (175B parameters) and GPT-4 (estimated >1T parameters) have the same fundamental limitation. The architecture runs forward and produces a token. It cannot pause, introspect, and decide to abstain.
- **Calibrated uncertainty**: the softmax probabilities are not calibrated -- a model that outputs P("earth")=0.6 for "capital of the Moon" is not expressing 60% confidence, it is expressing "earth is the most plausible completion I can generate." There is no mechanism to distinguish confident knowledge from confident fabrication.
- **Complex arithmetic beyond training distribution**: our model is tiny (64-dim, 2 layers) and character-level -- multi-digit arithmetic would require more capacity
- **Variable-depth reasoning**: with 2 layers, it can do at most 2 "steps" of reasoning. A 10-step problem is out of reach.

### Improvement Over Attention-Only (Chapter 04)

| Aspect | Attention-Only (Ch04) | Transformer (Ch05) |
|--------|----------------------|-------------------|
| Retrieval | Multi-head attention | Same -- multi-head attention |
| Computation | None -- weighted average only | FFN at every layer |
| Depth | 1 layer | $N$ stacked layers |
| Gradient flow | Direct through attention | Direct through attention + residuals |
| Training stability | Fragile for >1 layer | LayerNorm keeps it stable |
| Parameters | ~5K | ~30K (more capacity) |

## Human Lens

### Prompt 1: `"ADD 5 3 ="`

Humans parse the prompt, store `5` and `3` in working memory, recognize `ADD` as the operation, apply the addition algorithm ("5 + 3 = 8"), and verify the result ("does 8 make sense?").

The Transformer: embed, attend (retrieve `5` and `3`), FFN-compute (turn `5 + 3` into `8`), output. Similar result, fundamentally different mechanism. Humans understand addition as a rule -- a general algorithm that works on any two numbers. The Transformer learned a pattern -- a statistical regularity in its training data that maps certain digit combinations to certain outputs. It does not "understand" addition. It has memorized enough examples to interpolate correctly on small numbers.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"`

Humans scan the fact, identify it as relevant to the question, and retrieve "paris." This is goal-directed retrieval -- the human actively looks for the answer to the question.

The Transformer's attention mechanism does something structurally similar: it matches the question pattern against the fact pattern and retrieves the associated value. But machine retrieval is similarity-based, not goal-directed. The attention weights are computed by dot-product similarity between queries and keys -- there is no "understanding" of what the question is asking. It is pattern matching, not comprehension.

### Prompt 3: `"Q: What is the capital of the Moon?"`

This is where the gap is unbridgeable by architecture alone.

A human reads this question and immediately recognizes the problem: the Moon does not have a capital. The human can introspect -- "I have no knowledge of a capital of the Moon, therefore the answer is 'I do not know.'" This introspection is a meta-cognitive process: reasoning about the contents of one's own knowledge.

The Transformer has no introspection. It runs forward through its layers and produces output. The attention mechanism scans for relevant patterns, finds "capital of [place]" matches in its learned representations, and the FFN produces the most plausible completion. It cannot distinguish "I computed a real answer" from "I generated a plausible fabrication." It has no access to a representation of "what I know" versus "what I do not know." The forward pass always runs to completion and always produces a token.

This is not a matter of adding more layers or more parameters. The architecture itself lacks the mechanism. A human can say "I do not know" because the human has a model of their own knowledge. The Transformer has no model of its own knowledge -- it only has weights that produce outputs.

## What to Observe When Running

Run `python chapters/05_transformer/run.py` and notice:

1. **Loss drops faster and lower** than attention-only -- the FFN and residuals help
2. **Arithmetic improves dramatically** -- the FFN can compute, not just retrieve
3. **Layer-wise attention differs** -- Layer 0 and Layer 1 attend to different things
4. **Still 0% abstention** -- even with the full Transformer, no uncertainty mechanism
5. **More parameters** -- but weight tying keeps it efficient

### Generated Plots

**Training loss curve** (`results/ch05_loss_curve.png`):

![Loss curve](results/ch05_loss_curve.png)

Compare with Chapter 04's loss curve. The Transformer should reach a lower final loss thanks to the FFN's computation ability and the residual connections' gradient flow. The deeper architecture (2 layers vs 1) gives more capacity to fit the training data.

**Layer-wise attention heatmaps** (`results/ch05_attn_L*_*.png`):

The key insight of this chapter: different layers attend to different things. Layer 0 (early) tends to capture surface patterns -- attending to nearby tokens, delimiters, and positional neighbors. Layer 1 (later) builds on Layer 0's processed outputs and attends based on richer, more abstract representations. Compare the Layer 0 and Layer 1 heatmaps for the same input to see how attention evolves through depth.

**Task comparison** (`results/ch05_comparison.png`):

![Comparison bar chart](results/ch05_comparison.png)

This chart shows accuracy across task categories. The Transformer should show its biggest gains over Chapter 04 on tasks that need computation (arithmetic) and multi-step reasoning (compositional). Retrieval tasks should remain strong or improve thanks to residual connections preserving signal through depth.

## What's Next

We now have the core architecture of every modern LLM. Prompt 1 (computation) and Prompt 2 (retrieval) are solved. Prompt 3 (abstention) remains unsolved -- and it will stay unsolved until we add explicit mechanisms for uncertainty, calibration, or abstention in future chapters.

The evolution table tells the story clearly. Each chapter closed a gap: the bigram model could not handle context, the FFN added fixed-window context, the GRU added memory, attention added direct retrieval, and the Transformer added computation. But one gap has never closed: the gap between "generating a plausible answer" and "knowing when you do not know." That gap is architectural, not statistical. No amount of training data, no increase in model size, and no number of additional layers will close it within this architecture. It requires a fundamentally new mechanism -- and that is what comes next.
