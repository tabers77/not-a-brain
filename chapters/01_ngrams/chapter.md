# Chapter 01: N-gram Language Models

## Goal

Build the simplest possible language model — one that predicts the next character by counting how often characters appear together. No neural network, no gradients, just statistics.

## How N-grams Work

A **language model** assigns a probability to the next token given the previous tokens. N-grams do this by counting co-occurrences in training data.

### Bigram Model

A bigram predicts the next token based on **only the previous token**.

$$
P(c_t \mid c_{t-1}) = \frac{\text{count}(c_{t-1}, c_t)}{\text{count}(c_{t-1})}
$$

**Example**: If we train on `"ADD 5 3 =8"`, the bigram model learns:
- After `=`, the character `8` appeared → P(`8` | `=`) = count(`=`, `8`) / count(`=`)
- After `A`, `D` appeared → P(`D` | `A`) = count(`A`, `D`) / count(`A`)

**Generation**: Start with a prompt, then repeatedly pick the most likely next character:

```
Prompt: "ADD 5 3 ="
Step 1: After "=" → most common next char from training (e.g., "1")
Step 2: After "1" → most common next char (e.g., "9")
...
```

The problem: the model only sees **one character back**. It doesn't know what operation was requested or what the numbers were.

### Trigram Model

A trigram uses the **previous two tokens** as context.

$$
P(c_t \mid c_{t-2}, c_{t-1}) = \frac{\text{count}(c_{t-2}, c_{t-1}, c_t)}{\text{count}(c_{t-2}, c_{t-1})}
$$

This captures slightly longer patterns (e.g., `"= "` after a number), but still can't handle dependencies that span the full prompt.

### Fallback (Backoff)

When the trigram encounters a pair it hasn't seen before, it **falls back** to the bigram:

$$
P(c_t \mid c_{t-2}, c_{t-1}) =
\begin{cases}
\frac{\text{count}(c_{t-2}, c_{t-1}, c_t)}{\text{count}(c_{t-2}, c_{t-1})} & \text{if pair seen} \\[6pt]
P(c_t \mid c_{t-1}) & \text{otherwise (bigram fallback)}
\end{cases}
$$

## Step-by-Step: What Happens During Training

1. **Collect training data**: generate prompt+answer pairs from all tasks
   - `"ADD 5 3 =8"`, `"COPY: abc|abc"`, `"CHECK: ( )valid"`, etc.

2. **Fit the tokenizer**: map each character to an integer ID
   - `'A'→4, 'D'→5, ' '→6, '5'→7, ...`

3. **Count co-occurrences**: for every consecutive pair (bigram) or triple (trigram), increment a counter
   - Bigram: `counts[prev_char][next_char] += 1`
   - Trigram: `counts[(prev2, prev1)][next_char] += 1`

4. **That's it** — no optimization loop, no loss function, no gradients.

## Step-by-Step: What Happens During Generation

Given a prompt like `"ADD 5 3 ="`:

1. Encode the prompt as character IDs
2. Look up the last character (or last two for trigram) in the count table
3. Pick the character with the highest count
4. Append it to the sequence
5. Repeat until EOS or max length

## Why N-grams Fail on Our Tasks

| Task | Why It Fails | What Would Be Needed |
|------|-------------|---------------------|
| **Arithmetic** | After `=`, the model picks the globally most common digit — it doesn't know what `5 + 3` equals | Understanding of numbers and operations |
| **Copy** | After `\|`, it generates the most common character, not the specific sequence to copy | Memory of the full prompt |
| **Grammar** | Can predict common bracket pairs locally (`()`) but can't track nesting depth | A stack or counter |
| **Knowledge QA** | After `?`, it outputs common characters, not the specific fact from context | Attention to relevant context |
| **Compositional** | Can't chain operations — no intermediate state tracking | Working memory |
| **Unknown** | Always generates something — no mechanism to abstain | Uncertainty estimation |

The fundamental problem: **n-grams have a fixed, tiny context window** (1-2 characters). Every task requires understanding that spans the full prompt.

## Human Lens

Humans don't solve these tasks by counting character co-occurrences. They:

1. **Parse** the prompt to understand the task type (arithmetic, copy, etc.)
2. **Apply a learned algorithm** (addition rules, stack-based bracket matching)
3. **Hold intermediate results** in working memory
4. **Verify** the answer before responding
5. **Abstain** when the question has no answer

N-gram models have **none** of these capabilities. The gap between 0-14% (n-grams) and 100% (human) reflects the absence of understanding, memory, and reasoning.

## What to Observe When Running

Run `python chapters/01_ngrams/run.py` and notice:

1. **Bigram gets ~0%** — one character of context is useless
2. **Trigram gets ~14%** — two characters help slightly (e.g., guessing common bracket patterns)
3. **100% hallucination** for both — they always generate something, even for unanswerable questions
4. **0% abstention** — n-grams have no concept of "I don't know"
5. Look at the **sample generations** — they drift into repetitive patterns quickly

## What's Next

In **Chapter 02 (Feed-Forward LM)**, we replace counting with a neural network. A fixed-window MLP can learn more complex patterns within its window — but it's still limited by having no way to handle variable-length dependencies.
