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

**Worked example with a real training sentence**: Suppose our training corpus contains these three sequences:

```
"ADD 5 3 =8"
"ADD 2 7 =9"
"ADD 1 4 =5"
```

The bigram model counts every pair of consecutive characters. For the character `=`:
- `=` is followed by `8` once, `9` once, `5` once
- count(`=`) = 3 total
- P(`8` | `=`) = 1/3, P(`9` | `=`) = 1/3, P(`5` | `=`) = 1/3

Now when we prompt with `"ADD 6 2 ="`, the model looks at **only the `=`** and picks one of {8, 9, 5} — it has no idea the answer should be 8, because it never looks at the `6` and `2`.

Similarly, after `A`: `A` is always followed by `D` in these examples, so P(`D`|`A`) = 1.0. The model "knows" `A→D` but only because it memorized a pattern, not because it understands anything.

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

**Worked example**: Given the same three training sentences, the trigram counts pairs of two characters:
- The pair `" ="` (space then equals) is followed by `8`, `9`, `5` — same problem, still random.
- But the pair `"=8"` at the end is always followed by EOS (end of sequence) — so the trigram learns to stop after one digit.
- The pair `"D "` (D then space) is always followed by a digit — the trigram learns "after `D `, expect a number."

Two characters of context helps with local patterns (like "stop after the answer digit"), but `" ="` still doesn't know which digit to produce — the operands are 6+ characters back, far outside the 2-character window.

### Fallback (Backoff)

When the trigram encounters a pair it hasn't seen before, it **falls back** to the bigram:

$$
P(c_t \mid c_{t-2}, c_{t-1}) =
\begin{cases}
\frac{\text{count}(c_{t-2}, c_{t-1}, c_t)}{\text{count}(c_{t-2}, c_{t-1})} & \text{if pair seen} \\[6pt]
P(c_t \mid c_{t-1}) & \text{otherwise (bigram fallback)}
\end{cases}
$$

**Worked example**: Suppose we encounter the context `"x="` at generation time, but our training data never had `x` before `=`. The trigram has no counts for the pair `(x, =)`, so it falls back: instead of P(next | `x`, `=`), it uses P(next | `=`) from the bigram. This prevents the model from getting completely stuck on unseen contexts, but the fallback knows even less about what should come next.

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

Let's trace the bigram generating from the prompt `"COPY: hi|"`:

```
Step 0: Prompt = "COPY: hi|"
        Model looks at last char: "|"
        In training, "|" was followed by many different chars (a, b, c, h, x...)
        Picks most frequent: say "a"       →  "COPY: hi|a"

Step 1: Last char = "a"
        In training, "a" was often followed by "b", "l", "n"...
        Picks most frequent: say "l"       →  "COPY: hi|al"

Step 2: Last char = "l"
        Picks: "i"                         →  "COPY: hi|ali"
        ...and so on, drifting away from "hi" (the correct answer)
```

The model generated `"ali..."` instead of `"hi"` because it never looks back at what came before `"|"`. Each step only sees the single previous character — the prompt content is invisible.

For the trigram, replace "last char" with "last two chars." It would see `"|a"` at step 1 instead of just `"a"`, which helps slightly but still can't reach back to `"hi"`.

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

### Generated Plot

After running, check `results/ch01_comparison.png`:

![Comparison bar chart](results/ch01_comparison.png)

This chart shows accuracy per task for bigram, trigram, and human agent. The n-gram bars are nearly flat at zero across all tasks, with the trigram showing a small bump on grammar (it can guess common bracket patterns by chance). The human agent towers at 100% on every task. The massive gap visualizes what's missing: understanding, memory, and reasoning — none of which counting co-occurrences can provide.

## What's Next

In **Chapter 02 (Feed-Forward LM)**, we replace counting with a neural network. A fixed-window MLP can learn more complex patterns within its window — but it's still limited by having no way to handle variable-length dependencies.
