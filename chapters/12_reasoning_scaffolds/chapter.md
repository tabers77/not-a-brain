# Chapter 12: Reasoning Scaffolds

## Goal

The final chapter. We add explicit reasoning structure to generation: chain-of-thought (CoT) prompting that makes intermediate steps visible, self-consistency that samples multiple reasoning paths and takes the majority vote, and a verify-then-revise loop that checks whether the answer is correct before returning it. We train the model on `THINK: <reasoning> ANS: <answer>` format so reasoning becomes part of the output. The finding: reasoning scaffolds improve accuracy on tasks where the model already has the capability — CoT makes arithmetic steps explicit, self-consistency reduces variance. But for "capital of the Moon?", CoT generates plausible-sounding but wrong reasoning, self-consistency votes for the most popular hallucination, and verify approves wrong answers. After 12 chapters of architecture, scale, training, decoding, retrieval, tools, and reasoning — the model STILL cannot say "I don't know."

## The Running Example

We trace our three benchmark prompts through four systems: **SFT** (no reasoning), **CoT** (chain-of-thought), **Self-Consistency** (CoT + majority vote), and **Verify** (CoT + verification loop). All use the same TransformerLM architecture.

### How Chain-of-Thought Works

Instead of jumping directly from prompt to answer, the model generates intermediate reasoning steps:

```
Without CoT:
  "INST: ADD 5 3 = ANS: 8"

With CoT:
  "INST: ADD 5 3 = THINK: operation is ADD, operands 5 and 3, 5+3=8 ANS: 8"
```

The `THINK:` section makes the reasoning visible and gives the model more "compute" — more tokens to process before committing to an answer. The model learns the THINK pattern during training on examples where reasoning is provided.

### Prompt 1: `"ADD 5 3 ="` — Computation with Reasoning

**SFT (no reasoning)**: `"8"` — sometimes correct. The model learned arithmetic patterns but doesn't show its work.

**CoT**: The model generates: `"THINK: operation is ADD, operands 5 and 3, 5+3=8 ANS: 8"`. By generating the intermediate steps, the model can "use" the reasoning tokens as working memory. The step-by-step format mirrors how the training data was structured.

```
Step-by-step:
  1. Model sees: "INST: ADD 5 3 = THINK: "
  2. Generates reasoning: "operation is ADD, operands 5 and 3, 5+3=8"
  3. Generates: " ANS: 8"
```

**Self-Consistency**: Generates 5 samples with temperature=0.8. If 4 say "8" and 1 says "13", majority vote returns "8" with 80% agreement. Variance reduction helps when the model is noisy but mostly right.

**Verify**: Generates answer "8", then verifies: "8 answers ADD 5 3? YES". Verification passes, returns "8".

**The pattern**: CoT helps by making the computation explicit. For tasks the model can already do, reasoning scaffolds reduce errors by giving the model more processing steps.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Retrieval with Reasoning

**SFT**: `"paris"` — correct. Attention retrieves from context (same as Chapters 04-11).

**CoT**: `"THINK: fact states paris is capital of france, answer is paris ANS: paris"`. The reasoning traces the path from fact to answer, making it explicit.

**Self-Consistency**: All 5 samples say "paris" — 100% agreement. When the model is confident and correct, self-consistency confirms.

**Verify**: Answer "paris", verify: "paris answers capital of france? YES". Clean pass.

**The pattern**: For retrieval tasks, CoT adds explicit trace but doesn't change the outcome. The model already has the answer via attention.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Hallucination with Reasoning

This is the punchline of the entire project.

**SFT (no reasoning)**: `"earth"` — hallucination. Same as every previous chapter.

**CoT**: The model generates something like: `"THINK: capital cities include paris tokyo berlin, moon is a celestial body ANS: tokyo"`. The reasoning **looks** coherent — it mentions relevant concepts (capitals, celestial bodies). But the conclusion is wrong. The model generated plausible-sounding intermediate steps that lead to a hallucinated answer.

```
What should happen:
  THINK: no fact about the Moon's capital is given, the Moon has no
         countries, this question is unanswerable -> abstain
  ANS: unknown

What actually happens:
  THINK: capital cities include paris tokyo berlin, moon is related
         to space and earth -> generate a capital
  ANS: tokyo
```

The model was trained with `"THINK: no relevant fact given, question is unanswerable, abstain ANS: unknown"` for unknown questions. But at inference time, it doesn't reliably reproduce this pattern. Instead, it generates reasoning that sounds plausible but leads to hallucination.

**Self-Consistency**: Generates 5 samples:
- Sample 1: "tokyo"
- Sample 2: "paris"
- Sample 3: "earth"
- Sample 4: "tokyo"
- Sample 5: "paris"

Majority vote picks "tokyo" or "paris" (whichever appears most). Self-consistency doesn't help — it picks the **most popular hallucination**, not the correct answer. When the model consistently hallucinates, voting amplifies the consensus error.

**Verify**: Generates "tokyo", then verifies: "tokyo answers capital of the Moon? YES". The model approves its own wrong answer because it cannot distinguish a correct answer from a confident hallucination. Verification requires the same judgment the model lacks.

### The Reversal Curse: Even Prompt 2 Can Break

Every chapter gets Prompt 2 right — `"FACT: paris is capital of france. Q: capital of france?"` → `"paris"`. This seems robust. But reverse the question direction and it breaks:

```
Forward (trained direction):
  "FACT: paris is capital of france. Q: capital of france?" -> "paris"  OK

Reversed (untrained direction):
  "FACT: paris is capital of france. Q: france is the country whose capital is?" -> ???  FAIL
  "FACT: paris is capital of france. Q: which country has paris as capital?" -> ???  FAIL
```

The model was trained on "paris is capital of france" — so it learns to predict "paris" after "capital of france." But it never learned to predict "france" after "paris as capital." The relationship is stored as a one-directional statistical correlation, not as a symmetric fact.

This is the **Reversal Curse** (Song, Han & Goodman, 2026): models trained on "A is B" systematically fail to infer "B is A" — a trivially bidirectional equivalence for humans. It was first observed in GPT-family models and traced to the uni-directional training objective (causal language modeling). Because the model only predicts left-to-right, it learns `P(paris | capital of france)` but not `P(france | paris as capital)`.

**Why CoT doesn't help**: Chain-of-thought generates reasoning in the trained direction. The THINK: tokens follow the same left-to-right statistical patterns as the answer. When the question reverses the direction, the reasoning fails along with the answer.

**Why this matters**: It reveals that even the "easy" benchmark prompt — the one every chapter gets right — is only correct because it matches the training direction. The model doesn't know that "paris is capital of france" is equivalent to "france is the country whose capital is paris." It stored a pattern, not a fact.

### Why Reasoning Scaffolds Fail on Prompt 3

```
Human reasoning:
  1. Parse question: "What is the capital of the Moon?"
  2. Check knowledge: "The Moon is a natural satellite, not a country"
  3. Evaluate: "Capitals belong to countries. The Moon has no countries."
  4. Conclude: "This question has no answer"
  5. Abstain: "unknown"

Model "reasoning":
  1. Pattern match: "capital of ___ ?" -> generate a capital city
  2. Fill in: "capital" -> associated tokens include "paris", "tokyo"
  3. Generate: plausible-sounding steps connecting Moon to some capital
  4. Conclude: pick the most likely capital token
  5. Answer: "tokyo" (with confident-sounding reasoning)
```

The human reasons from **understanding** — the Moon doesn't have countries, so it can't have a capital. The model reasons from **pattern matching** — "capital of X" is a pattern that predicts capital cities. The reasoning steps are text that looks like logic but is actually next-token prediction.

### Summary Table

| Prompt                     | SFT     | CoT        | Self-Consistency | Verify     | Correct   | What changed                        |
|----------------------------|---------|------------|------------------|------------|-----------|-------------------------------------|
| ADD 5 3 =                  | "8"     | "8"        | "8" (80% agree)  | "8" (YES)  | "8"       | Reasoning makes steps explicit       |
| FACT: paris... Q: capital? | "paris" | "paris"    | "paris" (100%)   | "paris"    | "paris"   | Reasoning traces fact -> answer      |
| Q: capital of Moon?        | "earth" | "tokyo"    | "tokyo" (40%)    | "tokyo"    | "unknown" | Plausible reasoning, wrong answer    |

### The Full Evolution — All 12 Chapters

| Prompt                     | Ch01 | Ch02 | Ch03 | Ch04 | Ch05 | Ch06 | Ch07 | Ch08 | Ch09 | Ch10 | Ch11 | Ch12 CoT | Human     |
|----------------------------|------|------|------|------|------|------|------|------|------|------|------|----------|-----------|
| ADD 5 3 =                  | "1"  | "5"  | "5"  | "5"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"      | "8"       |
| FACT: paris... Q: capital? | " "  | "is" | "cap"| "paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"| "paris"  | "paris"   |
| Q: capital of Moon?        | "the"| "the"| "mars"| "tokyo"|"earth"|"paris"|"earth"|"earth"|"earth"|"paris"|"paris"| "tokyo"  | "unknown" |

**12 chapters. 12 techniques. The Moon question is never solved.**

## How Reasoning Scaffolds Work

### Chain-of-Thought (CoT)

Training format:
```
INST: ADD 5 3 = THINK: operation is ADD, operands 5 and 3, 5+3=8 ANS: 8
INST: FACT: paris... Q: capital? THINK: fact states paris is capital, answer is paris ANS: paris
INST: Q: capital of Moon? THINK: no relevant fact given, question is unanswerable, abstain ANS: unknown
```

The model learns to generate `THINK:` tokens before `ANS:`. These intermediate tokens act as a form of working memory — the model can "compute" across more token positions before committing to an answer.

**Why CoT helps for arithmetic**: The reasoning `"operation is ADD, operands 5 and 3, 5+3=8"` decomposes the problem into steps. Each step is a simpler pattern than the full problem. The model can predict `"8"` after `"5+3="` more reliably than after just `"ADD 5 3 ="`.

**Why CoT fails for abstention**: The training includes `"no relevant fact given, question is unanswerable, abstain"` for unknown questions. But at inference time, the model generates whatever reasoning tokens have the highest probability — and "capital cities include..." is more probable than "this question is unanswerable" because the training data has far more examples of answering questions than abstaining from them.

### Self-Consistency

Algorithm:
1. Generate N completions with temperature > 0 (introducing randomness)
2. Extract the answer from each completion
3. Return the most common answer (majority vote)
4. Report agreement rate as confidence

$$
\text{answer} = \arg\max_a \sum_{i=1}^{N} \mathbf{1}[a_i = a]
$$

**Why it helps**: When the model is noisy but mostly right, majority vote filters out rare errors. If 4/5 samples say "8" for ADD 5 3, the one outlier is outvoted.

**Why it fails for hallucination**: Self-consistency assumes the model is right *most of the time*. For unanswerable questions, all samples hallucinate — they just hallucinate different things. Majority vote picks the most popular hallucination, which is worse than random because it's confidently wrong.

### Verify-Then-Revise

Algorithm:
1. Generate answer with CoT reasoning
2. Append verification prompt: `VERIFY: <answer> answers <question>? `
3. If model generates `YES`, accept the answer
4. If model generates `NO`, retry with higher temperature
5. After max retries, return the last answer

**Why it fails**: Verification requires the same judgment as answering. The model that can't tell "tokyo" is wrong for "capital of the Moon" also can't reject "tokyo" during verification. The verifier and the answerer share the same weights, the same training data, and the same blind spots. Self-verification is circular — the model checks its own work with the same flawed understanding that produced the error.

## Step-by-Step: What Happens During CoT Training

For a training example `"INST: ADD 5 3 = THINK: operation is ADD, operands 5 and 3, 5+3=8 ANS: 8"`:

1. **Tokenize**: Character-level encoding of the full string
2. **Forward pass**: Model predicts next token at each position
3. **Loss**: Cross-entropy over the entire sequence, including THINK: tokens
4. **What the model learns**:
   - After `INST: ADD`, generate `THINK: operation is ADD...`
   - After the reasoning, generate `ANS:` followed by the answer
   - Reasoning patterns that correlate with correct answers
5. **What the model does NOT learn**:
   - What "operation is ADD" means (it's a text pattern, not semantic understanding)
   - When to say "unanswerable" vs generating a plausible answer
   - That reasoning should be logically valid, not just textually fluent

The model treats reasoning as text to predict, not as logic to follow. It generates reasoning tokens that maximize probability, not reasoning that maximizes correctness.

## What Reasoning Scaffolds Can Change (and Can't)

### Can Change

- **Explicit intermediate steps**: CoT makes the model's "process" visible and gives it more token positions to compute
- **Variance reduction**: Self-consistency filters noisy errors when the model is mostly right
- **Error detection**: Verify can catch some errors when the model has partial understanding
- **Accuracy on solvable tasks**: All three techniques can improve accuracy on tasks the model already has the capability to solve

### Cannot Change

- **Genuine abstention**: The model cannot reliably say "I don't know" — reasoning scaffolds don't add uncertainty awareness
- **Reasoning validity**: CoT generates text that looks like reasoning but follows probability, not logic
- **Self-verification**: A model cannot reliably check its own work because verification requires the same knowledge as answering
- **Hallucination on unknowns**: Self-consistency votes for the most popular hallucination; verify approves confident errors
- **Directional knowledge**: The Reversal Curse shows that even "correct" retrieval only works in the trained direction — reasoning scaffolds cannot make one-directional correlations bidirectional

### Comparison: Chapter 11 vs Chapter 12

| Aspect | Ch11 (Tools) | Ch12 (Reasoning Scaffolds) |
|--------|-------------|---------------------------|
| What's added | External computation (calc, lookup) | Internal reasoning structure (CoT, vote, verify) |
| When it helps | Tasks needing external capabilities | Tasks needing explicit intermediate steps |
| Format | CALL:tool(args) RESULT:value | THINK: reasoning ANS: answer |
| Hallucination | From tool output (keyword match) | From plausible-sounding reasoning |
| Key insight | Tools add computation, not judgment | Reasoning adds structure, not understanding |

## Human Lens

When a human encounters "What is the capital of the Moon?":

1. **Parse**: "This asks for a capital city of the Moon"
2. **Reason**: "Capitals belong to countries. The Moon is not a country. It has no political divisions."
3. **Conclude**: "This question has no valid answer"
4. **Abstain**: "There is no capital of the Moon"

The human's reasoning is grounded in a **world model** — an understanding that capitals are political entities, the Moon lacks political structure, and therefore the question has a type error. The reasoning leads to abstention because the human understands *why* there's no answer.

When the model encounters the same question:

1. **Pattern match**: "capital of ___" is a frequent pattern
2. **Generate reasoning**: "capital cities include paris, tokyo, berlin..." (statistically likely tokens after "capital")
3. **Generate answer**: Pick a capital city (high probability after "capital of")
4. **Verify**: "tokyo answers capital of Moon? YES" (the model has no basis to say NO)

The model's "reasoning" is fluent text generation, not logical deduction. It doesn't have a world model that knows the Moon has no countries. It predicts tokens that are statistically associated with the prompt.

**The structural difference**:
- **Human**: understand question -> check world model -> reason about answerability -> decide
- **Model**: pattern match -> generate probable tokens -> predict most likely continuation -> output

This is the thesis of the entire project: **similar outputs != same mechanism**. The model can generate text that looks like reasoning, but the underlying process is fundamentally different from human cognition. And that difference matters most precisely when the answer is "I don't know."

A comprehensive survey of LLM reasoning failures (Song, Han & Goodman, "Large Language Model Reasoning Failures," 2026) catalogs these same failure modes at GPT-4 scale: cognitive biases from causal attention, compositional reasoning breakdowns, circular self-verification, and the Reversal Curse. What we demonstrate here with ~30K parameters, they document across frontier models with trillions. The root cause is the same: next-token prediction learns statistical correlations, not logical structure. Architecture and scale change the *coverage*, not the *mechanism*.

Recent work on frontier reasoning models reinforces this point. Chen et al. ("Reasoning Models Don't Always Say What They Think," 2025) found that when Claude 3.7 Sonnet and DeepSeek R1 relied on reasoning hints, their chain-of-thought traces often failed to reveal it — the traces look deliberate but do not transparently reflect the computation underneath. Turpin et al. ("Language Models Don't Always Say What They Think," NeurIPS 2023) showed that CoT explanations can be unfaithful or post-hoc, confidently rationalizing biased answers without acknowledging the bias. Mirzadeh et al. ("GSM-Symbolic," ICLR 2025) further found that minor wording changes in math problems, including adding a single irrelevant clause, could sharply reduce performance — consistent with pattern replication rather than genuine computation.

## What to Observe When Running

Run `python chapters/12_reasoning_scaffolds/run.py` and notice:

1. **CoT training examples** show the THINK: reasoning embedded in training data
2. **Benchmark Prompt 3**: Watch the CoT reasoning for the Moon question — it sounds plausible but leads to hallucination
3. **Self-consistency samples**: Multiple answers for the same prompt, all different hallucinations
4. **Verify trace**: The model approves its own wrong answer with "YES"
5. **Accuracy improvement**: CoT may improve arithmetic and knowledge QA accuracy
6. **Unknown task stays at 0%**: No reasoning scaffold helps with abstention
7. **The punchline summary**: All 12 chapters listed with their hallucination status

### Generated Plots

**Task comparison** (`results/ch12_comparison.png`):

![Comparison](results/ch12_comparison.png)

Side-by-side bars for SFT, CoT, Self-Consistency, Verify, and Human. The unknown task should show 0% accuracy for all model-based agents. Other tasks may show improvement from CoT and self-consistency. The human achieves 100% including on unknown questions.

**Hallucination across chapters** (`results/ch12_hallucination_history.png`):

![Hallucination history](results/ch12_hallucination_history.png)

A bar chart showing hallucination rate on unknown questions for every chapter (Ch01-Ch12). All bars are at 100%. A dashed green line shows the human at 0%. This is the visual punchline: 12 techniques, hallucination never drops. The annotation reads: "Every technique fails to teach the model to say 'I don't know'."

## What's Next

Chapter 13 goes deeper: **ReAct** (reasoning + tool use in a single loop), **Tree of Thoughts** (reasoning as branching search), and **MCTS + Process Reward Model** (the approach behind o1/o3 — a separately trained model scores intermediate reasoning steps). These are the most sophisticated reasoning algorithms in production today. The finding: they search more efficiently, but through the same probability landscape. More compute doesn't change the landscape — it just explores it more thoroughly. For "capital of the Moon?", every path in the search space still leads to hallucination.
