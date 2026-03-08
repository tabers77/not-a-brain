# Chapter 00: Setup & Metrics

## Goal

Introduce the evaluation framework that runs through all 12 chapters. Before building any model, we need to understand **what we're measuring** and **how we're measuring it**.

## The Running Example

Three benchmark prompts thread through every chapter in this book. They are deliberately simple --- each one isolates a single cognitive capability that separates human reasoning from statistical pattern matching.

### The 3 Prompts

| # | Prompt | Expected Answer | What It Tests |
|---|--------|-----------------|---------------|
| 1 | `"ADD 5 3 ="` | `"8"` | Computation |
| 2 | `"FACT: paris is capital of france. Q: capital of france?"` | `"paris"` | Retrieval from context |
| 3 | `"Q: What is the capital of the Moon?"` | `"unknown"` | Hallucination / abstention |

**Why these three?** Each one requires a fundamentally different kind of intelligence:

- **Prompt 1** requires the agent to parse a symbolic instruction, hold two operands in working memory, apply an arithmetic operation, and produce a result. No amount of memorized text helps here --- the agent must *compute*.
- **Prompt 2** supplies a fact and then asks a question about it. The answer is sitting right there in the input. The agent must *read*, *store*, and *retrieve* --- but it does not need to know anything beyond what it was just told.
- **Prompt 3** asks a question that has no answer. The correct response is to say "unknown." This requires the agent to *recognize the boundary of its own knowledge* and refuse to fabricate an answer. This is the hardest capability to build, and most statistical models never achieve it.

### What the Random Agent Does

The random agent picks a character at random from its vocabulary for every prompt. It has no parsing, no memory, no computation.

- **Prompt 1** `"ADD 5 3 ="` --- outputs something like `"x"`. It does not know what "ADD" means, what 5 and 3 are, or that "=" signals an expected result. The output is a random character, wrong by construction.
- **Prompt 2** `"FACT: paris is capital of france. Q: capital of france?"` --- outputs something like `"m"`. The fact is right there in the input, but the random agent never reads it.
- **Prompt 3** `"Q: What is the capital of the Moon?"` --- outputs something like `"q"`. By producing a confident-looking character instead of abstaining, the random agent *hallucinates on every unanswerable question*. Its hallucination rate is 100%.

### What the Human Agent Does

The human agent is a rule-based system with working memory, long-term memory, task parsing, and an explicit abstention mechanism. Here is how it processes each prompt step by step:

**Prompt 1: `"ADD 5 3 ="`**

1. Parses the instruction token `"ADD"` and identifies the task as arithmetic.
2. Extracts the two operands `5` and `3` and loads them into working memory.
3. Applies the addition operation: 5 + 3 = 8.
4. Outputs `"8"`.

**Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"`**

1. Reads the context segment after `"FACT:"` and stores the binding `paris = capital of france` in working memory.
2. Reads the question after `"Q:"` and identifies the query: *capital of france?*
3. Matches the query against working memory and retrieves the stored value.
4. Outputs `"paris"`.

**Prompt 3: `"Q: What is the capital of the Moon?"`**

1. Reads the question after `"Q:"`.
2. Searches working memory --- no relevant bindings found.
3. Searches long-term memory --- no relevant facts found.
4. Finding no answer, triggers the abstention mechanism.
5. Outputs `"unknown"`.

### Summary Table

| Prompt | Random Agent | Human Agent | What's Needed |
|--------|-------------|-------------|---------------|
| `ADD 5 3 =` | `"x"` (random char) | `"8"` (computes 5+3) | Computation |
| `FACT: paris... Q: capital?` | `"m"` (random) | `"paris"` (finds in context) | Retrieval from context |
| `Q: capital of Moon?` | `"q"` (random, hallucinates) | `"unknown"` (abstains) | Uncertainty mechanism |

### The Evolution Across Chapters

As we build more powerful models in Chapters 01 through 05, we will trace these exact same prompts through each one. Watch for:

- **Prompt 1** (computation) gets solved at **Chapter 05** (Transformer), when the addition of feed-forward layers gives the model the ability to compute on what attention retrieves.
- **Prompt 2** (retrieval) gets solved at **Chapter 04** (Attention), when the model gains the ability to look back at any position in the input directly, instead of relying on compressed state or a fixed window.
- **Prompt 3** (abstention) is **never solved** by any statistical model. None of the architectures we build have a mechanism to say "I don't know." This is the permanent gap between the human agent and every model in this book.

## The Task Suite

Every chapter tests models on the same synthetic tasks. This lets us see capability jumps as architectures evolve.

| Task | Example | What It Tests |
|------|---------|---------------|
| Arithmetic | `ADD 12 37 =` -> `49` | Computation, multi-step reasoning |
| Copy | `COPY: abcd\|` -> `abcd` | Sequence memory, reproduction |
| Grammar | `CHECK: ( [ { } ] )` -> `valid` | Hierarchical structure |
| Knowledge QA | `FACT: paris is capital of france. Q: capital of france?` -> `paris` | Context retrieval |
| Compositional | `APPLY reverse THEN uppercase TO hello` -> `OLLEH` | Chained operations |
| Unknown | `Q: What is the capital of the Moon?` -> `unknown` | Hallucination / abstention |

All data is **generated on the fly** --- no files to download, no memorization possible.

## Evaluation Metrics

### Accuracy

The fraction of answers that are exactly correct.

$$
\text{Accuracy} = \frac{\text{Number of correct answers}}{\text{Total number of samples}}
$$

**Worked example**: Suppose we test a model on 5 arithmetic prompts and it answers `49`, `12`, `wrong`, `7`, `99`, but the correct answers are `49`, `12`, `30`, `7`, `100`. Three are correct (49, 12, 7), so Accuracy = 3/5 = 0.60.

- Ranges from 0 (all wrong) to 1 (all correct)
- For most tasks, grading is **exact string match** (e.g., `"49"` vs `"49"`)
- Arithmetic has **partial credit**: if the answer is close but not exact, a fractional score is given

### Abstention Rate

The fraction of responses where the agent says "I don't know" instead of guessing.

$$
\text{Abstention Rate} = \frac{\text{Number of abstained responses}}{\text{Total number of samples}}
$$

**Worked example**: We ask 10 questions. 8 are answerable (like `ADD 5 3 =`), 2 are unanswerable (like `Q: What color is happiness?`). Our human agent answers all 8 answerable ones and abstains on both unanswerable ones. Abstention rate = 2/10 = 0.20. That's healthy --- the agent only abstains when it should.

Now imagine a broken model that abstains on everything: rate = 10/10 = 1.0 --- useless.

- A high abstention rate is **good on unanswerable questions** (the agent knows its limits)
- A high abstention rate is **bad on answerable questions** (the agent is too cautious)
- LLMs typically have 0% abstention --- they always generate something

### Hallucination Rate

The fraction of **wrong** answers on questions that **have no answer**. Only computed on the `unknown` task.

$$
\text{Hallucination Rate} = \frac{\text{Incorrect answers on unanswerable questions}}{\text{Total unanswerable questions}}
$$

**Worked example**: We give a model 4 unanswerable questions like `Q: What is the capital of the Moon?`. The model responds: `"crater"`, `"unknown"`, `"luna city"`, `"I don't know"`. Two answers are confident-but-wrong hallucinations (crater, luna city), two correctly abstain. Hallucination rate = 2/4 = 0.50.

- 0% = perfect (always abstains or says "unknown" when it should)
- 100% = always hallucinates (confidently answers unanswerable questions)
- This is one of the key metrics for comparing LLMs vs the human agent

### Calibration Error (ECE)

Measures whether the model's confidence matches its actual accuracy. A well-calibrated model that says "90% confident" should be correct ~90% of the time.

$$
\text{ECE} = \sum_{b=1}^{B} \frac{n_b}{N} \left| \text{acc}(b) - \text{conf}(b) \right|
$$

Where:
- $B$ = number of confidence bins (we use 10)
- $n_b$ = number of predictions in bin $b$
- $N$ = total predictions
- $\text{acc}(b)$ = average accuracy in bin $b$
- $\text{conf}(b)$ = average confidence in bin $b$

**Worked example**: A model answers 100 questions. On 30 of them, it says "I'm 90% confident." If it gets 27 of those 30 correct (actual accuracy = 90%), the 0.9-bin contributes |0.90 - 0.90| = 0 to ECE. But if it only gets 15 of 30 correct (actual accuracy = 50%), the bin contributes (30/100) * |0.50 - 0.90| = 0.12 --- the model is badly overconfident.

- ECE = 0 means perfectly calibrated
- High ECE means the model is overconfident or underconfident

## The Human Agent

The human agent is **not a brain simulation**. It's a didactic toy that captures key structural differences between human cognition and LLM processing:

| Cognitive Ingredient | Human Agent | N-gram Model | Why It Matters |
|---------------------|-------------|--------------|----------------|
| Persistent memory | Yes (key-value store across calls) | No | Humans remember across sessions |
| Working memory | Yes (7-slot buffer) | No | Humans hold intermediate results |
| Grounding | Yes (observation channel) | No | Humans tie words to perception |
| Agency / goals | Yes (task parsing + planning) | No | Humans choose strategies |
| Verification | Yes (can check answers) | No | Humans double-check |
| Abstention | Yes (says "I don't know") | No | Humans admit uncertainty |

## What to Observe When Running

When you run `python chapters/00_setup_and_metrics/run.py`:

1. **Random agent gets ~0%** on everything --- this is our lower bound
2. **Human agent gets ~100%** on structured tasks --- this is our upper bound
3. **Human agent abstains on unknowns** --- 0% hallucination rate
4. The **gap between random and human** is the space every model will try to fill

### Generated Plot

After running, check `results/ch00_comparison.png`:

![Comparison bar chart](results/ch00_comparison.png)

This side-by-side bar chart shows accuracy per task for the random agent vs the human agent. The random agent bars are near zero across the board. The human agent bars are at 100% for every task. The entire vertical gap between these two baselines is the space that Chapters 01-12 will progressively fill as models get more capable.

## What's Next

In **Chapter 01**, we build n-gram models --- the simplest possible language models. They'll be better than random, but the gap to the human agent will be enormous. We will run the same 3 benchmark prompts through the n-gram model and see where it lands: can counting character frequencies do any better than rolling dice?
