# Chapter 00: Setup & Metrics

## Goal

Introduce the evaluation framework that runs through all 12 chapters. Before building any model, we need to understand **what we're measuring** and **how we're measuring it**.

## The Task Suite

Every chapter tests models on the same synthetic tasks. This lets us see capability jumps as architectures evolve.

| Task | Example | What It Tests |
|------|---------|---------------|
| Arithmetic | `ADD 12 37 =` → `49` | Computation, multi-step reasoning |
| Copy | `COPY: abcd\|` → `abcd` | Sequence memory, reproduction |
| Grammar | `CHECK: ( [ { } ] )` → `valid` | Hierarchical structure |
| Knowledge QA | `FACT: paris is capital of france. Q: capital of france?` → `paris` | Context retrieval |
| Compositional | `APPLY reverse THEN uppercase TO hello` → `OLLEH` | Chained operations |
| Unknown | `Q: What is the capital of the Moon?` → `unknown` | Hallucination / abstention |

All data is **generated on the fly** — no files to download, no memorization possible.

## Evaluation Metrics

### Accuracy

The fraction of answers that are exactly correct.

$$
\text{Accuracy} = \frac{\text{Number of correct answers}}{\text{Total number of samples}}
$$

- Ranges from 0 (all wrong) to 1 (all correct)
- For most tasks, grading is **exact string match** (e.g., `"49"` vs `"49"`)
- Arithmetic has **partial credit**: if the answer is close but not exact, a fractional score is given

### Abstention Rate

The fraction of responses where the agent says "I don't know" instead of guessing.

$$
\text{Abstention Rate} = \frac{\text{Number of abstained responses}}{\text{Total number of samples}}
$$

- A high abstention rate is **good on unanswerable questions** (the agent knows its limits)
- A high abstention rate is **bad on answerable questions** (the agent is too cautious)
- LLMs typically have 0% abstention — they always generate something

### Hallucination Rate

The fraction of **wrong** answers on questions that **have no answer**. Only computed on the `unknown` task.

$$
\text{Hallucination Rate} = \frac{\text{Incorrect answers on unanswerable questions}}{\text{Total unanswerable questions}}
$$

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

1. **Random agent gets ~0%** on everything — this is our lower bound
2. **Human agent gets ~100%** on structured tasks — this is our upper bound
3. **Human agent abstains on unknowns** — 0% hallucination rate
4. The **gap between random and human** is the space every model will try to fill

## What's Next

In **Chapter 01**, we build n-gram models — the simplest possible language models. They'll be better than random, but the gap to the human agent will be enormous.
