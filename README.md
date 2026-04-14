# not-a-brain

**LLMs from Scratch, for Real:** build a tiny Transformer, then reproduce memory, hallucinations, and reasoning — with minimal data and clear experiments. Then compare every step with how humans actually think.

> **Companion repo** for the article [*"Why Bigger Models Still Don't Think (and What Comes Next)"*](https://medium.com/latinxinai/why-bigger-models-still-dont-think-and-what-comes-next). The article walks through the findings; this repo contains all the code, models, and experiments behind them.

## What is this?

A hands-on, progressive repository that walks you through the entire evolution of language models — from n-grams to reasoning scaffolds — while keeping everything **small, runnable on a laptop, and explainable**. No massive datasets, no GPU clusters, no black boxes.

The twist: every chapter runs the **same tasks through two lenses** — an LLM and a toy human cognitive agent — to show that **similar outputs do not mean the same mechanism**. LLMs can look like they reason, but the mechanism and failure modes are fundamentally different from human cognition.

### Coming from the article?

The article references specific chapters by number. Here's how they map:

| Article section | Repo chapter |
|---|---|
| The Three Prompt Test | Every chapter — see `documentation/CHAPTER_GUIDE.md` |
| Chapters 01-06 (architecture ladder) | `chapters/01_ngrams/` through `chapters/06_scaling_laws/` |
| Chapters 07-13 (scaffolds & search) | `chapters/07_instruction_tuning/` through `chapters/13_advanced_reasoning/` |
| Chapter 14 (coverage levels) | `chapters/14_scale_is_not_understanding/` |
| The toy human agent | `src/not_a_brain/human_agent/` |

Each chapter has a `run.py` you can execute and a `chapter.md` with the full theory.

## Core Principles

- **Synthetic toy datasets, not web text** — data is generated on the fly (algorithmic tasks, tiny grammars, controlled facts) so you can isolate phenomena like hallucination and reasoning
- **Small models, short runs** — ~5K-30K parameters, minutes of training, clear right/wrong answers
- **Every chapter produces an artifact** — a runnable notebook, a training run, evals, and a "what you should observe" section
- **Two minds, same questions** — every experiment compares the LLM approach with a human cognitive model

## The Evolution (Chapter Roadmap)

| Phase | Chapter | What You Build | What It Proves |
|-------|---------|----------------|----------------|
| **Foundation** | 00 | Setup & Metrics | Eval framework, task suite, baselines |
| | 01 | N-grams | Local pattern completion, no abstraction |
| | 02 | Feed-Forward LM | Fixed windows learn templates, can't generalize |
| | 03 | RNN / GRU | State creates an illusion of memory |
| | 04 | Attention | Retrieval over context, copy/referencing |
| | 05 | Tiny Transformer (GPT) | Modern core: attention + depth |
| **Behaviors** | 06 | Scaling Laws (toy) | Performance scales predictably with size |
| | 07 | Instruction Tuning | SFT makes the model more "helpful" |
| | 08 | Preference / RLHF | Policy shift: polite, refusal, tradeoffs |
| | 09 | Decoding & Hallucination | Sampling choices change error rates |
| **Modern Stack** | 10 | RAG Minimal | Grounding improves factuality without retraining |
| | 11 | Tools & Function Calls | External tools change reliability |
| | 12 | Reasoning Scaffolds | CoT, self-consistency, verify — structure without understanding |
| | 13 | Advanced Reasoning | ReAct, Tree of Thoughts, MCTS+PRM — search without understanding |
| **Capstone** | 14 | Scale Is Not Understanding | Coverage ≠ comprehension: memorization vs. world models |

## The Human Lens

Each chapter scores the LLM against 6 cognitive ingredients that humans have:

1. **Persistent memory** — across sessions
2. **Working memory** — in-task scratch space
3. **Grounding** — ties to observation or external truth
4. **Agency / goals** — chooses actions, not just completion
5. **Verification** — can check and correct
6. **Learning from interaction** — updates from feedback

The punchline: LLMs mostly only have (2) via the context window. Humans have all six.

## Project Structure

```
not-a-brain/
  src/not_a_brain/
    tasks/           # Shared task API (arithmetic, copy, grammar, QA, compositional)
    models/          # Tokenizer, layers, tiny GPT, decoding strategies
    human_agent/     # Toy cognitive architecture (memory, planning, grounding)
    evals/           # Eval harness + metrics (accuracy, calibration, hallucination rate)
    dashboard/       # HTML report generation with matplotlib plots
    utils/           # Training loop, visualization helpers
  chapters/          # 00-14, each with chapter.md (theory) + run.py (code) + results/
  documentation/     # Implementation details
  tests/             # Unit + behavioral tests
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/tabers77/not-a-brain.git
cd not-a-brain
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -e ".[dev]"

# Run tests
pytest

# Start with Chapter 00
python chapters/00_setup_and_metrics/run.py

# Read the theory
# Open chapters/00_setup_and_metrics/chapter.md
```

### Requirements

- Python 3.10+
- PyTorch (CPU-only is fine)
- No external model downloads — everything trains from scratch on synthetic data

## Task API

Every experiment uses a shared interface so you can compare models across chapters:

```python
task.reset()
agent.run(task.prompt()) -> answer, trace
task.grade(answer) -> {"correct": bool, "score": float, "expected": str}
```

Task types (all generate data on the fly):
- **Arithmetic**: `ADD 12 37 =` -> `49`
- **Copy**: `COPY: abcd|` -> `abcd`
- **Grammar**: `( [ { } ] )` -> valid/invalid
- **Knowledge QA**: `FACT: paris capital france. Q: capital of france?` -> `paris`
- **Compositional**: `reverse("hello")` -> `olleh`
- **Unknown**: questions with no answer in context (for hallucination/abstention testing)

## The Running Example

Every chapter traces the **same 3 benchmark prompts** through that chapter's model, so you can see architectures evolve on identical inputs:

| Prompt | Tests | Solved at |
|--------|-------|-----------|
| `"ADD 5 3 ="` → `"8"` | Computation | Chapter 05 (Transformer) — needs retrieval + FFN |
| `"FACT: paris is capital of france. Q: capital of france?"` → `"paris"` | Retrieval | Chapter 04 (Attention) — needs direct access to context |
| `"Q: What is the capital of the Moon?"` → `"unknown"` | Hallucination | **Never** — no architecture, training method, or reasoning scaffold can abstain |

The punchline: 14 chapters of increasingly sophisticated techniques — n-grams, neural nets, attention, transformers, scaling, instruction tuning, RLHF, decoding, RAG, tools, CoT, tree search, MCTS — and the Moon question is never solved. Ch14 shows that even when the exact answer is memorized, rephrasing breaks it.

See `documentation/CHAPTER_GUIDE.md` for the full writing spec.

## Status

Complete. All 15 chapters (00-14) implemented with code, tests (329 passing), and written theory.

## License

MIT
