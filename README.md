# not-a-brain

**LLMs from Scratch, for Real:** build a tiny Transformer, then reproduce memory, hallucinations, and reasoning — with minimal data and clear experiments. Then compare every step with how humans actually think.

## What is this?

A hands-on, progressive repository that walks you through the entire evolution of language models — from n-grams to reasoning scaffolds — while keeping everything **small, runnable on a laptop, and explainable**. No massive datasets, no GPU clusters, no black boxes.

The twist: every chapter runs the **same tasks through two lenses** — an LLM and a toy human cognitive agent — to show that **similar outputs do not mean the same mechanism**. LLMs can look like they reason, but the mechanism and failure modes are fundamentally different from human cognition.

## Core Principles

- **Synthetic toy datasets, not web text** — data is generated on the fly (algorithmic tasks, tiny grammars, controlled facts) so you can isolate phenomena like hallucination and reasoning
- **Small models, short runs** — 50k-500k parameters, minutes of training, clear right/wrong answers
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
| | 12 | Reasoning Scaffolds | "Reasoning" as an algorithm around the model |

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
  chapters/          # 00-12, each with chapter.md (theory) + run.py (code) + results/
  documentation/     # Implementation details
  tests/             # Unit + behavioral tests
```

## Quick Start

```bash
# Clone and set up
git clone https://github.com/your-username/not-a-brain.git
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

## Status

This project is under active development. Phase 1 (Chapters 00-05) is the current focus.

## License

MIT
