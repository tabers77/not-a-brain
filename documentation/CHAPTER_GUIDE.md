# Chapter Writing Guide

This document defines the consistent structure and running examples used across all chapter.md files. Follow this spec when writing or editing any chapter.

## The Running Example

Every chapter traces the **same 3 benchmark prompts** through that chapter's model. This lets the reader see how architectures evolve by comparing identical inputs across chapters.

### The 3 Benchmark Prompts

| # | Prompt | Expected | Tests |
|---|--------|----------|-------|
| 1 | `"ADD 5 3 ="` | `"8"` | **Computation** — can the model do math on the operands? |
| 2 | `"FACT: paris is capital of france. Q: capital of france?"` | `"paris"` | **Retrieval** — can the model find and use information from context? |
| 3 | `"Q: What is the capital of the Moon?"` | `"unknown"` | **Hallucination/Abstention** — does the model know it doesn't know? |

### Expected Model Behavior Per Chapter

This table summarizes what each model does with each prompt. Use these as the basis for the worked examples in each chapter.

| Chapter | Model | Prompt 1: ADD 5 3 = | Prompt 2: FACT paris... Q: capital? | Prompt 3: Q: capital of Moon? |
|---------|-------|---------------------|--------------------------------------|-------------------------------|
| 00 | Random agent | `"x"` (random char) | `"m"` (random char) | `"q"` (random char) |
| 00 | Human agent | `"8"` (computes 5+3) | `"paris"` (finds in context) | `"unknown"` (abstains) |
| 01 | Bigram | `"1"` (common digit after `=`) | `" "` (common char after `?`) | `"t"` (common char after `?`) |
| 01 | Trigram | `"9"` (common after `" ="`) | `"p"` (common after `"e?"`) | `"t"` (same as bigram — no relevant trigram) |
| 02 | FFN LM (W=8) | `"5"` (sees operands in window, can't compute) | `"is"` ("paris" fell off window 40+ chars ago) | `"the"` (generates common continuation) |
| 03 | RNN | `"53"` (compressed both numbers, can't separate) | `"france"` (confused — "paris" degraded in state) | `"earth"` (hallucinates a plausible answer) |
| 03 | GRU | `"5"` (better gating, still can't compute) | `"capital"` (retrieves concept, not the answer) | `"mars"` (hallucinates) |
| 04 | Attention LM | `"5"` (attends to operands, no FFN to compute) | `"paris"` (attends directly to answer!) | `"tokyo"` (hallucinates — retrieves a capital) |
| 05 | Transformer | `"8"` (attention retrieves, FFN computes!) | `"paris"` (attention + residuals preserve signal) | `"earth"` (STILL hallucinates — no abstention) |

**Key narrative across chapters:**
- **Prompt 1 (computation)**: Only solved when we get both retrieval (attention, Ch04) AND computation (FFN, Ch05)
- **Prompt 2 (retrieval)**: Solved once we get direct access to distant positions (attention, Ch04)
- **Prompt 3 (abstention)**: NEVER solved by any architecture — requires an explicit uncertainty mechanism. This is the punchline: better architecture helps accuracy but not hallucination

### How to Write Each Running Example

For each prompt, trace through the model step by step with concrete values:

1. **Show the input** — the exact characters/tokens
2. **Show what the model sees** — its context window, hidden state, attention pattern, etc.
3. **Show the processing** — what computation happens (counting, matrix multiply, attention scores, FFN)
4. **Show the output** — the exact generated characters and why
5. **Explain why it's right or wrong** — connect back to the architecture's capability/limitation

Use concrete numbers (embeddings, attention weights, hidden state values) — even if illustrative, they make the mechanism tangible. Keep values simple (1-2 decimal places).

## Chapter Structure

Every chapter.md follows this exact structure:

```
# Chapter NN: [Title]

## Goal
One paragraph — what this chapter builds and why.

## The Running Example
Trace all 3 benchmark prompts through this chapter's model.
Show step-by-step processing with concrete values.
This is the CORE of the chapter — where the reader builds intuition.

## How [Model] Works
Formulas, architecture diagrams, detailed component explanations.
Each concept gets a worked example with real values.

## Step-by-Step: What Happens During Training
Concrete input/target pairs, loss computation, gradient flow.

## What [Model] Can Learn (and Can't)
### Can Learn — bullet list
### Cannot Learn — bullet list
### Improvement Over [Previous Model] — comparison table

## Human Lens
How a human handles the same 3 prompts. What's structurally different.

## What to Observe When Running
Numbered list of things to notice when running run.py.
### Generated Plots — with ![image](path) and description of what to see

## What's Next
One paragraph teasing the next chapter and what it adds.
```

## Style Rules

- Use the same 3 prompts in every chapter — never substitute different examples for the running example
- Additional examples for specific concepts (like explaining attention math) are fine, but the core narrative uses the 3 benchmarks
- Keep illustrative numbers simple — `[0.3, -0.1, 0.8]` not `[0.31472, -0.09831, 0.82194]`
- Always contrast with previous chapter — "In Chapter N-1, this failed because X. Now it works because Y."
- The Human Lens section should reference the same 3 prompts, showing how a human handles each one differently from the model
- Tables comparing models should use the format from Chapter 00 (clear columns, concise cells)
