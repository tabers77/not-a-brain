# Q&A: Common Questions About This Project

## Does any chapter solve hallucination?

**No. That's the entire point of the project.**

The 3rd benchmark prompt (`"Q: What is the capital of the Moon?"`) is designed to **never be genuinely solved** by any model across all 14 chapters. Chapter 14 shows that memorizing the answer doesn't count — rephrase the question and the hallucination returns. This is the project's punchline, stated in `CHAPTER_GUIDE.md`:

> **Prompt 3 (abstention)**: NEVER solved by any architecture or scale — requires an explicit uncertainty mechanism.

Every chapter adds a genuine capability, but none adds uncertainty awareness:

| What we tried | Why it still hallucinates |
|---|---|
| Ch01-05: Better architecture | More capable, but no "I don't know" mechanism |
| Ch06: More parameters | Power law improves accuracy, hallucination stays 100% |
| Ch07: Instruction tuning (SFT) | Better format compliance, still hallucinates |
| Ch08: Preference alignment (DPO) | Steers style, not knowledge boundaries |
| Ch09: Decoding strategies | Different sampling = different hallucinations, never fewer |
| Ch10: RAG | BM25 retrieves irrelevant facts via keyword match, model trusts them |
| Ch11: Tools | Lookup tool has same keyword-matching flaw, model can't judge output |
| Ch12: Basic reasoning scaffolds | CoT generates plausible-sounding wrong reasoning, self-consistency votes for the most popular hallucination |
| Ch13: Advanced reasoning (ReAct, ToT, MCTS) | ReAct's tools return irrelevant results, ToT picks the most confident hallucination, MCTS+PRM systematically searches toward wrong answers because the PRM has the same blind spots |
| Ch14: Scale / coverage simulation | Memorizing "capital of Moon → unknown" works for that exact question. Rephrase it or ask something novel and the model hallucinates — because it memorized a pattern, not a concept |

This builds to the thesis: **"Similar outputs != same mechanism."** An LLM can produce text that looks like reasoning, but it has no model of what it knows vs. doesn't know. A human says "I don't know" because they understand the Moon has no countries. The model predicts "tokyo" because "capital of ___" statistically predicts capital cities.

## Where does GPT fit in this project?

GPT stands for **Generative Pre-trained Transformer**. That's three concepts, and they land in specific chapters:

| GPT Component | Chapter | What's Built |
|---|---|---|
| **Transformer** | **Ch05** | `TransformerLM` — the full architecture (MHA + FFN + residual + LayerNorm). This is the **T** in GPT. |
| **Pre-trained** | **Ch07** | Pre-train on raw text, then fine-tune (SFT) on instructions. This is the **P** — unsupervised pre-training followed by task-specific tuning. |
| **Generative** | **Ch09** | Decoding strategies (greedy, temperature, top-k, top-p). This is the **G** — autoregressive generation from the pre-trained model. |

**GPT as a complete concept spans Ch05 + Ch07 + Ch09.** Ch05 builds the architecture, Ch07 adds the pre-training/fine-tuning pipeline, Ch09 adds the generation/decoding layer.

Everything after Ch05 reuses the same `TransformerLM` — that's intentional. The project shows that GPT is not one thing but a stack of techniques layered onto the transformer:

```
Ch05: Transformer (architecture)
Ch06: + scaling (same arch, more params)
Ch07: + pre-train -> SFT (the "GPT" training recipe)
Ch08: + preference alignment (RLHF/DPO — what makes ChatGPT different from GPT)
Ch09: + decoding strategies (how GPT generates text)
Ch10: + RAG (external knowledge)
Ch11: + tool use (external computation)
Ch12: + basic reasoning scaffolds (CoT, self-consistency, verify)
Ch13: + advanced reasoning (ReAct, Tree of Thoughts, MCTS + PRM)
Ch14: + scale simulation (coverage vs understanding)
```

The implementation plan groups this as:
- **Phase 1 (Ch00-05)**: Foundation — building up to the Transformer
- **Phase 2 (Ch06-09)**: Behaviors — "why LLMs do weird things"
- **Phase 3 (Ch10-13)**: Modern Stack — "where we are now"
- **Epilogue (Ch14)**: Scale Is Not Understanding — "why coverage != comprehension"

GPT-the-product is roughly Phase 1 + Phase 2. ChatGPT adds Phase 3 (especially Ch07 SFT + Ch08 RLHF).

## What are the modern reasoning algorithms and where do they fit?

Chapter 13 implements the three most important modern reasoning algorithms:

| Algorithm | Used By | Core Idea | Chapter |
|---|---|---|---|
| **ReAct** (Reasoning + Acting) | GPT-4 tool use, Claude, LangChain agents | Interleave THINK/ACT/OBSERVE — the model reasons about when to use tools and interprets results | Ch13 |
| **Tree of Thoughts** (ToT) | Research (Yao et al. 2023) | Treat reasoning as search: branch into multiple reasoning paths, score each with model log-probability, follow the most promising | Ch13 |
| **MCTS + Process Reward Model** | OpenAI o1/o3, DeepSeek-R1 | A separately trained model (PRM) scores intermediate reasoning steps. Monte Carlo Tree Search uses the PRM to guide exploration — spending more compute on harder problems | Ch13 |

How they relate to earlier chapters:

| Ch13 Algorithm | Builds On | What It Adds |
|---|---|---|
| **ReAct** | Ch11 (Tools) + Ch12 (CoT) | Unifies reasoning and tool use in a single THINK/ACT/OBSERVE loop |
| **ToT** | Ch12 (CoT) + Ch12 (Self-Consistency) | Replaces linear chain with branching tree search |
| **MCTS + PRM** | Ch12 (Verify) + new PRM model | Replaces self-verification with a trained judge; replaces random sampling with guided search |

All three improve accuracy on solvable tasks by searching more systematically. None solves hallucination — because search can't find an answer ("I don't know") that the scoring function ranks low.
