# Implementation Plan: `not-a-brain`

## Goal

Build a progressive, educational repository that walks ML practitioners through the evolution of language models (n-grams to reasoning scaffolds) in 15 chapters (00-14), each with runnable code, tiny trainable models, and a side-by-side comparison with a toy human cognitive agent. Every chapter produces scripts, evals, and visual outputs — culminating in a dashboard that shows capability jumps across the entire timeline.

**Thesis**: "Similar outputs != same mechanism" — LLMs can look like they reason but the mechanism and failure modes are fundamentally different from human cognition.

## Non-Goals

- **Not building a production LLM** — models are intentionally tiny (~5K-30K params)
- **Not simulating a real brain** — the human-agent is a didactic toy, not neuroscience
- **Not chasing SOTA** — the point is understanding, not benchmarks
- **Not requiring GPUs** — everything runs on a laptop CPU in minutes
- **Not covering every LLM paper** — curated evolution, not exhaustive survey

## Stack

- Python 3.10+
- PyTorch (CPU-only is fine)
- Jupyter
- matplotlib, jinja2 (for dashboard)
- No external model downloads — everything trains from scratch on synthetic data

---

## Phasing Strategy

Three phases, each delivering a complete, usable repo:

| Phase | Chapters | Delivers |
|-------|----------|----------|
| **Phase 1: Foundation** | 00-05 | Core evolution from n-grams to Transformer. Shared infra, task suite, human-agent v1, dashboard skeleton |
| **Phase 2: Behaviors** | 06-09 | Hallucination, memory, RAG, decoding — the "why LLMs do weird things" chapters |
| **Phase 3: Modern Stack** | 10-12 | RAG, tools, reasoning scaffolds — "where we are now" |
| **Phase 4: Advanced & Capstone** | 13-14 | Advanced reasoning (ReAct, ToT, MCTS+PRM) and the "Scale Is Not Understanding" capstone |

Each phase is self-contained and publishable. Phase 1 alone is a strong repo. The shared infrastructure (task API, eval harness, human-agent, dashboard) is designed from day 1 to support all 15 chapters.

---

## Repo Structure

```
not-a-brain/
  pyproject.toml
  README.md
  requirements.txt

  src/
    not_a_brain/
      __init__.py
      tasks/                        # Shared task API
        base.py                     # TaskBase class: reset(), prompt(), grade()
        synthetic/
          __init__.py
          arithmetic.py             # "ADD 12 37 =" -> "49"
          copy_task.py              # "COPY: abcd|" -> "abcd"
          grammar.py                # balanced parens, tiny grammar
          knowledge_qa.py           # "FACT: X. Q: Y?" -> answer
          compositional.py          # "reverse then uppercase"
        mini_kb/
          facts.json                # ~200 short facts for RAG chapters
        gridworld/
          gridworld.py              # text-based grid for grounding experiments

      models/                       # Shared model building blocks
        __init__.py
        tokenizer.py                # CharTokenizer + simple BPE
        layers.py                   # Attention, FFN, TransformerBlock, positional enc
        tiny_gpt.py                 # Configurable tiny GPT (reused ch05+)
        decoding.py                 # greedy, temperature, top-k, top-p

      human_agent/                  # Toy cognitive architecture
        __init__.py
        agent.py                    # HumanAgent class
        memory.py                   # WorkingMemory + LongTermMemory
        planner.py                  # hypothesis -> test -> decide loop
        grounding.py                # observation channel

      evals/
        __init__.py
        harness.py                  # run agent on task suite, collect metrics
        metrics.py                  # accuracy, calibration, abstention rate, hallucination rate
        behaviors/
          memory.yaml
          grounding.yaml
          hallucination.yaml
          reasoning.yaml

      dashboard/
        __init__.py
        generate.py                 # collect results -> render HTML
        template.html               # jinja2 template
        plots.py                    # matplotlib chart generators

      utils/
        __init__.py
        training.py                 # tiny training loop (shared)
        logging.py                  # experiment logging
        visualization.py            # attention heatmaps, loss curves, etc.

  chapters/
    00_setup_and_metrics/
      README.md
      notebook.ipynb
      run.py
    01_ngrams/
      ...
    02_ffn_lm/
      ...
    03_rnn_gru/
      ...
    04_attention/
      ...
    05_transformer/
      ...
    06_scaling_laws_toy/
      ...
    07_instruction_tuning_toy/
      ...
    08_preference_and_rlhf_toy/
      ...
    09_decoding_and_hallucination/
      ...
    10_rag_minimal/
      ...
    11_tools_and_function_calls/
      ...
    12_reasoning_scaffolds/
      ...
    13_advanced_reasoning/
      ...
    14_scale_is_not_understanding/
      ...

  human_lens/
    cognitive_ingredients.md        # The 6-ingredient framework
    tasks_as_humans.md              # How humans approach each task

  docs/
    timeline.md
    glossary.md
    why_llms_arent_brains.md

  tests/
    test_tasks.py
    test_models.py
    test_human_agent.py
    test_evals.py
```

---

## Chapter Template

Every chapter follows this exact structure:

```
chapters/XX_name/
  chapter.md          # Theory, formulas (LaTeX), step-by-step logic, human lens, what to observe
  run.py              # Runnable script: trains, evaluates, prints results, saves plots
  results/            # Generated after running (metrics JSON, plots)
```

Design decisions:
- **No notebooks** — scripts are simpler, version-control friendly, and always show output when run
- **chapter.md** contains all the educational content: math formulas, step-by-step explanations, and the "human lens" comparison
- **run.py** prints step-by-step explanations alongside results so the output is self-documenting

---

## Shared Task API

```python
class TaskBase:
    def reset(self, seed=None): ...
    def prompt(self) -> str: ...
    def grade(self, answer: str) -> dict:  # {"correct": bool, "score": float, "expected": str}
    def metadata(self) -> dict: ...        # difficulty, category, etc.
```

Task types (all generate data on the fly):
- **Arithmetic**: `ADD 12 37 =` -> `49` (configurable digits)
- **Copy**: `COPY: abcd|` -> `abcd`
- **Grammar**: `( [ { } ] )` -> valid/invalid
- **KnowledgeQA**: `FACT: paris capital france. Q: capital of france?` -> `paris`
- **Compositional**: `reverse("hello")` -> `olleh`
- **Unknown**: questions with no answer in context (for hallucination/abstention testing)

---

## Human Cognitive Agent Design

```python
class HumanAgent:
    working_memory: WorkingMemory      # small, structured, in-task
    long_term_memory: LongTermMemory   # persistent key-value store
    uncertainty_threshold: float        # abstains if confidence < threshold

    def run(self, task_prompt, observations=None) -> AgentResult:
        # 1. Parse prompt, check long-term memory
        # 2. If grounding observations available, use as trusted input
        # 3. Generate candidate answers (rule-based per task type)
        # 4. Estimate confidence
        # 5. If below threshold -> abstain
        # 6. Store new facts in long-term memory
        return AgentResult(answer, confidence, trace, abstained)
```

The 6 cognitive ingredients scored per chapter:

1. **Persistent memory** (across sessions)
2. **Working memory** (in-task scratch space)
3. **Grounding** (ties to observation or external truth)
4. **Agency/goals** (chooses actions, not just completion)
5. **Verification** (can check and correct)
6. **Learning from interaction** (updates from feedback)

---

## Implementation Steps

### Phase 1: Foundation (Chapters 00-05)

#### Step 0: Project skeleton + shared infrastructure — DONE
- **What**: Create repo structure, package setup, shared modules
- **Files**: `pyproject.toml`, `requirements.txt`, `src/not_a_brain/__init__.py`, all `__init__.py` files
- **Test**: `python -c "import not_a_brain; print('OK')"` + `pytest tests/`

#### Step 1: Task API + Synthetic Data Generators — DONE
- **What**: Implement the shared task interface and all synthetic data generators
- **Files**: `src/not_a_brain/tasks/base.py`, `src/not_a_brain/tasks/synthetic/` (arithmetic, copy, grammar, knowledge_qa, compositional, unknown)
- **Test**: `pytest tests/test_tasks.py` — 18 tests passing

#### Step 2: Tokenizer — DONE
- **What**: Character-level tokenizer + simple BPE
- **Files**: `src/not_a_brain/models/tokenizer.py`
- **Test**: `pytest tests/test_models.py` — 9 tests passing (roundtrip, special tokens, batch encode, BPE merges)

#### Step 3: Shared training utilities — DONE
- **What**: Reusable training loop + generation + visualization helpers
- **Files**: `src/not_a_brain/utils/training.py` (train loop, autoregressive generate), `src/not_a_brain/utils/visualization.py` (loss curves, attention heatmaps, comparison bars, evolution curves)

#### Step 4: Eval harness + metrics — DONE
- **What**: Run any agent on any task suite, collect standardized metrics
- **Files**: `src/not_a_brain/evals/harness.py` (AgentInterface, RandomAgent, evaluate, run_eval_suite, save/load results), `src/not_a_brain/evals/metrics.py` (accuracy, abstention, hallucination, calibration ECE)
- **Test**: `pytest tests/test_evals.py` — 9 tests passing

#### Step 5: Human cognitive agent (v1) — DONE
- **What**: Toy agent with working memory (7 slots), long-term memory, uncertainty threshold, planning loop, grounding channel
- **Files**: `src/not_a_brain/human_agent/` (agent.py, memory.py, planner.py, grounding.py)
- **Test**: `pytest tests/test_human_agent.py` — 22 tests passing (solves all task types, abstains on unknowns, persistent memory)

#### Step 6: Dashboard skeleton — DONE
- **What**: HTML report generator with matplotlib plots (evolution curves, hallucination bars, cognitive heatmap, calibration, radar)
- **Files**: `src/not_a_brain/dashboard/` (generate.py, plots.py, template.html)

#### Step 7: Chapter 00 — Setup & Metrics — DONE
- **What**: `chapter.md` with metric definitions (LaTeX formulas for accuracy, abstention, hallucination, ECE) + `run.py` that runs random agent and human agent baselines
- **Files**: `chapters/00_setup_and_metrics/chapter.md`, `chapters/00_setup_and_metrics/run.py`
- **Results**: Random agent 2.7% accuracy, Human agent 100% accuracy. Gap = 97.3%

#### Step 8: Chapter 01 — N-grams — DONE
- **What**: Bigram/trigram models (pure counting, no neural net) + NgramAgent wrapper for eval harness
- **Files**: `src/not_a_brain/models/ngram.py`, `chapters/01_ngrams/chapter.md` (LaTeX formulas for bigram/trigram probabilities, backoff), `chapters/01_ngrams/run.py`
- **Test**: `pytest tests/test_ngram.py` — 7 tests passing
- **Results**: Bigram 0%, Trigram ~14%, Human 100%. Both n-grams hallucinate 100% on unknowns.

---

#### Step 9: Chapter 02 — Feed-Forward LM (MLP) — DONE
- **What**: Fixed-window MLP language model. First neural model. First use of the shared training loop.
- **Implements**: `FFNLM(context_window, vocab_size, d_hidden)` in `src/not_a_brain/models/ffn_lm.py`
- **Chapter**: `chapters/02_ffn_lm/chapter.md` (formulas for embedding, linear layers, softmax, cross-entropy loss) + `chapters/02_ffn_lm/run.py`
- **Test**: `pytest tests/test_ffn_lm.py` — 8 tests passing (output shape, batch forward, param count, training loss, generation, multi-layer, agent)
- **Results**: FFN LM 23% accuracy (14,655 params, W=8), Human 100%. KnowledgeQA 36%, everything else near 0%.
- **Human lens**: "Humans generalize rules; MLPs memorize windows"

---

#### Step 10: Chapter 03 — RNN/GRU — DONE
- **What**: Recurrent models. "State" creates illusion of memory.
- **Implements**: `RNNLM`, `GRULM`, `RNNAgent` in `src/not_a_brain/models/rnn_lm.py`
- **Chapter**: `chapters/03_rnn_gru/chapter.md` (LaTeX for RNN/GRU equations, vanishing gradient, gates) + `chapters/03_rnn_gru/run.py`
- **Test**: `pytest tests/test_rnn_lm.py` — 13 tests passing (shape, training, generation, GRU>RNN params, agent)
- **Results**: RNN 6.3% (11,647 params), GRU 6.0% (22,143 params), Human 100%. Grammar 32-38%, rest near 0%. GRU lower training loss (0.41 vs 0.53).
- **Human lens**: "Human working memory is structured; RNN state is compressed soup"
- **Constraint**: sequences < 50 tokens, models < 100k params

---

#### Step 11: Chapter 04 — Attention — DONE
- **What**: Scaled dot-product attention in isolation. Then multi-head. Attention heatmap visualization.
- **Implements**: `SingleHeadAttention`, `MultiHeadAttention`, `AttentionLM`, `AttentionAgent` in `src/not_a_brain/models/layers.py`
- **Chapter**: `chapters/04_attention/chapter.md` (LaTeX for scaled dot-product, causal mask, multi-head, positional encoding) + `chapters/04_attention/run.py` (with attention heatmap generation)
- **Test**: `pytest tests/test_attention.py` — 15 tests passing (causal mask, weights sum to 1, multi-head shapes, training, generation, heatmap access)
- **Results**: Attention LM 5.0% (17,423 params), grammar 26%, arithmetic 4% (first model to score >0 on arithmetic). Heatmaps show per-head attention patterns.
- **Human lens**: "Humans attend guided by goals; attention is content-based similarity"

---

#### Step 12: Chapter 05 — Transformer (Tiny GPT) — DONE
- **What**: Full tiny GPT. 2-4 layers. Core model for all subsequent chapters.
- **Implements**: `TransformerLM`, `TransformerAgent` in `src/not_a_brain/models/transformer.py`
- **Chapter**: `chapters/05_transformer/chapter.md` + `chapters/05_transformer/run.py`
- **Test**: `pytest tests/test_transformer.py` — 14 tests passing
- **Human lens**: "Depth + attention = powerful pattern matching. Still not reasoning."

---

### Phase 2: Behaviors (Chapters 06-09)

#### Step 13: Chapter 06 — Scaling Laws (Toy) — DONE
- **What**: Train same Transformer at 4 sizes (tiny/small/medium/large), plot loss vs params
- **Chapter**: `chapters/06_scaling_laws/chapter.md` + `chapters/06_scaling_laws/run.py`
- **Test**: `pytest tests/test_scaling.py` — 7 tests passing
- **Results**: Power law confirmed. Hallucination stays 100% at all scales.
- **Human lens**: "Humans don't scale by parameter count"

#### Step 14: Chapter 07 — Instruction Tuning (Toy SFT) — DONE
- **What**: Pre-train Transformer on raw text, then fine-tune (SFT) on instruction-formatted data. Compare base vs SFT vs from-scratch.
- **Implements**: `SFTAgent` in `chapters/07_instruction_tuning/run.py`, `INST: ... ANS: ...` format
- **Chapter**: `chapters/07_instruction_tuning/chapter.md` + `chapters/07_instruction_tuning/run.py`
- **Test**: `pytest tests/test_instruction_tuning.py` — 12 tests passing
- **Results**: SFT improves format compliance, pre-training transfers. Hallucination stays 100%.
- **Human lens**: "Humans follow instructions via shared intent, not format fine-tuning"

#### Step 15: Chapter 08 — Preference / RLHF (Toy DPO) — DONE
- **What**: Build preference pairs (chosen vs rejected), implement DPO loss, fine-tune SFT model. Three-phase pipeline: pre-train → SFT → DPO.
- **Implements**: `dpo_loss()`, `train_dpo()`, `build_preference_pairs()`, wrong-answer generators per task in `chapters/08_preference_and_rlhf/run.py`
- **Chapter**: `chapters/08_preference_and_rlhf/chapter.md` + `chapters/08_preference_and_rlhf/run.py`
- **Test**: `pytest tests/test_preference.py` — 10 tests passing
- **Results**: DPO loss starts at ~0.693 (log 2) and decreases. Accuracy may improve slightly. Hallucination stays 100%.
- **Human lens**: "Human values are grounded; RLHF is statistical preference matching"

#### Step 16: Chapter 09 — Decoding & Hallucination — DONE
- **What**: Implement greedy, temperature, top-k, top-p decoding in `src/not_a_brain/models/decoding.py`. Train SFT model, run all strategies on same prompts, measure hallucination rate and diversity.
- **Implements**: `decode()`, `score_sequence()`, `_apply_top_k()`, `_apply_top_p()`, `STRATEGIES` dict, `DecodingAgent` in run.py
- **Chapter**: `chapters/09_decoding_and_hallucination/chapter.md` + `run.py`
- **Test**: `pytest tests/test_decoding.py` — 15 tests passing
- **Results**: All strategies hallucinate at ~100%. Higher temperature = more diverse hallucinations, never fewer.
- **Human lens**: "Humans say 'I don't know'; LLMs generate plausible text"

---

### Phase 3: Modern Stack (Chapters 10-12)

#### Step 17: Chapter 10 — RAG Minimal — DONE
- **What**: BM25 retrieval over mini knowledge base, prepend to context. Implements `BM25Retriever` (with IDF, TF saturation, length normalization), `RAGAgent`, `SFTAgent`, `build_rag_corpus()`.
- **Implements**: `BM25Retriever`, `RAGAgent`, `SFTAgent`, corpus builders with `CONTEXT:` prefix in `chapters/10_rag_minimal/run.py`
- **Chapter**: `chapters/10_rag_minimal/chapter.md` + `run.py`
- **Tests**: `tests/test_rag.py` (24 tests — BM25 retriever, IDF, corpus builders, agents, sequences)
- **Key finding**: RAG improves knowledge QA but still hallucinates on unanswerable questions — BM25 retrieves by keyword match, not semantic relevance
- **Human lens**: "Humans seek evidence and reconcile contradictions; RAG only does step 1 (seek)"

#### Step 18: Chapter 11 — Tools & Function Calling — DONE
- **What**: Give tiny GPT a calculator and lookup tool via `CALL:tool(args) RESULT:value` format
- **Implements**: `ToolCallingAgent`, `tool_calc`, `tool_lookup` with BM25 in `chapters/11_tools_and_function_calls/run.py`
- **Chapter**: `chapters/11_tools_and_function_calls/chapter.md` + `run.py`
- **Test**: `pytest tests/test_tools.py`
- **Results**: Tools add external computation but still hallucinate on unknowns — keyword-match retrieval returns wrong context
- **Human lens**: "Humans naturally use tools; LLMs need explicit interfaces"

#### Step 19: Chapter 12 — Reasoning Scaffolds — DONE
- **What**: CoT (THINK: marker), self-consistency (majority vote over N temperature samples), verify-then-revise, and reversal curse probe
- **Implements**: `SFTAgent`, `CoTAgent`, `SelfConsistencyAgent`, `VerifyAgent` in `chapters/12_reasoning_scaffolds/run.py`
- **Chapter**: `chapters/12_reasoning_scaffolds/chapter.md` + `run.py`
- **Test**: `pytest tests/test_reasoning.py`
- **Results**: CoT helps solvable tasks, self-consistency reduces variance. All scaffolds hallucinate on unknowns. Reversal curse shows even Prompt 2 breaks when question direction is reversed.
- **Human lens**: "Humans also search and verify, but with richer world models"

#### Step 20: Chapter 13 — Advanced Reasoning — DONE
- **What**: ReAct (reasoning + tool use), Tree of Thoughts (branching search), MCTS + Process Reward Model (the o1/o3 approach)
- **Implements**: `ReActAgent`, `ToTAgent`, `MCTSAgent`, `ProcessRewardModel` in `chapters/13_advanced_reasoning/run.py`
- **Chapter**: `chapters/13_advanced_reasoning/chapter.md` + `run.py`
- **Test**: `pytest tests/test_advanced_reasoning.py`
- **Results**: All algorithms improve accuracy on solvable tasks. For Moon question, every search path leads to hallucination. PRM has the same blind spots as the generator.
- **Human lens**: "Human search is guided by understanding; model search is guided by probability"

#### Step 21: Chapter 14 — Scale Is Not Understanding — DONE
- **What**: Train models at 5 coverage levels (0-30 abstention patterns). Test on in-distribution, rephrased, and novel questions. Shows scale = coverage ≠ understanding.
- **Implements**: Coverage-level agents, rephrase/novel test suites in `chapters/14_scale_is_not_understanding/run.py`
- **Chapter**: `chapters/14_scale_is_not_understanding/chapter.md` + `run.py`
- **Test**: `pytest tests/test_scale.py`
- **Results**: In-distribution accuracy climbs with coverage. Rephrase accuracy lags. Novel accuracy stays near zero. Gap between Level 4 and Human on novel questions = the thesis.
- **Human lens**: "O(1) reasoning from principles vs O(N) memorization of patterns"

#### Step 22: Final dashboard + docs — DONE
- **What**: Full dashboard across all 15 chapters, polished docs, README
- **Files**: `documentation/`, `README.md`

---

## Dashboard Design

Generated via `python -m not_a_brain.dashboard.generate`, the HTML report includes:

1. **Evolution curve**: accuracy per task type across all chapters (line chart)
2. **Calibration plots**: confidence vs correctness per chapter (scatter)
3. **Hallucination rate**: bar chart across chapters on "unknown" questions
4. **Memory retention**: line chart showing cross-session memory scores
5. **Cognitive ingredients heatmap**: 6 ingredients x 13 rows (chapters + human), color-coded
6. **Per-chapter detail pages**: traces, attention visualizations, loss curves

---

## Verification Plan

1. `pytest tests/` — all unit tests pass (currently 329 passing)
2. `python -m not_a_brain.dashboard.generate` — produces valid HTML with all chapters
3. Each `chapters/XX/run.py` completes in < 5 minutes on CPU
4. Each `chapters/XX/chapter.md` has LaTeX formulas, step-by-step logic, and human lens
5. Dashboard shows clear capability progression across chapters

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| RNN/GRU chapter too slow on CPU | Keep sequences short (< 50 tokens), model tiny (< 100k params) |
| RLHF toy chapter too complex | Use DPO (simpler than PPO), or reduce to reward-weighted SFT |
| Human-agent feels fake/strawman | Be explicit in docs that it's didactic, not neuroscience. Focus on structural differences |
| Chapters take too long to build | Phase 1 (ch00-05) is priority and standalone-publishable |
| Scope creep per chapter | Strict template: 1 chapter.md, 1 run.py, bounded experiments |
