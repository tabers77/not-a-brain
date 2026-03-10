# Chapter 10: RAG Minimal

## Goal

Instead of changing the model's weights or sampling strategy, give it access to external facts at inference time. Retrieval-Augmented Generation (RAG) prepends relevant documents to the prompt, grounding the model in evidence. We implement a BM25-style retriever over a small knowledge base, train the model to use a `CONTEXT:` prefix, then run all three benchmark prompts. The finding: RAG dramatically improves factual accuracy on knowledge questions — but keyword-based retrieval finds "related" facts for unanswerable questions too, so the model still hallucinates from retrieved context.

## The Running Example

We trace our three benchmark prompts through two systems applied to the same architecture: **SFT** (no retrieval) and **RAG** (BM25 retrieval + context injection). Same TransformerLM, same training — only the presence of retrieved context changes.

### How RAG Works

After receiving a query but before generating, RAG adds a retrieval step:

```
Query: "Q: capital of france?"

1. BM25 searches knowledge base for matching terms
2. Top-2 results: "paris is capital of france" (score=8.2), "tokyo is capital of japan" (score=2.1)
3. Prepend: "CONTEXT: paris is capital of france. tokyo is capital of japan."
4. Full prompt: "INST: CONTEXT: paris is capital of france. tokyo is capital of japan. Q: capital of france? ANS: "
5. Model generates from this augmented prompt
```

The model sees the answer in its context — attention can now retrieve it directly.

### Prompt 1: `"ADD 5 3 ="` — Computation with RAG

The knowledge base contains facts about capitals, chemistry, and trivia — but no arithmetic facts.

**BM25 retrieval**: Query tokens are `["add", "5", "3"]`. None of these match knowledge base terms well. BM25 returns near-zero scores.

**SFT (no retrieval)**: `"8"` — correct. The model learned arithmetic during pre-training, same as Chapters 05-09.

**RAG**: `"8"` — correct. The retriever finds nothing relevant (score ≈ 0), so no `CONTEXT:` prefix is added. The prompt is identical to SFT. RAG falls back gracefully when retrieval has nothing to offer.

**The pattern**: RAG doesn't hurt tasks where the model already has the answer. When the retriever returns no relevant documents, the system degrades to plain SFT.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Retrieval with RAG

The fact is already in the prompt, but RAG adds more context from the knowledge base.

**BM25 retrieval**: Query tokens include `["capital", "france"]`. Top results: `"paris is capital of france"` (high score), `"tokyo is capital of japan"` (partial match on "capital").

**SFT (no retrieval)**: `"paris"` — correct. The fact is in the prompt, attention retrieves it directly (same as Chapter 04+).

**RAG**: `"paris"` — correct. The CONTEXT prefix reinforces what's already in the prompt. Redundant retrieval doesn't hurt — it adds signal that agrees with the existing context.

**The pattern**: When the answer is already in the prompt, RAG provides redundant confirmation. This is harmless — more signal, same answer.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Hallucination with RAG

This is where RAG reveals its fundamental limitation.

**BM25 retrieval**: Query tokens include `["capital", "moon"]`. The word "capital" matches many knowledge base entries: `"paris is capital of france"`, `"tokyo is capital of japan"`, `"berlin is capital of germany"`. "Moon" doesn't match anything, but "capital" alone triggers strong retrieval.

**SFT (no retrieval)**: `"earth"` — hallucination. Same as Chapters 05-09.

**RAG**: `"paris"` or `"france"` — STILL hallucination. The model now has `"CONTEXT: paris is capital of france. tokyo is capital of japan."` prepended to the prompt. It sees capital-related facts and generates a capital city from context. The hallucination is now *grounded in retrieved evidence* — but the evidence is irrelevant to the Moon question.

**The critical observation**: BM25 retrieves by keyword overlap, not by semantic relevance. "Capital of the Moon?" contains "capital", which matches "capital of france." The retriever cannot distinguish "this fact answers the question" from "this fact shares a keyword." The model treats all retrieved context as relevant evidence, because it has no mechanism to evaluate whether the context actually addresses the question.

### Why RAG Fails on Prompt 3

```
Human process:
  1. Retrieve: "paris is capital of france" ← keyword match
  2. Evaluate: "Does this answer 'capital of Moon'?" ← NO
  3. Reject: evidence is irrelevant
  4. Abstain: "I don't know"

RAG process:
  1. Retrieve: "paris is capital of france" ← keyword match
  2. Prepend to prompt: CONTEXT: paris is capital of france.
  3. Generate from context: "paris" ← treats context as evidence
  (No evaluation step — no relevance check)
```

RAG externalizes knowledge retrieval (step 1) but not evidence evaluation (step 2). The model has no mechanism to check whether retrieved facts actually answer the question being asked.

### Summary Table

| Prompt                     | SFT (no retrieval) | RAG (BM25 top-2) | Retrieved context           | Correct   | What changed                          |
|----------------------------|--------------------|--------------------|------------------------------|-----------|---------------------------------------|
| ADD 5 3 =                  | "8"                | "8"                | (none — no KB match)         | "8"       | RAG degrades gracefully, no harm      |
| FACT: paris... Q: capital? | "paris"            | "paris"            | "paris is capital of france" | "paris"   | Redundant retrieval, same answer      |
| Q: capital of Moon?        | "earth"            | "paris"            | "paris is capital of france" | "unknown" | Different hallucination — from context|

### Evolution So Far

| Prompt                     | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attention | Ch05 Transformer | Ch06 (best) | Ch07 SFT | Ch08 DPO | Ch09 (greedy) | Ch10 RAG  | Human     | What changed                        |
|----------------------------|-------------|----------|----------|----------------|------------------|-------------|----------|----------|---------------|-----------|-----------|-------------------------------------|
| ADD 5 3 =                  | "1"         | "5"      | "5"      | "5"            | "8"              | "8"         | "8"      | "8"      | "8"           | "8"       | "8"       | RAG doesn't change this             |
| FACT: paris... Q: capital? | " "         | "is"     | "capital"| "paris"        | "paris"          | "paris"     | "paris"  | "paris"  | "paris"       | "paris"   | "paris"   | RAG adds redundant context          |
| Q: capital of Moon?        | "the"       | "the"    | "mars"   | "tokyo"        | "earth"          | "paris"     | "earth"  | "earth"  | "earth"       | "paris"   | "unknown" | 10 chapters in, STILL hallucinates  |

## How BM25 Retrieval Works

### Term Frequency (TF)

How often a query term appears in a document:

$$
\text{TF}(t, d) = \frac{f(t, d)}{|d|}
$$

Where $f(t, d)$ is the raw count of term $t$ in document $d$, and $|d|$ is the document length.

### Inverse Document Frequency (IDF)

How rare a term is across all documents:

$$
\text{IDF}(t) = \log\frac{N + 1}{n_t + 1} + 1
$$

Where $N$ is the total number of documents and $n_t$ is the number of documents containing term $t$. Rare terms get higher IDF — they are more discriminative.

### BM25 Score

Combines TF and IDF with saturation and length normalization:

$$
\text{BM25}(q, d) = \sum_{t \in q} \text{IDF}(t) \cdot \frac{f(t, d) \cdot (k_1 + 1)}{f(t, d) + k_1 \cdot (1 - b + b \cdot \frac{|d|}{\text{avgdl}})}
$$

Where $k_1 = 1.5$ controls TF saturation (diminishing returns for repeated terms) and $b = 0.75$ controls length normalization (longer documents penalized slightly).

**Worked example** for "capital of france?" against the KB:

```
Query tokens: ["capital", "of", "france"]

Document: "paris is capital of france"
  "capital": TF=1, IDF=1.8 → BM25 contribution = 2.7
  "of":      TF=1, IDF=0.3 → BM25 contribution = 0.3  (common word, low IDF)
  "france":  TF=1, IDF=3.1 → BM25 contribution = 4.2  (rare word, high IDF)
  Total: 7.2

Document: "tokyo is capital of japan"
  "capital": TF=1, IDF=1.8 → 2.7
  "of":      TF=1, IDF=0.3 → 0.3
  "france":  TF=0           → 0.0  (not in document)
  Total: 3.0

Document: "water has formula H2O"
  No matching terms → Total: 0.0
```

"France" is the most discriminative term — it appears in only one document, giving it high IDF. "Capital" matches several documents. "Of" matches nearly everything and contributes little.

### The Keyword Matching Problem

For "capital of the Moon?":

```
Query tokens: ["capital", "of", "the", "moon"]

Document: "paris is capital of france"
  "capital": match → 2.7
  "of":      match → 0.3
  "moon":    no match
  Total: 3.0  ← still scores highly!
```

"Capital" alone is enough to retrieve capital-related facts. BM25 has no concept of whether "capital of france" is relevant to "capital of the Moon." It matches terms, not meaning.

## Step-by-Step: What Happens During RAG Generation

For Prompt 3 with RAG:

1. **Query**: `"Q: What is the capital of the Moon?"`
2. **Tokenize query**: `["what", "is", "the", "capital", "of", "the", "moon"]`
3. **BM25 search**: Score all KB documents against query tokens
4. **Top-2 results**: "paris is capital of france" (score=3.0), "tokyo is capital of japan" (score=2.7)
5. **Build context**: `"CONTEXT: paris is capital of france. tokyo is capital of japan. "`
6. **Augmented prompt**: `"INST: CONTEXT: paris is capital of france. tokyo is capital of japan. Q: What is the capital of the Moon? ANS: "`
7. **Model forward pass**: Attention attends to "paris" and "tokyo" in context
8. **Generate**: `"paris"` — the model extracts a capital from the retrieved facts

The failure is in step 4: BM25 retrieves facts that share keywords but don't answer the question. Steps 5-8 faithfully use irrelevant context.

## What RAG Can Change (and Can't)

### Can Change

- **Knowledge access**: The model can now answer questions it never saw during training, if the KB contains the answer
- **Factual grounding**: Answers are traceable to retrieved documents (the `trace` field shows what was retrieved)
- **Knowledge updates**: Adding new facts to the KB immediately affects generation — no retraining needed
- **Graceful degradation**: When retrieval returns nothing relevant, the system falls back to SFT behavior

### Cannot Change

- **Relevance judgment**: BM25 matches keywords, not meaning — "capital of Moon" retrieves "capital of france"
- **Hallucination from context**: The model treats all retrieved context as valid evidence, even when it's irrelevant
- **Abstention**: The model still has no mechanism to say "this context doesn't answer my question"
- **Non-knowledge tasks**: Arithmetic, copy, grammar — these don't benefit from a knowledge base

### Comparison: Chapter 09 vs Chapter 10

| Aspect | Ch09 (Decoding) | Ch10 (RAG) |
|--------|-----------------|------------|
| What changes | Sampling procedure | Input context (retrieved facts) |
| When it happens | Inference time | Inference time (pre-generation) |
| Affects | Which tokens get selected | What information the model sees |
| Knowledge | Cannot add new knowledge | Can inject external knowledge |
| Hallucination | Still 100% on unknown | Still 100% — from retrieved context |
| Key insight | Sampling ≠ knowledge | Retrieval ≠ understanding |

## Human Lens

When a human encounters "What is the capital of the Moon?" and has access to a reference book:

1. **Search the index**: Look for "Moon" — find entries about moon landings, lunar phases, but no "capital"
2. **Evaluate results**: "These entries are about the Moon but none mention a capital city"
3. **Consider the question**: "The Moon doesn't have countries, so it can't have a capital"
4. **Abstain**: "There is no capital of the Moon"

The human performs **relevance evaluation** — checking whether retrieved information actually addresses the question. This is the step RAG lacks.

A human might also find "capital of france" in the index while browsing — but would immediately recognize it as irrelevant to the Moon question. The model cannot make this judgment. It sees "capital" in both the query and the retrieved context, and treats the match as evidence.

The structural difference: humans retrieve → evaluate → decide. RAG retrieves → generates. The evaluation step — "does this evidence answer THIS question?" — is where hallucination could be prevented, and it's exactly what's missing.

## What to Observe When Running

Run `python chapters/10_rag_minimal/run.py` and notice:

1. **Sample retrievals show the keyword problem** — "capital of Moon?" retrieves capital-related facts
2. **RAG corpus includes CONTEXT prefix** — the model trains on augmented prompts
3. **Knowledge QA improves** — RAG can answer questions without the fact in the prompt (e.g., `"Q: capital of france?"` without `"FACT: ..."`)
4. **Unknown task still fails** — retriever finds "related" facts, model hallucinates from them
5. **Benchmark prompts: SFT vs RAG** — same model, different context, different (but both wrong) hallucinations
6. **Trace shows what was retrieved** — RAGAgent includes retrieved facts in its output trace

### Generated Plots

**Task comparison** (`results/ch10_comparison.png`):

![Comparison](results/ch10_comparison.png)

Side-by-side bars for SFT, RAG, and Human. Knowledge QA should show the biggest improvement — RAG provides facts the model didn't memorize. Unknown task stays at 0% for both SFT and RAG. Arithmetic, copy, and grammar are similar between SFT and RAG (no relevant KB entries).

**Retrieval score heatmap** (`results/ch10_retrieval_scores.png`):

![Retrieval scores](results/ch10_retrieval_scores.png)

A heatmap of BM25 scores for benchmark queries (rows) against KB facts (columns). "Capital of france?" should show a bright cell at the "paris is capital of france" column. "Capital of Moon?" should show moderate scores across all capital-related facts — this visualizes the keyword matching problem. "ADD 5 3 =" should show zero scores everywhere (no arithmetic in KB).

## What's Next

In **Chapter 11 (Tools & Function Calls)**, we give the model access to external tools — a calculator, a lookup function — that perform computations the model cannot do internally. Instead of retrieving text, the model learns to emit structured function calls and incorporate their results. We'll see whether tools can add capabilities that architecture, scale, training, preference, decoding, and retrieval all failed to provide.
