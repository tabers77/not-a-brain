# Chapter 11: Tools & Function Calling

## Goal

Give the model external tools — a calculator and a knowledge lookup — that perform tasks the model cannot do internally. The model learns to emit structured `CALL:tool(args)` markers during generation; the system intercepts these, executes the tool, and injects `RESULT:value` back into the context. We compare three approaches: plain SFT (no tools), a ToolAgent (model decides when to call tools), and an OracleToolAgent (orchestrator always calls the right tool). The finding: tools genuinely extend capabilities (the calculator solves arithmetic), but the model cannot judge whether a tool's output actually answers the question — for "capital of the Moon?", the lookup tool returns irrelevant capital facts via keyword match, and the model treats them as valid evidence.

## The Running Example

We trace our three benchmark prompts through three systems: **SFT** (no tools), **ToolAgent** (model-initiated tool calls), and **OracleTool** (orchestrator-initiated tool calls). All use the same TransformerLM architecture and training — only tool access differs.

### How Tool Calling Works

After generating a few tokens, the model may emit a `CALL:` marker:

```
Input:  "INST: ADD 5 3 = "
Model generates: "CALL:calc(ADD 5 3)"
                  ^--- system intercepts here

System executes: calc("ADD 5 3") -> "8"
System injects:  "RESULT:8 ANS: "
                  ^--- generation continues

Model generates: "8"
Final answer:    "8"
```

The key difference from RAG (Chapter 10): instead of always prepending retrieved text, the model learns to **request specific computations** and the system executes them. The model decides whether, when, and which tool to call.

### Prompt 1: `"ADD 5 3 ="` — Computation with Tools

This is where tools shine. Arithmetic requires exact computation that neural networks struggle to learn reliably.

**SFT (no tools)**: `"8"` — sometimes correct, sometimes wrong. The model learned arithmetic patterns during training but doesn't reliably compute. Small digit problems sometimes work; larger ones fail.

**ToolAgent**: The model generates `CALL:calc(ADD 5 3)`, the system computes `5 + 3 = 8` and injects `RESULT:8`. The model then outputs `"8"`. The calculator is deterministic — it always returns the correct answer regardless of operand size.

```
Step-by-step:
  1. Model sees: "INST: ADD 5 3 = "
  2. Model generates: " CALL:calc(ADD 5 3)"
  3. System intercepts: executes calc("ADD 5 3") -> "8"
  4. Context becomes: "INST: ADD 5 3 = CALL:calc(ADD 5 3) RESULT:8 ANS: "
  5. Model generates: "8"
```

**OracleTool**: Same result — the orchestrator recognizes the arithmetic pattern and calls `calc` before the model generates anything.

**The pattern**: Tools add genuine capabilities. The calculator performs exact computation that the model's weights cannot — this is not retrieval or pattern matching, it's outsourced computation.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Retrieval with Tools

The fact is already in the prompt. The lookup tool provides redundant confirmation.

**SFT (no tools)**: `"paris"` — correct. Attention retrieves the answer from context (same as Chapters 04-10).

**ToolAgent**: If the model generates `CALL:lookup(capital of france?)`, the system searches the KB and returns `"paris"`. Redundant but harmless — tool confirms what's in the prompt.

**OracleTool**: Always calls `lookup(capital of france?)` -> `"paris"`. Same redundant confirmation.

**The pattern**: When the model already has the answer, tools provide redundant verification. Like RAG, this is harmless — more signal, same answer.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Hallucination with Tools

This is where tools reveal the same fundamental limitation as RAG.

**SFT (no tools)**: `"earth"` — hallucination. Same as Chapters 05-10.

**ToolAgent**: The model may generate `CALL:lookup(capital of the Moon?)`. The lookup tool uses BM25 to search the KB. "Capital" matches many entries: "paris is capital of france", "tokyo is capital of japan". The top match returns `"paris"`. The model sees `RESULT:paris` and outputs `"paris"` — hallucination laundered through a tool.

```
Step-by-step:
  1. Model sees: "INST: Q: What is the capital of the Moon? "
  2. Model generates: " CALL:lookup(What is the capital of the Moon?)"
  3. System executes BM25 search:
     - "capital" matches "paris is capital of france" (score=3.0)
     - "capital" matches "tokyo is capital of japan" (score=2.7)
     - "moon" matches nothing in KB
     - Top result: "paris" (score=3.0)
  4. Context becomes: "... CALL:lookup(...) RESULT:paris ANS: "
  5. Model generates: "paris"
```

**OracleTool**: Same problem. The orchestrator calls `lookup(What is the capital of the Moon?)`, gets `"paris"` from BM25, and the model uses it.

**Why tools fail here**: The lookup tool uses the same BM25 keyword matching as RAG. "Capital of the Moon" retrieves "capital of france" because they share the word "capital." The tool cannot distinguish "this fact answers the question" from "this fact shares a keyword." And the model cannot evaluate whether the tool's output is relevant.

### Why Tools Fail on Prompt 3

```
Human process:
  1. Choose tool: "I need to look this up" -> lookup
  2. Execute: search for "capital of Moon" -> no relevant results
  3. Evaluate result: "The results are about France, not the Moon"
  4. Decide: "The tool didn't help. I don't know."
  5. Abstain: "unknown"

Tool agent process:
  1. Generate tool call: CALL:lookup(capital of the Moon?)
  2. System executes: BM25 returns "paris" (keyword match on "capital")
  3. Model sees RESULT:paris
  4. Generate from result: "paris"
  (No evaluation step — no relevance check on tool output)
```

The model outsources the search but not the judgment. Tools add capabilities (what you can compute) but not understanding (whether the result answers your question).

### Training with NOT_FOUND

During training, unknown-question examples include `RESULT:NOT_FOUND ANS: unknown`, teaching the model that `NOT_FOUND` should map to abstention. But at inference time, BM25 returns capital-related facts instead of `NOT_FOUND` (because "capital" is a strong keyword match). The model never sees `NOT_FOUND`, so it never triggers the abstention pattern.

Even if the lookup tool returned `NOT_FOUND`, our tiny model doesn't robustly learn the `NOT_FOUND -> unknown` mapping — it sometimes hallucinates anyway. The mapping requires understanding what `NOT_FOUND` means, not just memorizing the pattern.

### Summary Table

| Prompt                     | SFT (no tools) | ToolAgent         | OracleTool        | Tool called          | Correct   | What changed                    |
|----------------------------|----------------|-------------------|-------------------|----------------------|-----------|----------------------------------|
| ADD 5 3 =                  | "8" (fragile)  | "8" (via calc)    | "8" (via calc)    | calc(ADD 5 3) -> 8   | "8"       | Calculator adds exact computation|
| FACT: paris... Q: capital? | "paris"        | "paris"           | "paris"           | lookup -> paris      | "paris"   | Redundant tool confirmation      |
| Q: capital of Moon?        | "earth"        | "paris"           | "paris"           | lookup -> paris (!)  | "unknown" | Different hallucination source   |

### Evolution So Far

| Prompt                     | Ch01 Bigram | Ch02 FFN | Ch03 GRU | Ch04 Attn | Ch05 Transformer | Ch06 (best) | Ch07 SFT | Ch08 DPO | Ch09 Decode | Ch10 RAG | Ch11 Tools | Human     | What changed                   |
|----------------------------|-------------|----------|----------|-----------|------------------|-------------|----------|----------|-------------|----------|------------|-----------|--------------------------------|
| ADD 5 3 =                  | "1"         | "5"      | "5"      | "5"       | "8"              | "8"         | "8"      | "8"      | "8"         | "8"      | "8" (calc) | "8"       | Tools guarantee correct arithmetic|
| FACT: paris... Q: capital? | " "         | "is"     | "capital"| "paris"   | "paris"          | "paris"     | "paris"  | "paris"  | "paris"     | "paris"  | "paris"    | "paris"   | Tools add redundant confirmation  |
| Q: capital of Moon?        | "the"       | "the"    | "mars"   | "tokyo"   | "earth"          | "paris"     | "earth"  | "earth"  | "earth"     | "paris"  | "paris"    | "unknown" | 11 chapters in, STILL hallucinates|

## How Tool Calling Works

### The Tool Interface

Each tool is a simple function:

```python
def tool_calc(args: str) -> str:
    # Parse "ADD 5 3" -> compute 5 + 3 -> return "8"

def tool_lookup(query: str) -> str:
    # BM25 search KB -> return best answer or "NOT_FOUND"
```

Tools are deterministic: same input always produces same output. They add capabilities the model's weights don't have — exact arithmetic, structured knowledge access.

### The CALL/RESULT Protocol

Training examples embed tool calls as text markers:

```
Arithmetic:
  "INST: ADD 5 3 = CALL:calc(ADD 5 3) RESULT:8 ANS: 8"

Knowledge (known):
  "INST: Q: capital of france? CALL:lookup(capital of france?) RESULT:paris ANS: paris"

Knowledge (unknown):
  "INST: Q: capital of Moon? CALL:lookup(capital of Moon?) RESULT:NOT_FOUND ANS: unknown"

No tool needed:
  "INST: COPY: abc| ANS: abc"
```

The model learns three patterns:
1. **When to call**: Arithmetic -> calc, questions -> lookup, copy/grammar -> no call
2. **Call format**: `CALL:tool_name(arguments)`
3. **Result integration**: Use `RESULT:value` to generate the final answer

### ToolAgent vs OracleToolAgent

| Aspect | ToolAgent | OracleToolAgent |
|--------|-----------|-----------------|
| Who decides to call | The model (generates CALL:) | The orchestrator (pattern matching) |
| Tool selection | Learned from training data | Hard-coded rules |
| Can fail to call | Yes — model may not generate CALL: | No — always calls if pattern matches |
| Can call wrong tool | Yes — model may confuse tools | No — deterministic rules |
| Shows | What the model actually learned | Ceiling of tool-augmented performance |

The gap between ToolAgent and OracleToolAgent shows how much the model loses from imperfect tool selection — a measure of how well it learned the calling protocol.

### Why the Calculator is Different from Lookup

The calculator is a **true capability extension**:
- Input: a well-formed expression (ADD 5 3)
- Output: the exact correct answer (8)
- No ambiguity, no keyword matching, no relevance judgment needed

The lookup tool is **retrieval with structure**:
- Input: a natural language query
- Output: best keyword match from KB (which may be irrelevant)
- Same BM25 flaw as RAG — "capital of Moon" retrieves "capital of france"
- Wrapping retrieval in a tool interface doesn't fix the retrieval quality

This distinction matters: tools that perform exact computation (calculator, code executor) genuinely solve problems. Tools that perform fuzzy search (lookup, web search) inherit all the limitations of their search algorithm.

## Step-by-Step: What Happens During Tool-Augmented Training

For a training example `"INST: ADD 5 3 = CALL:calc(ADD 5 3) RESULT:8 ANS: 8"`:

1. **Tokenize**: Character-level encoding of the full string including markers
2. **Forward pass**: Model sees all tokens, predicts next token at each position
3. **Loss**: Cross-entropy over entire sequence — the model learns to predict `CALL:`, `calc`, `(`, `ADD 5 3`, `)`, `RESULT:`, `8`, `ANS:`, `8`
4. **What the model learns**:
   - After arithmetic prompts, emit `CALL:calc(...)`
   - After `RESULT:`, the next content is the answer
   - After `ANS:`, repeat the result value
5. **What the model does NOT learn**:
   - What `calc` actually computes (it just sees input/output pairs)
   - Why `RESULT:8` is correct for `ADD 5 3`
   - That `RESULT:NOT_FOUND` means "no answer exists"

The model treats tool calls as text patterns to imitate, not as requests to an external system. It doesn't understand that `CALL:` triggers computation — it just predicts the most likely next character.

## What Tools Can Change (and Can't)

### Can Change

- **Exact computation**: Calculator provides correct arithmetic regardless of model capacity
- **Structured knowledge access**: Lookup searches KB with explicit queries, not implicit attention
- **Capability ceiling**: OracleToolAgent shows the maximum benefit of tools — model capacity is no longer the bottleneck for computation
- **Composability**: Multiple tools can be called in sequence (calc for arithmetic, lookup for facts)

### Cannot Change

- **Tool output judgment**: The model treats all tool results as valid — it cannot evaluate whether `RESULT:paris` actually answers "capital of the Moon?"
- **Tool selection reliability**: The tiny model doesn't always call the right tool or format the call correctly
- **Abstention**: Even with `RESULT:NOT_FOUND` in training, the model doesn't robustly learn to abstain
- **Non-tool tasks**: Copy, grammar, compositional tasks don't benefit from a calculator or lookup

### Comparison: Chapter 10 vs Chapter 11

| Aspect | Ch10 (RAG) | Ch11 (Tools) |
|--------|-----------|-------------|
| What the model gets | Retrieved text prepended | Tool results injected |
| Who initiates | System always retrieves | Model generates CALL: (or oracle decides) |
| Computation | No computation — just text | Calculator performs real arithmetic |
| Knowledge | BM25 retrieves text | BM25 lookup returns answers |
| Format | CONTEXT: prefix | CALL:tool(args) RESULT:value |
| Hallucination | From retrieved context | From tool output (same BM25 flaw) |
| Key insight | Retrieval != understanding | Computation != judgment |

## Human Lens

When a human encounters "ADD 5 3 =" and has a calculator:

1. **Recognize the task**: "This is arithmetic — I should use the calculator"
2. **Formulate the query**: Type `5 + 3` into the calculator
3. **Read the result**: `8`
4. **Verify**: "Does 8 make sense for 5 + 3? Yes."
5. **Answer**: "8"

When a human encounters "What is the capital of the Moon?" and has an encyclopedia:

1. **Recognize the task**: "This is a factual question — I should look it up"
2. **Formulate the query**: Look up "Moon" in the index
3. **Read the result**: Entries about moon landings, lunar phases — no "capital" entry
4. **Evaluate**: "The Moon doesn't have countries, so it can't have a capital"
5. **Abstain**: "There is no capital of the Moon"

The human performs **result evaluation** at step 4 — checking whether the tool's output actually answers the question. This is the step the model lacks.

The structural difference:
- **Human**: select tool -> execute -> evaluate result -> decide
- **Model**: emit CALL: pattern -> receive result -> generate from result

The model skips the evaluation step. It doesn't check whether the calculator's input was well-formed, whether the lookup's result is relevant, or whether the tool addressed the actual question. Tools extend what the model can compute, but not what it can judge.

## What to Observe When Running

Run `python chapters/11_tools_and_function_calls/run.py` and notice:

1. **Tool demos show exact vs fuzzy**: `calc("ADD 5 3")` always returns `"8"`, but `lookup("capital of Moon?")` returns a capital-of-somewhere answer
2. **Tool corpus examples include CALL/RESULT markers** embedded in the training text
3. **SFT vs ToolAgent**: Where does tool access help? Arithmetic should improve with calc
4. **ToolAgent vs OracleToolAgent**: The gap shows how well the model learned to call tools
5. **Tool usage by task**: The tool usage plot shows which tasks trigger which tools
6. **Benchmark Prompt 3**: The Moon question still fails — different hallucination source, same outcome
7. **Trace output**: ToolAgent includes `trace` showing which tools were called and what they returned

### Generated Plots

**Task comparison** (`results/ch11_comparison.png`):

![Comparison](results/ch11_comparison.png)

Side-by-side bars for SFT, ToolAgent, OracleTool, and Human. Arithmetic should show the biggest improvement from tools (calculator is deterministic). Knowledge QA may improve (lookup provides facts). Unknown task stays at 0% for all model-based agents. The gap between ToolAgent and OracleTool shows the cost of imperfect tool selection.

**Tool usage by task** (`results/ch11_tool_usage.png`):

![Tool usage](results/ch11_tool_usage.png)

Stacked bars showing what fraction of prompts trigger calc, lookup, or no tool call, broken out by task. Arithmetic prompts should mostly trigger calc. Knowledge/unknown prompts should trigger lookup. Copy/grammar/compositional should show no tool calls. Deviations from this ideal pattern reveal where the model's tool selection fails.

## What's Next

In **Chapter 12 (Reasoning Scaffolds)**, we add explicit reasoning steps — chain-of-thought prompting, self-consistency (generate multiple answers, vote), and a verify loop. Instead of calling external tools, the model reasons through intermediate steps in its own output. We'll see whether structured reasoning can provide the judgment that architecture, scale, training, preference, decoding, retrieval, and tools all failed to deliver — and whether the Moon question finally gets an honest "I don't know."
