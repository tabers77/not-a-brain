# Chapter 13: Advanced Reasoning — ReAct, Tree of Thoughts, MCTS

## Goal

Implement the three reasoning algorithms that power today's frontier LLMs. **ReAct** (Reasoning + Acting) interleaves thinking with tool use — this is how GPT-4 and Claude use tools in practice. **Tree of Thoughts** (ToT) treats reasoning as search, branching into multiple paths and following the most promising. **MCTS + Process Reward Model** is the approach behind OpenAI o1/o3 and DeepSeek-R1 — a separately trained model scores intermediate reasoning steps, and Monte Carlo Tree Search uses those scores to explore reasoning paths efficiently. The finding: these algorithms improve accuracy on solvable tasks by searching more systematically through reasoning paths. But for "capital of the Moon?", every path in the search space leads to hallucination. Search can't find an answer that doesn't exist.

## The Running Example

We trace our three benchmark prompts through four systems: **SFT** (baseline), **ReAct** (reasoning + tools), **ToT** (branching search), and **MCTS** (guided search with PRM). All use the same TransformerLM architecture.

### How ReAct Works

ReAct interleaves reasoning (THINK) with action (ACT) and observation (OBSERVE):

```
INST: ADD 5 3 =
  THINK: need calculator
  ACT: calc(ADD 5 3)
  OBSERVE: 8              <-- system executes tool, injects result
  THINK: result is 8
  ANS: 8
```

This combines Ch11's tool use with Ch12's reasoning. The model decides what to think about, when to use a tool, and how to incorporate the result — all in a single generation loop.

### How Tree of Thoughts Works

Instead of a single reasoning chain, ToT generates multiple branches and picks the best:

```
Branch 1: THINK: capital cities... paris  (score: -1.2)
Branch 2: THINK: Moon has no countries   (score: -1.8)
Branch 3: THINK: space... earth          (score: -1.4)
Branch 4: THINK: tokyo is a capital      (score: -1.1)  <-- highest score

Winner: Branch 4 -> "tokyo"
```

Branches are scored by the model's own log-probability — how confident the model is in each reasoning path. The highest-scoring path becomes the answer.

### How MCTS + PRM Works

The approach behind o1/o3. A separately trained **Process Reward Model** (PRM) scores reasoning steps:

```
Iteration 1: THINK: need to find capital...     PRM score: 0.6
Iteration 2: THINK: capital cities include...   PRM score: 0.7
Iteration 3: THINK: Moon is not a country...    PRM score: 0.4  (lower!)
Iteration 4: THINK: paris is a capital city     PRM score: 0.8  (highest!)

MCTS selects: path through iterations 1->2->4
Answer: "paris"
```

The PRM guides which reasoning paths to explore further (high score = promising) and which to abandon (low score = unpromising). UCB1 balances exploitation (follow high-scoring paths) with exploration (try new paths).

### Prompt 1: `"ADD 5 3 ="` — Computation

**SFT**: `"8"` — sometimes correct, fragile on harder arithmetic.

**ReAct**: Generates `THINK: need calculator ACT: calc(ADD 5 3) OBSERVE: 8 THINK: result is 8 ANS: 8`. The model reasons about what tool to use, executes it, observes the result, and incorporates it. Deterministically correct — the calculator always returns the right answer.

**ToT**: Generates 4 branches. Most branches that include the reasoning `"5+3=8"` score higher than branches with wrong arithmetic. The winner is `"8"`.

**MCTS**: The PRM scores reasoning paths containing `"5+3=8"` higher than paths with wrong answers (because it was trained on correct examples). MCTS follows the high-scoring path to `"8"`.

**The pattern**: All three algorithms get this right, but for different reasons. ReAct outsources computation to a tool. ToT picks the most confident reasoning. MCTS follows the PRM's judgment. For solvable tasks, more systematic search helps.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Retrieval

**SFT**: `"paris"` — correct via attention.

**ReAct**: `THINK: look up fact ACT: lookup(capital of france?) OBSERVE: paris THINK: answer is paris ANS: paris`. Redundant tool use (fact is already in the prompt), but correct.

**ToT**: All branches attend to "paris" in context. High agreement, correct answer.

**MCTS**: PRM scores paths with "paris" highest. Correct.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Hallucination

This is where even advanced reasoning fails.

**ReAct**:
```
THINK: look up fact
ACT: lookup(What is the capital of the Moon?)
OBSERVE: paris              <-- BM25 keyword match on "capital"
THINK: lookup returned paris
ANS: paris
```
ReAct combines reasoning with tools, but the lookup tool has the same BM25 keyword-matching flaw from Ch10-11. "Capital" matches "capital of france", the tool returns "paris", and the model trusts it.

**ToT**:
```
Branch 1: THINK: capital cities... paris    (score: -1.1)
Branch 2: THINK: Moon is in space... earth  (score: -1.3)
Branch 3: THINK: tokyo is a capital         (score: -1.2)
Branch 4: THINK: no known capital           (score: -1.9)  <-- lowest!
```
Branch 4 ("no known capital") is the correct reasoning — but it scores **lowest** because the model rarely saw "no known capital" in training. The model is most confident in patterns it's seen most: "capital of X" -> name a capital city. ToT picks the most confident wrong answer.

**MCTS**:
```
Path A: "capital cities include paris"  -> PRM: 0.78
Path B: "Moon has no countries"         -> PRM: 0.35
Path C: "tokyo is a major capital"      -> PRM: 0.72
```
The PRM scores "capital cities include paris" **higher** than "Moon has no countries" because the PRM was trained on the same data distribution. In training, reasoning that mentions capital cities and arrives at a city name correlates with correct answers. The PRM has no world model — it learned statistical patterns, just like the generator.

MCTS follows the PRM's guidance toward "paris" — systematically, efficiently, and wrongly.

### Why Advanced Reasoning Fails on Prompt 3

```
What search algorithms need to work:
  - The correct answer must EXIST in the search space
  - The scoring function must rank it higher than wrong answers

What actually happens:
  - "unknown" exists in the search space (some branches generate it)
  - But "unknown" scores LOWER than capital city names
  - Because the model/PRM learned that capital questions -> capital answers
  - The scoring function has the same blind spots as the generator
```

Search algorithms are only as good as their search space and scoring function. When the scoring function systematically prefers hallucination over abstention, more search means more efficient hallucination.

This connects to a broader pattern documented in the reasoning failures literature (Song, Han & Goodman, "Large Language Model Reasoning Failures," 2026): compositional reasoning failures, where models succeed on individual facts but fail when combining them, stem from the same root cause as our search failures. The model can't compose "the Moon is not a country" with "capitals belong to countries" because it stores these as independent statistical patterns, not composable logical facts. More search iterations don't help — they just compose the same non-composable patterns more efficiently.

### Summary Table

| Prompt                     | SFT     | ReAct        | ToT          | MCTS         | Correct   | What changed                      |
|----------------------------|---------|--------------|--------------|--------------|-----------|-----------------------------------|
| ADD 5 3 =                  | "8"     | "8" (calc)   | "8" (best branch) | "8" (PRM-guided) | "8"       | Search finds correct path         |
| FACT: paris... Q: capital? | "paris" | "paris" (lookup) | "paris" (all agree) | "paris" (high PRM) | "paris"   | Redundant search, same answer     |
| Q: capital of Moon?        | "earth" | "paris" (tool) | "tokyo" (highest score) | "paris" (PRM-guided) | "unknown" | Search finds the best hallucination |

### The Full Evolution — All 13 Chapters

| Prompt                     | Ch01 | Ch02 | Ch03 | Ch04 | Ch05 | Ch06 | Ch07 | Ch08 | Ch09 | Ch10 | Ch11 | Ch12 | Ch13 | Human |
|----------------------------|------|------|------|------|------|------|------|------|------|------|------|------|------|-------|
| ADD 5 3 =                  | "1"  | "5"  | "5"  | "5"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"   |
| FACT: paris... Q: capital? | " "  | "is" | "cap"| "paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"| "paris"|
| Q: capital of Moon?        | "the"| "the"| "mars"|"tokyo"|"earth"|"paris"|"earth"|"earth"|"earth"|"paris"|"paris"|"tokyo"|"paris"| "unknown"|

**13 chapters. Every major technique in modern AI. The Moon question is never solved.**

## How Each Algorithm Works

### ReAct: Reasoning + Acting

ReAct unifies the reasoning of Ch12 with the tool use of Ch11 into a single generation format:

```
THINK: <internal reasoning>
ACT: <tool_name>(<arguments>)
OBSERVE: <tool result>           <-- injected by system
THINK: <reasoning about result>
ANS: <final answer>
```

The model alternates between reasoning (THINK) and acting (ACT). After each ACT, the system executes the tool and injects the result as OBSERVE. The model then reasons about the observation before continuing.

**Why ReAct matters**: This is how modern LLM agents (GPT-4 with tools, Claude with tools, LangChain agents) actually work. The model decides not just what to answer, but what actions to take and how to interpret results.

**Why ReAct still fails**: The model's tool selection and result interpretation are learned patterns, not reasoned decisions. When `lookup("capital of Moon")` returns "paris" via keyword match, the model has no basis to question the tool's output.

### Tree of Thoughts: Reasoning as Search

ToT generalizes chain-of-thought from a single path to a tree:

1. **Branch**: Generate N different reasoning steps from the current state (temperature sampling)
2. **Evaluate**: Score each branch using the model's log-probability
3. **Select**: Keep the top-k highest-scoring branches
4. **Expand**: Continue generating from the selected branches
5. **Return**: Answer from the highest-scoring complete path

$$
\text{score}(path) = \frac{1}{|path|} \sum_{t \in path} \log P(t | t_{<})
$$

The score is the mean per-token log-probability — how surprised the model is by its own reasoning. Lower surprise (higher log-prob) = more confident path.

**Why ToT helps for solvable tasks**: When the model sometimes gets the right answer and sometimes doesn't, searching multiple paths increases the chance of finding a correct one. The scoring function correctly ranks "5+3=8" higher than "5+3=13" because the model has seen more examples of correct arithmetic.

**Why ToT fails for abstention**: The scoring function ranks "tokyo" (a confident, common pattern) higher than "unknown" (a rare pattern). ToT systematically prefers the most common type of response, which for "capital of X" questions is a capital city name — not "I don't know."

### MCTS + Process Reward Model: The o1/o3 Approach

This is the most sophisticated reasoning algorithm and the one behind OpenAI's o1/o3 and DeepSeek-R1.

**Process Reward Model (PRM)**: A separately trained neural network that scores intermediate reasoning steps:
- Input: a partial reasoning trace
- Output: score in [0, 1] — how likely this reasoning leads to a correct answer
- Trained on (correct reasoning, 1.0) and (wrong reasoning, 0.0) pairs

The PRM is trained differently from the generator. It doesn't predict next tokens — it evaluates whether a reasoning path is on track. This is the "trained judge."

**MCTS (Monte Carlo Tree Search)**: Uses the PRM to guide exploration:

1. **Select**: Pick the most promising node using UCB1:
$$
\text{UCB1}(node) = \frac{\text{total\_score}}{\text{visits}} + c \sqrt{\frac{\ln(\text{total\_visits})}{\text{visits}}}
$$
   The first term exploits high-scoring paths. The second term explores under-visited paths.

2. **Expand**: Generate a new reasoning step from the selected node
3. **Evaluate**: PRM scores the new step
4. **Backpropagate**: Update scores up the tree

This is the same algorithm that powered AlphaGo — but applied to reasoning instead of Go moves. The key insight: spend more compute on harder problems by searching deeper.

**Why MCTS+PRM is powerful for solvable tasks**: The PRM provides step-level feedback (not just final-answer feedback). If the first reasoning step is wrong, the PRM catches it early and MCTS explores other paths. This is more efficient than ToT, which only evaluates complete paths.

**Why MCTS+PRM still fails for abstention**: The PRM was trained on the same data distribution as the generator. It learned that reasoning mentioning capital cities scores high (because it usually leads to correct answers for "capital of X" questions). For the Moon question, the PRM scores "paris is a capital city" (0.78) higher than "Moon has no countries" (0.35) — because in training, the first pattern correlated with correctness and the second didn't appear.

The PRM has the same blind spot as the generator: it learned statistical patterns, not world knowledge. A separate model can be more accurate, but it can't know things the training data didn't teach it.

## Step-by-Step: What Happens During PRM Training

The PRM is trained as a binary classifier on reasoning quality:

**Positive example** (label = 1.0):
```
"INST: ADD 5 3 = THINK: op is ADD, 5+3=8 ANS: 8"
```
This is correct reasoning leading to a correct answer.

**Negative example** (label = 0.0):
```
"INST: ADD 5 3 = THINK: steps: fish ANS: fish"
```
This is wrong reasoning (shuffled from a different task) leading to a wrong answer.

The PRM learns to distinguish good reasoning patterns from bad ones:
- Arithmetic reasoning mentioning the correct operation scores high
- Knowledge reasoning mentioning the right fact scores high
- Mismatched or nonsensical reasoning scores low

**What the PRM does NOT learn**: Whether a question is answerable. The training data has many examples of questions with correct answers and few examples of "unanswerable" patterns. The PRM can't score "I don't know" as correct if that pattern is rare in training.

## What Advanced Reasoning Can Change (and Can't)

### Can Change

- **Systematic search**: ToT and MCTS explore multiple reasoning paths instead of committing to one
- **Tool-augmented reasoning**: ReAct combines thinking with action in a unified framework
- **Step-level evaluation**: PRM catches bad intermediate steps before they compound into wrong final answers
- **Test-time compute scaling**: MCTS can spend more iterations on harder problems — this is how o1 "thinks longer" on difficult questions
- **Accuracy on solvable tasks**: All three algorithms improve performance when the correct answer exists in the search space

### Cannot Change

- **Search space limitations**: If "unknown" scores low in the model's probability space, search won't find it
- **Scoring blind spots**: The PRM has the same training data distribution — it can't score patterns it hasn't learned
- **Tool quality**: ReAct is limited by its tools. BM25 lookup returns keyword matches, not semantic matches
- **Fundamental abstention**: No amount of search compensates for the absence of genuine uncertainty awareness

### Comparison: Chapter 12 vs Chapter 13

| Aspect | Ch12 (Basic Reasoning) | Ch13 (Advanced Reasoning) |
|--------|----------------------|--------------------------|
| CoT | Single chain | Multiple chains (ToT) or guided search (MCTS) |
| Self-Consistency | Majority vote (unguided) | PRM-guided selection |
| Verify | Self-check (same model) | PRM check (separate model) |
| Tool use | Not combined | ReAct unifies reasoning + tools |
| Search | None | Tree search (ToT) and MCTS |
| Compute | Fixed | Adaptive (more iterations for harder problems) |
| Hallucination | From fluent-sounding reasoning | From systematically searched reasoning |

## Human Lens

When a human encounters "What is the capital of the Moon?":

1. **Reason**: "Capitals are political entities. The Moon has no political structure."
2. **Search knowledge**: Check memory for Moon facts — landings, phases, composition. No "capital" entry.
3. **Evaluate**: "Every path I consider leads to 'the Moon has no capital.' This isn't a gap in my knowledge — the question has a type error."
4. **Decide**: "The correct answer is that there is no answer."
5. **Abstain**: "unknown"

The human's search is guided by a **world model** — an understanding of what capitals are, what the Moon is, and why they're incompatible. Every search path the human explores reinforces the conclusion that the question is unanswerable.

The model's search (ToT, MCTS) is guided by **statistical patterns**:
- "Capital of X" -> high probability of capital city names
- "Moon" -> associated with space, earth, planets
- "Unknown/unanswerable" -> rare pattern, low probability

The model searches more efficiently than Ch12's single-chain approach, but it searches through the same probability landscape. More compute doesn't change the landscape — it just explores it more thoroughly.

**The structural difference**:
- **Human search**: guided by understanding, converges on "unanswerable"
- **Model search**: guided by probability, converges on "most common pattern"

This is the deepest version of the project's thesis. Even with search algorithms from AlphaGo, trained reward models from RLHF, and the reasoning patterns behind o1/o3 — the model cannot arrive at "I don't know" because its search space and scoring function don't support it.

The same survey that documents the Reversal Curse (Song et al., 2026) finds that these failures persist at GPT-4 scale and beyond. Their taxonomy distinguishes *fundamental architectural failures* (intrinsic to next-token prediction) from *application-specific limitations* (domain-tied). What we show here is that ToT, MCTS, and PRM are application-level mitigations — they can't fix what's fundamentally missing at the architectural level. The scoring function and the generator share the same architectural blind spot: probability over tokens, not understanding of concepts.

## What to Observe When Running

Run `python chapters/13_advanced_reasoning/run.py` and notice:

1. **ReAct traces** show THINK/ACT/OBSERVE interleaving — the model reasons about when to use tools
2. **ToT branches** show multiple reasoning paths with scores — the "best" path isn't always the correct one
3. **PRM sanity check** — good reasoning should score higher than bad reasoning for solvable tasks
4. **MCTS traces** show iterations, nodes explored, and PRM scores — more search doesn't help the Moon question
5. **Benchmark Prompt 3**: All three algorithms produce different hallucinations — ReAct via tool, ToT via scoring, MCTS via PRM
6. **Unknown task stays at 0%** across all agents
7. **The final punchline**: 13 chapters, every technique, hallucination persists

### Generated Plots

**Task comparison** (`results/ch13_comparison.png`):

![Comparison](results/ch13_comparison.png)

Side-by-side bars for SFT, ReAct, ToT, MCTS, and Human. The unknown task shows 0% for all model-based agents. Arithmetic and knowledge QA may show improvement from ReAct (tool use) and MCTS (guided search). The gap between any model agent and the human on the unknown task is the project's thesis visualized.

**Search comparison** (`results/ch13_search_depth.png`):

![Search depth](results/ch13_search_depth.png)

ToT vs MCTS accuracy per task. MCTS (with PRM guidance) may outperform ToT (with log-prob scoring) on solvable tasks, showing the value of a trained judge. But on the unknown task, both are at 0% — more sophisticated search doesn't help when the correct answer is "I don't know."

## What's Next

Chapter 14 asks the question that haunts every chapter: "But GPT-5.2 gets the Moon question right — doesn't that mean scale solves it?" We'll train models at increasing levels of abstention coverage and show that scale solves **coverage** (memorizing more patterns) but not **understanding** (reasoning from principles). The model that says "unknown" for "capital of the Moon" hallucinates for "WiFi password of the Bermuda Triangle" — because it memorized an answer, not a concept.
