# Chapter 14: Scale Is Not Understanding

## Goal

Answer the question that haunts every chapter: "If GPT-5.2 correctly says 'The Moon does not have a capital city,' doesn't that mean scale solves hallucination?" No. This chapter demonstrates experimentally that scale solves **coverage** — the model has memorized enough patterns to handle that specific question — not **understanding**. We train models at five levels of abstention coverage (0 to 30 patterns) and test them on in-distribution questions, rephrased versions, and novel nonsensical questions. The finding: in-distribution accuracy climbs with coverage (memorization works), rephrased accuracy lags behind (generalization is limited), and novel questions barely improve at all (understanding is absent). This is the difference between a lookup table with billions of entries and a mind that knows what it doesn't know.

## The Running Example

We trace our three benchmark prompts through five models, each trained with increasing amounts of "unknown" patterns — simulating what happens as an LLM sees more and more data.

### The Five Coverage Levels

| Level | Abstention Patterns | What It Simulates |
|-------|--------------------|--------------------|
| Level 0 | 0 patterns | Ch01-09: no abstention training |
| Level 1 | 1 pattern ("capital of Moon") | Memorizing one specific answer |
| Level 2 | 5 patterns | Limited RLHF coverage |
| Level 3 | 20 patterns | Moderate coverage |
| Level 4 | 30 patterns | Simulating scale — many patterns |

### Prompt 1: `"ADD 5 3 ="` — Computation

All five models answer `"8"`. Abstention training doesn't hurt solvable tasks — the base corpus still contains arithmetic examples, and the model learns them regardless of how many "unknown" patterns are added. This confirms that abstention training is additive, not destructive.

### Prompt 2: `"FACT: paris is capital of france. Q: capital of france?"` — Retrieval

All five models answer `"paris"`. Same reasoning — attention retrieves the answer from context, and abstention patterns don't interfere with this capability.

### Prompt 3: `"Q: What is the capital of the Moon?"` — Hallucination

This is where coverage levels matter:

**Level 0 (no abstention)**:
```
INST: Q: What is the capital of the Moon? ANS: tokyo
```
The model has never seen "unknown" as an answer to this type of question. It predicts a capital city — the most common pattern following "capital of X".

**Level 1 (Moon only)**:
```
INST: Q: What is the capital of the Moon? ANS: unknown
```
The model has memorized this exact question → "unknown". It gets it right. But this is pure memorization, not understanding.

**Level 2-4 (increasing coverage)**:
```
INST: Q: What is the capital of the Moon? ANS: unknown
```
All levels with Moon in training get it right. More patterns don't help for a question that's already memorized.

### The Real Test: Rephrased Questions

Now rephrase the Moon question in ways the model hasn't seen:

```
"Q: Which city is the Moon's capital?"
"Q: Name the capital city of the Moon."
"Q: The Moon's seat of government is located in which city?"
"Q: Tell me the capital of Earth's Moon."
```

**Level 0**: Hallucinates on all — no abstention concept at all.

**Level 1**: Hallucinates on all rephrases — memorized "What is the capital of the Moon?" exactly, but "Which city is the Moon's capital?" is a different token sequence.

**Level 2-3**: May catch some rephrases — patterns like "capital of Sun" and "capital of Jupiter" create partial generalization. The model starts to associate "capital of [celestial body]" with "unknown".

**Level 4**: Better on rephrases — 30 patterns create more overlap. But still fails on phrasing that doesn't match any trained pattern.

### The Hardest Test: Novel Questions

Questions the model has never seen in any form:

```
"Q: What is the WiFi password of the Bermuda Triangle?"
"Q: What is the shoe size of mathematics?"
"Q: How many teeth does Wednesday have?"
"Q: What is the email address of gravity?"
```

These are category errors — applying a physical attribute to an abstract concept, or a human attribute to a day of the week. A human instantly recognizes these as nonsensical. The model has no basis for abstention because:
- "WiFi password of X" never appeared in training
- "Shoe size of X" never appeared in training
- The model can't reason "mathematics is abstract, therefore it has no shoe size"

**All levels hallucinate on novel questions.** Even Level 4 with 30 patterns. The model memorized "capital of Moon → unknown" and many variations, but it can't generalize to "this entire category of question is unanswerable."

### Summary Table

| Test Type | Level 0 | Level 1 | Level 2 | Level 3 | Level 4 | Human |
|-----------|---------|---------|---------|---------|---------|-------|
| In-distribution | 0% | 25%+ | 50%+ | 75%+ | 75%+ | 100% |
| Rephrased | 0% | 0% | ~25% | ~25% | ~50% | 100% |
| Novel | 0% | 0% | 0% | 0% | ~0% | 100% |

The gap between Level 4 and Human on novel questions is the thesis of this entire project.

### The Full Evolution — All 14 Chapters

| Prompt                     | Ch01 | Ch02 | Ch03 | Ch04 | Ch05 | Ch06 | Ch07 | Ch08 | Ch09 | Ch10 | Ch11 | Ch12 | Ch13 | Ch14 | Human |
|----------------------------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|-------|
| ADD 5 3 =                  | "1"  | "5"  | "5"  | "5"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"  | "8"   |
| FACT: paris... Q: capital? | " "  | "is" | "cap"| "paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"|"paris"| "paris"|
| Q: capital of Moon?        | "the"| "the"| "mars"|"tokyo"|"earth"|"paris"|"earth"|"earth"|"earth"|"paris"|"paris"|"tokyo"|"paris"|"unknown"*|"unknown"|

\*Level 1+ only — because the exact question was in training. Rephrase the question and the hallucination returns.

**14 chapters. The Moon question is "solved" only by memorizing the answer. Change the wording and it breaks.**

## How Coverage Simulates Scale

### What GPT-5.2 Actually Does

When GPT-5.2 correctly answers "The Moon does not have a capital city," it's doing the same thing our Level 4 model does — but with incomprehensibly more coverage:

| Our Model (Level 4) | GPT-5.2 |
|---------------------|---------|
| 30 abstention patterns | Millions of abstention patterns from RLHF |
| Character-level tokenizer | 100K subword tokens |
| ~30K parameters | Trillions of parameters |
| 5 categories of nonsense | Virtually every category humans have discussed online |
| Fails on novel rephrasing | Fails on sufficiently novel rephrasing |

The difference is quantitative, not qualitative. GPT-5.2 has seen so many patterns that it's hard to find a gap. But the mechanism is the same: pattern matching against training data.

### Evidence That It's Coverage, Not Understanding

Real-world evidence that frontier LLMs still pattern-match rather than understand:

1. **Adversarial rephrasing**: Researchers routinely find ways to make LLMs hallucinate by rephrasing questions in unusual ways
2. **Novel category errors**: LLMs can be tricked by questions that combine familiar concepts in unfamiliar ways
3. **Instruction following vs. reasoning**: LLMs follow the *format* of abstention ("I'm not sure about...") without genuine uncertainty awareness
4. **Sycophancy**: When users push back, LLMs often abandon correct abstention — suggesting the "I don't know" was a learned pattern, not a reasoned conclusion
5. **The Reversal Curse**: Models trained on "A is B" cannot infer "B is A" — even frontier models store directional correlations, not symmetric facts (Song, Han & Goodman, 2026)
6. **Perturbation brittleness**: Logic-preserving transformations (reordering options, renaming entities, rephrasing questions) cause performance drops on benchmarks the model supposedly "solved" — the same pattern we demonstrate with our rephrased Moon questions (Song et al., 2026)

The first systematic survey of LLM reasoning failures (Song, Han & Goodman, "Large Language Model Reasoning Failures," 2026) provides large-scale evidence for exactly this thesis. Their two-axis taxonomy — reasoning type (formal/informal/embodied) crossed with failure type (fundamental/application-specific/robustness) — shows that the failures we demonstrate at toy scale with ~30K parameters persist identically at frontier scale with trillions of parameters. The root causes they identify are the same ones we trace through 14 chapters: the next-token prediction objective, causal attention's directional bias, and training data distributions that encode correlation without causation. Scale changes the coverage. It does not change the mechanism.

### The Coverage-Understanding Spectrum

```
More coverage          ←──────────────────────→       Understanding
(bigger model,          (world model,
 more RLHF data,        causal reasoning,
 more training)         meta-cognition)

Our Level 0  ...  Level 4  ...  GPT-5.2  ...  Human

|----------pattern matching---------|----???----|
```

The question mark is intentional. We don't know if there's a path from "enough coverage" to "genuine understanding." This project demonstrates the gap exists. Whether it's bridgeable is an open question.

## Step-by-Step: What Happens During Training

### Base Corpus (all levels)

Standard SFT examples from all task types:
```
"INST: ADD 5 3 = ANS: 8"
"INST: COPY: abc| ANS: abc"
"INST: FACT: paris is capital of france. Q: capital of france? ANS: paris"
```

### Abstention Corpus (level-dependent)

Each pattern is repeated 50 times for emphasis:
```
Level 1: "INST: Q: What is the capital of the Moon? ANS: unknown" × 50
Level 2: + "INST: Q: What is the population of Atlantis? ANS: unknown" × 50
          + "INST: Q: Who won the 2087 World Cup? ANS: unknown" × 50
          + ... (5 patterns × 50 = 250 examples)
Level 4: 30 patterns × 50 = 1,500 abstention examples
```

### What the Model Learns

At each level, the model learns two things:
1. **Task competence**: Arithmetic, copy, knowledge QA (from base corpus)
2. **Abstention patterns**: Specific question→"unknown" mappings (from abstention corpus)

The model does NOT learn:
- "Questions about non-existent things are unanswerable"
- "If I haven't seen evidence for X, I should say unknown"
- "Category errors are nonsensical"

It learns specific (input, output) pairs. This is the fundamental limitation.

## What Coverage Can Change (and Can't)

### Can Change

- **In-distribution abstention**: Adding "capital of Moon → unknown" directly teaches the model to say "unknown" for that exact question
- **Nearby generalization**: With enough similar patterns ("capital of Sun", "capital of Jupiter", "capital of Neptune"), the model learns a local pattern: "capital of [celestial body] → unknown"
- **Format compliance**: The model learns to produce "unknown" as a valid answer format
- **Apparent reliability**: With enough coverage, the model rarely encounters questions outside its training distribution — creating the *illusion* of understanding

### Cannot Change

- **True generalization**: The model can't reason "the Moon is not a political entity, therefore it has no capital" — it can only match patterns it has seen
- **Novel category errors**: "Shoe size of mathematics" requires understanding that mathematics is abstract and shoe sizes are physical — not pattern matching
- **Adversarial robustness**: Rephrasing a trained question in a new way breaks the pattern match
- **Genuine uncertainty**: The model doesn't *know* it doesn't know — it matches an input pattern to an output pattern. "Unknown" is just another token it predicts

### Comparison: Pattern Matching vs Understanding

| Aspect | Pattern Matching (LLM) | Understanding (Human) |
|--------|----------------------|----------------------|
| "Capital of Moon?" | Matches training pattern → "unknown" | Moon isn't a country → no capital possible |
| "Capital of Zorgon?" | If trained: "unknown". If not: hallucinates | Zorgon isn't real → no capital possible |
| "WiFi password of math?" | No matching pattern → hallucinates | Math isn't a network → question is nonsensical |
| Mechanism | Input similarity → output | Concept + reasoning → conclusion |
| Failure mode | Unseen patterns | Genuinely hard reasoning |
| Scaling behavior | More patterns → fewer gaps | Same principle → all gaps covered |

## Human Lens

A human handles all three test categories identically:

**In-distribution**: "What is the capital of the Moon?" → "The Moon has no capital. It's not a country." The human doesn't recall a memorized answer — they reason from what they know about the Moon and about capitals.

**Rephrased**: "Which city is the Moon's seat of government?" → Same reasoning, same answer. The rephrasing doesn't matter because the human reasons from concepts, not token patterns.

**Novel**: "What is the WiFi password of the Bermuda Triangle?" → "That's nonsensical. The Bermuda Triangle isn't a network and doesn't have WiFi." The human has never been asked this question before, but immediately recognizes it as a category error.

The human uses **one principle**: "Does this entity have this attribute?" The model needs **one training example per pattern**. That's the difference between O(1) reasoning and O(N) memorization.

## What to Observe When Running

Run `python chapters/14_scale_is_not_understanding/run.py` and notice:

1. **In-distribution accuracy climbs** with coverage level — more patterns = better performance on trained questions
2. **Rephrase accuracy lags behind** — even Level 4 struggles with unseen phrasings of the same question
3. **Novel accuracy stays near zero** — 30 abstention patterns don't generalize to new categories of nonsense
4. **The gap widens** between in-distribution and novel as coverage increases — the model gets better at what it's seen without getting better at what it hasn't
5. **Level 1 vs Level 0** on the Moon question — memorization works perfectly for the exact trained input
6. **Level 4 on rephrases vs novel** — some generalization from similar patterns, but no transfer to new question types

### Generated Plots

**Coverage vs Understanding** (`results/ch14_coverage.png`):

![Coverage](results/ch14_coverage.png)

Three lines showing abstention rate vs coverage level for in-distribution (green, climbing), rephrased (orange, rising slowly), and novel (red, flat near zero). The growing gap between green and red IS the thesis: more coverage helps memorization but not understanding.

**Brittleness** (`results/ch14_brittleness.png`):

![Brittleness](results/ch14_brittleness.png)

Bar chart for the highest-coverage model showing performance on original (high), rephrased (medium), and novel (near zero) questions. Same model, same capability, completely different performance — because it memorized answers, not concepts.

## What's Next

There is no Chapter 15.

This project traced the full modern AI stack: from character counting (n-grams) through neural networks, attention, transformers, scaling, instruction tuning, preference alignment, decoding, retrieval, tool use, basic reasoning, advanced reasoning, and finally the question of whether scale bridges the gap.

The answer: scale bridges the *coverage* gap. Given enough training data, enough parameters, and enough RLHF, a model can handle virtually any question a human has discussed online. This is genuinely useful and commercially valuable.

But scale does not bridge the *understanding* gap. The model that says "The Moon has no capital" does so because it matches a pattern, not because it understands what capitals and moons are. Change the question slightly and the pattern breaks. Ask something truly novel and the model hallucinates confidently.

The thesis: **similar outputs != same mechanism**. A model that says "unknown" and a human that says "unknown" arrived there by completely different paths. The model matched a training pattern. The human reasoned from a world model.

That difference is invisible when the coverage is large enough. It becomes visible at the edges — in novel questions, adversarial rephrasing, and category errors that no training data anticipated.

That's not a brain.
