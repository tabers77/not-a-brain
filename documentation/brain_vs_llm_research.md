# The Human Brain in Code & Formulas vs. LLM Architectures

Research survey comparing attempts to model the human brain computationally against modern LLM reasoning architectures. This document supports the project's thesis: "Similar outputs != same mechanism."

---

## 1. Biologically Detailed Simulations: Modeling Actual Neurons

### The Hodgkin-Huxley Model (1952)

The most literal attempt to "code the brain." A set of 4 nonlinear differential equations describing how a **single neuron** fires:

```
C_m (dV/dt) = -g_Na * m³h * (V - V_Na) - g_K * n⁴ * (V - V_K) - g_L * (V - V_L) + I_ext
dm/dt = α_m(V)(1-m) - β_m(V)m
dh/dt = α_h(V)(1-h) - β_h(V)h
dn/dt = α_n(V)(1-n) - β_n(V)n
```

Where:
- `V` = membrane potential
- `g_Na`, `g_K`, `g_L` = conductances for sodium, potassium, and leak channels
- `m`, `h`, `n` = gating variables (probability that ion channel gates are open)
- `α`, `β` = voltage-dependent rate functions
- `I_ext` = external current input

This models ion channels, membrane potential, sodium/potassium gates — for **one neuron**. The human brain has **86 billion** of these, each with ~7,000 synaptic connections (~100 trillion synapses total).

**Source**: [Hodgkin-Huxley Model — Wikipedia](https://en.wikipedia.org/wiki/Hodgkin%E2%80%93Huxley_model)

### The Blue Brain Project (EPFL)

Attempted to simulate at scale using an IBM Blue Gene supercomputer (22 trillion operations/second). Results:

- Simulated **100,000 neurons** of a single cortical column — that is 0.0001% of the brain
- Software: `NEURON` simulator (Michael Hines), `BBP-SDK` (C++, wrapped in Java/Python), `RT Neuron` (3D visualization)
- Discovered neural structures connected in up to **11 mathematical dimensions** using algebraic topology (2017)
- Released first digital 3D brain cell atlas: 737 brain regions with cell types, numbers, and positions (2018)

The computational and technological hurdles of simulating the full brain remain enormous.

**Sources**:
- [Blue Brain Project — EPFL](https://bluebrain.epfl.ch/)
- [Blue Brain Project — PMC Overview](https://pmc.ncbi.nlm.nih.gov/articles/PMC10767063/)

---

## 2. Cognitive Architectures: Modeling How Humans Think

Rather than simulating neurons, these model cognition at a higher level — reasoning, memory, decision-making.

### ACT-R (Adaptive Control of Thought-Rational)

Developed by John R. Anderson at Carnegie Mellon (1980s-present). Models the mind as interacting modules for declarative memory, procedural memory, perception, and motor control. Written in **Common Lisp**.

**Memory retrieval formula**:

```
Activation(i) = Base-level(i) + Σ W_j * S_ji + noise
```

Where:
- `Base-level(i)` decays with time (you forget things)
- `W_j` = attentional weight of source `j`
- `S_ji` = associative strength between source `j` and chunk `i`
- `noise` = stochastic variability in retrieval

**Critical property**: If activation falls below a threshold, retrieval **fails** — the system says "I don't know." This is the built-in abstention mechanism that LLMs lack (the thesis of Chapters 04-14 in this project).

**Source**: [ACT-R — Wikipedia](https://en.wikipedia.org/wiki/ACT-R)

### SOAR (State, Operator, And Result)

Developed at University of Michigan. Represents knowledge through goals, states, and operators.

**Key mechanism — Impasse detection**: When SOAR cannot proceed (no operator applies), it creates a **subgoal** to resolve the impasse. This is goal-directed problem-solving with explicit failure handling — fundamentally different from an LLM's forward pass that always produces output.

**Chunking**: SOAR compiles reasoning in substates into rules. Learned rules fire automatically in similar situations, converting complex reasoning into automatic/reactive processing — analogous to how humans shift from deliberate to automatic thinking.

**Sources**:
- [SOAR Cognitive Architecture — ArXiv](https://arxiv.org/pdf/2205.03854)
- [ACT-R vs SOAR Comparison — ArXiv](https://arxiv.org/abs/2201.09305)

### Relevance to This Project

ACT-R and SOAR have built-in **retrieval failure** and **impasse detection** — exactly the "I don't know" mechanism that Chapters 04-14 show LLMs lack. This is not accidental: cognitive architectures were designed to model human cognition, **including its limits**. LLMs were designed to predict the next token — abstention is not part of that objective.

---

## 3. Brain-Inspired Alternatives to Transformers

### Spiking Neural Networks (SNNs)

Neurons communicate via discrete **spikes** (like real neurons) rather than continuous activations. They are event-driven and vastly more energy-efficient.

- Up to **100x less energy** than traditional transformers
- **Spike-driven Transformer**: replaces softmax attention with spike-based mechanisms — achieves **linear complexity** (vs quadratic for standard attention)
- **Meta-SpikeFormer**: 80.0% top-1 accuracy on ImageNet-1K (55M params), surpassing SNN baselines by 3.7%
- State Space Model-based SNNs can **outperform Transformers** on long-range sequence benchmarks

**Sources**:
- [SpikingBrain Technical Report — ArXiv](https://arxiv.org/html/2509.05276v1)
- [Spike-driven Transformer — NeurIPS 2023](https://papers.neurips.cc/paper_files/paper/2023/file/ca0f5358dbadda74b3049711887e9ead-Paper-Conference.pdf)

### Numenta's Thousand Brains Theory

Based on the idea that each cortical column in the brain builds a **complete model of objects** using spatial reference frames.

**Key principles**:
- Each cortical column is a semi-independent **learning module** that can model entire objects
- Information is represented through spatially structured **reference frames**
- Thousands of columns work in parallel and **vote on consensus** — not a monolithic forward pass
- Open-source implementation: **"Monty"** (released 2024)

Built on Vernon Mountcastle's discovery that the neocortex is composed of repeating cortical columns — the brain's primary computational unit.

**Sources**:
- [Thousand Brains Project — ArXiv](https://arxiv.org/html/2412.18354v1)
- [Numenta Thousand Brains Theory](https://www.numenta.com/blog/2019/01/16/the-thousand-brains-theory-of-intelligence/)

---

## 4. Recent Hybrid Approaches (2025-2026)

### Hierarchical Reasoning Model (HRM)

Inspired by multi-timescale processing in the brain. A recurrent architecture that achieves significant computational depth while maintaining training stability and efficiency.

**Source**: [Hierarchical Reasoning Model — ArXiv](https://arxiv.org/abs/2506.21734)

### Brain-Inspired Agentic Architecture (Nature Communications, 2025)

Takes inspiration from specific brain regions responsible for:
- Conflict monitoring
- State prediction
- State evaluation
- Task decomposition
- Task coordination

Shows that planning in the brain is modular — different regions handle different aspects. This contrasts with LLMs where planning emerges implicitly from next-token prediction.

**Source**: [Brain-Inspired Agentic Architecture — Nature Communications 2025](https://www.nature.com/articles/s41467-025-63804-5)

### Cognitive LLMs (2025)

Hybrid decision-making architectures that embed ACT-R/SOAR-style cognitive processes into LLM adapter layers. A knowledge transfer mechanism extracts the cognitive architecture's internal decision-making process as latent neural representations and injects them into trainable LLM layers.

**Source**: [Cognitive LLMs — SAGE Journals 2025](https://journals.sagepub.com/doi/10.1177/29498732251377341)

### LLM-Brain Alignment (Nature Computational Science, 2025)

MIT researchers found that LLMs process diverse data modalities (languages, audio, images) similarly to how the human brain abstracts information in a generalized way. However, the underlying **mechanisms** remain fundamentally different — similar outputs, different processes.

**Sources**:
- [LLM-Brain Alignment — Nature Computational Science 2025](https://www.nature.com/articles/s43588-025-00863-0)
- [MIT: LLMs Reason Like Human Brains](https://news.mit.edu/2025/large-language-models-reason-about-diverse-data-general-way-0219)

### Transformer Emulating Human Mental States (2025)

A new transformer architecture that attempts to emulate imagination and higher-level human mental states, going beyond pattern matching toward more human-like cognitive processes.

**Source**: [Transformer Emulates Human Mental States — TechXplore](https://techxplore.com/news/2025-05-architecture-emulates-higher-human-mental.html)

---

## 5. The Complexity Gap — By the Numbers

| Aspect | Human Brain | GPT-4 (~SOTA LLM) | Ratio / Gap |
|--------|------------|-------------------|-------------|
| Processing units | 86 billion neurons | ~1.8 trillion parameters | Brain has ~100T synapses (100x more connections) |
| Connections per unit | ~7,000 synapses per neuron | 1 weight per connection | Each synapse is a complex biochemical system, not a single float |
| Energy consumption | 20 watts | Megawatts (training), kilowatts (inference) | Brain is ~1,000x more efficient |
| Structural dimensionality | Up to 11D topological structures | Flat matrix multiplications | Qualitatively different computation |
| Retrieval failure / "I don't know" | Built-in (ACT-R threshold, SOAR impasse) | None — always generates output | The central thesis of this project |
| Learning paradigm | Continuous, one-shot capable, sleep consolidation | Batch training on billions of examples | Fundamentally different |
| Metacognition | Can reason about own knowledge gaps | No introspection mechanism | Architectural gap, not scale gap |
| Modularity | Specialized brain regions + cortical columns | Monolithic forward pass | Brain is massively modular |

---

## 6. Connection to This Project's Thesis

This research directly reinforces the thesis traced across Chapters 00-14:

1. **Retrieval with failure detection** (ACT-R's activation threshold) — Chapter 04 shows attention retrieves perfectly but **cannot fail gracefully**. ACT-R's formula includes a threshold below which retrieval explicitly fails. No LLM architecture has this.

2. **Goal-directed search with impasse detection** (SOAR) — Chapter 13 shows that ToT, MCTS, and ReAct search for answers but converge on the "best hallucination." SOAR would detect an impasse ("no valid operator applies") and create a subgoal to handle it.

3. **World models in reference frames** (Thousand Brains) — LLMs have statistical patterns over token sequences, not spatial/conceptual models of what things are. The Thousand Brains theory argues intelligence requires reference frames — structured representations of objects and their properties.

4. **20 watts for 86 billion neurons** — Even tiny transformers need significant compute. The brain's efficiency suggests a fundamentally different computational substrate, not just a scaled-up version of matrix multiplication.

5. **The Hodgkin-Huxley gap** — A single biological neuron requires 4 coupled differential equations. An artificial neuron is `y = activation(Wx + b)`. The simplification is enormous, and what is lost in that simplification may include the mechanisms that enable metacognition and abstention.

The gap between LLMs and the human brain is not primarily about scale (more parameters, more data, more compute). It is about **architecture and mechanism**: the brain was evolved with metacognition, retrieval failure, world models, and goal-directed reasoning as first-class mechanisms. LLMs bolt reasoning capabilities on top of next-token prediction — which is exactly what 14 chapters of this project demonstrate.

---

## Additional Reading

- [LLMs and Cognitive Science: Comprehensive Review — ArXiv](https://arxiv.org/html/2409.02387v1)
- [Cognitive Architectures for Language Agents — ArXiv](https://arxiv.org/pdf/2309.02427)
- [Can Cognitive Architectures Enhance LLMs? — ArXiv](https://arxiv.org/pdf/2401.10444)
- [Brain-Inspired Concept Learning — ArXiv](https://arxiv.org/html/2401.06471v1)
- [Cognitive Computational Neuroscience — PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC6706072/)
- [Whole-Brain Network Models — Frontiers](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2022.866517/full)
- [System-Level Brain Modeling — Frontiers 2025](https://www.frontiersin.org/journals/computational-neuroscience/articles/10.3389/fncom.2025.1607239/full)
- [Brain vs GPT-4 Comparison](https://seifeur.com/human-brain-vs-gpt-4-comparison/)
