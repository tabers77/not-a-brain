"""Toy human cognitive agent.

NOT a brain simulation. A didactic model that captures key structural
differences between human cognition and LLM processing:
- Persistent memory across sessions
- Uncertainty-aware abstention
- Grounding via observations
- Deliberate planning (hypothesis -> verify -> decide)

The point: show that humans solve tasks via fundamentally different
mechanisms than next-token prediction.
"""

import re
from not_a_brain.evals.harness import AgentInterface, AgentResult
from not_a_brain.human_agent.memory import WorkingMemory, LongTermMemory
from not_a_brain.human_agent.planner import Planner, Hypothesis
from not_a_brain.human_agent.grounding import GroundingChannel


class HumanAgent(AgentInterface):
    """Toy human-like cognitive agent.

    Solves tasks using:
    1. Parse the prompt and extract task structure
    2. Check long-term memory for relevant knowledge
    3. Use grounding observations if available
    4. Generate candidate answers via rule-based reasoning
    5. Estimate confidence
    6. Abstain if uncertain
    7. Store new facts in long-term memory
    """

    def __init__(self, uncertainty_threshold: float = 0.3):
        self.working_memory = WorkingMemory(capacity=7)
        self.long_term_memory = LongTermMemory()
        self.grounding = GroundingChannel()
        self.planner = Planner()
        self.uncertainty_threshold = uncertainty_threshold

    @property
    def name(self) -> str:
        return "human_agent"

    def run(self, prompt: str) -> AgentResult:
        """Process a task prompt through the cognitive pipeline."""
        self.working_memory.clear()
        trace = [f"Received: {prompt[:80]}..."]

        # 1. Parse the prompt to identify task type
        task_type = self._identify_task(prompt)
        trace.append(f"Identified task type: {task_type}")

        # 2. Solve based on task type
        if task_type == "arithmetic":
            return self._solve_arithmetic(prompt, trace)
        elif task_type == "copy":
            return self._solve_copy(prompt, trace)
        elif task_type == "grammar":
            return self._solve_grammar(prompt, trace)
        elif task_type == "knowledge_qa":
            return self._solve_knowledge_qa(prompt, trace)
        elif task_type == "compositional":
            return self._solve_compositional(prompt, trace)
        else:
            return self._solve_unknown(prompt, trace)

    def _identify_task(self, prompt: str) -> str:
        prompt_upper = prompt.upper()
        if any(prompt_upper.startswith(op) for op in ("ADD ", "SUB ", "MUL ")):
            return "arithmetic"
        if prompt_upper.startswith("COPY:"):
            return "copy"
        if prompt_upper.startswith("CHECK:"):
            return "grammar"
        if "FACT:" in prompt_upper and "Q:" in prompt_upper:
            return "knowledge_qa"
        if prompt_upper.startswith("APPLY "):
            return "compositional"
        return "unknown"

    def _solve_arithmetic(self, prompt: str, trace: list[str]) -> AgentResult:
        """Humans can do arithmetic deliberately — step by step."""
        match = re.match(r"(ADD|SUB|MUL)\s+(\d+)\s+(\d+)\s*=", prompt)
        if not match:
            trace.append("Failed to parse arithmetic prompt")
            return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

        op, a_str, b_str = match.groups()
        a, b = int(a_str), int(b_str)
        self.working_memory.store("operand_a", str(a))
        self.working_memory.store("operand_b", str(b))
        self.working_memory.store("operation", op)

        if op == "ADD":
            result = a + b
            trace.append(f"Computing {a} + {b} = {result}")
        elif op == "SUB":
            result = a - b
            trace.append(f"Computing {a} - {b} = {result}")
        elif op == "MUL":
            result = a * b
            trace.append(f"Computing {a} * {b} = {result}")
        else:
            return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

        return AgentResult(answer=str(result), confidence=0.99, trace=trace)

    def _solve_copy(self, prompt: str, trace: list[str]) -> AgentResult:
        """Copy task: extract and reproduce the sequence."""
        match = re.match(r"COPY:\s*(.+)\|", prompt)
        if not match:
            trace.append("Failed to parse copy prompt")
            return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

        seq = match.group(1).strip()
        self.working_memory.store("sequence", seq)
        trace.append(f"Stored sequence in working memory: '{seq}'")
        trace.append(f"Reproducing from working memory: '{seq}'")
        return AgentResult(answer=seq, confidence=0.95, trace=trace)

    def _solve_grammar(self, prompt: str, trace: list[str]) -> AgentResult:
        """Bracket matching: use a stack (deliberate algorithm)."""
        match = re.match(r"CHECK:\s*(.+)", prompt)
        if not match:
            return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

        seq = match.group(1).strip()
        trace.append(f"Checking bracket sequence: {seq}")

        stack = []
        close_to_open = {")": "(", "]": "[", "}": "{"}
        opens = {"(", "[", "{"}

        for token in seq.split():
            if token in opens:
                stack.append(token)
                trace.append(f"  Push '{token}' -> stack: {stack}")
            elif token in close_to_open:
                if not stack or stack[-1] != close_to_open[token]:
                    trace.append(f"  Mismatch: '{token}' vs stack top '{stack[-1] if stack else 'empty'}'")
                    return AgentResult(answer="invalid", confidence=0.95, trace=trace)
                stack.pop()
                trace.append(f"  Pop for '{token}' -> stack: {stack}")

        is_valid = len(stack) == 0
        answer = "valid" if is_valid else "invalid"
        trace.append(f"Final stack: {stack} -> {answer}")
        return AgentResult(answer=answer, confidence=0.95, trace=trace)

    def _solve_knowledge_qa(self, prompt: str, trace: list[str]) -> AgentResult:
        """Extract facts from context, answer from them."""
        # Parse facts and question
        fact_match = re.search(r"FACT:\s*(.+?)\.\s*Q:", prompt)
        q_match = re.search(r"Q:\s*(.+?)(?:\?|$)", prompt)

        if not fact_match or not q_match:
            return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

        facts_str = fact_match.group(1)
        question = q_match.group(1).strip().lower()
        trace.append(f"Question: {question}")

        # Parse individual facts into working memory
        for fact in facts_str.split("."):
            fact = fact.strip()
            if " is " in fact:
                parts = fact.split(" is ", 1)
                key = parts[1].strip().lower()
                value = parts[0].strip().lower()
                self.working_memory.store(key, value)
                self.long_term_memory.remember(key, value, source="context")
                trace.append(f"  Stored fact: '{key}' -> '{value}'")

        # Search working memory for answer
        results = self.working_memory.search(question)
        if results:
            answer = results[0][1]
            trace.append(f"Found in working memory: '{answer}'")
            return AgentResult(answer=answer, confidence=0.9, trace=trace)

        # Check long-term memory
        lt_results = self.long_term_memory.search(question)
        if lt_results:
            answer = lt_results[0].value
            trace.append(f"Found in long-term memory: '{answer}'")
            return AgentResult(answer=answer, confidence=0.8, trace=trace)

        trace.append("No matching fact found — abstaining")
        return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

    def _solve_compositional(self, prompt: str, trace: list[str]) -> AgentResult:
        """Apply chained operations deliberately."""
        match = re.match(r"APPLY\s+(.+?)\s+TO\s+(\w+)", prompt)
        if not match:
            return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

        ops_str = match.group(1)
        word = match.group(2)
        ops = [op.strip().lower() for op in ops_str.split("THEN")]
        trace.append(f"Input: '{word}', Operations: {ops}")

        OPS = {
            "reverse": lambda s: s[::-1],
            "uppercase": lambda s: s.upper(),
            "lowercase": lambda s: s.lower(),
            "first3": lambda s: s[:3] if len(s) >= 3 else s,
            "last3": lambda s: s[-3:] if len(s) >= 3 else s,
            "double": lambda s: s + s,
            "sort": lambda s: "".join(sorted(s)),
        }

        result = word
        for op in ops:
            if op in OPS:
                prev = result
                result = OPS[op](result)
                trace.append(f"  {op}('{prev}') -> '{result}'")
            else:
                trace.append(f"  Unknown operation '{op}' — abstaining")
                return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)

        return AgentResult(answer=result, confidence=0.95, trace=trace)

    def _solve_unknown(self, prompt: str, trace: list[str]) -> AgentResult:
        """When task type is unrecognized or question is unanswerable: abstain.

        This is a key human behavior — saying "I don't know" when uncertain.
        """
        # Check if there's a question we might answer from long-term memory
        q_match = re.search(r"Q:\s*(.+?)(?:\?|$)", prompt)
        if q_match:
            question = q_match.group(1).strip().lower()
            lt_results = self.long_term_memory.search(question)
            if lt_results:
                answer = lt_results[0].value
                trace.append(f"Found in long-term memory: '{answer}'")
                return AgentResult(answer=answer, confidence=0.6, trace=trace)

        trace.append("Insufficient information — abstaining (humans say 'I don't know')")
        return AgentResult(answer="unknown", confidence=0.1, trace=trace, abstained=True)
