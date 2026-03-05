"""Tests for the toy human cognitive agent."""

from not_a_brain.human_agent.agent import HumanAgent
from not_a_brain.human_agent.memory import WorkingMemory, LongTermMemory
from not_a_brain.tasks import (
    ArithmeticTask, CopyTask, GrammarTask,
    KnowledgeQATask, CompositionalTask, UnknownTask,
)


class TestWorkingMemory:
    def test_store_and_retrieve(self):
        wm = WorkingMemory(capacity=3)
        wm.store("a", "1")
        assert wm.retrieve("a") == "1"

    def test_capacity_eviction(self):
        wm = WorkingMemory(capacity=2)
        wm.store("a", "1")
        wm.store("b", "2")
        wm.store("c", "3")
        # "a" should be evicted (oldest)
        assert wm.retrieve("a") is None
        assert wm.retrieve("b") == "2"
        assert wm.retrieve("c") == "3"

    def test_search(self):
        wm = WorkingMemory()
        wm.store("capital of france", "paris")
        wm.store("capital of japan", "tokyo")
        results = wm.search("france")
        assert len(results) == 1
        assert results[0][1] == "paris"


class TestLongTermMemory:
    def test_remember_and_recall(self):
        ltm = LongTermMemory()
        ltm.remember("capital of france", "paris")
        entry = ltm.recall("capital of france")
        assert entry is not None
        assert entry.value == "paris"

    def test_persistence(self):
        ltm = LongTermMemory()
        ltm.remember("key", "value1")
        ltm.remember("key", "value2")
        assert ltm.recall("key").value == "value2"

    def test_search(self):
        ltm = LongTermMemory()
        ltm.remember("fact1", "paris is capital of france")
        ltm.remember("fact2", "tokyo is capital of japan")
        results = ltm.search("france")
        assert len(results) == 1


class TestHumanAgent:
    def setup_method(self):
        self.agent = HumanAgent()

    def test_arithmetic(self):
        result = self.agent.run("ADD 12 37 =")
        assert result.answer == "49"
        assert result.confidence > 0.9
        assert not result.abstained

    def test_arithmetic_sub(self):
        result = self.agent.run("SUB 50 23 =")
        assert result.answer == "27"

    def test_arithmetic_mul(self):
        result = self.agent.run("MUL 6 7 =")
        assert result.answer == "42"

    def test_copy(self):
        result = self.agent.run("COPY: abcdef|")
        assert result.answer == "abcdef"
        assert not result.abstained

    def test_grammar_valid(self):
        result = self.agent.run("CHECK: ( [ { } ] )")
        assert result.answer == "valid"

    def test_grammar_invalid(self):
        result = self.agent.run("CHECK: ( [ )")
        assert result.answer == "invalid"

    def test_knowledge_qa(self):
        result = self.agent.run("FACT: paris is capital of france. Q: capital of france?")
        assert result.answer == "paris"
        assert not result.abstained

    def test_compositional_reverse(self):
        result = self.agent.run("APPLY reverse TO hello")
        assert result.answer == "olleh"

    def test_compositional_chained(self):
        result = self.agent.run("APPLY reverse THEN uppercase TO hello")
        assert result.answer == "OLLEH"

    def test_unknown_abstains(self):
        result = self.agent.run("Q: What is the population of Atlantis?")
        assert result.abstained
        assert result.confidence < 0.3

    def test_unknown_with_context_abstains(self):
        result = self.agent.run(
            "CONTEXT: The weather is sunny. Q: What is the capital of the Moon?"
        )
        assert result.abstained

    def test_long_term_memory_persists(self):
        """Key difference from LLMs: memory persists across calls."""
        agent = HumanAgent()
        # First call stores facts
        agent.run("FACT: paris is capital of france. Q: capital of france?")
        # Second call should still know from long-term memory
        result = agent.run("Q: capital of france?")
        # Should find it in long-term memory (not abstain)
        assert result.answer == "paris" or not result.abstained

    def test_traces_are_recorded(self):
        result = self.agent.run("ADD 1 2 =")
        assert result.trace is not None
        assert len(result.trace) > 0


class TestHumanAgentOnTasks:
    """Run the human agent on actual task instances."""

    def test_arithmetic_task(self):
        agent = HumanAgent()
        task = ArithmeticTask(seed=42)
        correct = 0
        n = 20
        for _ in range(n):
            sample = task.generate()
            result = agent.run(sample.prompt)
            grade = task.grade(result.answer, sample)
            if grade.correct:
                correct += 1
        # Human agent should get 100% on arithmetic
        assert correct == n

    def test_copy_task(self):
        agent = HumanAgent()
        task = CopyTask(seed=42)
        correct = 0
        n = 20
        for _ in range(n):
            sample = task.generate()
            result = agent.run(sample.prompt)
            grade = task.grade(result.answer, sample)
            if grade.correct:
                correct += 1
        assert correct == n

    def test_unknown_task_abstains(self):
        agent = HumanAgent()
        task = UnknownTask(seed=42)
        correct = 0
        n = 10
        for _ in range(n):
            sample = task.generate()
            result = agent.run(sample.prompt)
            grade = task.grade(result.answer, sample)
            if grade.correct:
                correct += 1
        # Human agent should abstain (= correct) on most unknowns
        assert correct >= 8
