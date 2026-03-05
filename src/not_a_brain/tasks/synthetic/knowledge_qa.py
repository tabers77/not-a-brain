"""Knowledge QA task: answer questions from facts provided in context."""

from not_a_brain.tasks.base import TaskBase, TaskSample

# Small, controlled fact set — enough to test memory/retrieval behaviors
FACTS = [
    ("paris", "capital of france", "paris"),
    ("tokyo", "capital of japan", "tokyo"),
    ("berlin", "capital of germany", "berlin"),
    ("canberra", "capital of australia", "canberra"),
    ("ottawa", "capital of canada", "ottawa"),
    ("rome", "capital of italy", "rome"),
    ("madrid", "capital of spain", "madrid"),
    ("lisbon", "capital of portugal", "lisbon"),
    ("water", "chemical formula of water", "H2O"),
    ("oxygen", "chemical formula of oxygen", "O2"),
    ("earth", "third planet from the sun", "earth"),
    ("mars", "fourth planet from the sun", "mars"),
    ("jupiter", "largest planet in solar system", "jupiter"),
    ("7", "number of continents", "7"),
    ("5", "number of oceans", "5"),
    ("python", "language created by guido van rossum", "python"),
    ("linux", "kernel created by linus torvalds", "linux"),
    ("1969", "year of first moon landing", "1969"),
    ("1989", "year the berlin wall fell", "1989"),
    ("blue", "color of the sky on a clear day", "blue"),
]


class KnowledgeQATask(TaskBase):
    """Generates fact-based QA with context.

    Examples:
        "FACT: paris is capital of france. Q: capital of france?" -> "paris"

    The fact is always in the prompt (testing context use, not memorization).
    Distractor facts can be added to test attention/retrieval.
    """

    def __init__(self, seed: int | None = None, n_distractors: int = 0):
        super().__init__(seed)
        self.n_distractors = n_distractors
        self.facts = FACTS

    @property
    def name(self) -> str:
        return "knowledge_qa"

    @property
    def description(self) -> str:
        return f"Fact QA from context ({self.n_distractors} distractors)"

    def _generate(self) -> TaskSample:
        # Pick the target fact
        idx = self.rng.randint(0, len(self.facts) - 1)
        answer, question_text, _ = self.facts[idx]

        # Build context with optional distractors
        fact_str = f"{answer} is {question_text}"
        context_parts = [fact_str]

        if self.n_distractors > 0:
            other_indices = [i for i in range(len(self.facts)) if i != idx]
            self.rng.shuffle(other_indices)
            for d_idx in other_indices[:self.n_distractors]:
                d_answer, d_question, _ = self.facts[d_idx]
                context_parts.append(f"{d_answer} is {d_question}")

        # Shuffle context order so target fact isn't always first
        self.rng.shuffle(context_parts)
        context = ". ".join(context_parts)

        prompt = f"FACT: {context}. Q: {question_text}?"
        return TaskSample(
            prompt=prompt,
            expected=answer,
            metadata={
                "target_fact_idx": idx,
                "n_distractors": self.n_distractors,
                "question": question_text,
            },
        )
