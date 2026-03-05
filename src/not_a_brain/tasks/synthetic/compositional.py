"""Compositional task: apply chained string operations."""

from not_a_brain.tasks.base import TaskBase, TaskSample, GradeResult

# Operations that can be composed
OPERATIONS = {
    "reverse": lambda s: s[::-1],
    "uppercase": lambda s: s.upper(),
    "lowercase": lambda s: s.lower(),
    "first3": lambda s: s[:3] if len(s) >= 3 else s,
    "last3": lambda s: s[-3:] if len(s) >= 3 else s,
    "double": lambda s: s + s,
    "sort": lambda s: "".join(sorted(s)),
}

WORDS = [
    "hello", "world", "brain", "model", "token", "query",
    "learn", "train", "think", "magic", "alpha", "delta",
    "gamma", "sigma", "omega", "robot", "agent", "stack",
]


class CompositionalTask(TaskBase):
    """Generates chained string transformation tasks.

    Examples:
        "APPLY reverse TO hello"           -> "olleh"
        "APPLY reverse THEN uppercase TO hello" -> "OLLEH"

    Tests: compositional generalization — can the model chain operations?
    """

    def __init__(self, seed: int | None = None, max_ops: int = 2):
        super().__init__(seed)
        self.max_ops = max_ops

    @property
    def name(self) -> str:
        return "compositional"

    @property
    def description(self) -> str:
        return f"Chained string ops (up to {self.max_ops} operations)"

    def _generate(self) -> TaskSample:
        n_ops = self.rng.randint(1, self.max_ops)
        op_names = self.rng.sample(list(OPERATIONS.keys()), n_ops)
        word = self.rng.choice(WORDS)

        # Apply operations in sequence
        result = word
        for op_name in op_names:
            result = OPERATIONS[op_name](result)

        # Build prompt
        ops_str = " THEN ".join(op_names)
        prompt = f"APPLY {ops_str} TO {word}"

        return TaskSample(
            prompt=prompt,
            expected=result,
            metadata={"operations": op_names, "input_word": word},
        )

    def _grade(self, answer: str, sample: TaskSample) -> GradeResult:
        correct = answer == sample.expected
        # Partial credit: fraction of characters correct
        score = 1.0 if correct else 0.0
        if not correct and len(sample.expected) > 0:
            matches = sum(a == b for a, b in zip(answer, sample.expected))
            score = matches / max(len(answer), len(sample.expected))
        return GradeResult(correct=correct, score=score, expected=sample.expected)
