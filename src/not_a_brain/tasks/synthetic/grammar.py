"""Grammar task: validate bracket sequences."""

from not_a_brain.tasks.base import TaskBase, TaskSample


class GrammarTask(TaskBase):
    """Generates bracket matching tasks.

    Examples:
        "CHECK: ( [ { } ] )" -> "valid"
        "CHECK: ( [ )"       -> "invalid"

    Tests: structural/hierarchical pattern recognition.
    """

    BRACKETS = [("(", ")"), ("[", "]"), ("{", "}")]

    def __init__(self, seed: int | None = None, max_depth: int = 3,
                 max_pairs: int = 4, invalid_ratio: float = 0.5):
        super().__init__(seed)
        self.max_depth = max_depth
        self.max_pairs = max_pairs
        self.invalid_ratio = invalid_ratio

    @property
    def name(self) -> str:
        return "grammar"

    @property
    def description(self) -> str:
        return f"Bracket matching (depth<={self.max_depth}, pairs<={self.max_pairs})"

    def _generate_valid(self) -> str:
        """Generate a valid bracket sequence recursively."""
        return self._gen_nested(depth=0)

    def _gen_nested(self, depth: int) -> str:
        if depth >= self.max_depth or self.rng.random() < 0.3:
            return ""
        open_b, close_b = self.rng.choice(self.BRACKETS)
        inner = self._gen_nested(depth + 1)
        sibling = self._gen_nested(depth) if self.rng.random() < 0.5 else ""
        parts = [open_b]
        if inner:
            parts.append(inner)
        parts.append(close_b)
        if sibling:
            parts.append(sibling)
        return " ".join(parts)

    def _corrupt(self, seq: str) -> str:
        """Make a valid sequence invalid by one of several mutations."""
        tokens = seq.split()
        if len(tokens) < 2:
            return "( ]"

        mutation = self.rng.choice(["swap", "delete", "insert"])

        if mutation == "swap":
            i = self.rng.randint(0, len(tokens) - 1)
            all_brackets = [b for pair in self.BRACKETS for b in pair]
            tokens[i] = self.rng.choice(all_brackets)
        elif mutation == "delete":
            i = self.rng.randint(0, len(tokens) - 1)
            tokens.pop(i)
        elif mutation == "insert":
            all_brackets = [b for pair in self.BRACKETS for b in pair]
            i = self.rng.randint(0, len(tokens))
            tokens.insert(i, self.rng.choice(all_brackets))

        result = " ".join(tokens)
        # Verify it's actually invalid
        if self._is_valid(result):
            return "( ]"
        return result

    def _is_valid(self, seq: str) -> bool:
        stack = []
        close_to_open = {c: o for o, c in self.BRACKETS}
        for token in seq.split():
            if token in {o for o, _ in self.BRACKETS}:
                stack.append(token)
            elif token in close_to_open:
                if not stack or stack[-1] != close_to_open[token]:
                    return False
                stack.pop()
        return len(stack) == 0

    def _generate(self) -> TaskSample:
        make_invalid = self.rng.random() < self.invalid_ratio

        valid_seq = self._generate_valid()
        if not valid_seq:
            valid_seq = "( )"

        if make_invalid:
            seq = self._corrupt(valid_seq)
            expected = "invalid"
        else:
            seq = valid_seq
            expected = "valid"

        prompt = f"CHECK: {seq}"
        return TaskSample(
            prompt=prompt,
            expected=expected,
            metadata={"sequence": seq, "is_valid": expected == "valid"},
        )
