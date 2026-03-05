"""Planning loop for the toy human cognitive agent.

Implements: generate hypotheses -> test -> decide.
This is a simplified model of deliberate human reasoning.
"""

from dataclasses import dataclass


@dataclass
class Hypothesis:
    answer: str
    confidence: float
    reasoning: str


class Planner:
    """Simple hypothesis-test-decide loop.

    Unlike LLMs which produce a single forward pass, this models
    the human ability to consider multiple candidates and verify.
    """

    def __init__(self, verification_fn=None):
        self.verification_fn = verification_fn

    def generate_hypotheses(self, candidates: list[tuple[str, float, str]]) -> list[Hypothesis]:
        """Convert raw candidates into hypotheses.

        Args:
            candidates: list of (answer, confidence, reasoning) tuples
        """
        return [
            Hypothesis(answer=a, confidence=c, reasoning=r)
            for a, c, r in candidates
        ]

    def verify(self, hypothesis: Hypothesis, context: dict) -> Hypothesis:
        """Optional verification step — check if answer is consistent.

        This is what humans do when they "double-check" their answer.
        LLMs don't do this unless explicitly scaffolded.
        """
        if self.verification_fn:
            passed, new_conf = self.verification_fn(hypothesis.answer, context)
            if not passed:
                hypothesis.confidence *= 0.5
            else:
                hypothesis.confidence = min(1.0, hypothesis.confidence * 1.1)
        return hypothesis

    def decide(self, hypotheses: list[Hypothesis],
               uncertainty_threshold: float = 0.3) -> tuple[str | None, float, list[str]]:
        """Pick the best hypothesis, or abstain if uncertain.

        Returns:
            (answer_or_None, confidence, reasoning_trace)
        """
        if not hypotheses:
            return None, 0.0, ["No hypotheses generated"]

        # Sort by confidence
        ranked = sorted(hypotheses, key=lambda h: h.confidence, reverse=True)
        best = ranked[0]

        trace = [f"Considered {len(ranked)} hypotheses"]
        for h in ranked:
            trace.append(f"  - '{h.answer}' (conf={h.confidence:.2f}): {h.reasoning}")

        # Abstain if best confidence is too low
        if best.confidence < uncertainty_threshold:
            trace.append(f"Abstaining: best confidence {best.confidence:.2f} < threshold {uncertainty_threshold}")
            return None, best.confidence, trace

        trace.append(f"Selected: '{best.answer}' with confidence {best.confidence:.2f}")
        return best.answer, best.confidence, trace
