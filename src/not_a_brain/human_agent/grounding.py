"""Grounding channel for the toy human cognitive agent.

Humans tie words to perception — they can verify claims against
observations. LLMs have no such channel unless explicitly given tools.
"""

from dataclasses import dataclass


@dataclass
class Observation:
    content: str
    source: str = "direct"
    trusted: bool = True


class GroundingChannel:
    """Provides trusted observations that the agent can reference.

    This models the human ability to check claims against reality.
    When an observation is available, the agent treats it as ground truth.
    When absent, the agent must rely on memory/reasoning alone.
    """

    def __init__(self):
        self.observations: list[Observation] = []

    def observe(self, content: str, source: str = "direct",
                trusted: bool = True) -> None:
        self.observations.append(
            Observation(content=content, source=source, trusted=trusted)
        )

    def get_relevant(self, query: str) -> list[Observation]:
        """Find observations relevant to a query."""
        query_lower = query.lower()
        return [
            obs for obs in self.observations
            if query_lower in obs.content.lower()
        ]

    def has_grounding_for(self, query: str) -> bool:
        return len(self.get_relevant(query)) > 0

    def clear(self) -> None:
        self.observations.clear()
