"""Memory systems for the toy human cognitive agent."""

from dataclasses import dataclass, field
from collections import OrderedDict


@dataclass
class MemoryEntry:
    key: str
    value: str
    source: str = "observation"
    confidence: float = 1.0


class WorkingMemory:
    """Small, structured, in-task scratch space.

    Like human working memory: limited capacity, structured slots,
    actively maintained during a task.
    """

    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.slots: OrderedDict[str, str] = OrderedDict()

    def store(self, key: str, value: str) -> None:
        if len(self.slots) >= self.capacity and key not in self.slots:
            # Evict oldest item (like human working memory decay)
            self.slots.popitem(last=False)
        self.slots[key] = value
        # Move to end (recently accessed = more available)
        self.slots.move_to_end(key)

    def retrieve(self, key: str) -> str | None:
        if key in self.slots:
            self.slots.move_to_end(key)
            return self.slots[key]
        return None

    def search(self, query: str) -> list[tuple[str, str]]:
        """Simple keyword search across all slots."""
        results = []
        query_lower = query.lower()
        for k, v in self.slots.items():
            if query_lower in k.lower() or query_lower in v.lower():
                results.append((k, v))
        return results

    def clear(self) -> None:
        self.slots.clear()

    @property
    def contents(self) -> dict[str, str]:
        return dict(self.slots)


class LongTermMemory:
    """Persistent key-value store — survives across "sessions".

    Unlike LLM context, this persists. Unlike human memory,
    it's perfect (no decay). That's intentional: it shows the
    structural advantage, not biological accuracy.
    """

    def __init__(self):
        self.store: dict[str, MemoryEntry] = {}

    def remember(self, key: str, value: str, source: str = "observation",
                 confidence: float = 1.0) -> None:
        self.store[key] = MemoryEntry(
            key=key, value=value, source=source, confidence=confidence
        )

    def recall(self, key: str) -> MemoryEntry | None:
        return self.store.get(key)

    def search(self, query: str) -> list[MemoryEntry]:
        """Keyword search across all stored memories."""
        query_lower = query.lower()
        results = []
        for entry in self.store.values():
            if (query_lower in entry.key.lower() or
                    query_lower in entry.value.lower()):
                results.append(entry)
        return results

    def forget(self, key: str) -> None:
        self.store.pop(key, None)

    def clear(self) -> None:
        self.store.clear()

    @property
    def size(self) -> int:
        return len(self.store)
