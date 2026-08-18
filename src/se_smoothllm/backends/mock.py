"""Deterministic backend for tests and examples."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from se_smoothllm.backends.base import Backend
from se_smoothllm.models import Generation

ExhaustionPolicy = Literal["raise", "cycle", "repeat_last"]
MockResponse: TypeAlias = str | Generation


@dataclass(slots=True)
class MockBackend(Backend):
    """Return configured generations and record every prompt received.

    The default ``raise`` exhaustion policy makes unexpected model calls fail
    immediately. ``cycle`` and ``repeat_last`` are available only when a test
    explicitly needs an unlimited deterministic backend.
    """

    responses: MockResponse | Sequence[MockResponse] = "I cannot help with that request."
    exhaustion: ExhaustionPolicy = "raise"
    received_prompts: list[str] = field(default_factory=list, init=False)
    returned_generations: list[Generation] = field(default_factory=list, init=False)
    _items: tuple[Generation, ...] = field(init=False, repr=False)
    _index: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        configured = (
            (self.responses,)
            if isinstance(self.responses, (str, Generation))
            else tuple(self.responses)
        )
        if not configured:
            raise ValueError("responses must contain at least one item")
        if self.exhaustion not in {"raise", "cycle", "repeat_last"}:
            raise ValueError(f"unsupported exhaustion policy: {self.exhaustion}")
        self._items = tuple(_normalize_response(item) for item in configured)

    @property
    def call_count(self) -> int:
        """Number of generation attempts, including an exhausted call that failed."""

        return len(self.received_prompts)

    @property
    def successful_call_count(self) -> int:
        """Number of generations successfully returned."""

        return len(self.returned_generations)

    @property
    def remaining_responses(self) -> int:
        """Number of configured responses not yet consumed."""

        return max(len(self._items) - self._index, 0)

    def generate(self, prompt: str) -> Generation:
        self.received_prompts.append(prompt)
        generation = self._next_generation()
        self.returned_generations.append(generation)
        return generation

    def reset(self) -> None:
        """Rewind configured responses and clear the recorded trace."""

        self._index = 0
        self.received_prompts.clear()
        self.returned_generations.clear()

    def _next_generation(self) -> Generation:
        if self._index < len(self._items):
            generation = self._items[self._index]
        elif self.exhaustion == "cycle":
            generation = self._items[self._index % len(self._items)]
        elif self.exhaustion == "repeat_last":
            generation = self._items[-1]
        else:
            raise RuntimeError(
                f"MockBackend exhausted after {len(self._items)} configured responses"
            )

        self._index += 1
        return generation


def _normalize_response(response: MockResponse) -> Generation:
    if isinstance(response, Generation):
        return response
    if isinstance(response, str):
        return Generation(text=response, latency_ms=0.0)
    raise TypeError("mock responses must be strings or Generation instances")
