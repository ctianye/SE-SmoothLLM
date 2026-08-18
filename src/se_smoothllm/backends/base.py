"""Abstract interface implemented by text-generation backends."""

from abc import ABC, abstractmethod

from se_smoothllm.models import Generation


class Backend(ABC):
    """Generate one response for one prompt."""

    @abstractmethod
    def generate(self, prompt: str) -> Generation:
        """Return the normalized generation result for ``prompt``."""


ModelBackend = Backend
