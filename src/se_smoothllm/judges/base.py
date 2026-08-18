"""Abstract interface implemented by response judges."""

from abc import ABC, abstractmethod

from se_smoothllm.models import Generation, JudgeResult


class Judge(ABC):
    """Classify whether one prompt-generation pair is a successful jailbreak."""

    @abstractmethod
    def classify(self, prompt: str, generation: Generation) -> JudgeResult:
        """Return a normalized jailbreak classification for one prompt and response."""


ResponseJudge = Judge
