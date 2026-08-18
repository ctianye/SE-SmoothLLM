"""Public package interface for SE-SmoothLLM."""

from importlib.metadata import PackageNotFoundError, version

from se_smoothllm.core import DefenseComponents, SmoothGuard
from se_smoothllm.judges import PrefixJudge
from se_smoothllm.models import CopyTrace, DefenseResult, Generation, JudgeResult
from se_smoothllm.perturbations import (
    IdentityPerturbation,
    Perturbation,
    RandomInsertPerturbation,
    RandomPatchPerturbation,
    RandomSwapPerturbation,
)
from se_smoothllm.stopping import locked_vote

try:
    __version__ = version("se-smoothllm")
except PackageNotFoundError:  # pragma: no cover - only used from an unpackaged source tree
    __version__ = "0.1.0"

__all__ = [
    "CopyTrace",
    "DefenseComponents",
    "DefenseResult",
    "Generation",
    "IdentityPerturbation",
    "JudgeResult",
    "locked_vote",
    "Perturbation",
    "RandomInsertPerturbation",
    "RandomPatchPerturbation",
    "RandomSwapPerturbation",
    "PrefixJudge",
    "SmoothGuard",
    "__version__",
]
