"""Validated data contracts shared by backends, judges, and future defenses."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

VoteLabel = Literal["safe", "jailbroken"]


class Generation(BaseModel):
    """One model generation and the metadata reported by its backend."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    latency_ms: float = Field(ge=0)


class JudgeResult(BaseModel):
    """A judge's jailbreak classification for one generation.

    ``score`` is judge-specific. A judge should document its scale and direction;
    ``None`` means that it only produces a binary decision.
    """

    model_config = ConfigDict(frozen=True)

    jailbroken: bool
    score: float | None = None
    reason: str | None = None

    @property
    def label(self) -> VoteLabel:
        """返回便于展示和计票的稳定标签。"""

        return "jailbroken" if self.jailbroken else "safe"


class CopyTrace(BaseModel):
    """一个扰动副本从输入到判定的完整执行记录。"""

    model_config = ConfigDict(frozen=True)

    copy_index: int = Field(ge=1)
    perturbed_prompt: str
    response: str
    model: str | None = None
    judge_result: JudgeResult
    latency_ms: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)


class DefenseResult(BaseModel):
    """固定预算或早停防御执行后的汇总结果。"""

    model_config = ConfigDict(frozen=True)

    response: str
    jailbroken: bool
    copies_used: int = Field(ge=1)
    votes: dict[VoteLabel, int] = Field(default_factory=dict)
    stopped_early: bool
    latency_ms: float = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    completion_tokens: int | None = Field(default=None, ge=0)
    trace: tuple[CopyTrace, ...] = ()
