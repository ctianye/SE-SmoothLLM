"""与执行流程解耦的精确早停判定。"""

from se_smoothllm.models import VoteLabel


def locked_vote(
    safe: int,
    jailbroken: int,
    remaining: int,
) -> VoteLabel | None:
    """返回已经锁定的最终投票结果，否则返回 ``None``。

    SmoothLLM 只有在越狱票数严格超过总票数一半时才判定越狱，平票
    归为安全。因此越狱锁定使用严格大于号，安全锁定包含等号。
    """

    _validate_vote_count("safe", safe)
    _validate_vote_count("jailbroken", jailbroken)
    _validate_vote_count("remaining", remaining)

    if jailbroken > safe + remaining:
        return "jailbroken"
    if safe >= jailbroken + remaining:
        return "safe"
    return None


def _validate_vote_count(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value < 0:
        raise ValueError(f"{name} must be non-negative")
