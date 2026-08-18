"""可复现的提示词字符扰动接口与实现。"""

from __future__ import annotations

import math
import random
import string
from abc import ABC, abstractmethod
from dataclasses import dataclass


class Perturbation(ABC):
    """使用调用方提供的独立随机数生成器修改输入文本。"""

    @abstractmethod
    def apply(self, text: str, *, rng: random.Random) -> str:
        """返回一份新的扰动文本。"""


class IdentityPerturbation(Perturbation):
    """保持输入不变，主要用于接口测试和后端调试。"""

    def apply(self, text: str, *, rng: random.Random) -> str:
        del rng
        return text


@dataclass(frozen=True, slots=True)
class _RandomCharacterPerturbation(Perturbation):
    """三种字符扰动共享的百分比和字符表配置。"""

    q: float
    alphabet: str = string.printable

    def __post_init__(self) -> None:
        if isinstance(self.q, bool) or not isinstance(self.q, (int, float)):
            raise TypeError("q must be a numeric percentage")
        if not math.isfinite(self.q) or not 0 <= self.q <= 100:
            raise ValueError("q must be between 0 and 100")
        if not self.alphabet:
            raise ValueError("alphabet must not be empty")

    def _change_count(self, text: str) -> int:
        return int(len(text) * self.q / 100)

    @staticmethod
    def _require_rng(rng: random.Random) -> None:
        if not isinstance(rng, random.Random):
            raise TypeError("rng must be an instance of random.Random")


class RandomSwapPerturbation(_RandomCharacterPerturbation):
    """随机选择约 ``q%`` 的位置，并将这些字符替换为随机字符。"""

    def apply(self, text: str, *, rng: random.Random) -> str:
        self._require_rng(rng)
        change_count = self._change_count(text)
        if change_count == 0:
            return text

        characters = list(text)
        sampled_indices = rng.sample(range(len(characters)), change_count)
        for index in sampled_indices:
            characters[index] = rng.choice(self.alphabet)
        return "".join(characters)


class RandomPatchPerturbation(_RandomCharacterPerturbation):
    """将一个长度约为原文本 ``q%`` 的连续片段替换为随机字符。"""

    def apply(self, text: str, *, rng: random.Random) -> str:
        self._require_rng(rng)
        patch_width = self._change_count(text)
        if patch_width == 0:
            return text

        characters = list(text)
        start_index = rng.randint(0, len(characters) - patch_width)
        patch = [rng.choice(self.alphabet) for _ in range(patch_width)]
        characters[start_index : start_index + patch_width] = patch
        return "".join(characters)


class RandomInsertPerturbation(_RandomCharacterPerturbation):
    """在随机位置插入数量约为原文本 ``q%`` 的随机字符。"""

    def apply(self, text: str, *, rng: random.Random) -> str:
        self._require_rng(rng)
        insert_count = self._change_count(text)
        if insert_count == 0:
            return text

        characters = list(text)
        sampled_indices = rng.sample(range(len(characters)), insert_count)
        for index in sampled_indices:
            characters.insert(index, rng.choice(self.alphabet))
        return "".join(characters)
