"""OCR 抽象层。

技术方案第二十、二十一、三十二节的核心要求：把 OCR 做成可替换组件。

原因是第 32 节那条建议——永远保存 crop 和中间产物。以后换模型（v5 → v6、
通用 → 手写微调）时，不需要重跑渲染/配准/差分/ROI，只要拿现成的 crop 重新
识别即可。所以这里的接口边界划在"输入一张 crop 图，输出文本"这一层。
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


@dataclass
class OcrLine:
    """一行识别结果。"""

    text: str
    confidence: float
    #: 相对 crop 的局部坐标 [x1, y1, x2, y2]；rec_only 模式下可能为 None。
    bbox: tuple[int, int, int, int] | None = None
    #: 该行与手写 mask 的重叠比例，用于过滤残留印刷内容。
    mask_ratio: float = 1.0
    dropped: bool = False
    drop_reason: str = ""


@dataclass
class OcrBatchResult:
    lines: list[OcrLine] = field(default_factory=list)
    elapsed: float = 0.0
    engine: str = ""
    model: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "".join(l.text for l in self.lines if not l.dropped)

    @property
    def kept(self) -> list[OcrLine]:
        return [l for l in self.lines if not l.dropped]

    @property
    def confidence(self) -> float:
        """整块答案的置信度：取各行最小值，而不是平均值。

        平均值会掩盖"一行很好一行很糟"的情况，而后者恰恰是必须人工看的。
        """
        kept = self.kept
        if not kept:
            return 0.0
        return min(l.confidence for l in kept)


class OcrEngine(abc.ABC):
    """所有 OCR 后端实现这个接口。"""

    name: str = "base"

    @abc.abstractmethod
    def recognize(self, images: Sequence[np.ndarray], use_det: bool = True) -> list[OcrBatchResult]:
        """批量识别。

        use_det=True  -> 引擎自己做检测+识别（V1，稳定）
        use_det=False -> 调用方已经切好了行，只做识别（V2，更快更稳）
        返回列表与 images 一一对应。
        """

    @abc.abstractmethod
    def describe(self) -> str:
        """人类可读的引擎/模型标识，写进结果 JSON 便于追溯。"""

    def warmup(self) -> None:
        """可选：预热。首次推理会做 ORT 图优化，冷启动很慢。"""
        return None


class OcrUnavailable(RuntimeError):
    """OCR 引擎不可用。"""
