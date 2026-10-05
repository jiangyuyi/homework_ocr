"""OCR 后端工厂。

配置里 ocr.engine 决定用哪个实现，pipeline 只认 OcrEngine 接口。
"""

from __future__ import annotations

from pathlib import Path

from ..config import OcrConfig
from .base import OcrBatchResult, OcrEngine, OcrLine, OcrUnavailable
from .rapid import RapidOcrEngine

__all__ = [
    "OcrBatchResult",
    "OcrEngine",
    "OcrLine",
    "OcrUnavailable",
    "create_engine",
]


def create_engine(cfg: OcrConfig, model_root: str | Path | None = None) -> OcrEngine:
    name = (cfg.engine or "rapidocr").lower()
    root = str(model_root) if model_root else None

    if name == "rapidocr":
        return RapidOcrEngine(cfg, model_root=model_root)
    if name in {"paddleocr", "paddle"}:
        from .paddle_adapter import PaddleOcrEngine

        return PaddleOcrEngine(cfg, model_root=root)

    raise ValueError(f"未知的 OCR 引擎: {cfg.engine}（可选: rapidocr / paddleocr）")
