"""PaddleOCR + PaddlePaddle 后端（可选）。

保留这个适配器有三个理由：

1. 想用官方最新模型、或需要 GPU 加速时可以直接切过去，业务代码不用改。
2. 文档里"换模型做 A/B 测试"这件事需要一个现成的对照实现。
3. 迁移成本已经付掉了——以后真需要时不用重新研究 API。

注意 macOS 上装 PaddlePaddle 的坑（见 rapid.py 的说明）。本适配器在
paddleocr 缺失时只会在真正被调用时才报错，不影响默认的 rapidocr 路径。
"""

from __future__ import annotations

import logging
import time
from typing import Any, Sequence

import numpy as np

from ..config import OcrConfig
from .base import OcrBatchResult, OcrEngine, OcrLine, OcrUnavailable

log = logging.getLogger(__name__)


class PaddleOcrEngine(OcrEngine):
    name = "paddleocr"

    def __init__(self, cfg: OcrConfig, model_root: str | None = None):
        self.cfg = cfg
        self.model_root = model_root
        self._engine: Any | None = None
        self._model_label = f"{cfg.rec_ocr_version}-{cfg.model_type}-{cfg.lang}"

    def _params(self) -> dict[str, Any]:
        params: dict[str, Any] = {
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": False,
            "device": self.cfg.device,
        }
        if self.model_root:
            # 显式指定本地模型目录，隔离网络环境下不会触发下载。
            params["text_detection_model_dir"] = f"{self.model_root}/det"
            params["text_recognition_model_dir"] = f"{self.model_root}/rec"
        return params

    def _get_engine(self) -> Any:
        if self._engine is None:
            try:
                from paddleocr import PaddleOCR
            except ImportError as exc:  # pragma: no cover
                raise OcrUnavailable(
                    "未安装 paddleocr/paddlepaddle。Windows: pip install paddlepaddle paddleocr；"
                    "macOS 需锁 PaddleOCR<=3.3 + PaddlePaddle==3.0（3.1+ 无 macOS x86_64 wheel）。"
                ) from exc
            try:
                self._engine = PaddleOCR(**self._params())
            except Exception as exc:  # pragma: no cover
                raise OcrUnavailable(f"PaddleOCR 初始化失败: {str(exc)[:300]}") from exc
        return self._engine

    def describe(self) -> str:
        return f"paddleocr det={self.cfg.det_ocr_version} rec={self._model_label} device={self.cfg.device}"

    def recognize(self, images: Sequence[np.ndarray], use_det: bool = True) -> list[OcrBatchResult]:
        results: list[OcrBatchResult] = []
        if not images:
            return results
        engine = self._get_engine()

        for img in images:
            started = time.perf_counter()
            lines: list[OcrLine] = []
            notes: list[str] = []
            if img is None or img.size == 0:
                results.append(OcrBatchResult([], 0.0, self.name, self._model_label, ["输入图像为空"]))
                continue

            try:
                raw = engine.predict(img)
                payload = _extract_payload(raw)
            except Exception as exc:  # pragma: no cover
                results.append(
                    OcrBatchResult([], time.perf_counter() - started, self.name, self._model_label,
                                   [f"OCR 异常: {str(exc)[:200]}"])
                )
                continue

            for text, score, quad in payload:
                text = str(text).strip()
                if not text:
                    continue
                bbox = None
                if quad is not None:
                    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
                    bbox = (int(pts[:, 0].min()), int(pts[:, 1].min()),
                            int(pts[:, 0].max()), int(pts[:, 1].max()))
                lines.append(OcrLine(text=text, confidence=round(float(score), 4), bbox=bbox))

            results.append(
                OcrBatchResult(
                    lines=lines,
                    elapsed=round(time.perf_counter() - started, 4),
                    engine=self.name,
                    model=self._model_label,
                    notes=notes,
                )
            )
        return results

    def warmup(self) -> None:
        try:
            self._get_engine()
        except OcrUnavailable:  # pragma: no cover
            pass


def _extract_payload(raw: Any) -> list[tuple[str, float, Any | None]]:
    """兼容 PaddleOCR 3.x 的 predict() 返回结构。"""
    rows: list[tuple[str, float, Any | None]] = []
    for res in raw or []:
        data = getattr(res, "json", None)
        payload = res if isinstance(res, dict) else (data or {}).get("res", {}) if data else {}
        texts = payload.get("rec_texts") or []
        scores = payload.get("rec_scores") or []
        polys = payload.get("rec_polys") or payload.get("dt_polys") or []
        for i, text in enumerate(texts):
            score = scores[i] if i < len(scores) else 0.0
            quad = polys[i] if i < len(polys) else None
            rows.append((text, float(score), quad))
    return rows
