"""RapidOCR + ONNXRuntime 后端。

选它的原因（对应"Windows 或 mac 皆可"这条硬要求）：

* 同一批 PP-OCR 模型，跑在 ONNXRuntime 上，Windows / macOS(含 Apple Silicon) /
  Linux 通用，纯 CPU 可用，不依赖 PaddlePaddle。
* PaddlePaddle 在 macOS 上有实打实的坑：PaddleOCR ≥3.4 依赖 PaddlePaddle 3.1+
  才有的算子，而 3.1+ 没有 macOS x86_64 wheel；paddlepaddle 3.3.1 在
  macOS 26 上还有导入即崩的已知问题。

模型路径通过 Global.model_root_dir 固定到项目内 models/，配合 offline_guard
的 socket 封禁，保证运行时不会偷偷联网取模型。
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..config import OcrConfig
from .base import OcrBatchResult, OcrEngine, OcrLine, OcrUnavailable

log = logging.getLogger(__name__)


def _version_enum(enum_cls: Any, value: str, default: Any) -> Any:
    """把 'PP-OCRv5' / 'v5' 之类的写法归一到 RapidOCR 的枚举成员。

    RapidOCR 要求这些字段必须是 Enum 类型，传裸字符串会抛
    "must be Enum Type"，所以这里不能只做字符串替换。
    """
    raw = (value or "").strip()
    if not raw:
        return default

    normalized = raw.upper().replace("-", "").replace("PPOCRV", "PPOCRV")
    for member in enum_cls:
        name = getattr(member, "name", str(member)).upper()
        value_str = str(member.value).upper() if hasattr(member, "value") else ""
        if normalized in {name, value_str, name.replace("PPOCRV", "PPOCRV")}:
            return member
        # "PP-OCRv5" -> "PPOCRV5"
        if normalized.replace("PPOCRV", "PPOCRV") == name and normalized:
            return member
    # 退化：按 v4/v5/v6 尾号匹配
    tail = raw[-1] if raw else ""
    for member in enum_cls:
        if tail and str(getattr(member, "name", "")).upper().endswith(f"V{tail}"):
            return member
    log.warning("无法识别的 OCR 版本 %r，回退到 %s", value, default)
    return default


class RapidOcrEngine(OcrEngine):
    name = "rapidocr"

    def __init__(self, cfg: OcrConfig, model_root: str | Path | None = None):
        self.cfg = cfg
        self.model_root = Path(model_root) if model_root else None
        self._engine: Any | None = None
        self._model_label = f"{cfg.rec_ocr_version}-{cfg.model_type}-{cfg.lang}"

    # ---------- 初始化 ----------
    def _params(self, use_det: bool) -> dict[str, Any]:
        from rapidocr import EngineType, LangDet, LangRec, ModelType, OCRVersion

        # 必须是 Enum 成员，传普通字符串 RapidOCR 会直接拒绝。
        det_ver = _version_enum(OCRVersion, self.cfg.det_ocr_version, OCRVersion.PPOCRV5)
        rec_ver = _version_enum(OCRVersion, self.cfg.rec_ocr_version, OCRVersion.PPOCRV5)
        model_type = getattr(ModelType, self.cfg.model_type.upper(), ModelType.SERVER)
        lang_rec = getattr(LangRec, self.cfg.lang.upper(), LangRec.CH)

        params: dict[str, Any] = {
            "Global.log_level": "error",
            "Global.use_det": use_det,
            "Global.use_cls": False,  # 学生答案不会是倒置的，关掉省一次前向
            "Global.use_rec": True,
            "Global.text_score": self.cfg.min_det_score,
            "Det.engine_type": EngineType.ONNXRUNTIME,
            "Det.lang_type": LangDet.CH,
            "Det.model_type": model_type,
            "Det.ocr_version": det_ver,
            "Det.limit_side_len": self.cfg.det_limit_side_len,
            "Det.limit_type": "max",
            "Rec.engine_type": EngineType.ONNXRUNTIME,
            "Rec.lang_type": lang_rec,
            "Rec.model_type": model_type,
            "Rec.ocr_version": rec_ver,
            "Rec.rec_batch_num": self.cfg.rec_batch,
        }
        if self.cfg.threads > 0:
            params["EngineConfig.onnxruntime.intra_op_num_threads"] = self.cfg.threads
        if self.model_root is not None:
            params["Global.model_root_dir"] = str(self.model_root)
        return params

    def _get_engine(self, use_det: bool = True) -> Any:
        # use_det 不同 -> params 不同 -> RapidOCR 内部缓存 key 不同，
        # 所以这里按 use_det 分别缓存实例。
        cache = getattr(self, "_cache", None)
        if cache is None:
            cache = {}
            self._cache = cache  # type: ignore[attr-defined]
        if use_det not in cache:
            try:
                from rapidocr import RapidOCR
            except ImportError as exc:  # pragma: no cover
                raise OcrUnavailable(
                    "未安装 rapidocr。请先执行: pip install rapidocr onnxruntime"
                ) from exc
            try:
                cache[use_det] = RapidOCR(params=self._params(use_det))
            except Exception as exc:  # pragma: no cover
                raise OcrUnavailable(
                    "RapidOCR 初始化失败。多半是模型没下载——先在联网环境执行 "
                    "`homework-ocr fetch-models`。原始错误: " + str(exc)[:300]
                ) from exc
        return cache[use_det]

    def describe(self) -> str:
        root = str(self.model_root) if self.model_root else "<site-packages 默认目录>"
        return f"rapidocr/onnxruntime det={self.cfg.det_ocr_version} rec={self._model_label} models={root}"

    # ---------- 预热 ----------
    def warmup(self) -> None:
        engine = self._get_engine(True)
        blank = np.full((64, 320, 3), 255, dtype=np.uint8)
        try:
            engine(blank)
        except Exception as exc:  # pragma: no cover
            log.warning("OCR 预热失败（不影响正确性）: %s", str(exc)[:120])

    # ---------- 识别 ----------
    def recognize(self, images: Sequence[np.ndarray], use_det: bool = True) -> list[OcrBatchResult]:
        results: list[OcrBatchResult] = []
        if not images:
            return results

        engine = self._get_engine(use_det)
        for img in images:
            started = time.perf_counter()
            notes: list[str] = []
            lines: list[OcrLine] = []

            if img is None or img.size == 0:
                results.append(OcrBatchResult([], 0.0, self.name, self._model_label, ["输入图像为空"]))
                continue

            try:
                out = engine(img, use_det=use_det, use_cls=False, use_rec=True)
            except Exception as exc:  # pragma: no cover
                log.warning("OCR 失败: %s", str(exc)[:200])
                results.append(
                    OcrBatchResult([], time.perf_counter() - started, self.name, self._model_label,
                                   [f"OCR 异常: {str(exc)[:200]}"])
                )
                continue

            texts = list(getattr(out, "txts", ()) or ())
            scores = list(getattr(out, "scores", ()) or ())
            boxes = getattr(out, "boxes", None)

            for i, text in enumerate(texts):
                if text is None:
                    continue
                score = float(scores[i]) if i < len(scores) else 0.0
                bbox = None
                if boxes is not None and i < len(boxes):
                    bbox = _quad_to_bbox(boxes[i])
                text = str(text).strip()
                if not text:
                    continue
                lines.append(OcrLine(text=text, confidence=round(score, 4), bbox=bbox))

            if use_det and not lines:
                notes.append("检测阶段未找到文本行：mask 判定的位置没有可识别内容")

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

    def ensure_models(self) -> list[Path]:
        """触发一次加载，确认模型已在本地。返回模型目录。"""
        self._get_engine(True)
        if self.model_root is not None:
            return sorted(self.model_root.glob("*.onnx"))
        try:
            import rapidocr

            return sorted((Path(rapidocr.__file__).parent / "models").glob("*.onnx"))
        except Exception:  # pragma: no cover
            return []


def _quad_to_bbox(quad: Any) -> tuple[int, int, int, int]:
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    return (
        int(round(pts[:, 0].min())),
        int(round(pts[:, 1].min())),
        int(round(pts[:, 0].max())),
        int(round(pts[:, 1].max())),
    )
