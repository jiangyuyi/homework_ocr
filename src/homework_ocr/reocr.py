"""从已保存的中间产物重跑 OCR。

对应技术方案第 32 节的建议：aligned 图和答案 crop 永远保存。以后换 OCR
模型（v5 -> v6、通用 -> 手写微调）、或改后处理规则时，只需要拿现成的 crop
重新识别，不必把渲染/配准/差分/ROI 全部重跑一遍——这几步是最慢也最容易
受参数影响的部分。

前提是 run 时保存了 answer crop 和对齐后的页面图。
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2

from .config import Config
from .debugviz import save_image
from .geometry import as_bbox
from .ocr import create_engine
from .pdfio import imread_unicode
from .pipeline import DocumentResult, PageResult, QuestionResult
from .postprocess import (
    evaluate_review,
    filter_lines_by_mask,
    group_text_blocks,
    mask_for_region,
    sort_reading_order,
    stitch_lines,
)
from .template import Question

log = logging.getLogger(__name__)


def rerun_from_run_dir(run_dir: Path, cfg: Config, model_root: Path | None = None):
    run_dir = Path(run_dir)
    result_path = run_dir / "result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"找不到 {result_path}，这不是一个 run 输出目录")

    data = json.loads(result_path.read_text(encoding="utf-8"))
    engine = create_engine(cfg.ocr, model_root=model_root)
    engine.warmup()

    masks: dict[int, "object"] = {}
    grays: dict[int, "object"] = {}
    for page_data in data.get("pages", []):
        number = int(page_data["page"])
        paths = page_data.get("debug_paths", {})
        mask_path = run_dir / paths.get("mask", "")
        masks[number] = imread_unicode(mask_path, cv2.IMREAD_GRAYSCALE) if mask_path.exists() else None
        # 对齐图用来判断「两块之间那段空白里有没有印刷内容」，
        # 拿不到就退回把所有行连成一整块。
        aligned_path = run_dir / paths.get("aligned", "")
        if aligned_path.exists():
            img = imread_unicode(aligned_path)
            grays[number] = img if img is not None and img.ndim == 2 else (
                cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img is not None else None)

    for page_data in data.get("pages", []):
        number = int(page_data["page"])
        mask = masks.get(number)
        for q in page_data.get("questions", []):
            crop_name = q.get("crop_path")
            if not crop_name:
                continue
            crop_path = run_dir / crop_name
            if not crop_path.exists():
                log.warning("缺少 crop，跳过: %s", crop_name)
                continue

            # answer crop 上方有一条信息栏（标题+上次识别文字），识别时要裁掉，
            # 否则重跑会把上一轮的识别结果也当成答案识别进去。
            image = imread_unicode(crop_path)
            if image is None:
                log.warning("无法读取 crop: %s", crop_path)
                continue
            bar = int(q.get("crop_bar_px") or (46 if q.get("review_required") else 26))
            body = image[bar:, :, :] if image.shape[0] > bar + 5 else image

            res = engine.recognize([body], use_det=cfg.ocr.use_det)[0]
            # OCR 出的 bbox 是 crop 局部坐标，所以必须用 ROI 局部 mask 比对，
            # 直接拿整页 mask 过滤会把所有行都误判成"不在手写上"。
            local_mask = None
            if mask is not None and mask.size and q.get("processing_area"):
                local_mask = mask_for_region(mask, tuple(int(v) for v in q["processing_area"]))
            if local_mask is not None and local_mask.size:
                res = filter_lines_by_mask(res, local_mask, cfg.ocr)

            text = stitch_lines(res.lines)
            # 和主流程保持同一套分块规则，否则换模型重跑会把段落结构弄丢。
            roi = q.get("processing_area")
            gray = grays.get(number)
            blocks: list[str] = []
            if roi and gray is not None:
                rx, ry = int(roi[0]), int(roi[1])
                items: list[tuple[tuple[int, int, int, int], str]] = []
                for line in sort_reading_order(res.lines):
                    if line.dropped or not line.text or line.bbox is None:
                        continue
                    x1, y1, x2, y2 = line.bbox
                    items.append((as_bbox((x1 + rx, y1 + ry, x2 + rx, y2 + ry)), line.text))
                blocks = group_text_blocks(
                    items,
                    page_gray=gray,
                    hand_mask=mask,
                    foreign_ink_limit=cfg.output.block_foreign_ink_px,
                    hgap_ratio=cfg.output.block_hgap_ratio,
                )
                text = cfg.output.block_joiner.join(blocks)
            confidence = res.confidence
            verdict = evaluate_review(
                detected_handwriting=q.get("detected_handwriting", True),
                ocr=res,
                alignment_score=float(page_data.get("alignment", {}).get("score", 0.0)),
                fragments=int(q.get("fragments", 0)),
                cfg=cfg.review,
            )
            q["text"] = text
            if blocks:
                q["blocks"] = blocks
            q["confidence"] = round(confidence, 4)
            q["review_required"] = verdict.required
            q["review_reasons"] = verdict.reasons
            q["review_priority"] = verdict.priority
            q["answers"] = [
                {"text": line.text, "confidence": line.confidence,
                 "bbox": list(line.bbox) if line.bbox else None,
                 "mask_ratio": line.mask_ratio}
                for line in res.kept
            ]

    result = DocumentResult(
        document=data.get("document", ""),
        template_id=data.get("template_id", ""),
        template_dpi=int(data.get("template_dpi", 300)),
        status=data.get("status", "ok"),
        engine=engine.describe(),
        pages=[_page_from_dict(p) for p in data.get("pages", [])],
    )
    payload = result.to_dict()
    payload["reocr_note"] = "本次结果由 reocr 从已保存的 crop 重新生成，未重跑配准/差分"
    (run_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = run_dir / "result.csv"
    if csv_path.exists():
        from .pipeline import HomeworkPipeline

        HomeworkPipeline._export_csv(result, csv_path)
    txt_path = run_dir / "result.txt"
    if txt_path.exists():
        from .pipeline import HomeworkPipeline

        HomeworkPipeline._export_txt(result, txt_path)

    _ = (Question, as_bbox, save_image)
    return result


def _page_from_dict(d: dict) -> PageResult:
    return PageResult(
        page=int(d["page"]),
        status=d.get("status", "ok"),
        alignment=d.get("alignment", {}),
        questions=[
            QuestionResult(
                question_id=str(q["question_id"]),
                question_type=q.get("question_type", "text"),
                answer_area=tuple(q.get("answer_area", (0, 0, 0, 0))),  # type: ignore[arg-type]
                processing_area=tuple(q.get("processing_area", (0, 0, 0, 0))),  # type: ignore[arg-type]
                detected_handwriting=bool(q.get("detected_handwriting", False)),
                text=q.get("text", ""),
                confidence=float(q.get("confidence", 0.0)),
                review_required=bool(q.get("review_required", False)),
                review_reasons=q.get("review_reasons", []),
                review_priority=int(q.get("review_priority", 0)),
                ink_coverage=float(q.get("ink_coverage", 0.0)),
                fragments=int(q.get("fragments", 0)),
                region_count=int(q.get("region_count", 0)),
                crop_bar_px=int(q.get("crop_bar_px", 26)),
                crop_path=q.get("crop_path"),
            )
            for q in d.get("questions", [])
        ],
        notes=d.get("notes", []),
        timings_ms=d.get("timings_ms", {}),
        debug_paths=d.get("debug_paths", {}),
    )
