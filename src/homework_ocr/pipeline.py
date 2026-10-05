"""流水线编排。

链路（技术方案的 MVP 组合，补上了 mask 过滤与方向探测）：

    PyMuPDF 300 DPI
      -> 方向探测 + ORB + Homography + ECC
      -> 题目 ROI
      -> 光照归一化 + 自适应阈值
      -> 模板相减（保守抑制）
      -> 形态学 + 连通域 + 切行
      -> 用原图 crop 做 OCR
      -> 按 mask 过滤印刷残留
      -> 低置信度标记
      -> JSON / CSV / TXT / debug 图

关键约定：**OCR 吃的是原图 crop，不是 mask**。mask 只用来定位和过滤。
"""

from __future__ import annotations

import csv
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from . import debugviz
from .alignment import AlignResult, align_page
from .config import Config
from .difference import detect_handwriting_no_template, subtract_template
from .geometry import BBox, as_bbox, crop, expand_bbox, offset_bbox
from .imaging import binarize_pair, estimate_stroke_width
from .ocr import OcrBatchResult, OcrEngine, create_engine
from .ocr.base import OcrLine
from .pdfio import RenderedPage, compare_geometry, render_pdf
from .postprocess import (
    ReviewVerdict,
    build_absolute_lines,
    evaluate_review,
    filter_lines_by_mask,
    mask_for_region,
    stitch_lines,
)
from .regions import Region, RegionSet, count_shards, group_regions, sort_regions
from .template import Question, Template

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1


# ---------------------------------------------------------------- 结果结构
@dataclass
class AnswerBox:
    text: str
    confidence: float
    bbox: BBox | None = None
    mask_ratio: float = 1.0


@dataclass
class QuestionResult:
    question_id: str
    question_type: str = "text"
    answer_area: BBox = (0, 0, 0, 0)
    processing_area: BBox = (0, 0, 0, 0)
    detected_handwriting: bool = False
    answers: list[AnswerBox] = field(default_factory=list)
    text: str = ""
    #: 按区域拆开的文本块。行内用 line_joiner，块间用 block_joiner。
    #: 保留它是为了将来做「按段落对齐」或只重新拼接其中一块。
    blocks: list[str] = field(default_factory=list)
    confidence: float = 0.0
    review_required: bool = False
    review_reasons: list[str] = field(default_factory=list)
    review_priority: int = 0
    ink_coverage: float = 0.0
    fragments: int = 0       # 碎裂块数（明显偏矮的小碎片），不是行数
    region_count: int = 0    # 该题区域内检测到的行/块总数
    #: 答案图顶部信息栏的高度。reocr 时要按它裁掉，
    #: 否则重跑会把标题栏的旧识别文字也识别进去。
    crop_bar_px: int = 0
    crop_path: str | None = None
    notes: list[str] = field(default_factory=list)


@dataclass
class PageResult:
    page: int
    status: str = "ok"  # ok | alignment_failed | empty | error
    alignment: dict[str, Any] = field(default_factory=dict)
    questions: list[QuestionResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)
    debug_paths: dict[str, str] = field(default_factory=dict)

    @property
    def review_count(self) -> int:
        return sum(1 for q in self.questions if q.review_required)


@dataclass
class DocumentResult:
    document: str
    template_id: str
    template_dpi: int
    status: str = "ok"
    pages: list[PageResult] = field(default_factory=list)
    engine: str = ""
    config_fingerprint: str = ""
    elapsed_s: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def total_questions(self) -> int:
        return sum(len(p.questions) for p in self.pages)

    @property
    def review_questions(self) -> int:
        return sum(p.review_count for p in self.pages)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema_version"] = SCHEMA_VERSION
        d["summary"] = {
            "total_questions": self.total_questions,
            "review_required": self.review_questions,
            "pages": len(self.pages),
        }
        return d


# ---------------------------------------------------------------- 流水线
class HomeworkPipeline:
    """模板驱动的作业手写提取流水线。"""

    def __init__(
        self,
        template: Template,
        ocr: OcrEngine,
        config: Config,
        model_root: str | Path | None = None,
    ):
        self.template = template
        self.ocr = ocr
        self.cfg = config
        self.model_root = Path(model_root) if model_root else None
        self._template_pages: list[RenderedPage] | None = None

    # ---------- 模板 ----------
    def load_template_pages(self) -> list[RenderedPage]:
        """模板只渲染一次，批量处理时复用。"""
        if self._template_pages is None:
            self._template_pages = render_pdf(self.template.pdf_path, dpi=self.template.dpi)
        return self._template_pages

    # ---------- 入口 ----------
    def process_pdf(self, pdf_path: str | Path, output_dir: str | Path | None = None) -> DocumentResult:
        started = time.perf_counter()
        pdf_path = Path(pdf_path)
        template_pages = self.load_template_pages()
        student_pages = render_pdf(pdf_path, dpi=self.template.dpi)

        result = DocumentResult(
            document=pdf_path.name,
            template_id=self.template.template_id,
            template_dpi=self.template.dpi,
            engine=self.ocr.describe(),
            config_fingerprint=fingerprint_config(self.cfg),
        )

        report = compare_geometry(template_pages, student_pages)
        result.notes.extend(report.notes)

        for sp in student_pages:
            tpl_page = template_pages[sp.index] if sp.index < len(template_pages) else template_pages[-1]
            try:
                page_result = self.process_page(sp, tpl_page, sp.index + 1, output_dir)
            except Exception as exc:  # 单页失败不该让整份文档报废
                log.exception("第 %d 页处理失败", sp.index + 1)
                page_result = PageResult(
                    page=sp.index + 1,
                    status="error",
                    notes=[f"处理异常: {type(exc).__name__}: {str(exc)[:200]}"],
                )
                if not self.cfg.run.continue_on_error:
                    raise
            result.pages.append(page_result)

        result.elapsed_s = round(time.perf_counter() - started, 3)
        if any(p.status == "error" for p in result.pages):
            result.status = "partial"
        elif all(p.status == "alignment_failed" for p in result.pages) and result.pages:
            result.status = "alignment_failed"

        if output_dir:
            self._export(result, Path(output_dir))
        return result

    # ---------- 单页 ----------
    def process_page(
        self,
        student: RenderedPage,
        template: RenderedPage,
        page_number: int,
        output_dir: str | Path | None = None,
    ) -> PageResult:
        timings: dict[str, float] = {}
        out = PageResult(page=page_number)
        outdir = Path(output_dir) if output_dir else None

        # 1) 配准
        t0 = time.perf_counter()
        align = align_page(student.image, template.image, self.cfg.alignment)
        timings["align"] = round((time.perf_counter() - t0) * 1000, 1)

        out.alignment = {
            "success": align.ok,
            "score": align.score,
            "method": align.method,
            "good_matches": align.good_matches,
            "inliers": align.inliers,
            "inlier_ratio": align.inlier_ratio,
            "rotation_applied": align.rotation_applied,
            "reason": align.reason,
            "warnings": align.warnings,
        }
        if not align.ok or align.aligned is None:
            out.status = "alignment_failed"
            out.notes.append(
                "配准失败，差分结果无意义，已跳过本页 OCR。" + (f" 原因: {align.reason}" if align.reason else "")
            )
            out.timings_ms = timings
            if outdir and self.cfg.output.save_debug:
                debugviz.save_image(outdir / f"page_{page_number:03d}_failed.jpg", student.image,
                                    self.cfg.output.jpeg_quality)
                out.debug_paths["failed_page"] = f"page_{page_number:03d}_failed.jpg"
            return out

        aligned = align.aligned
        use_template = self.cfg.difference.method == "template_subtract"

        # 2) 二值化 + 差分（整页一次，省时间）
        t0 = time.perf_counter()
        student_bin, template_bin = binarize_pair(aligned, template.image, self.cfg.difference)
        if use_template:
            diff = subtract_template(student_bin, template_bin, self.cfg.difference)
            mask = diff.mask
            out.notes.extend(diff.notes)
        elif self.cfg.difference.method == "none":
            mask = student_bin
        else:
            # 无模板回退：靠形态学/连通域特征筛出候选笔迹（弱方案，会标注）
            mask = detect_handwriting_no_template(student_bin, self.cfg.noise_filter).mask
            out.notes.append("未使用模板相分，手写检测为启发式结果，误检率偏高")
        timings["difference"] = round((time.perf_counter() - t0) * 1000, 1)

        stroke_w = estimate_stroke_width(mask)
        if stroke_w:
            out.notes.append(f"估计笔画宽度 {stroke_w:.1f}px（{self.template.dpi} DPI）")

        # 3) 中间产物
        if outdir and self.cfg.output.save_aligned:
            debugviz.save_image(outdir / f"page_{page_number:03d}_aligned.png", aligned)
            out.debug_paths["aligned"] = f"page_{page_number:03d}_aligned.png"
        if outdir and self.cfg.output.save_mask:
            debugviz.save_image(outdir / f"page_{page_number:03d}_mask.png", mask)
            out.debug_paths["mask"] = f"page_{page_number:03d}_mask.png"

        # 4) 逐题处理
        tpl_page_spec = self.template.page(page_number)
        questions = list(tpl_page_spec.questions) if tpl_page_spec else []
        if not questions:
            if self.cfg.roi.allow_full_page:
                out.notes.append(
                    "模板未定义该页题目 ROI，已退化为整页处理。识别结果无法按题号归属，"
                    "建议先跑 `annotate` 补齐 ROI。"
                )
                questions = [Question(id="page", answer_area=[0, 0, aligned.shape[1], aligned.shape[0]])]
            else:
                out.status = "empty"
                out.notes.append("模板未定义该页题目 ROI，且 allow_full_page=false，跳过。")
                out.timings_ms = timings
                return out

        t0 = time.perf_counter()
        overlay_items: list[debugviz.RegionDebug] = []
        for q in questions:
            qr = self._process_question(q, aligned, mask, align, page_number, outdir)
            out.questions.append(qr)
            overlay_items.append(
                debugviz.RegionDebug(
                    question_id=q.id,
                    roi_bbox=as_bbox(q.answer_area),
                    ocr_lines=_debug_lines(qr),
                    review_required=qr.review_required,
                    text=qr.text,
                )
            )
        timings["questions"] = round((time.perf_counter() - t0) * 1000, 1)

        # 5) debug overlay
        if outdir and self.cfg.output.save_debug:
            overlay = debugviz.render_overlay(aligned, mask, overlay_items, page_number,
                                              self.cfg.output.overlay_scale)
            debugviz.save_image(outdir / f"page_{page_number:03d}_debug.jpg", overlay,
                                self.cfg.output.jpeg_quality)
            out.debug_paths["debug"] = f"page_{page_number:03d}_debug.jpg"

        out.timings_ms = timings
        return out

    # ---------- 单题 ----------
    def _process_question(
        self,
        question: Question,
        aligned: np.ndarray,
        mask: np.ndarray,
        align: AlignResult,
        page_number: int,
        outdir: Path | None,
    ) -> QuestionResult:
        h, w = aligned.shape[:2]
        roi = expand_bbox(question.answer_area, self.cfg.roi.padding, w, h)
        qr = QuestionResult(
            question_id=question.id,
            question_type=question.type,
            answer_area=as_bbox(question.answer_area),
            processing_area=roi,
        )

        region_mask = mask_for_region(mask, roi)
        total_ink = int(cv2.countNonZero(region_mask))
        qr.ink_coverage = round(total_ink / float(region_mask.size or 1), 5)

        regions: RegionSet = group_regions(region_mask, self.cfg.grouping, self.cfg.noise_filter)
        qr.notes.extend(regions.notes)

        # group_regions 拿到的是 ROI 局部 mask，返回的框也是 ROI 局部坐标。
        # 后面要和整页 mask / 整页原图比对，必须先平移到整页坐标系。
        for region in regions.regions:
            region.bbox = offset_bbox(region.bbox, roi[0], roi[1])
        # 同一行的左右两段可能差几像素，必须按「垂直重叠聚行 + x 排序」，
        # 否则一段会跳到句首（实测「我也」被排到了答案最前面）。
        regions.regions = sort_regions(regions.regions)

        if not regions.regions:
            verdict = evaluate_review(
                detected_handwriting=False, ocr=None, alignment_score=align.score,
                fragments=0, cfg=self.cfg.review, crop_mask=region_mask,
            )
            qr.detected_handwriting = False
            _apply_verdict(qr, verdict)
            return qr

        qr.detected_handwriting = True

        # 送 OCR 的是原图 crop，不是 mask。
        crops: list[np.ndarray] = []
        crop_boxes: list[BBox] = []
        crop_regions: list[Region] = []
        for region in regions.regions:
            box = expand_bbox(region.bbox, 2, w, h)
            piece = crop(aligned, box)
            if piece.size == 0:
                continue
            crops.append(piece)
            crop_boxes.append(box)
            crop_regions.append(region)

        if not crops:
            qr.notes.append("区域有效面积为空")
            verdict = evaluate_review(
                detected_handwriting=True, ocr=None, alignment_score=align.score,
                fragments=0, cfg=self.cfg.review, crop_mask=region_mask,
            )
            _apply_verdict(qr, verdict)
            return qr

        qr.region_count = len(crop_regions)

        # V1（默认）: 引擎自己做检测+识别。
        # V2: 我们已经按行切好了 region，关掉引擎的检测，只做识别。
        use_det = self.cfg.ocr.use_det
        batch = self.ocr.recognize(crops, use_det=use_det)

        answers: list[AnswerBox] = []
        all_lines: list[OcrLine] = []
        productive: list[Region] = []  # 真正产出文本的区域
        worst_conf = 1.0
        for region, piece, box, res in zip(crop_regions, crops, crop_boxes, batch):
            # 每个 region 自己的局部 mask —— 过滤印刷残留必须用局部坐标比对。
            local_mask = mask_for_region(mask, box)
            res = filter_lines_by_mask(res, local_mask, self.cfg.ocr)
            res = build_absolute_lines(res, box)  # 换算回整页坐标
            res.notes = [f"[region {box}] {n}" for n in res.notes]
            all_lines.extend(res.lines)
            answers.extend(
                AnswerBox(text=l.text, confidence=l.confidence, bbox=l.bbox, mask_ratio=l.mask_ratio)
                for l in res.kept
            )
            if res.kept:
                worst_conf = min(worst_conf, res.confidence)
                productive.append(region)

        # 碎裂只统计"产出了答案文本"的区域。没有产出的多半是被正确丢弃的
        # 印刷残影，把它们算成"笔迹碎裂"会制造大量假警报。
        _, shard_count = count_shards(productive, question.expected_lines)
        qr.fragments = shard_count

        # 整题拼成一段。
        #
        # 分隔规则（语文作业通常是多行、多段的）：
        #   同一区域内的多行  -> 换一行
        #   不同区域之间      -> 换行 + 空行
        # 之前所有行直接首尾相连，段落全糊在一起，人根本看不出哪句是哪句。
        region_texts: list[str] = []
        for res in batch:
            text = stitch_lines(res.lines, line_joiner=self.cfg.output.line_joiner)
            if text:
                region_texts.append(text)

        joined = (self.cfg.output.block_joiner).join(region_texts)
        qr.text = joined
        qr.blocks = region_texts
        qr.answers = answers
        qr.confidence = round(worst_conf if answers else 0.0, 4)

        combined = OcrBatchResult(
            lines=all_lines,
            engine=batch[0].engine if batch else "",
            model=batch[0].model if batch else "",
            notes=[n for b in batch for n in b.notes],
        )
        verdict = evaluate_review(
            detected_handwriting=True,
            ocr=combined,
            alignment_score=align.score,
            fragments=qr.fragments,
            cfg=self.cfg.review,
            crop_mask=region_mask,
        )
        _apply_verdict(qr, verdict)
        qr.notes.extend(dict.fromkeys(combined.notes))

        # 保存答案 crop —— 技术方案第 32 节：换模型时只重跑这些。
        if outdir and self.cfg.output.save_crops:
            qr.crop_bar_px = 46 if qr.review_required else 26
            canvas = debugviz.render_question_crop(
                aligned, question.id, roi, qr.text, qr.confidence,
                qr.review_required, qr.review_reasons,
            )
            name = f"page_{page_number:03d}_q{question.id}_answer.png"
            debugviz.save_image(outdir / name, canvas)
            qr.crop_path = name

        return qr

    # ---------- 导出 ----------
    def _export(self, result: DocumentResult, outdir: Path) -> None:
        outdir.mkdir(parents=True, exist_ok=True)
        formats = set(self.cfg.output.formats)

        if "json" in formats:
            path = outdir / "result.json"
            path.write_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

        if "csv" in formats:
            self._export_csv(result, outdir / "result.csv")

        if "txt" in formats:
            self._export_txt(result, outdir / "result.txt")

        if "xlsx" in formats:
            # 逐份导出，方便 Excel 逐个打开。整班汇总用 `homework-ocr export`。
            from pathlib import Path as _Path

            from .excel import export_xlsx
            from .review import ReviewStore

            store = ReviewStore.load(outdir)
            stem = _Path(result.document).stem or outdir.name
            export_xlsx(outdir / f"{stem}.xlsx", [result.to_dict()], {outdir.name: store})

    @staticmethod
    def _export_csv(result: DocumentResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as fh:
            writer = csv.writer(fh)
            writer.writerow(["page", "question_id", "type", "detected", "text",
                             "confidence", "review_required", "review_reasons", "ink_coverage"])
            for page in result.pages:
                for q in page.questions:
                    writer.writerow([
                        page.page, q.question_id, q.question_type, int(q.detected_handwriting),
                        q.text, f"{q.confidence:.4f}", int(q.review_required),
                        " | ".join(q.review_reasons), f"{q.ink_coverage:.5f}",
                    ])

    @staticmethod
    def _export_txt(result: DocumentResult, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        out = [
            f"文档: {result.document}",
            f"模板: {result.template_id} (DPI {result.template_dpi})",
            f"引擎: {result.engine}",
            f"状态: {result.status}   耗时: {result.elapsed_s}s",
            "=" * 60,
        ]
        for page in result.pages:
            out.append(f"[第 {page.page} 页] status={page.status} "
                       f"align={page.alignment.get('score', 0):.3f} "
                       f"({page.alignment.get('method', '-')})")
            for note in page.notes:
                out.append(f"    ! {note}")
            for q in page.questions:
                flag = " [需复核]" if q.review_required else ""
                mark = "+" if q.detected_handwriting else "-"
                # 答案可能是多行多段，按原样输出，缩进能看出结构
                body = (q.text or "(空)").split("\n")
                body[0] = f"  {mark} Q{q.question_id}{flag}  {body[0]}"
                out.extend(body[1:])
                for reason in q.review_reasons:
                    out.append(f"        - {reason}")
        out.append("=" * 60)
        out.append(f"共 {result.total_questions} 题，其中 {result.review_questions} 题需人工复核")
        path.write_text("\n".join(out), encoding="utf-8")


def _apply_verdict(qr: QuestionResult, verdict: ReviewVerdict) -> None:
    qr.review_required = verdict.required
    qr.review_reasons = verdict.reasons
    qr.review_priority = verdict.priority


def _debug_lines(qr: QuestionResult) -> list[OcrLine] | None:
    """从结果里还原出用于 overlay 的行。"""
    if not qr.answers:
        return None
    return [OcrLine(text=a.text, confidence=a.confidence, bbox=a.bbox) for a in qr.answers]


def iter_pdfs(path: str | Path) -> list[Path]:
    """收集输入路径下的所有 PDF（文件或目录）。"""
    p = Path(path)
    if p.is_file():
        return [p] if p.suffix.lower() == ".pdf" else []
    if p.is_dir():
        return sorted(x for x in p.rglob("*.pdf") if x.is_file())
    return []


def fingerprint_config(cfg: Config) -> str:
    """配置指纹。写进结果里，方便追溯某次结果是用什么参数跑出来的。"""
    import hashlib

    payload = json.dumps(cfg.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def build_pipeline(
    template: Template,
    cfg: Config,
    model_root: str | Path | None = None,
) -> HomeworkPipeline:
    """按模板的实际 DPI 换算像素类阈值后再建流水线。

    阈值在 config.yaml 里是按 300 DPI 写的；模板如果是 200 或 600 DPI，
    在这里统一换算，避免用户改完扫描仪分辨率还要重新调参。
    """
    cfg = cfg.scaled_for_dpi(template.dpi)
    engine = create_engine(cfg.ocr, model_root=model_root)
    return HomeworkPipeline(template, engine, cfg, model_root=model_root)
