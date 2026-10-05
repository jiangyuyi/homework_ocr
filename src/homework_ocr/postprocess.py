"""结果后处理。

针对语文作业（看拼音写词语、阅读理解、作文）做了专门处理，另外补了
技术方案里没有但很关键的一步：**用 mask 过滤 OCR 结果**。

为什么需要这一步：
mask 只用来"定位哪里有手写"，而送进 OCR 的是原始对齐图——图里还有印刷的
拼音、题目、方格线。如果不做过滤，OCR 会把残留的印刷内容一起读出来，
读成 "pin yin 词语" 这种结果。所以：

    OCR 检出的每一行 -> 与该 crop 的手写 mask 求重叠比例
                    -> 低于阈值就丢掉，并记录丢弃原因

同时按技术方案第 24 节的要求，把"是否需要人工复核"独立出来算，而不是
直接拿 OCR confidence 当可信度。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import OcrConfig, ReviewConfig
from .ocr.base import OcrBatchResult, OcrLine
from .geometry import bbox_mask, mask_ratio, offset_bbox

# 只清理"几乎不可能出现在语文手写答案里、且几乎必然是检测噪声"的字符。
# 刻意保守：引号、加号、等号、下划线在阅读理解引用和填空题里是真实内容，
# 不能删——宁可留着一点噪声，也不要改坏学生写的原文。
_NOISE_CHARS = frozenset({"|", "~", "`", "^"})


@dataclass
class ReviewVerdict:
    required: bool = False
    reasons: list[str] = field(default_factory=list)
    #: 供人工复核界面排序，数值越大越需要优先看。
    priority: int = 0


def filter_lines_by_mask(result: OcrBatchResult, mask: np.ndarray, cfg: OcrConfig) -> OcrBatchResult:
    """丢掉不落在手写 mask 上的识别行。

    这是"mask 负责判断，OCR 负责识别"这条分工的收口：如果不这么做，
    差分残留的印刷文字会混进答案里。
    """
    if result.lines:
        if mask is None or mask.size == 0:
            for line in result.lines:
                line.mask_ratio = 0.0
                line.dropped = True
                line.drop_reason = "缺少手写 mask，无法确认来源"
            result.notes.append("缺少 mask，所有识别行都被标记为待确认")
            return result

        for line in result.lines:
            if line.bbox is None:
                # rec_only 模式没有 bbox，假定调用方已按行切好，直接保留。
                line.mask_ratio = 1.0
                continue
            ratio = mask_ratio(line.bbox, mask)
            line.mask_ratio = round(ratio, 4)
            if ratio < cfg.min_box_overlap:
                line.dropped = True
                line.drop_reason = (
                    f"与手写区域重叠仅 {ratio * 100:.0f}%（阈值 {cfg.min_box_overlap * 100:.0f}%），"
                    "判定为印刷内容残留"
                )

    dropped = sum(1 for line in result.lines if line.dropped)
    if dropped:
        result.notes.append(f"按 mask 过滤掉 {dropped}/{len(result.lines)} 行印刷内容残留")
    return result


def sort_reading_order(lines: list[OcrLine], line_height_hint: float | None = None) -> list[OcrLine]:
    """按阅读顺序排序：先分行，行内从左到右。

    分行用「垂直区间重叠」而不是「上边缘接近」：
    学生写错字、涂改、或墨点溅高，都可能在正文上方留下一个小碎块。
    按上边缘排序时那个碎块会跳到答案最前面（实测「我也」跑到了句首）。
    按区间重叠聚行，碎块和它下方的主行属于同一行，交给 x 坐标决定先后。
    """
    with_box = [l for l in lines if l.bbox is not None]
    without = [l for l in lines if l.bbox is None]

    if not with_box:
        return list(lines)

    if line_height_hint is None:
        heights = [l.bbox[3] - l.bbox[1] for l in with_box]  # type: ignore[index]
        line_height_hint = float(np.median(heights)) if heights else 0.0

    with_box.sort(key=lambda l: (l.bbox[1], l.bbox[0]))  # type: ignore[index]

    rows: list[list[OcrLine]] = []
    row_extent: list[tuple[int, int]] = []  # 每行当前覆盖的 [top, bottom]

    for line in with_box:
        top, bottom = line.bbox[1], line.bbox[3]  # type: ignore[index]
        # 和已有行比对，取重叠最多的那一行，避免误并到相邻行
        best_i, best_ov = -1, 0
        for i, (rt, rb) in enumerate(row_extent):
            ov = min(bottom, rb) - max(top, rt)
            if ov > best_ov:
                best_i, best_ov = i, ov
        # 短碎块用「重叠占自身高度」判定：短行和长行要分别用不同标准
        need = min(max(2, int((line_height_hint or 0.0) * 0.25)), max(1, (bottom - top) * 0.6))
        if best_i >= 0 and best_ov >= need:
            rows[best_i].append(line)
            rt, rb = row_extent[best_i]
            row_extent[best_i] = (min(rt, top), max(rb, bottom))
        else:
            rows.append([line])
            row_extent.append((top, bottom))

    ordered: list[OcrLine] = []
    for row in rows:
        row.sort(key=lambda l: l.bbox[0])  # type: ignore[index]
        ordered.extend(row)
    ordered.extend(without)
    return ordered


def clean_text(text: str) -> str:
    """轻量清洗。只删确定是噪声的东西，不做会改变语义的重写。"""
    if not text:
        return ""
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = "".join(ch for ch in text if ch not in _NOISE_CHARS)
    return text.strip()


def stitch_lines(lines: list[OcrLine], line_joiner: str = "") -> str:
    """把多行按阅读顺序拼成一段文本。

    line_joiner 控制"同一区域内的多行"怎么分隔。默认空串（直接相连）适合
    单词/数字这类无空格语言；语文作业必须用换行，否则句子会糊成一坨。
    """
    ordered = sort_reading_order(lines)
    kept = [l.text for l in ordered if not l.dropped and l.text]
    return clean_text(line_joiner.join(kept))


def build_absolute_lines(
    result: OcrBatchResult,
    crop_bbox: tuple[int, int, int, int],
) -> OcrBatchResult:
    """把 crop 内的局部坐标换算回整页坐标，方便可视化与人工核对。"""
    dx = crop_bbox[0]
    dy = crop_bbox[1]
    for line in result.lines:
        if line.bbox is not None:
            line.bbox = offset_bbox(line.bbox, dx, dy)
    return result


def evaluate_review(
    *,
    detected_handwriting: bool,
    ocr: OcrBatchResult | None,
    alignment_score: float,
    fragments: int,
    cfg: ReviewConfig,
    crop_mask: np.ndarray | None = None,
) -> ReviewVerdict:
    """决定这个答案是否需要人工复核，并给出可读的 reasons。"""
    reasons: list[str] = []
    priority = 0

    if not detected_handwriting:
        reasons.append("未在该题区域检测到新增笔迹（学生可能未作答，或笔迹过淡）")
        priority += 2

    if alignment_score < cfg.min_alignment_score:
        reasons.append(f"页面配准分数偏低（{alignment_score:.2f} < {cfg.min_alignment_score}），差分结果可能不可靠")
        priority += 3

    if fragments > cfg.max_fragments:
        reasons.append(f"笔迹碎成 {fragments} 段（阈值 {cfg.max_fragments}），可能被错误切分或字迹过淡")
        priority += 1
    if ocr is None:
        reasons.append("未执行 OCR")
        priority += 1
    else:
        kept = ocr.kept
        if not kept:
            reasons.append("未识别出任何答案文本")
            priority += 3
        else:
            conf = ocr.confidence
            if conf < cfg.min_ocr_confidence:
                worst = min(kept, key=lambda l: l.confidence)
                reasons.append(
                    f"识别置信度偏低（最低 {conf:.2f} < {cfg.min_ocr_confidence}，"
                    f"出现在 \"{worst.text}\"）"
                )
                priority += 2

        # 只有"识别得很确信、却被 mask 判掉"的行才值得复核。
        # 丢掉低置信度的印刷残影是预期行为，不是问题——不该把每个
        # 有残影的答案都推给人工，那样复核队列会淹掉真正需要看的。
        suspicious = [
            l for l in ocr.lines
            if l.dropped and l.confidence >= cfg.min_ocr_confidence and l.mask_ratio > 0.05
        ]
        if suspicious:
            worst = max(suspicious, key=lambda l: l.confidence)
            reasons.append(
                f"有 {len(suspicious)} 行识别置信度较高却被判为非手写而丢弃"
                f"（如 \"{worst.text}\" conf={worst.confidence:.2f}），请确认是否漏掉了真实作答"
            )
            priority += 2

    if crop_mask is not None and crop_mask.size:
        # 涂改：墨迹异常浓重，通常是反复涂画。OCR 几乎必然出错。
        ratio = cv2.countNonZero(crop_mask) / float(crop_mask.size)
        if ratio > 0.55:
            reasons.append("该区域墨迹覆盖率异常高，疑似涂改或大面积涂黑")
            priority += 2

    return ReviewVerdict(required=bool(reasons), reasons=reasons, priority=priority)


def mask_for_region(full_mask: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """从整页 mask 中裁出某个区域的局部 mask。"""
    if full_mask is None or full_mask.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    x1, y1, x2, y2 = bbox
    return full_mask[y1:y2, x1:x2]


def local_mask_to_page(mask: np.ndarray, bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> np.ndarray:
    """把局部 mask 贴回整页，用于 debug overlay。"""
    page = np.zeros(shape[:2], dtype=np.uint8)
    if mask is None or mask.size == 0:
        return page
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    ph, pw = shape[:2]
    x2, y2 = min(pw, x1 + w), min(ph, y1 + h)
    if x2 > x1 and y2 > y1:
        page[y1:y2, x1:x2] = mask[: y2 - y1, : x2 - x1]
    return page


def overlay_roi_mask(image: np.ndarray, bbox: tuple[int, int, int, int], mask: np.ndarray) -> np.ndarray:
    """把该区域的手写像素高亮叠在原图上（用于确认 mask 准不准）。"""
    out = image.copy()
    if mask is None or mask.size == 0:
        return out
    x1, y1, x2, y2 = bbox
    h, w = mask.shape[:2]
    ph, pw = out.shape[:2]
    x2, y2 = min(pw, x1 + w), min(ph, y1 + h)
    if x2 <= x1 or y2 <= y1:
        return out
    sub_mask = mask[: y2 - y1, : x2 - x1] > 0
    sub_img = out[y1:y2, x1:x2]
    tint = np.zeros_like(sub_img)
    tint[..., 1] = 255  # 绿色
    sub_img[sub_mask] = (0.45 * sub_img[sub_mask] + 0.55 * tint[sub_mask]).astype(np.uint8)
    return out


def bbox_outline(image: np.ndarray, bbox: tuple[int, int, int, int], color: tuple[int, int, int],
                 thickness: int = 2) -> np.ndarray:
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
    return image


def empty_bbox_mask(shape: tuple[int, int]) -> np.ndarray:
    return bbox_mask((0, 0, 0, 0), shape)
