"""手写提取：模板差分。

技术方案第十一~十三节。要点：

    手写 = student_binary AND NOT template_binary

比 absdiff 更符合业务含义——落在印刷文字上的笔迹才是我们要的，
印刷文字本身不是新内容。

第十二节提到的坑也在这里处理：学生笔画压在横线上时，直接减模板会把
笔画啃掉。做法是"保守抑制"——模板只做极小膨胀（默认 1px），宁可留下
一点鬼影，也不要吃掉真实笔画；鬼影靠后续的 mask 过滤和 OCR 重叠判定清掉。
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .config import DifferenceConfig, NoiseFilterConfig

log = logging.getLogger(__name__)


class DifferenceResult:
    __slots__ = ("mask", "method", "notes")

    def __init__(self, mask: np.ndarray, method: str, notes: list[str] | None = None):
        self.mask = mask
        self.method = method
        self.notes = notes or []


def subtract_template(
    student_bin: np.ndarray,
    template_bin: np.ndarray,
    cfg: DifferenceConfig,
) -> DifferenceResult:
    """核心差分。"""
    notes: list[str] = []

    if student_bin.shape != template_bin.shape:
        template_bin = cv2.resize(template_bin, (student_bin.shape[1], student_bin.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        notes.append("模板尺寸与学生页不一致，已缩放到学生页尺寸")

    if cfg.method == "absdiff":
        # 仅用于排查问题时对照，不推荐作为生产路径。
        notes.append("当前使用 absdiff，对光照和 1px 对齐误差敏感")
        mask = cv2.bitwise_and(cv2.absdiff(student_bin, template_bin), student_bin)
        return DifferenceResult(mask, "absdiff", notes)

    if cfg.method == "none":
        return DifferenceResult(student_bin.copy(), "none", ["未做模板差分，直接使用学生页二值图"])

    template_guard = template_bin
    if cfg.template_dilate_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (2 * cfg.template_dilate_px + 1,) * 2)
        template_guard = cv2.dilate(template_bin, k)
        if cfg.template_dilate_px >= 3:
            notes.append(f"模板膨胀 {cfg.template_dilate_px}px 偏大，可能吃掉贴近印刷线的笔画")

    mask = cv2.bitwise_and(student_bin, cv2.bitwise_not(template_guard))
    return DifferenceResult(mask, "template_subtract", notes)


def detect_handwriting_no_template(binary: np.ndarray, cfg: NoiseFilterConfig) -> DifferenceResult:
    """无模板回退：靠形态学和连通域特征猜哪里是手写。

    这是弱方案。印刷体和手写体在"墨迹连通域"层面的差别只有统计性的
    （手写更松散、笔画更细、更不规则），没有模板做基准时误检率明显偏高。
    存在的意义是：拿到一份没登记过的作业时至少能给出候选区域供人工确认，
    而不是直接失败。
    """
    notes = [
        "无模板模式：结果为启发式候选，未做印刷内容剔除，误检率显著高于模板模式",
    ]

    # 1) 先干掉细长的线框（作业本横线、方格线、表格边框）。
    h, w = binary.shape[:2]
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, w // 25), 1))
    vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, h // 25)))
    lines = cv2.dilate(binary, horizontal) | cv2.dilate(binary, vertical)
    line_mask = cv2.morphologyEx(lines, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9)))
    candidate = cv2.bitwise_and(binary, cv2.bitwise_not(line_mask))

    # 2) 尺寸/形状过滤。
    cleaned = _filter_components(candidate, cfg)
    notes.append("线框已剔除；印刷正文与手写在此模式下无法区分")
    return DifferenceResult(cleaned, "no_template_heuristic", notes)


def _filter_components(binary: np.ndarray, cfg: NoiseFilterConfig) -> np.ndarray:
    """按面积、长宽比剔除噪点，保留有可能是字的连通域。"""
    if cfg.open_kernel > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.open_kernel,) * 2))
    if cfg.close_kernel > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE,
                                  cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.close_kernel,) * 2))

    total_area = binary.size or 1
    num, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):  # 0 是背景
        x, y, w, h, area = stats[i]
        if area < cfg.min_component_area:
            continue
        if area > total_area * cfg.max_component_area_ratio:
            continue  # 阴影/污渍/订书钉
        aspect = max(w, h) / max(1, min(w, h))
        if aspect > cfg.max_aspect_ratio:
            continue  # 细长残留
        keep[i] = True

    out = np.zeros_like(binary)
    out[keep[labels]] = 255
    return out


def mask_coverage(mask: np.ndarray) -> float:
    return float(cv2.countNonZero(mask)) / float(mask.size or 1)
