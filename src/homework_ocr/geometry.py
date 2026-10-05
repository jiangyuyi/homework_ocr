"""矩形与二值图的基础工具。

坐标一律用 [x1, y1, x2, y2]，左上原点，单位像素，和技术方案保持一致。
"""

from __future__ import annotations

from typing import Iterable, Sequence

import cv2
import numpy as np

BBox = tuple[int, int, int, int]


def as_bbox(values: Sequence[float]) -> BBox:
    x1, y1, x2, y2 = (int(round(v)) for v in values)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_bbox(bbox: Sequence[float], padding: int, width: int, height: int) -> BBox:
    """外扩并裁到画面内。

    padding 会按 300 DPI 的经验值给，但换 DPI 时最好同步调。
    """
    x1, y1, x2, y2 = as_bbox(bbox)
    return (
        max(0, x1 - padding),
        max(0, y1 - padding),
        min(width, x2 + padding),
        min(height, y2 + padding),
    )


def bbox_area(bbox: Sequence[float]) -> int:
    x1, y1, x2, y2 = as_bbox(bbox)
    return max(0, x2 - x1) * max(0, y2 - y1)


def bbox_iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = as_bbox(a)
    bx1, by1, bx2, by2 = as_bbox(b)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    union = bbox_area(a) + bbox_area(b) - inter
    return inter / union if union else 0.0


def bbox_overlap_area(a: Sequence[float], mask: np.ndarray) -> int:
    """box 与二值 mask 的重叠像素数。"""
    if mask.size == 0:
        return 0
    h, w = mask.shape[:2]
    x1, y1, x2, y2 = as_bbox(a)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return 0
    return int(cv2.countNonZero(mask[y1:y2, x1:x2]))


def crop(image: np.ndarray, bbox: Sequence[float], pad: int = 0) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1, x2, y2 = expand_bbox(bbox, pad, w, h)
    if x2 <= x1 or y2 <= y1:
        return image[0:0, 0:0]
    return image[y1:y2, x1:x2]


def merge_bbox(boxes: Iterable[Sequence[float]]) -> BBox:
    boxes = list(boxes)
    if not boxes:
        return (0, 0, 0, 0)
    arr = np.array([as_bbox(b) for b in boxes], dtype=np.int64)
    return (
        int(arr[:, 0].min()),
        int(arr[:, 1].min()),
        int(arr[:, 2].max()),
        int(arr[:, 3].max()),
    )


def offset_bbox(bbox: Sequence[float], dx: int, dy: int) -> BBox:
    x1, y1, x2, y2 = as_bbox(bbox)
    return as_bbox((x1 + dx, y1 + dy, x2 + dx, y2 + dy))


def bbox_mask(bbox: Sequence[float], shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    m = np.zeros((h, w), dtype=np.uint8)
    x1, y1, x2, y2 = as_bbox(bbox)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 255
    return m


def mask_ratio(bbox: Sequence[float], mask: np.ndarray) -> float:
    """box 区域内被 mask 覆盖的比例——用于判断 OCR 结果是否落在手写上。"""
    area = bbox_area(bbox)
    if area == 0 or mask.size == 0:
        return 0.0
    return bbox_overlap_area(bbox, mask) / area
