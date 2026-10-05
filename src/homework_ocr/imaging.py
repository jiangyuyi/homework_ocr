"""二值化与光照归一化。

技术方案第九、十节。核心观点：不要用 RGB absdiff。

实际扫描/拍照的干扰源：纸张偏色、背面透印、阴影渐变、JPEG 压缩、
墨粉/铅笔深浅。这里先用大核形态学估计背景并做除法，把"局部光照"消掉，
再做自适应阈值，这样模板和学生页的阈值行为才一致，差分才有意义。
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import DifferenceConfig
from .alignment import to_gray


def _odd(value: int, minimum: int = 3) -> int:
    value = max(minimum, int(value))
    return value if value % 2 == 1 else value + 1


def normalize_illumination(gray: np.ndarray, kernel_size: int = 51) -> np.ndarray:
    """除以局部背景，消除阴影和纸张渐变。

    用形态学闭运算估计"纸背亮度"：白纸区域闭合后接近白色，笔画区域被填上，
    于是 gray / background 之后背景≈1、笔画≈低值，阈值就稳定了。
    """
    k = _odd(kernel_size, 15)
    background = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    background = cv2.GaussianBlur(background, (0, 0), sigmaX=k / 4.0, sigmaY=k / 4.0)
    # 分母加一个下限，避免纯黑区域除零放大噪声。
    safe = np.maximum(background.astype(np.float32), 1.0)
    normalized = gray.astype(np.float32) / safe * 255.0
    return np.clip(normalized, 0, 255).astype(np.uint8)


def binarize(image: np.ndarray, cfg: DifferenceConfig) -> np.ndarray:
    """返回"墨迹为 255、纸白为 0"的二值图。"""
    gray = to_gray(image)
    if cfg.illumination:
        gray = normalize_illumination(gray, cfg.illumination_kernel)

    gray = cv2.medianBlur(gray, 3)
    block = _odd(cfg.adaptive_block_size)
    binary = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block,
        float(cfg.adaptive_c),
    )
    return binary


def estimate_stroke_width(binary: np.ndarray) -> float:
    """用距离变换的中位数估计笔画宽度（中值意义下对图像尺寸不敏感）。"""
    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 3)
    nonzero = dist[dist > 0]
    if nonzero.size == 0:
        return 0.0
    return float(np.median(nonzero) * 2.0)


def ink_ratio(binary: np.ndarray) -> float:
    return float(cv2.countNonZero(binary)) / float(binary.size or 1)


def binarize_pair(student: np.ndarray, template: np.ndarray, cfg: DifferenceConfig) -> tuple[np.ndarray, np.ndarray]:
    """模板和学生页必须用完全相同的参数二值化，否则阈值差异会变成整片"差异"。"""
    return binarize(student, cfg), binarize(template, cfg)
