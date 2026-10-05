"""PDF 渲染。

技术方案第四节。这里在原方案基础上补了两件事：

1. 页面尺寸一致性检查（页数、尺寸、方向），不一致时给出明确原因而不是
   让它在后面某一层莫名其妙地失败。
2. 渲染结果的 dtype 修正。PyMuPDF 的 pix.samples 在不同版本/色彩空间下
   可能是 RGB 也可能是 BGR，这里显式转 RGB 再交给 OpenCV，避免颜色通道
   在后面某一步悄悄反了。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import fitz
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class RenderedPage:
    index: int  # 0-based
    image: np.ndarray  # BGR, uint8
    width_pt: float
    height_pt: float
    dpi: int

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def height(self) -> int:
        return int(self.image.shape[0])


@dataclass
class RenderReport:
    page_count: int
    size_mismatch: bool
    size_diff_ratio: float
    notes: list[str]


class PdfError(RuntimeError):
    pass


def render_pdf(path: str | Path, dpi: int = 300, max_pages: int | None = None) -> list[RenderedPage]:
    """把 PDF 渲染成 BGR 图像列表。

    page.get_pixmap 会自动应用页面的 /Rotate，所以倒置页在这里就被纠正了。
    """
    p = Path(path)
    if not p.exists():
        raise PdfError(f"文件不存在: {p}")

    pages: list[RenderedPage] = []
    with fitz.open(str(p)) as doc:
        if doc.needs_pass:
            raise PdfError(f"PDF 已加密，无法读取: {p}")
        total = doc.page_count
        for i, page in enumerate(doc):
            if max_pages is not None and i >= max_pages:
                break
            try:
                pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
            except Exception as exc:  # pragma: no cover - 依赖底层异常类型
                raise PdfError(f"第 {i + 1} 页渲染失败: {exc}") from exc

            buf = np.frombuffer(pix.samples, dtype=np.uint8)
            expected = pix.height * pix.width * pix.n
            if buf.size != expected:  # pragma: no cover - 防御性
                raise PdfError(f"第 {i + 1} 页像素缓冲区异常: {buf.size} != {expected}")

            rgb = buf.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:  # alpha=False 理论上不会走到，兜底
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
            elif pix.n == 1:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)

            rect = page.rect
            pages.append(
                RenderedPage(
                    index=i,
                    image=cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2BGR),
                    width_pt=float(rect.width),
                    height_pt=float(rect.height),
                    dpi=dpi,
                )
            )
        if not pages:
            raise PdfError(f"PDF 没有任何可渲染页面: {p}")
        if max_pages is None and doc.page_count == 0:
            raise PdfError(f"PDF 没有任何页面: {p}")
        _ = total

    return pages


def compare_geometry(
    template: Sequence[RenderedPage],
    student: Sequence[RenderedPage],
    tolerance: float = 0.02,
) -> RenderReport:
    """比较模板与学生 PDF 的页数与尺寸。

    尺寸差小于 2% 视为一致——扫描件和电子版在同一 DPI 下渲染出完全相同的
    尺寸才是常态，有微小差异不该阻断流程。
    """
    notes: list[str] = []
    if len(template) != len(student):
        notes.append(
            f"页数不一致: 模板 {len(template)} 页, 学生 {len(student)} 页。"
            "将按最大重叠页配对，多出的页单独标记。"
        )

    diffs: list[float] = []
    for t, s in zip(template, student):
        for label, tw, th, sw, sh in (
            ("宽", t.width, t.height, s.width, s.height),
            ("高", t.width, t.height, s.width, s.height),
        ):
            base = max(tw, th)
            other = max(sw, sh)
            if base == 0:
                continue
            diffs.append(abs(other - base) / base)
            _ = label

    max_diff = max(diffs) if diffs else 0.0
    if max_diff > tolerance:
        notes.append(
            f"页面尺寸差异 {max_diff * 100:.1f}% 超过阈值 {tolerance * 100:.0f}%，"
            "扫描分辨率可能与模板不同。"
        )

    return RenderReport(
        page_count=len(student),
        size_mismatch=max_diff > tolerance,
        size_diff_ratio=max_diff,
        notes=notes,
    )


def imwrite_unicode(path: str | Path, image: np.ndarray,
                    params: Sequence[int] | None = None) -> bool:
    """写图片，路径含中文也能成功。

    Windows 上 cv2.imwrite 走的是窄字符路径接口，遇到非 ASCII 路径会
    直接返回 False 而且**不落盘、不报错**。调用方通常不检查返回值，于是
    「图存不下来」就变成了一次完全静默的失败——输出目录叫
    `识别结果\\第一次\\...` 时，管线记下了文件名却什么都没写，
    核对页的图片自然全是裂图。

    先在内存里编码，再交给 Python 的文件接口写出，就不受路径字符集影响了。
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(p.suffix or ".png", image,
                           list(params) if params else [])
    if not ok:
        return False
    buf.tofile(str(p))  # ndarray.tofile 走 Python 文件接口，支持中文路径
    return True


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """读图片，路径含中文也能成功。cv2.imread 同理会直接返回 None。"""
    p = Path(path)
    if not p.exists():
        return None
    try:
        data = np.fromfile(str(p), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def render_to_file(page: RenderedPage, out_path: str | Path, jpeg_quality: int = 88) -> Path:
    p = Path(out_path)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if p.suffix.lower() in {".jpg", ".jpeg"} else None
    if not imwrite_unicode(p, page.image, params):
        log.warning("图片写入失败: %s", p)
    return p
