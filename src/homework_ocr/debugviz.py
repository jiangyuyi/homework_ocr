"""Debug overlay。

技术方案第二十五节。这层图的价值在于：出问题时能立刻看出是哪一层坏了。

    蓝框 = 模板定义的答案区域
    绿字 = 检测到的手写区域（mask 高亮）
    橙框 = OCR 识别出的文本行
    红字 = 需要人工复核的识别结果

一张图就能回答："是没对齐、mask 没提干净、还是 OCR 不认识这个字"。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .geometry import BBox
from .ocr.base import OcrLine
from .pdfio import imwrite_unicode

log = logging.getLogger(__name__)

COLOR_ROI = (255, 160, 0)      # BGR: 蓝
COLOR_MASK = (0, 200, 0)       # 绿
COLOR_OK = (0, 200, 255)       # 橙
COLOR_REVIEW = (0, 0, 255)     # 红
COLOR_TEXT = (40, 40, 40)
COLOR_BG = (255, 255, 255)

# 覆盖层中文标注需要的中文字体。找不到就退回英文标签——不因为字体问题
# 让整个可视化流程挂掉。
_CJK_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _try_load_cjk_font() -> str | None:
    """找一个能显示中文的字体文件路径。

    不用 cv2.putText：它对中文只能靠 Hershey 矢量字（画不出汉字），
    而 cv2.freetype 在部分 opencv-python 构建里根本没编进来。
    这里统一走 PIL 渲染，再贴回 BGR 图。
    """
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
        "C:/Windows/Fonts/simsun.ttc",
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/Library/Fonts/Arial Unicode.ttf",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    ]
    for path in candidates:
        if Path(path).exists():
            try:
                from PIL import ImageFont

                ImageFont.truetype(path, 16)
                return path
            except Exception:
                continue
    return None


_FONT_PATH: str | None = None
_FONT_READY = False
_PIL_FONT_CACHE: dict[int, object] = {}


def _font_path() -> str | None:
    global _FONT_PATH, _FONT_READY
    if not _FONT_READY:
        _FONT_PATH = _try_load_cjk_font()
        _FONT_READY = True
    return _FONT_PATH


def has_cjk_support() -> bool:
    return _font_path() is not None


def _pil_font(size: int):
    from PIL import ImageFont

    key = max(8, int(size))
    if key not in _PIL_FONT_CACHE:
        path = _font_path()
        _PIL_FONT_CACHE[key] = ImageFont.truetype(path, key) if path else ImageFont.load_default()
    return _PIL_FONT_CACHE[key]


def _put(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    color: tuple[int, int, int],
    scale: float = 0.5,
    thickness: int = 1,
) -> None:
    """在 BGR 图上写文本。能显示中文就用 PIL，否则退回 cv2.putText。"""
    if not text:
        return
    if not has_cjk_support():
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return

    from PIL import Image, ImageDraw

    size = max(9, int(round(scale * 34)))
    font = _pil_font(size)
    try:
        left, top, right, bottom = font.getbbox(text)
    except Exception:  # pragma: no cover
        left, top, right, bottom = (0, 0, int(len(text) * size * 0.6), size)
    tw, th = max(1, right - left), max(1, bottom - top)

    x, y = int(org[0]), int(org[1])
    if y - th < 0:
        y = th + int(org[1]) + 4

    patch = Image.new("RGBA", (tw + 4, th + 4), (0, 0, 0, 0))
    pd = ImageDraw.Draw(patch)
    # PIL 用 RGB，OpenCV 是 BGR。
    pd.text((2 - left, 2 - top), text, font=font, fill=(int(color[2]), int(color[1]), int(color[0]), 255))

    ih, iw = img.shape[:2]
    if x >= iw or y >= ih:
        return
    region_h = min(patch.height, ih - y)
    region_w = min(patch.width, iw - x)
    if region_h <= 0 or region_w <= 0:
        return
    # 文字可能超出画布右/下边缘，裁到实际可用的区域再合成。
    if (region_h, region_w) != (patch.height, patch.width):
        patch = patch.crop((0, 0, region_w, region_h))

    region = img[y:y + region_h, x:x + region_w]
    if region.size == 0:
        return
    if region.shape[2] == 3:
        # OpenCV 是 BGR，PIL 期望 RGB，直接反转通道。
        rgb = np.ascontiguousarray(region[..., ::-1])
        base = Image.fromarray(rgb).convert("RGBA")
        merged = Image.alpha_composite(base, patch)
        img[y:y + region_h, x:x + region_w] = cv2.cvtColor(np.array(merged.convert("RGB")), cv2.COLOR_RGB2BGR)
    _ = thickness


def _put_bg(img: np.ndarray, text: str, org: tuple[int, int],
            color: tuple[int, int, int], scale: float = 0.5) -> None:
    """白底黑边标签，保证叠在图像上也读得清。"""
    if not text:
        return
    h = max(9, int(round(scale * 34)))
    try:
        box = _pil_font(h).getbbox(text)
        tw = max(1, box[2] - box[0])
    except Exception:  # pragma: no cover
        tw = int(len(text) * h * 0.6)
    th = int(h * 1.25)

    x, y = int(org[0]), int(org[1]) - th
    if y < 0:
        y = 0
    x = max(0, min(x, img.shape[1] - tw - 4))
    y = min(y, max(0, img.shape[0] - th - 2))

    cv2.rectangle(img, (x, y), (x + tw + 4, y + th), (255, 255, 255), -1)
    cv2.rectangle(img, (x, y), (x + tw + 4, y + th), color, 1)
    _put(img, text, (x + 2, y + th - 3), color, scale)


@dataclass
class RegionDebug:
    """渲染一个题目区域所需的信息。"""

    question_id: str
    roi_bbox: BBox
    region_bbox: BBox | None = None
    ocr_lines: list[OcrLine] | None = None
    review_required: bool = False
    text: str = ""


def render_overlay(
    aligned: np.ndarray,
    mask: np.ndarray,
    items: list[RegionDebug],
    page_number: int,
    scale: float = 0.4,
) -> np.ndarray:
    """生成一页的 debug overlay（BGR 图）。"""
    canvas = aligned.copy()
    h, w = canvas.shape[:2]

    # 1) 手写区域高亮（绿色半透明）
    if mask is not None and mask.size and mask.shape[:2] == (h, w):
        sel = mask > 0
        if sel.any():
            tint = np.zeros_like(canvas)
            tint[..., 1] = 255
            canvas[sel] = (0.62 * canvas[sel] + 0.38 * tint[sel]).astype(np.uint8)

    # 2) 模板 ROI 蓝框
    for item in items:
        cv2.rectangle(canvas, (item.roi_bbox[0], item.roi_bbox[1]),
                      (item.roi_bbox[2], item.roi_bbox[3]), COLOR_ROI, 3)
        _put_bg(canvas, f"Q{item.question_id}", (item.roi_bbox[0] + 4, item.roi_bbox[1] + 20),
                COLOR_ROI, 0.6)

    # 3) 识别出的文本行
    for item in items:
        for line in item.ocr_lines or []:
            if line.bbox is None:
                continue
            color = COLOR_REVIEW if line.dropped or item.review_required else COLOR_OK
            cv2.rectangle(canvas, (line.bbox[0], line.bbox[1]), (line.bbox[2], line.bbox[3]), color, 2)
            if line.dropped:
                tag = "X" if not has_cjk_support() else "丢弃"
            else:
                tag = f"{line.text} {line.confidence:.2f}"
            _put_bg(canvas, tag, (line.bbox[0], line.bbox[1] - 4), color, 0.45)

    # 4) 顶部信息条
    header = f"page {page_number}  {w}x{h}px"
    cv2.rectangle(canvas, (0, 0), (w, 30), COLOR_BG, -1)
    _put(canvas, header, (10, 21), COLOR_TEXT, 0.6, 1)

    if scale != 1.0:
        canvas = cv2.resize(canvas, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return canvas


def render_question_crop(
    aligned: np.ndarray,
    question_id: str,
    bbox: BBox,
    text: str,
    confidence: float,
    review_required: bool,
    reasons: list[str],
    scale: float = 1.0,
) -> np.ndarray:
    """单题答案图：答案区域原图 + 识别文字 + 复核标记。

    技术方案第十九节强调送 OCR 的是原图而不是 mask，这里输出的也是原图 crop，
    方便人工直接核对。
    """
    x1, y1, x2, y2 = bbox
    crop = aligned[y1:y2, x1:x2].copy()
    if crop.size == 0:
        crop = np.full((80, 400, 3), 255, np.uint8)

    bar = 46 if review_required else 26
    canvas = np.full((crop.shape[0] + bar, crop.shape[1], 3), 255, np.uint8)
    canvas[bar:, :, :] = crop

    label = f"Q{question_id}"
    color = COLOR_REVIEW if review_required else COLOR_OK
    _put(canvas, label, (6, bar - 12 if review_required else 19), color, 0.55, 1)

    info = f"{text}  (conf {confidence:.2f})" if text else "(未识别出文本)"
    _put(canvas, info, (46, bar - 12 if review_required else 19), COLOR_TEXT, 0.5, 1)

    if review_required and reasons:
        _put(canvas, "REVIEW: " + reasons[0][:60], (6, 20), COLOR_REVIEW, 0.45, 1)

    if scale != 1.0:
        canvas = cv2.resize(canvas, (int(canvas.shape[1] * scale), int(canvas.shape[0] * scale)),
                            interpolation=cv2.INTER_AREA)
    return canvas


def side_by_side(panels: list[tuple[str, np.ndarray]], scale: float = 0.35) -> np.ndarray | None:
    """把多个中间结果拼成一张对比图（对齐后 / 模板 / mask）。"""
    usable = [(t, p) for t, p in panels if p is not None and p.size]
    if not usable:
        return None

    target_h = min(int(p.shape[0] * scale) for _, p in usable)
    resized = []
    for title, panel in usable:
        ratio = target_h / panel.shape[0]
        r = cv2.resize(panel, (int(panel.shape[1] * ratio), target_h), interpolation=cv2.INTER_AREA)
        resized.append((title, r))

    gap = 12
    width = sum(p.shape[1] for _, p in resized) + gap * (len(resized) - 1)
    canvas = np.full((target_h + 26, width, 3), 245, np.uint8)
    x = 0
    for title, panel in resized:
        canvas[26:26 + target_h, x:x + panel.shape[1]] = panel
        _put(canvas, title, (x + 4, 18), COLOR_TEXT, 0.5, 1)
        x += panel.shape[1] + gap
    return canvas


def put_text(img: np.ndarray, text: str, org: tuple[int, int],
             color: tuple[int, int, int], scale: float = 0.5) -> None:
    """公开的文字绘制入口，自动处理中文字体。"""
    _put(img, text, org, color, scale)


def save_image(path: str | Path, image: np.ndarray, jpeg_quality: int = 88) -> Path:
    p = Path(path)
    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)] if p.suffix.lower() in {".jpg", ".jpeg"} else None
    # 必须用 pdfio.imwrite_unicode：直接 cv2.imwrite 遇到中文路径会静默失败，
    # 结果就是核对页全是裂图，而日志里一个字都没有。
    if not imwrite_unicode(p, image, params):
        log.warning("图片写入失败: %s", p)
    return p
