"""手写区域合并与切行。

技术方案第十七、十八节：

* 连通域过滤去掉孤立噪点，但阈值不能太狠——小数点、句号、"i"的点、
  数字 1 的顶部、中文的小撇捺都可能只有几十像素。
* 不能逐个连通域做 OCR。几十个 component 逐个识别既慢又会打乱语序。
  先用水平膨胀把同一行的字粘成一块，再按块识别。

语文作业的答案经常是整段文字，所以这里额外做了"过高的块按水平投影再切"，
否则一段话会被当成一行送进识别模型，高度被压扁后准确率会掉很多。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import GroupingConfig, NoiseFilterConfig
from .geometry import BBox, as_bbox


@dataclass
class Region:
    bbox: BBox
    area: int = 0
    fragments: int = 0  # 该区域由多少个原始连通域组成

    @property
    def width(self) -> int:
        return self.bbox[2] - self.bbox[0]

    @property
    def height(self) -> int:
        return self.bbox[3] - self.bbox[1]


@dataclass
class RegionSet:
    regions: list[Region] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def clean_mask(mask: np.ndarray, cfg: NoiseFilterConfig) -> np.ndarray:
    """去噪点 + 补断笔，但不改变整体形状。"""
    if cfg.open_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.open_kernel,) * 2))
    if cfg.close_kernel > 1:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                cv2.getStructuringElement(cv2.MORPH_RECT, (cfg.close_kernel,) * 2))
    return mask


def remove_line_residue(mask: np.ndarray, cfg: NoiseFilterConfig) -> np.ndarray:
    """剔除细长的线框残留（作业本横线、方格线、框边在差分后的残影）。

    这些残影又长又扁，会把后面统计的中位行高拉偏，进而让真正的字迹被误判成
    "过高的块"而被切碎。所以必须先清掉再统计。
    """
    if cv2.countNonZero(mask) == 0:
        return mask

    num, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(num, dtype=bool)
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < cfg.min_component_area:
            continue
        short = max(1, min(w, h))
        aspect = max(w, h) / short
        if aspect > cfg.max_aspect_ratio:
            continue  # 又长又扁 -> 线框残留
        keep[i] = True

    out = np.zeros_like(mask)
    out[keep[labels]] = 255
    return out


def group_regions(mask: np.ndarray, cfg: GroupingConfig, noise: NoiseFilterConfig) -> RegionSet:
    """把 mask 合并成按行/按块的答案区域。"""
    notes: list[str] = []
    if cv2.countNonZero(mask) == 0:
        return RegionSet([], ["该区域未检测到任何新增笔迹（可能学生未作答）"])

    mask = clean_mask(mask, noise)
    before = cv2.countNonZero(mask)
    mask = remove_line_residue(mask, noise)
    removed = before - cv2.countNonZero(mask)
    if removed > 0 and before > 0:
        notes.append(f"已剔除细长线框残留 {removed}px")
    if cv2.countNonZero(mask) == 0:
        return RegionSet([], ["剔除线框残留后该区域为空（可能只是印刷线条，未作答）"])

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                       (max(1, cfg.dilation_width), max(1, cfg.dilation_height)))
    grouped = cv2.dilate(mask, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(grouped, connectivity=8)

    boxes: list[tuple[BBox, int]] = []
    for i in range(1, num):
        x, y, w, h, area = stats[i]
        if area < noise.min_component_area:
            continue
        boxes.append((as_bbox((x, y, x + w, y + h)), area))

    if not boxes:
        return RegionSet([], ["连通域全部被尺寸过滤条件剔除，mask 可能过碎"])

    # 中位行高只用"看起来像文字"的块来算：太扁的块（残留噪声）先排掉，
    # 否则会把基准压到几像素，真字迹全被误判成"超高块"。
    plausible = [b for b, _ in boxes if (b[3] - b[1]) > 0]
    heights = np.array([b[3] - b[1] for b in plausible], dtype=np.float64)
    median_h = float(np.median(heights)) if heights.size else 0.0
    if median_h <= 2:
        median_h = max(heights) if heights.size else 0.0
    if median_h <= 0:
        return RegionSet([], ["无法估计行高，区域可能过小"])

    regions: list[Region] = []
    min_h = max(3.0, median_h * cfg.min_region_height_ratio)
    for bbox, area in boxes:
        height = bbox[3] - bbox[1]
        width = bbox[2] - bbox[0]
        box_area = max(1, width * height)
        # 又长又扁的块在这一步再筛一次（膨胀后 aspect 会变小，所以用更宽的阈值）
        if height < median_h * 0.25 and width > max(8, height * 4):
            continue
        # 空心框（作文框/表格边框残影）：外接框很大但墨迹只占周长，填充率极低。
        if area / box_area < noise.min_fill_ratio and box_area > (median_h * median_h):
            notes.append(f"已剔除疑似框线残影 {width}x{height}（填充率 {area / box_area:.1%}）")
            continue
        if height < min_h:
            continue
        if height > median_h * cfg.tall_region_ratio:
            pieces = _split_by_projection(mask, bbox, median_h * cfg.line_gap_ratio, median_h)
            if pieces:
                regions.extend(pieces)
                notes.append(f"检测到 {height / median_h:.1f} 倍行高的连通块，已按水平投影切分")
            else:
                # 切不开就原样保留。返回空列表会让整块凭空消失。
                regions.append(Region(bbox=bbox, area=area))
        else:
            regions.append(Region(bbox=bbox, area=area))

    regions = _split_by_gap(mask, regions, median_h * cfg.line_gap_ratio, median_h)
    min_keep = max(3.0, median_h * cfg.min_region_height_ratio * 0.6)
    regions = [r for r in regions if (r.bbox[3] - r.bbox[1]) >= min_keep]
    regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))
    if not regions:
        return RegionSet([], ["所有候选区域的高度都低于阈值，视为无有效笔迹"])
    return RegionSet(regions, notes)


def _count_fragments(mask: np.ndarray) -> int:
    num, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return max(0, num - 1)


def _split_by_projection(mask: np.ndarray, bbox: BBox, min_gap: float, line_height: float) -> list[Region]:
    """按水平投影的行间空白切分一个过高的块。"""
    x1, y1, x2, y2 = bbox
    sub = mask[y1:y2, x1:x2]
    if sub.size == 0:
        return []
    profile = (sub > 0).sum(axis=1).astype(np.float32)
    return _profile_to_regions(profile, bbox, min_gap, line_height)


def _split_by_gap(mask: np.ndarray, regions: list[Region], min_gap: float, line_height: float) -> list[Region]:
    """相邻块之间如果存在明显行间空白，拆开。

    水平膨胀把上下两行粘在一起的情况很常见，尤其是小学生字偏大、行距偏小。
    """
    if len(regions) < 2:
        return regions
    out: list[Region] = []
    for r in regions:
        x1, y1, x2, y2 = r.bbox
        sub = mask[y1:y2, x1:x2]
        if sub.size == 0:
            out.append(r)
            continue
        profile = (sub > 0).sum(axis=1).astype(np.float32)
        pieces = _profile_to_regions(profile, r.bbox, min_gap, line_height)
        out.extend(pieces if pieces else [r])
    return out


def _profile_to_regions(profile: np.ndarray, bbox: BBox, min_gap: float, line_height: float) -> list[Region]:
    """把行投影曲线切成若干行段，gap 小于 min_gap 的合并为同一行。"""
    if profile.size == 0 or not profile.any():
        return []
    x1, y1, x2, y2 = bbox
    active = profile > 0
    gap_px = max(2.0, float(min_gap))

    segments: list[tuple[int, int]] = []
    start: int | None = None
    blank = 0
    for idx, is_on in enumerate(active):
        if is_on:
            if start is None:
                start = idx
            blank = 0
        else:
            if start is not None:
                blank += 1
                if blank >= gap_px:
                    segments.append((start, idx - blank + 1))
                    start = None
                    blank = 0
    if start is not None:
        segments.append((start, len(active) - blank))

    if len(segments) <= 1:
        return []

    regions: list[Region] = []
    min_h = max(2, line_height * 0.25)
    kept: list[Region] = []
    for s, e in segments:
        if e - s < min_h:
            continue
        kept.append(Region(bbox=as_bbox((x1, y1 + s, x2, y1 + e)),
                           area=int(profile[s:e].sum()),
                           fragments=0))
    if not kept and segments:
        # 全都"太矮"时，宁可保留最长的一段，也不要整块丢掉
        s, e = max(segments, key=lambda seg: seg[1] - seg[0])
        kept.append(Region(bbox=as_bbox((x1, y1 + s, x2, y1 + e)),
                           area=int(profile[s:e].sum()), fragments=0))
    regions.extend(kept)
    return regions


def count_shards(regions: list[Region], expected_lines: int | None = None) -> tuple[int, int]:
    """返回 (区域总数, 碎裂数)。

    "碎裂"指明显比同题其它区域矮得多的小碎块——那通常意味着 mask 碎了或字迹
    过淡，而不是学生真的写了那么多行。所以不能直接拿"区域总数"当碎裂指标：
    一段三行的阅读理解答案本来就会产生 3 个区域，那是正常的。
    """
    if not regions:
        return 0, 0

    heights = np.array([r.height for r in regions], dtype=np.float64)
    median_h = float(np.median(heights))
    if median_h <= 0:
        return len(regions), 0

    short = heights < median_h * 0.45
    shards = int(short.sum())

    if expected_lines:
        # 标注了预期行数时，区域数明显超出预期也算异常。
        if len(regions) > max(expected_lines * 2, expected_lines + 2):
            shards = max(shards, len(regions) - expected_lines)

    return len(regions), shards


def sort_regions(regions: list[Region]) -> list[Region]:
    """按阅读顺序排列区域：先分行（按垂直重叠），行内从左到右。

    为什么不能直接按上边缘 y 排：学生手写的行不会绝对水平，同一行的左右两段
    可能差几像素。实测「我也」在 y=652、主行在 y=654，只差 2px 却会被排到
    句首，答案读起来就乱了。按垂直重叠聚成一行、再用 x 决定先后才对。
    """
    if len(regions) <= 1:
        return list(regions)

    heights = np.array([r.height for r in regions], dtype=np.float64)
    heights = heights[heights > 0]
    line_h = float(np.median(heights)) if heights.size else 1.0

    ordered = sorted(regions, key=lambda r: (r.bbox[1], r.bbox[0]))
    rows: list[list[Region]] = []
    extents: list[list[int]] = []

    for region in ordered:
        top, bottom = region.bbox[1], region.bbox[3]
        best_i, best_ov = -1, 0
        for i, (rt, rb) in enumerate(extents):
            ov = min(bottom, rb) - max(top, rt)
            if ov > best_ov:
                best_i, best_ov = i, ov
        need = min(max(2, int(line_h * 0.25)), max(1, (bottom - top) * 0.6))
        if best_i >= 0 and best_ov >= need:
            rows[best_i].append(region)
            extents[best_i][0] = min(extents[best_i][0], top)
            extents[best_i][1] = max(extents[best_i][1], bottom)
        else:
            rows.append([region])
            extents.append([top, bottom])

    out: list[Region] = []
    for row in rows:
        row.sort(key=lambda r: r.bbox[0])
        out.extend(row)
    return out


def region_mask(mask: np.ndarray, bbox: BBox) -> np.ndarray:
    """取出某个区域对应的 mask 子图，坐标已对齐到该 crop 的局部坐标系。"""
    x1, y1, x2, y2 = bbox
    return mask[y1:y2, x1:x2]
