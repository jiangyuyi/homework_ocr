"""交互式 ROI 标注工具。

技术方案第三十节：ROI 人工标一次，所有学生复用。这是整个系统性价比最高的
一步——比让程序去猜"哪个框属于哪道题"可靠得多，也快得多。

用法：
    homework-ocr annotate --template templates/yuwen_g3_1

窗口内操作：
    鼠标拖拽    框选一个答案区域
    滚轮 / [ ]  调整框选精度（放大观察）
    n           下一个题号（1, 2, 3 ...）
    d           删除当前题号
    Enter / s   保存 template.json
    q / Esc     退出（不保存）
    0           跳到第 0 页之外的所有页（切换页用 PageUp/PageDown）

保存时坐标会按当前显示缩放还原成模板 DPI 下的真实像素坐标。
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .debugviz import put_text
from .template import Question, Template, TemplatePage

log = logging.getLogger(__name__)

HELP_LINES = [
    "drag=box  n=next id  d=del  [ ]=zoom  s=save  q=quit",
    "PgUp/PgDn=page",
]


@dataclass
class DraftBox:
    """正在拖拽中的临时框。"""

    x1: int = 0
    y1: int = 0
    x2: int = 0
    y2: int = 0
    active: bool = False

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (
            min(self.x1, self.x2),
            min(self.y1, self.y2),
            max(self.x1, self.x2),
            max(self.y1, self.y2),
        )


@dataclass
class Viewer:
    template: Template
    pages: list[np.ndarray]
    page_index: int = 0
    scale: float = 0.5
    current_id: str = "1"
    draft: DraftBox = field(default_factory=DraftBox)
    dirty: bool = False
    message: str = "拖拽框选答案区域"

    # ---------- 坐标换算 ----------
    def to_view(self, x: int, y: int) -> tuple[int, int]:
        return int(x * self.scale), int(y * self.scale)

    def to_page(self, x: int, y: int) -> tuple[int, int]:
        s = 1.0 / self.scale if self.scale else 1.0
        return int(x * s), int(y * s)

    def page_spec(self) -> TemplatePage | None:
        return self.template.page(self.page_index + 1)

    def next_id(self) -> None:
        try:
            self.current_id = str(int(self.current_id) + 1)
        except ValueError:
            self.current_id = "x"

    def delete_current(self) -> str:
        page = self.page_spec()
        if not page:
            return "无此页"
        before = len(page.questions)
        page.questions = [q for q in page.questions if q.id != self.current_id]
        return f"已删除 Q{self.current_id}" if len(page.questions) != before else f"本页没有 Q{self.current_id}"

    def commit_draft(self) -> str:
        if not self.draft.active:
            return "没有待保存的框"
        x1, y1, x2, y2 = self.draft.as_tuple()
        if x2 - x1 < 8 or y2 - y1 < 8:
            return "框太小，已忽略"

        page = self.page_spec()
        if page is None:
            return "无此页"

        x1, y1 = max(0, x1), max(0, y1)
        x2 = min(page.width, x2)
        y2 = min(page.height, y2)

        page.questions = [q for q in page.questions if q.id != self.current_id]
        page.questions.append(Question(id=self.current_id, answer_area=[x1, y1, x2, y2]))
        page.questions.sort(key=lambda q: (q.answer_area[1], q.answer_area[0]))
        self.dirty = True
        self.draft.active = False
        return f"Q{self.current_id} = [{x1}, {y1}, {x2}, {y2}]"

    # ---------- 渲染 ----------
    def render(self) -> np.ndarray:
        page = self.pages[self.page_index]
        h, w = page.shape[:2]
        view_w, view_h = int(w * self.scale), int(h * self.scale)
        canvas = cv2.resize(page, (view_w, view_h), interpolation=cv2.INTER_AREA)

        spec = self.page_spec()
        color_for = {q.id: (255, 160, 0) for q in (spec.questions if spec else [])}
        if spec:
            for q in spec.questions:
                x1, y1 = self.to_view(q.answer_area[0], q.answer_area[1])
                x2, y2 = self.to_view(q.answer_area[2], q.answer_area[3])
                color = (0, 220, 0) if q.id == self.current_id else color_for.get(q.id, (255, 160, 0))
                cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3 if q.id == self.current_id else 2)
                cv2.putText(canvas, q.id, (x1 + 6, y1 + 26), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if self.draft.active:
            x1, y1 = self.to_view(self.draft.x1, self.draft.y1)
            x2, y2 = self.to_view(self.draft.x2, self.draft.y2)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 0, 255), 2)

        _header(canvas, self)
        return canvas

    def zoom(self, factor: float) -> str:
        self.scale = float(np.clip(self.scale * factor, 0.1, 4.0))
        return f"缩放 {self.scale:.2f}x"

    def turn_page(self, delta: int) -> str:
        target = self.page_index + delta
        if not 0 <= target < len(self.pages):
            return f"只有 {len(self.pages)} 页"
        self.page_index = target
        return f"第 {self.page_index + 1} 页"

    def summary(self) -> str:
        total = sum(len(p.questions) for p in self.template.pages)
        return f"共 {len(self.pages)} 页, 已标注 {total} 题, 当前题号 Q{self.current_id}"


def _header(canvas: np.ndarray, viewer: Viewer) -> None:
    spec = viewer.page_spec()
    count = len(spec.questions) if spec else 0
    lines = [
        f"page {viewer.page_index + 1}/{len(viewer.pages)}  Q数={count}  zoom={viewer.scale:.2f}x  "
        f"[{viewer.current_id}]",
        viewer.message,
        "  |  ".join(HELP_LINES),
    ]
    # PIL 渲染的中文字体行高明显大于字号（字号 17px 时行高可达 30px+），
    # 行距按经验值给，否则三行会叠在一起。
    line_h = 38
    h = line_h * len(lines) + 12
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], h), (250, 250, 250), -1)
    for i, line in enumerate(lines):
        color = (0, 0, 160) if i == 1 else (40, 40, 40)
        # 必须走 debugviz.put_text：cv2.putText 用的 Hershey 字体画不出汉字。
        put_text(canvas, line, (10, 12 + line_h * i), color, 0.5)
    # 顶栏盖住内容区，标注时注意别把答案框画在最上面。
    _ = cv2.rectangle(canvas, (0, h), (canvas.shape[1], h + 1), (180, 180, 180), -1)


class _MouseState:
    def __init__(self) -> None:
        self.start: tuple[int, int] | None = None
        self.dragging = False


def annotate(template: Template, pages: list[np.ndarray], initial_scale: float = 0.5) -> bool:
    """打开标注窗口。返回 True 表示已保存。"""
    if not pages:
        log.error("模板没有可标注的页面")
        return False

    viewer = Viewer(template=template, pages=pages, scale=initial_scale)
    state = _MouseState()
    win = "annotate"

    try:
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    except cv2.error as exc:  # pragma: no cover
        log.error("无法创建 OpenCV 窗口（无图形界面？）: %s", exc)
        log.error("无 GUI 环境请直接手写 template.json，字段格式见 README。")
        return False

    saved = False
    try:
        while True:
            cv2.imshow(win, viewer.render())
            key = cv2.waitKey(30) & 0xFF

            if key == 27 or key == ord("q"):
                break
            if key in (ord("s"), 13, 10):
                path = template.save()
                print(f"已保存: {path}")
                saved = True
                break
            if key == ord("n"):
                viewer.next_id()
                viewer.message = f"当前题号 Q{viewer.current_id}"
            elif key == ord("d"):
                viewer.message = viewer.delete_current()
            elif key == ord("["):
                viewer.message = viewer.zoom(1 / 1.25)
            elif key == ord("]"):
                viewer.message = viewer.zoom(1.25)
            elif key == 21:  # PageUp
                viewer.message = viewer.turn_page(-1)
            elif key == 22:  # PageDown
                viewer.message = viewer.turn_page(1)
            elif key in (ord("w"), ord("W")):
                viewer.message = viewer.turn_page(-1)
            elif key in (ord("e"), ord("E")):
                viewer.message = viewer.turn_page(1)
    finally:
        cv2.destroyAllWindows()

    if not saved and viewer.dirty:
        print("有未保存的标注（已丢弃）。")
    print(viewer.summary())
    return saved


def auto_detect_boxes(
    page: np.ndarray,
    dpi: int = 300,
    min_lines: int = 2,
    merge_gap_ratio: float = 2.2,
    min_line_ratio: float = 0.35,
) -> list[list[int]]:
    """针对"印好的横线"版式，给 ROI 标注一个起点。

    作业本、试卷的答案区通常就是若干条印好的空白横线。这里先检测出这些横线，
    再把**相邻的横线合并成一个答案块**——因为一道题的答案区往往占好几行，
    按单条线去框会框出一个个细条，没法直接用。

    min_line_ratio 控制"多长才算一条答案横线"。默认 0.35（页面宽度的 35%），
    这样填空用的短下划线、题号后的横杠不会被误当成答案区。若你的版式是
    "看拼音写词语"那种分段短格，检测不到是正常的，直接手画更快。

    返回的是候选块，标注时仍需人工核对位置和题号归属。这只是省掉从零画框，
    不是自动答案。
    """
    gray = cv2.cvtColor(page, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]

    # 只保留够长的水平线
    min_len = max(int(w * min_line_ratio), 30)
    horizontal = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    binv = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                 cv2.THRESH_BINARY_INV, 31, 12)
    lines_mask = cv2.morphologyEx(binv, cv2.MORPH_OPEN, horizontal)

    num, _, stats, _ = cv2.connectedComponentsWithStats(lines_mask, connectivity=8)
    max_thickness = max(2, int(dpi / 40))
    baselines: list[tuple[int, int, int]] = []  # (y_center, x1, x2)
    for i in range(1, num):
        x, y, lw, lh, _ = stats[i]
        if lw < min_len or lh > max_thickness:
            continue
        baselines.append((y + lh // 2, x, x + lw))

    if not baselines:
        return []

    baselines.sort()
    gaps = [baselines[i + 1][0] - baselines[i][0] for i in range(len(baselines) - 1)]
    # 相邻横线的行距：用中位数代表"正常行距"，超过它的间隔才是块与块的分界。
    typical_gap = float(np.median(gaps)) if gaps else float(dpi / 20)
    merge_gap = max(typical_gap * merge_gap_ratio, dpi / 100)

    blocks: list[list[tuple[int, int, int]]] = [[baselines[0]]]
    for prev, cur in zip(baselines, baselines[1:]):
        if cur[0] - prev[0] <= merge_gap:
            blocks[-1].append(cur)
        else:
            blocks.append([cur])

    boxes: list[list[int]] = []
    pad = max(4, int(dpi / 150))
    for block in blocks:
        if len(block) < min_lines:
            continue  # 单条线太可能是下划线/填空线，不是答案块
        y_top = block[0][0] - pad
        y_bottom = block[-1][0] + pad
        x1 = int(np.percentile([b[1] for b in block], 10)) - pad
        x2 = int(np.percentile([b[2] for b in block], 90)) + pad
        boxes.append([
            int(max(0, x1)), int(max(0, y_top)),
            int(min(w, x2)), int(min(h, y_bottom)),
        ])

    boxes.sort(key=lambda b: b[1])
    return boxes


def ensure_template_dir(template_dir: str | Path) -> Path:
    p = Path(template_dir)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _has_display() -> bool:  # pragma: no cover
    if sys.platform.startswith("win") or sys.platform == "darwin":
        return True
    return bool(sys.stdout.isatty() and "DISPLAY" in __import__("os").environ)
