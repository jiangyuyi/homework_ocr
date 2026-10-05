"""模板描述文件的读写。

技术方案第三、三十节。设计原则：

* 坐标是 DPI 相关的像素坐标，所以 template.json 里必须记下渲染时的 DPI。
  换 DPI 后所有 ROI 都会错位——这是最容易踩且最难发现的坑。
* 题目 ROI 人工标一次，所有学生复用。不做自动题号识别。
* 支持"空白区域"以外的题型字段（text/number/choice/formula/drawing），
  为后面分流留出位置，第一版不实现自动判型。
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Literal

log = logging.getLogger(__name__)

QuestionType = Literal["text", "number", "choice", "formula", "drawing"]
BBox = list[int]


@dataclass
class Question:
    id: str
    answer_area: BBox
    type: QuestionType = "text"
    expected_lines: int | None = None
    label: str | None = None
    #: 题型特化的额外坐标，例如选择题每个选项的框。
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        if out.get("expected_lines") is None:
            out.pop("expected_lines")
        if not out.get("label"):
            out.pop("label")
        if not out.get("extras"):
            out.pop("extras")
        return out


@dataclass
class TemplatePage:
    page: int  # 1-based
    width: int
    height: int
    questions: list[Question] = field(default_factory=list)


@dataclass
class Template:
    template_id: str
    dpi: int
    pdf_path: str
    pages: list[TemplatePage] = field(default_factory=list)
    version: int = 1
    notes: str = ""

    # ---------- 路径 ----------
    @property
    def dir(self) -> Path:
        return Path(self.pdf_path).resolve().parent

    @property
    def json_path(self) -> Path:
        return self.dir / "template.json"

    def page(self, number: int) -> TemplatePage | None:
        for p in self.pages:
            if p.page == number:
                return p
        return None

    def question(self, page_number: int, question_id: str) -> Question | None:
        p = self.page(page_number)
        if not p:
            return None
        for q in p.questions:
            if q.id == question_id:
                return q
        return None

    def all_questions(self) -> Iterable[tuple[int, Question]]:
        for p in self.pages:
            for q in p.questions:
                yield p.page, q

    def is_empty(self) -> bool:
        return not any(p.questions for p in self.pages)

    # ---------- 序列化 ----------
    def to_dict(self) -> dict[str, Any]:
        return {
            "template_id": self.template_id,
            "version": self.version,
            "dpi": self.dpi,
            "pdf": Path(self.pdf_path).name,
            "notes": self.notes,
            "pages": [
                {
                    "page": p.page,
                    "width": p.width,
                    "height": p.height,
                    "questions": [q.to_dict() for q in p.questions],
                }
                for p in self.pages
            ],
        }

    def save(self, path: str | Path | None = None) -> Path:
        target = Path(path) if path else self.json_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        log.info("模板已保存: %s", target)
        return target

    @classmethod
    def load(cls, path: str | Path) -> "Template":
        p = Path(path)
        if p.is_dir():
            p = p / "template.json"
        if not p.exists():
            raise FileNotFoundError(f"找不到模板描述文件: {p}")

        data = json.loads(p.read_text(encoding="utf-8"))
        pages: list[TemplatePage] = []
        for pd in data.get("pages", []):
            questions = [
                Question(
                    id=str(q["id"]),
                    answer_area=[int(v) for v in q["answer_area"]],
                    type=q.get("type", "text"),
                    expected_lines=q.get("expected_lines"),
                    label=q.get("label"),
                    extras=q.get("extras", {}) or {},
                )
                for q in pd.get("questions", [])
            ]
            pages.append(
                TemplatePage(
                    page=int(pd["page"]),
                    width=int(pd["width"]),
                    height=int(pd["height"]),
                    questions=questions,
                )
            )

        pdf_field = data.get("pdf") or ""
        pdf_path = (p.parent / pdf_field) if pdf_field else p.parent
        if pdf_field and not Path(pdf_path).exists():
            raise FileNotFoundError(f"模板 PDF 不存在: {pdf_path}（template.json 里记录的是 {pdf_field}）")

        tpl = cls(
            template_id=data.get("template_id") or p.parent.name,
            dpi=int(data.get("dpi", 300)),
            pdf_path=str(pdf_path),
            pages=pages,
            version=int(data.get("version", 1)),
            notes=data.get("notes", ""),
        )
        _validate(tpl)
        return tpl


def _validate(tpl: Template) -> None:
    if tpl.dpi <= 0:
        raise ValueError("template.json 的 dpi 非法")
    for page in tpl.pages:
        for q in page.questions:
            x1, y1, x2, y2 = q.answer_area
            if not (0 <= x1 < x2 <= page.width and 0 <= y1 < y2 <= page.height):
                raise ValueError(
                    f"第 {page.page} 页题目 {q.id} 的 answer_area {q.answer_area} 越界 "
                    f"(页面 {page.width}x{page.height})。常见原因：改了 dpi 却没重新标 ROI。"
                )


def create_from_pdf(pdf_path: str | Path, template_id: str | None = None, dpi: int = 300) -> Template:
    """从模板 PDF 生成骨架 template.json（先不标 ROI）。"""
    from .pdfio import render_pdf

    p = Path(pdf_path).resolve()
    pages = render_pdf(p, dpi=dpi)
    tpl = Template(
        template_id=template_id or p.parent.name,
        dpi=dpi,
        pdf_path=str(p),
        pages=[TemplatePage(page=pg.index + 1, width=pg.width, height=pg.height) for pg in pages],
        notes="ROI 由 `annotate` 命令人工标注后写入本文件。",
    )
    return tpl


def scale_rois(tpl: Template, old_dpi: int, new_dpi: int) -> Template:
    """换 DPI 时按比例缩放已有 ROI。"""
    if old_dpi == new_dpi:
        return tpl
    k = new_dpi / float(old_dpi)
    for page in tpl.pages:
        page.width = int(round(page.width * k))
        page.height = int(round(page.height * k))
        for q in page.questions:
            q.answer_area = [int(round(v * k)) for v in q.answer_area]
    tpl.dpi = new_dpi
    return tpl
