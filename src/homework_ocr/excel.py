"""Excel 导出。

给老师的最终交付物应该是"能直接看/能直接改"的表，而不是一堆 JSON。

三个工作表：

    答题明细  每题一行，含识别文本、置信度、复核状态、人工修正后的最终文本
    复核队列  只放需复核的题，按优先级排序——这就是当天要干活的清单
    学生汇总  每人一行，配准成功率、需复核数，用来快速筛出可疑文件

"最终文本"列 = 人工修正优先，没有修正就取 OCR 结果。老师改过的内容在
"人工修正"列里单独留痕，方便回溯哪些是机器认的、哪些是人确认的。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.worksheet import Worksheet

from .i18n import t
from .review import ReviewEntry, ReviewStore

log = logging.getLogger(__name__)

HEADER_FILL = PatternFill("solid", fgColor="DDEBF7")
REVIEW_FILL = PatternFill("solid", fgColor="FFF2CC")
CORRECTED_FILL = PatternFill("solid", fgColor="E2EFDA")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

STATUS_CODES = ("pending", "accepted", "corrected", "rejected")


def _reasons(codes: Sequence[dict[str, Any]], fallback: Sequence[str]) -> str:
    """复核原因：有 code 就翻译，没 code（老数据）就用原文。"""
    out: list[str] = []
    for c in codes or ():
        if isinstance(c, dict) and c.get("code"):
            params = c.get("params") or {}
            out.append(t(f"reason.{c['code']}", **params))
    return " | ".join(out or list(fallback))


@dataclass
class Row:
    document: str
    page: int
    question_id: str
    question_type: str
    detected: bool
    ocr_text: str
    confidence: float
    review_required: bool
    review_reasons: list[str]
    review_reason_codes: list[dict[str, Any]]
    final_text: str
    status: str
    note: str
    ink_coverage: float


def _flatten(
    documents: Sequence[dict[str, Any]],
    reviews: dict[str, ReviewStore] | None = None,
) -> tuple[list[Row], list[dict[str, Any]]]:
    """把多份 result.json + review.json 摊平成行。"""
    rows: list[Row] = []
    summaries: list[dict[str, Any]] = []

    for doc in documents:
        name = doc.get("document", "")
        store = (reviews or {}).get(Path(name).stem)
        total = answered = need_review = aligned = pages = 0

        for page in doc.get("pages", []):
            pages += 1
            if page.get("alignment", {}).get("success"):
                aligned += 1
            for q in page.get("questions", []):
                total += 1
                detected = bool(q.get("detected_handwriting"))
                answered += int(detected)
                need = bool(q.get("review_required"))
                need_review += int(need)

                ocr_text = q.get("text", "") or ""
                entry = store.get(int(page.get("page", 1)), str(q.get("question_id"))) if store else None

                final_text = ocr_text
                status = "pending"
                note = ""
                if entry is not None:
                    if entry.status in {"corrected", "accepted"} and entry.final_text:
                        final_text = entry.final_text
                        status = entry.status
                    elif entry.status == "rejected":
                        status = "rejected"
                    note = entry.note
                elif not need:
                    # 没标记需复核、也没人工记录 -> 视为机器已通过
                    status = "auto"

                rows.append(
                    Row(
                        document=name,
                        page=int(page.get("page", 1)),
                        question_id=str(q.get("question_id", "")),
                        question_type=q.get("question_type", "text"),
                        detected=detected,
                        ocr_text=ocr_text,
                        confidence=float(q.get("confidence", 0.0)),
                        review_required=need,
                        review_reasons=list(q.get("review_reasons", []) or []),
                        review_reason_codes=[c for c in (q.get("review_reason_codes") or [])
                                             if isinstance(c, dict) and c.get("code")],
                        final_text=final_text,
                        status=status,
                        note=note,
                        ink_coverage=float(q.get("ink_coverage", 0.0)),
                    )
                )

        summaries.append(
            {
                "document": name,
                "pages": pages,
                "aligned": aligned,
                "total": total,
                "answered": answered,
                "need_review": need_review,
                "elapsed": doc.get("elapsed_s", 0.0),
                "status": doc.get("status", ""),
            }
        )

    return rows, summaries


def _style_header(ws: Worksheet, headers: Sequence[str], widths: Sequence[int]) -> None:
    ws.append(list(headers))
    for idx, width in enumerate(widths, start=1):
        ws.column_dimensions[get_column_letter(idx)].width = width
    for cell in ws[1]:
        cell.fill = HEADER_FILL
        cell.font = Font(bold=True)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = BORDER
    ws.freeze_panes = "A2"
    ws.auto_filter.ref = f"A1:{get_column_letter(len(headers))}1"


def export_xlsx(
    path: str | Path,
    documents: Sequence[dict[str, Any]],
    reviews: dict[str, ReviewStore] | None = None,
) -> Path:
    """把若干份 run 的结果写成一个 .xlsx。"""
    rows, summaries = _flatten(documents, reviews)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wb = Workbook()

    # ---------------- 答题明细 ----------------
    ws = wb.active
    ws.title = t("excel.sheet.detail")
    _style_header(
        ws,
        [t("excel.h.document"), t("excel.h.page"), t("excel.h.qid"), t("excel.h.qtype"),
         t("excel.h.detected"), t("excel.h.ocr"), t("excel.h.conf"),
         t("excel.h.final"), t("excel.h.status"), t("excel.h.needs"),
         t("excel.h.reason"), t("excel.h.note"), t("excel.h.ink")],
        [22, 6, 8, 8, 12, 46, 9, 46, 10, 9, 52, 24, 11],
    )
    for r in rows:
        ws.append([
            r.document, r.page, r.question_id, r.question_type,
            t("common.yes") if r.detected else t("common.no"),
            r.ocr_text, round(r.confidence, 4), r.final_text,
            t(f"excel.st.{r.status}") if r.status in STATUS_CODES + ("auto",) else r.status,
            t("common.yes") if r.review_required else "",
            _reasons(r.review_reason_codes, r.review_reasons), r.note,
            round(r.ink_coverage, 5),
        ])
        row_idx = ws.max_row
        for cell in ws[row_idx]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = BORDER
        if r.status == "corrected":
            for cell in ws[row_idx]:
                cell.fill = CORRECTED_FILL
        elif r.review_required and r.status in {"pending", "auto"}:
            for cell in ws[row_idx]:
                cell.fill = REVIEW_FILL

    _autosize_rows(ws, max_lines=4)

    # ---------------- 复核队列 ----------------
    ws2 = wb.create_sheet(t("excel.sheet.queue"))
    _style_header(
        ws2,
        [t("excel.h.document"), t("excel.h.page"), t("excel.h.qid"), t("excel.h.ocr"),
         t("excel.h.conf"), t("excel.h.reason"), t("excel.h.status"), t("excel.h.note")],
        [22, 6, 8, 52, 9, 60, 10, 24],
    )
    queue = [r for r in rows if r.review_required and r.status in {"pending", "auto"}]
    queue.sort(key=lambda r: (r.document, r.page, _qnum(r.question_id)))
    if not queue:
        ws2.append(["", t("excel.allDone"), "", "", "", "", "", ""])
    for r in queue:
        ws2.append([r.document, r.page, r.question_id, r.ocr_text,
                    round(r.confidence, 4),
                    _reasons(r.review_reason_codes, r.review_reasons),
                    t(f"excel.st.{r.status}") if r.status in STATUS_CODES + ("auto",) else r.status,
                    r.note])
        row_idx = ws2.max_row
        for cell in ws2[row_idx]:
            cell.alignment = Alignment(vertical="top", wrap_text=True)
            cell.border = BORDER
            cell.fill = REVIEW_FILL
    _autosize_rows(ws2, max_lines=4)

    # ---------------- 学生汇总 ----------------
    ws3 = wb.create_sheet(t("excel.sheet.summary"))
    _style_header(
        ws3,
        [t("excel.h.document"), t("excel.h.pages"), t("excel.h.aligned"), t("excel.h.total"),
         t("excel.h.answered"), t("excel.h.needs"), t("excel.h.elapsed"), t("excel.h.docStatus")],
        [24, 8, 12, 8, 9, 9, 11, 12],
    )
    for s in summaries:
        ws3.append([s["document"], s["pages"], s["aligned"], s["total"],
                    s["answered"], s["need_review"], round(float(s["elapsed"]), 2), s["status"]])
        for cell in ws3[ws3.max_row]:
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = BORDER
            if s["need_review"] > 0:
                cell.fill = REVIEW_FILL

    wb.save(str(path))
    log.info("已导出 Excel: %s", path)
    return path


def _autosize_rows(ws: Worksheet, max_lines: int = 4) -> None:
    """按内容给带换行的单元格设行高，否则长文本会被截断显示。"""
    for row in ws.iter_rows(min_row=2):
        longest = 1
        for cell in row:
            if isinstance(cell.value, str) and "\n" in cell.value:
                longest = max(longest, min(max_lines, cell.value.count("\n") + 1))
        if longest > 1:
            ws.row_dimensions[row[0].row].height = 15 * longest


def _qnum(qid: str) -> tuple[int, str]:
    """题号排序：'2' < '10'，非数字的排在后面按字典序。"""
    try:
        return (0, f"{int(qid):09d}")
    except (TypeError, ValueError):
        return (1, str(qid))
