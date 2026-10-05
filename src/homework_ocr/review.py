"""人工复核结果的存储。

为什么要单独存：OCR 结果是机器猜的，老师改过的才是最终答案。两者必须分开
保存，否则没法回答"这题是机器认的，还是人确认过的"。

每份学生作业一个 review.json，和 run 的输出放在一起：

    output/张三/
    ├── result.json      # 机器识别结果（可被 reocr 覆盖重写）
    └── review.json      # 人工修正（永不覆盖）

这样 reocr 换模型重跑不会冲掉人工劳动；导出 Excel 时两份一起读，
"最终文本"= 人工修正优先，没有修正就用 OCR 结果。
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Literal

log = logging.getLogger(__name__)

ReviewStatus = Literal["pending", "accepted", "corrected", "rejected"]

REVIEW_FILENAME = "review.json"


def make_key(page: int, question_id: str) -> str:
    return f"p{page}_q{question_id}"


@dataclass
class ReviewEntry:
    #: 机器识别出的原文
    original_text: str = ""
    #: 人工修正后的文本。空串表示"未修正"。
    corrected_text: str = ""
    status: ReviewStatus = "pending"
    note: str = ""
    updated_at: str = ""
    #: 复核时机器的置信度，留档用
    confidence: float = 0.0
    #: 复核前机器给出的复核原因
    review_reasons: list[str] = field(default_factory=list)
    #: 复核原因的语言中立版本（{"code","params"}）。界面/Excel 靠它翻译；
    #: 旧记录没有这个字段，就直接用 review_reasons 里的原文。
    review_reason_codes: list[dict[str, Any]] = field(default_factory=list)

    @property
    def final_text(self) -> str:
        if self.status == "corrected" and self.corrected_text.strip():
            return self.corrected_text.strip()
        return self.original_text

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ReviewEntry":
        return cls(
            original_text=str(data.get("original_text", "")),
            corrected_text=str(data.get("corrected_text", "")),
            status=data.get("status", "pending"),  # type: ignore[arg-type]
            note=str(data.get("note", "")),
            updated_at=str(data.get("updated_at", "")),
            confidence=float(data.get("confidence", 0.0)),
            review_reasons=list(data.get("review_reasons", []) or []),
            review_reason_codes=[c for c in (data.get("review_reason_codes") or [])
                                 if isinstance(c, dict) and c.get("code")],
        )


class ReviewStore:
    """一份学生作业的人工复核记录。线程安全。"""

    def __init__(self, run_dir: str | Path, entries: dict[str, ReviewEntry] | None = None):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / REVIEW_FILENAME
        self._entries: dict[str, ReviewEntry] = entries or {}
        self._lock = threading.Lock()

    # ---------- 读写 ----------
    @classmethod
    def load(cls, run_dir: str | Path) -> "ReviewStore":
        run_dir = Path(run_dir)
        path = run_dir / REVIEW_FILENAME
        entries: dict[str, ReviewEntry] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                for key, value in (raw.get("entries", {}) or {}).items():
                    entries[key] = ReviewEntry.from_dict(value)
            except (json.JSONDecodeError, OSError, AttributeError) as exc:
                log.warning("复核记录损坏，已忽略: %s (%s)", path, exc)
        return cls(run_dir, entries)

    def save(self) -> Path:
        with self._lock:
            payload = {
                "version": 1,
                "updated_at": _now(),
                "entries": {k: v.to_dict() for k, v in self._entries.items()},
            }
            self.run_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            # 原子替换：避免 GUI 写到一半崩掉时把已有复核记录弄坏。
            tmp.replace(self.path)
            return self.path

    # ---------- 访问 ----------
    def get(self, page: int, question_id: str) -> ReviewEntry | None:
        return self._entries.get(make_key(page, question_id))

    def ensure(self, page: int, question_id: str, *, original_text: str = "",
               confidence: float = 0.0, review_reasons: list[str] | None = None,
               review_reason_codes: list[dict[str, Any]] | None = None) -> ReviewEntry:
        """取现有记录，没有就用机器结果建一条。"""
        key = make_key(page, question_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = ReviewEntry(
                    original_text=original_text,
                    confidence=confidence,
                    review_reasons=list(review_reasons or []),
                    review_reason_codes=list(review_reason_codes or []),
                )
                self._entries[key] = entry
            return entry

    def update(
        self,
        page: int,
        question_id: str,
        *,
        corrected_text: str | None = None,
        status: ReviewStatus | None = None,
        note: str | None = None,
        original_text: str | None = None,
        confidence: float | None = None,
        review_reasons: list[str] | None = None,
        review_reason_codes: list[dict[str, Any]] | None = None,
    ) -> ReviewEntry:
        key = make_key(page, question_id)
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                entry = ReviewEntry()
                self._entries[key] = entry

            if original_text is not None:
                entry.original_text = original_text
            if confidence is not None:
                entry.confidence = confidence
            if review_reasons is not None:
                entry.review_reasons = list(review_reasons)
            if review_reason_codes is not None:
                entry.review_reason_codes = list(review_reason_codes)

            if corrected_text is not None:
                entry.corrected_text = corrected_text
                # 改了文字就自动标成 corrected；改回和原文一样则算 accepted。
                if corrected_text.strip() == entry.original_text.strip():
                    entry.status = "accepted"
                elif corrected_text.strip():
                    entry.status = "corrected"
                else:
                    entry.status = "rejected"

            if status is not None:
                entry.status = status
            if note is not None:
                entry.note = note

            entry.updated_at = _now()
            return entry

    def final_text(self, page: int, question_id: str, ocr_text: str) -> str:
        entry = self.get(page, question_id)
        if entry is None:
            return ocr_text
        if entry.status in {"corrected", "accepted"} and entry.final_text:
            return entry.final_text
        return ocr_text

    def is_resolved(self, page: int, question_id: str) -> bool:
        entry = self.get(page, question_id)
        return bool(entry and entry.status in {"accepted", "corrected", "rejected"})

    def __iter__(self) -> Iterator[ReviewEntry]:
        return iter(list(self._entries.values()))

    def __len__(self) -> int:
        return len(self._entries)

    # ---------- 统计 ----------
    def stats(self) -> dict[str, int]:
        out = {"total": len(self._entries), "accepted": 0, "corrected": 0, "rejected": 0, "pending": 0}
        for e in self._entries.values():
            if e.status in out:
                out[e.status] += 1
        return out


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")
