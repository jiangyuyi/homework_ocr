"""用户级配置与路径。

CLI 参数是给开发者的；普通用户不该被要求记住 `--template xxx --input yyy`。
所以把选择过的模板/输入/输出目录记下来，下次启动直接用。

写入位置遵循「程序目录只读、用户目录可写」的原则（参考 asr_mm 的 paths 拆分）：

    Windows  %LOCALAPPDATA%\\homework_ocr\\
    macOS    ~/Library/Application Support/homework_ocr/
    Linux    ~/.local/share/homework_ocr/

模板和学生作业永远不往程序目录里塞——那是只读的（尤其打包成 exe 之后）。
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

APP_NAME = "homework_ocr"


def user_data_dir() -> Path:
    """用户可写的数据目录。"""
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share"
    path = Path(base) / APP_NAME
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as exc:  # pragma: no cover
        log.warning("无法创建用户数据目录 %s: %s", path, exc)
    return path


def settings_path() -> Path:
    return user_data_dir() / "settings.json"


def default_workspace() -> Path:
    """默认工作目录（模板/输入/输出）。放用户目录，不放程序目录。"""
    return user_data_dir() / "workspace"


@dataclass
class Batch:
    """一个识别批次 = 一套模板 + 一个作业文件夹。

    为什么要有批次：期中/期末/周测的作业版式不同，模板也不一样。
    把模板固定成一个全局设置，混着处理就会用错模板，结果全错。
    所以模板的选择粒度是「批次」而不是「程序」。
    """

    id: str
    name: str
    template: str = ""
    input_dir: str = ""
    #: 浏览器上传的文件落在这里（路线 A：读字节上传，不依赖本机路径）。
    #: 程序自管，落在用户数据目录下，用户不用管它在哪。
    inbox_dir: str = ""
    created_at: str = ""
    last_run_at: str = ""

    @property
    def template_path(self) -> Path | None:
        return Path(self.template) if self.template else None

    @property
    def input_path(self) -> Path | None:
        return Path(self.input_dir) if self.input_dir else None

    @property
    def inbox_path(self) -> Path | None:
        return Path(self.inbox_dir) if self.inbox_dir else None

    def source_dirs(self) -> list[Path]:
        """这个批次的作业从哪来。上传的在前，本机文件夹在后。"""
        out: list[Path] = []
        for d in (self.inbox_path, self.input_path):
            if d and d.exists() and d not in out:
                out.append(d)
        return out

    def is_valid(self) -> bool:
        t = self.template_path
        if not t or not (t / "template.json").exists():
            return False
        # 路线 A：只上传过文件也算就绪，不必再指定本机文件夹。
        if self.inbox_dir:
            return True
        i = self.input_path
        return bool(i and i.exists())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Batch":
        return cls(
            id=str(data.get("id") or _slug(data.get("name", ""))),
            name=str(data.get("name", "")),
            template=str(data.get("template", "")),
            input_dir=str(data.get("input_dir", "")),
            inbox_dir=str(data.get("inbox_dir", "")),
            created_at=str(data.get("created_at", "")),
            last_run_at=str(data.get("last_run_at", "")),
        )


def _slug(text: str) -> str:
    """把批次名变成安全的目录名。中文保留，只清掉文件系统不允许的字符。"""
    bad = '<>:"/\\|?*'
    out = "".join("_" if c in bad else c for c in text).strip().strip(".")
    return out or "batch"


@dataclass
class Settings:
    """GUI 记住的选择。全部可空——空了就走引导页。"""

    template: str = ""      # 单批次模式的兼容字段
    input_dir: str = ""
    output_dir: str = ""    # 输出根目录，各批次在其下建子目录
    config_file: str = ""
    port: int = 8000
    last_template_id: str = ""
    onboarded: bool = False
    #: 识别批次列表
    batches: list[Batch] = field(default_factory=list)
    #: 当前选中的批次 id
    active_batch: str = ""
    extras: dict[str, Any] = field(default_factory=dict)

    # ---------- 批次 ----------
    def batch(self, batch_id: str) -> Batch | None:
        for b in self.batches:
            if b.id == batch_id:
                return b
        return None

    def add_batch(self, name: str, template: str, input_dir: str) -> Batch:
        from datetime import datetime

        bid = _slug(name)
        # 同名时加后缀，避免覆盖
        if any(b.id == bid for b in self.batches):
            n = 2
            while any(b.id == f"{bid}-{n}" for b in self.batches):
                n += 1
            bid = f"{bid}-{n}"
        batch = Batch(
            id=bid,
            name=name or bid,
            template=template,
            input_dir=input_dir,
            created_at=datetime.now().isoformat(timespec="seconds"),
        )
        self.batches.append(batch)
        self.active_batch = batch.id
        return batch

    def remove_batch(self, batch_id: str) -> None:
        self.batches = [b for b in self.batches if b.id != batch_id]
        if self.active_batch == batch_id:
            self.active_batch = self.batches[0].id if self.batches else ""

    def get_batch(self) -> Batch | None:
        """当前生效的批次。没有多批次数据时，从单批次字段合成一个。"""
        if self.batches:
            return self.batch(self.active_batch) or self.batches[0]
        if self.template or self.input_dir:
            return Batch(id="default", name="默认", template=self.template,
                         input_dir=self.input_dir)
        return None

    def batch_output_dir(self, batch: Batch | None = None) -> Path | None:
        b = batch or self.get_batch()
        root = Path(self.output_dir) if self.output_dir else None
        if root is None:
            return None
        return root / b.id if b else root

    # ---------- 读写 ----------
    @classmethod
    def load(cls) -> "Settings":
        p = settings_path()
        if not p.exists():
            return cls()
        try:
            raw = json.loads(p.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            log.warning("配置损坏，使用默认值: %s (%s)", p, exc)
            return cls()
        known = {f for f in cls.__dataclass_fields__ if f not in {"extras", "batches"}}
        extras = {k: v for k, v in raw.items() if k not in known and k != "batches"}
        data = {k: v for k, v in raw.items() if k in known}
        s = cls(**data)
        s.extras = extras
        s.batches = [Batch.from_dict(b) for b in (raw.get("batches") or []) if isinstance(b, dict)]
        s._migrate()
        return s

    def _migrate(self) -> None:
        """旧版本只有单批次字段，升级成批次列表。"""
        if not self.batches and (self.template or self.input_dir):
            from datetime import datetime

            self.batches = [Batch(
                id="default",
                name="默认",
                template=self.template,
                input_dir=self.input_dir,
                created_at=datetime.now().isoformat(timespec="seconds"),
            )]
            self.active_batch = "default"

    def save(self) -> Path:
        p = settings_path()
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["batches"] = [b.to_dict() for b in self.batches]
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(p)
        return p

    # ---------- 派生路径 ----------
    @property
    def template_path(self) -> Path | None:
        b = self.get_batch()
        return b.template_path if b else (Path(self.template) if self.template else None)

    @property
    def input_path(self) -> Path | None:
        b = self.get_batch()
        return b.input_path if b else (Path(self.input_dir) if self.input_dir else None)

    @property
    def output_path(self) -> Path | None:
        return self.batch_output_dir()

    def is_ready(self) -> bool:
        b = self.get_batch()
        return bool(b and b.is_valid() and self.output_dir)


# ---------------------------------------------------------------- 模板发现
def find_templates(search_roots: list[Path] | None = None) -> list[dict[str, Any]]:
    """扫描已存在的模板，供 GUI 下拉选择。

    扫描位置：显式目录 + 用户工作区 + 程序目录下的 templates/。
    """
    roots = list(search_roots or [])
    roots.append(default_workspace() / "templates")
    roots.append(Path.cwd() / "templates")
    roots.append(Path(__file__).resolve().parents[3] / "templates")

    seen: set[Path] = set()
    out: list[dict[str, Any]] = []

    for root in roots:
        if not root or not root.exists():
            continue
        for json_path in sorted(root.glob("*/template.json")):
            folder = json_path.parent
            try:
                resolved = folder.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                data = json.loads(json_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            questions = sum(len(p.get("questions", []) or []) for p in data.get("pages", []))
            out.append({
                "id": data.get("template_id") or folder.name,
                "path": str(folder),
                "dpi": data.get("dpi"),
                "pages": len(data.get("pages", [])),
                "questions": questions,
                "has_rois": questions > 0,
                "source": str(root),
            })
    return out
