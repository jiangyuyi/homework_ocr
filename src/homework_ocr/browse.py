"""服务端目录浏览。

为什么要自己写：浏览器的安全模型**不允许网页拿到绝对路径**。
`<input type="file" webkitdirectory>` 只给文件名列表和相对路径，JS 拿不到
"C:\\Users\\..." 这样的绝对路径，也没法把它交给后端。

所以目录选择必须由服务端来做：前端请求列目录，后端读文件系统返回条目，
用户在弹窗里点着往下走。这是本地工具绕开浏览器限制的标准做法。

同时提供自由输入路径的兜底——用户可以直接粘贴路径，这对知道路径的老师更快。
"""

from __future__ import annotations

import os
import string
from pathlib import Path
from typing import Any

# 这些是"没什么用"的目录：不是选作业/模板时要去的地方，列出来只会碍事。
_NOISE_DIRS = {
    "$recycle.bin", "system volume information", "windows", "program files",
    "program files (x86)", "programdata", "appdata", "node_modules",
    ".git", "__pycache__", ".venv", "venv", "site-packages", ".cache",
}


def _is_hidden(name: str) -> bool:
    return name.startswith(".") or name.lower() in _NOISE_DIRS


def _drives() -> list[dict[str, str]]:
    """Windows 上列出可用盘符，方便从 D:\ 之类的盘开始找。"""
    if os.name != "nt":
        return []
    out = []
    for letter in string.ascii_uppercase:
        root = f"{letter}:\\"
        if os.path.exists(root):
            out.append({"name": root, "path": root, "kind": "drive"})
    return out


def _shortcuts() -> list[dict[str, str]]:
    """常用位置，作为「从这里开始」的入口。"""
    home = Path.home()
    candidates = [
        ("用户目录", home),
        ("桌面", home / "Desktop"),
        ("文档", home / "Documents"),
        ("下载", home / "Downloads"),
        ("扫描件", home / "Pictures"),
    ]
    out = []
    for name, path in candidates:
        if path.exists():
            out.append({"name": name, "path": str(path), "kind": "shortcut"})
    return out


def list_dir(
    path: str | None,
    show_hidden: bool = False,
    want_files: bool = False,
    suffixes: tuple[str, ...] = (".pdf",),
) -> dict[str, Any]:
    """列出一个目录下的条目。

    选文件夹时只列子目录（want_files=False）。选 PDF 文件时要置
    want_files=True，让后缀匹配的文件一起列出来——否则弹窗里只有文件夹，
    用户根本看不到、也点不中要选的那个 PDF，只能手动敲路径。
    """
    if not path:
        return {
            "path": "",
            "parent": None,
            "dirs": [],
            "files": [],
            "drives": _drives(),
            "shortcuts": _shortcuts(),
            "error": None,
        }

    try:
        p = Path(path).expanduser()
    except (OSError, RuntimeError) as exc:
        return {"path": str(path), "parent": None, "dirs": [], "files": [],
                "drives": [], "shortcuts": [], "error": f"路径无效: {exc}"}

    if not p.exists():
        return {"path": str(p), "parent": None, "dirs": [], "files": [],
                "drives": _drives(), "shortcuts": _shortcuts(), "error": "目录不存在"}

    if p.is_file():
        p = p.parent

    if not os.access(p, os.R_OK):
        return {"path": str(p), "parent": None, "dirs": [], "files": [],
                "drives": [], "shortcuts": [], "error": "没有读取权限"}

    suffixes = tuple(s.lower() for s in suffixes)
    dirs: list[dict[str, Any]] = []
    files: list[dict[str, Any]] = []
    try:
        for entry in sorted(p.iterdir(), key=lambda e: e.name.lower()):
            try:
                if not show_hidden and _is_hidden(entry.name):
                    continue
                if entry.is_dir():
                    dirs.append({
                        "name": entry.name,
                        "path": str(entry),
                        "kind": "dir",
                        "has_template": (entry / "template.json").exists(),
                    })
                elif want_files and entry.suffix.lower() in suffixes:
                    files.append({
                        "name": entry.name,
                        "path": str(entry),
                        "kind": "file",
                        "size": entry.stat().st_size,
                    })
            except (PermissionError, OSError):
                continue
    except (PermissionError, OSError) as exc:
        return {"path": str(p), "parent": None, "dirs": [], "files": [], "drives": [],
                "shortcuts": [], "error": f"无法列出目录: {exc.strerror or exc}"}

    parent = str(p.parent) if p.parent != p else None

    # 有 template.json 的目录排前面——用户往往就是来找模板的
    dirs.sort(key=lambda d: (not d["has_template"], d["name"].lower()))

    return {
        "path": str(p),
        "parent": parent,
        "dirs": dirs,
        "files": files,
        "drives": _drives(),
        "shortcuts": _shortcuts(),
        "error": None,
    }


def validate_template_dir(path: str) -> dict[str, Any]:
    """检查一个目录能不能当模板用，并返回摘要。"""
    from .template import Template

    p = Path(path).expanduser()
    json_path = p if p.is_file() and p.suffix == ".json" else p / "template.json"
    if not json_path.exists():
        # 目录里只有一个 PDF？直接告诉他可以建模板
        pdfs = [x for x in p.glob("*.pdf")] if p.is_dir() else []
        return {
            "ok": False,
            "reason": "没有找到 template.json",
            "pdfs": [str(x) for x in pdfs],
            "hint": "选空白模板 PDF 可以在这里直接创建模板" if pdfs else "",
        }
    try:
        tpl = Template.load(json_path)
    except (FileNotFoundError, ValueError, OSError) as exc:
        return {"ok": False, "reason": f"模板读取失败: {exc}"}

    questions = sum(len(pg.questions) for pg in tpl.pages)
    return {
        "ok": True,
        "id": tpl.template_id,
        "dpi": tpl.dpi,
        "pages": len(tpl.pages),
        "questions": questions,
        "reason": "" if questions else "模板还没有标注任何答案区域（需要先做标注）",
    }
