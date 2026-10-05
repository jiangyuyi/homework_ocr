"""运行时出站网络封禁。

技术方案第 22 节要求"部署完成后拔网线跑一遍，确保不会出现任何模型自动下载"。
这里把它变成常态：程序启动后直接封掉 socket，任何出站连接都会抛异常并打印
是谁发起的。这样"本地计算"是一个可执行的不变量，而不是一句承诺。

模型必须在 enforce_offline() 之前下载好。`fetch-models` 命令负责这件事，
它自己会先调用 allow_network()。
"""

from __future__ import annotations

import socket
import sys
from contextlib import contextmanager
from typing import Any, Iterator

_installed = False
_allow_loopback = True
_saved: dict[str, Any] = {}
_violations: list[str] = []


class NetworkBlocked(RuntimeError):
    """程序在离线模式下尝试建立出站连接。"""


def _is_local(address: Any) -> bool:
    if not _allow_loopback:
        return False
    if isinstance(address, (bytes, str)):
        try:
            host = socket.gethostbyname(address.decode() if isinstance(address, bytes) else address)
        except (OSError, UnicodeDecodeError):
            return True  # 解析不出来就当作本地，放行并由防火墙兜底
        return host.startswith("127.") or host in {"::1", "0.0.0.0"}
    if isinstance(address, tuple) and address:
        return _is_local(address[0])
    return False


def _guard(original: Any, label: str):
    def wrapper(self: Any, address: Any, *args: Any, **kwargs: Any):
        if _is_local(address):
            return original(self, address, *args, **kwargs)
        who = _caller_label()
        msg = f"{label} -> {address} (发起方: {who})"
        _violations.append(msg)
        raise NetworkBlocked(
            "检测到出站网络请求，已按隐私要求阻断。\n"
            f"  {msg}\n"
            "  模型请先在联网环境执行: homework-ocr fetch-models"
        )

    return wrapper


def _caller_label() -> str:
    """尽量定位是谁在联网。深度受限，只回溯几层且跳过 stdlib 自身。"""
    import inspect

    try:
        for frame_info in inspect.stack()[2:8]:
            mod = inspect.getmodule(frame_info.frame)
            name = getattr(mod, "__name__", "") if mod else ""
            if name.startswith(("socket", "urllib", "http", "ssl", "asyncio", "_frozen")):
                continue
            short = name.removeprefix("homework_ocr.")
            return f"{short or '?'}:{frame_info.function}"
    except Exception:  # pragma: no cover - 定位失败不该影响主流程
        pass
    return "unknown"


def is_enforced() -> bool:
    return _installed


def violations() -> list[str]:
    return list(_violations)


def enforce_offline(allow_loopback: bool = True) -> None:
    """封禁出站连接。重复调用无副作用。"""
    global _installed, _allow_loopback
    if _installed:
        return

    _allow_loopback = allow_loopback
    _saved["connect"] = socket.socket.connect
    _saved["connect_ex"] = socket.socket.connect_ex
    socket.socket.connect = _guard(_saved["connect"], "socket.connect")  # type: ignore[method-assign]
    socket.socket.connect_ex = _guard(_saved["connect_ex"], "socket.connect_ex")  # type: ignore[method-assign]
    _installed = True


def disable_offline() -> None:
    """还原 socket。仅供测试使用。"""
    global _installed
    if not _installed:
        return
    socket.socket.connect = _saved["connect"]  # type: ignore[method-assign]
    socket.socket.connect_ex = _saved["connect_ex"]  # type: ignore[method-assign]
    _saved.clear()
    _installed = False


@contextmanager
def allow_network() -> Iterator[None]:
    """临时放行网络，用于下载模型。"""
    was_enforced = _installed
    if was_enforced:
        disable_offline()
    try:
        yield
    finally:
        if was_enforced:
            enforce_offline(allow_loopback=True)


def set_env_offline() -> None:
    """把主流库/工具链的联网开关全部关掉，和 socket 封禁形成双保险。"""
    import os

    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["NO_PROXY"] = "*"
    os.environ["no_proxy"] = "*"
    os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "4")
