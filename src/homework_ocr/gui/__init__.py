"""本地图形界面。

从 `homework_ocr.gui` 导入 `serve` 即可启动；`create_app` 供测试用。
"""

from .server import create_app, serve

__all__ = ["create_app", "serve"]
