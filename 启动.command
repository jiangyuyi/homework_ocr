#!/bin/bash
# ============================================================
#  作业手写提取 - 一键启动（macOS / Linux）
#  双击「启动.command」即可。第一次会引导你选择模板和文件夹。
# ============================================================
cd "$(dirname "$0")" || exit 1

PY=""
if [ -x ".venv/bin/python" ]; then
  PY=".venv/bin/python"
else
  for c in python3 python; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi

if [ -z "$PY" ]; then
  echo
  echo "  找不到 Python。"
  echo
  echo "  请先安装 Python 3.10 或更高版本。"
  echo "  也可以用 Homebrew:  brew install python@3.11"
  echo
  read -r -p "按回车键关闭…" _
  exit 1
fi

echo
echo "  正在启动作业手写提取..."
echo

if [ -d "models" ]; then
  "$PY" -m homework_ocr gui "$@"
else
  # 还没下载过模型：先联网拉一次
  "$PY" -m homework_ocr fetch-models --out models || {
    echo
    echo "  模型下载失败，请检查网络后重试。"
    read -r -p "按回车键关闭…" _
    exit 1
  }
  "$PY" -m homework_ocr gui "$@"
fi

if [ $? -ne 0 ]; then
  echo
  echo "  启动失败。请把上面的错误信息发出来。"
  read -r -p "按回车键关闭…" _
fi
