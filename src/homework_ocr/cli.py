"""命令行入口。

    homework-ocr init-template  扫描件模板.pdf --id yuwen_g3_1
    homework-ocr annotate       --template templates/yuwen_g3_1
    homework-ocr run            --template templates/yuwen_g3_1 --input input/ --output output/
    homework-ocr reocr          --run output/张三
    homework-ocr fetch-models
    homework-ocr doctor
    homework-ocr make-demo      --out demo     （生成自测用模板+模拟手写扫描件）
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer

from . import offline_guard
from .config import Config
from .template import Template

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="本地离线的学生手写作答提取工具（模板驱动 + RapidOCR）",
)

log = logging.getLogger("homework_ocr")
MODELS_DIR = Path("models")

DEFAULT_CONFIG = Path("config.yaml")


# ------------------------------------------------------------------ helpers
def _setup_logging(verbose: bool = False) -> None:
    _fix_console_encoding()
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)-7s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    # 第三方库日志压到 WARNING，避免刷屏。
    for noisy in ("rapidocr", "urllib3", "filelock", "matplotlib"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _fix_console_encoding() -> None:
    """Windows 控制台默认 GBK，中文输出会变乱码。这里强制 UTF-8。"""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is None:
            continue
        try:
            reconfigure(encoding="utf-8", errors="replace")
        except (ValueError, OSError):  # pragma: no cover - 某些终端不支持
            pass


def _load_config(path: Optional[Path]) -> Config:
    return Config.load(path if path and path.exists() else None)


def _resolve_template(ref: str) -> Template:
    p = Path(ref)
    if p.suffix == ".json":
        return Template.load(p)
    if p.suffix.lower() == ".pdf":
        raise typer.BadParameter(f"--template 需要指向模板目录或 template.json，收到的是 PDF: {p}")
    return Template.load(p)


def _apply_offline(cfg: Config) -> None:
    if cfg.offline.enforce:
        offline_guard.enforce_offline(allow_loopback=cfg.offline.allow_loopback)
        log.info("已启用出站网络封禁（本地计算模式）")


# ------------------------------------------------------------------ commands
@app.command("init-template")
def init_template(
    pdf: Path = typer.Argument(..., help="空白模板 PDF"),
    template_id: Optional[str] = typer.Option(None, "--id", help="模板标识，缺省用 PDF 所在目录名"),
    dpi: int = typer.Option(300, help="渲染 DPI，必须与后续处理一致"),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="模板目录，缺省用 PDF 所在目录"),
) -> None:
    """从空白模板 PDF 生成 template.json 骨架。"""
    from .template import create_from_pdf

    _setup_logging()
    if not pdf.exists():
        raise typer.BadParameter(f"找不到 PDF: {pdf}")

    target_dir = Path(out_dir) if out_dir else pdf.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    dest_pdf = target_dir / pdf.name
    if pdf.resolve() != dest_pdf.resolve():
        dest_pdf.write_bytes(pdf.read_bytes())

    tpl = create_from_pdf(dest_pdf, template_id=template_id or target_dir.name, dpi=dpi)
    path = tpl.save()
    typer.echo(f"已创建 {path}")
    typer.echo(f"共 {len(tpl.pages)} 页, DPI={dpi}")
    typer.echo("下一步: homework-ocr annotate --template " + str(target_dir))


@app.command("annotate")
def annotate_cmd(
    template: Path = typer.Option(..., "--template", "-t", help="模板目录或 template.json"),
    scale: float = typer.Option(0.5, help="初始显示缩放"),
    suggest: bool = typer.Option(False, "--suggest", help="先用横线检测预生成 ROI 供微调"),
    min_lines: int = typer.Option(2, help="几个相邻横线合并成一个答案块"),
    min_line_ratio: float = typer.Option(0.35, help="横线长度至少占页面宽度的比例"),
) -> None:
    """交互式标注题目答案区域。人工标一次，所有学生复用。"""
    from .annotate import annotate, auto_detect_boxes
    from .pdfio import render_pdf

    _setup_logging()
    tpl = _resolve_template(str(template))
    pages = render_pdf(tpl.pdf_path, dpi=tpl.dpi)

    if suggest:
        from .template import Question

        for spec in tpl.pages:
            idx = spec.page - 1
            if idx >= len(pages):
                continue
            guesses = auto_detect_boxes(pages[idx].image, tpl.dpi,
                                        min_lines=min_lines, min_line_ratio=min_line_ratio)
            typer.echo(f"第 {spec.page} 页: 检测到 {len(guesses)} 个候选答案块（仅供起点，请人工核对题号归属）")
            for i, g in enumerate(guesses, start=1):
                typer.echo(f"   候选 {i}: {g}")
                if any(q.id == str(i) for q in spec.questions):
                    continue
                spec.questions.append(Question(id=str(i), answer_area=[int(v) for v in g]))
        tpl.save()
        typer.echo(f"已写入 {tpl.json_path}，现在打开标注窗口微调位置。")

    ok = annotate(tpl, [p.image for p in pages], initial_scale=scale)
    if not ok:
        raise typer.Exit(code=1)


@app.command("run")
def run_cmd(
    template: Path = typer.Option(..., "--template", "-t", help="模板目录或 template.json"),
    input_path: Path = typer.Option(..., "--input", "-i", help="学生 PDF 文件或目录"),
    output: Path = typer.Option(Path("output"), "--output", "-o", help="输出根目录"),
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG, "--config", "-c", help="配置文件"),
    engine: Optional[str] = typer.Option(None, help="覆盖 ocr.engine (rapidocr / paddleocr)"),
    diff_method: Optional[str] = typer.Option(None, help="覆盖 difference.method"),
    no_debug: bool = typer.Option(False, "--no-debug", help="不输出中间图（省时间，但出问题难查）"),
    max_pages: Optional[int] = typer.Option(None, help="每份 PDF 只处理前 N 页（调试用）"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="只输出汇总"),
) -> None:
    """批量提取学生手写作答。"""
    from .pipeline import build_pipeline, iter_pdfs

    _setup_logging()

    overrides: dict = {}
    if engine:
        overrides.setdefault("ocr", {})["engine"] = engine
    if diff_method:
        overrides.setdefault("difference", {})["method"] = diff_method
    if no_debug:
        overrides.setdefault("output", {})["save_debug"] = False
        overrides["output"]["save_aligned"] = False
        overrides["output"]["save_mask"] = False
        overrides["output"]["save_crops"] = False

    cfg = Config.load(config_path, overrides=overrides or None)
    tpl = _resolve_template(str(template))

    if tpl.is_empty() and not cfg.roi.allow_full_page:
        raise typer.BadParameter("模板没有定义任何 ROI，且 allow_full_page=false，无从处理。")

    pdfs = iter_pdfs(input_path)
    if not pdfs:
        raise typer.BadParameter(f"在 {input_path} 下没有找到 PDF")

    _apply_offline(cfg)

    typer.echo(f"模板: {tpl.template_id} (DPI {tpl.dpi}, "
               f"{sum(len(p.questions) for p in tpl.pages)} 题)")
    typer.echo(f"待处理: {len(pdfs)} 份 PDF")
    typer.echo("-" * 66)

    pipe = build_pipeline(tpl, cfg, model_root=MODELS_DIR if MODELS_DIR.exists() else None)
    pipe.ocr.warmup()

    total_q = 0
    total_review = 0
    failures: list[tuple[str, str]] = []

    for pdf in pdfs:
        outdir = Path(output) / pdf.stem
        try:
            if max_pages:
                from .pdfio import render_pdf as _render

                pages = _render(pdf, dpi=tpl.dpi, max_pages=max_pages)
                del pages  # 只为提前暴露渲染错误
            result = pipe.process_pdf(pdf, output_dir=outdir)
        except Exception as exc:
            log.exception("处理失败: %s", pdf.name)
            failures.append((pdf.name, f"{type(exc).__name__}: {str(exc)[:120]}"))
            if not cfg.run.continue_on_error:
                raise
            continue

        total_q += result.total_questions
        total_review += result.review_questions

        if not quiet:
            for page in result.pages:
                flag = "" if page.status == "ok" else f"  [{page.status}]"
                typer.echo(f"{pdf.stem} p{page.page}{flag} "
                           f"align={page.alignment.get('score', 0):.3f}")
                for q in page.questions:
                    mark = "!" if q.review_required else " "
                    text = q.text if q.text else "(空)"
                    typer.echo(f"  {mark} Q{q.question_id}: {text}  [{q.confidence:.2f}]")
                    for r in q.review_reasons:
                        typer.echo(f"      - {r}")
        typer.echo(f"-> {outdir}")

    typer.echo("-" * 66)
    typer.echo(f"完成: {len(pdfs) - len(failures)}/{len(pdfs)} 份, "
               f"共 {total_q} 题, {total_review} 题需人工复核")
    if failures:
        typer.echo("失败:")
        for name, why in failures:
            typer.echo(f"  {name}: {why}")
        raise typer.Exit(code=2)


@app.command("reocr")
def reocr_cmd(
    run_dir: Path = typer.Argument(..., help="之前 run 生成的某个输出目录"),
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    engine: Optional[str] = typer.Option(None, help="换引擎重跑，例如 paddleocr"),
    model_type: Optional[str] = typer.Option(None, help="覆盖 ocr.model_type (server / mobile)"),
    ocr_version: Optional[str] = typer.Option(None, help="覆盖 ocr.rec_ocr_version"),
) -> None:
    """只重跑 OCR，不重跑渲染/配准/差分。

    技术方案第 32 节：中间产物（aligned + crop）都存下来了，换模型时
    只需要拿现成 crop 重新识别，不用把前面的流程再跑一遍。
    """
    from .reocr import rerun_from_run_dir

    _setup_logging()
    overrides: dict = {}
    if engine:
        overrides.setdefault("ocr", {})["engine"] = engine
    if model_type:
        overrides.setdefault("ocr", {})["model_type"] = model_type
    if ocr_version:
        overrides.setdefault("ocr", {})["rec_ocr_version"] = ocr_version
    if not overrides:
        raise typer.BadParameter("请至少指定 --engine / --model-type / --ocr-version 之一")

    cfg = Config.load(config_path, overrides=overrides)
    _apply_offline(cfg)
    result = rerun_from_run_dir(run_dir, cfg, model_root=MODELS_DIR if MODELS_DIR.exists() else None)
    typer.echo(f"重跑完成: {result.total_questions} 题, {result.review_questions} 题需复核 -> {run_dir}")


@app.command("fetch-models")
def fetch_models(
    out_dir: Path = typer.Option(MODELS_DIR, "--out", help="模型存放目录"),
    model_type: str = typer.Option("server", help="server / mobile"),
    ocr_version: str = typer.Option("PP-OCRv5", help="PP-OCRv5 / PP-OCRv6 / PP-OCRv4"),
) -> None:
    """联网环境下预下载 OCR 模型到本地。

    之后就可以完全断网运行。建议在联网机器上先跑一次这个命令。
    """
    from .ocr import create_engine

    _setup_logging()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = Config.load(None, overrides={
        "ocr": {"engine": "rapidocr", "model_type": model_type,
                "rec_ocr_version": ocr_version, "det_ocr_version": ocr_version},
        "offline": {"enforce": False},
    })
    typer.echo(f"正在下载 {model_type} / {ocr_version} 模型到 {out_dir.resolve()} ...")
    engine = create_engine(cfg.ocr, model_root=out_dir)
    files = engine.ensure_models()
    for f in files:
        typer.echo(f"  {f.name}  {f.stat().st_size / 1024 / 1024:.1f} MB")
    if not files:
        typer.echo("未在目标目录找到模型文件，请检查 RapidOCR 的下载日志。")
        raise typer.Exit(code=1)
    typer.echo("完成。现在可以断网运行 homework-ocr run。")


@app.command("doctor")
def doctor_cmd(
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
) -> None:
    """环境自检：依赖、模型、离线封禁、字体。"""
    from . import debugviz

    _setup_logging()
    cfg = _load_config(config_path)
    problems: list[str] = []
    notes: list[str] = []

    typer.echo("== Python 依赖 ==")
    for mod, required in (("cv2", True), ("numpy", True), ("fitz", True), ("yaml", True),
                          ("typer", True), ("rapidocr", True), ("onnxruntime", True),
                          ("paddleocr", False)):
        try:
            m = __import__(mod)
            ver = getattr(m, "__version__", "?")
            mark = "OK " if required else "-- "
            typer.echo(f"  {mark}{mod:14s} {ver}")
            if required and mod in {"rapidocr", "onnxruntime"}:
                try:
                    import importlib.metadata as md

                    typer.echo(f"     version: {md.version('rapidocr' if mod == 'rapidocr' else 'onnxruntime')}")
                except Exception:
                    pass
        except ImportError:
            typer.echo(f"  !! {mod:14s} 未安装" + ("" if required else " (可选)"))
            if required:
                problems.append(f"缺少依赖 {mod}")

    typer.echo("\n== 模型 ==")
    local = list(MODELS_DIR.glob("*.onnx")) if MODELS_DIR.exists() else []
    for f in local:
        typer.echo(f"  OK  {f.name}  {f.stat().st_size / 1024 / 1024:.1f} MB")
    if not local:
        try:
            import rapidocr

            site = Path(rapidocr.__file__).parent / "models"
            for f in sorted(site.glob("*.onnx")):
                typer.echo(f"  --  {f.name}  {f.stat().st_size / 1024 / 1024:.1f} MB  (site-packages)")
        except ImportError:
            pass
        notes.append("未发现本地模型，请先执行 homework-ocr fetch-models")

    typer.echo("\n== 可视化 ==")
    if debugviz.has_cjk_support():
        typer.echo("  OK  中文字体可用，debug 图能显示中文标签")
    else:
        typer.echo("  --  未找到中文字体，debug 图标签会退回英文（不影响识别结果）")

    typer.echo("\n== 离线封禁 ==")
    offline_guard.set_env_offline()
    _apply_offline(cfg)
    typer.echo("  OK  出站网络已封禁（allow_loopback=%s）" % cfg.offline.allow_loopback)
    try:
        import socket

        s = socket.socket()
        s.settimeout(2.0)
        s.connect(("pypi.org", 443))
        s.close()
        problems.append("出站网络竟然连通了，隐私要求无法满足")
        typer.echo("  !! 仍然可以联网，offline_guard 失效")
    except offline_guard.NetworkBlocked as exc:
        typer.echo("  OK  实测出站连接已被拦截")
        _ = exc
    except OSError as exc:
        notes.append(f"外网本来就不可达（{exc.__class__.__name__}），封禁未触发")
        typer.echo("  --  外网不可达，未能实测封禁")
    finally:
        offline_guard.disable_offline()

    typer.echo("\n== 结论 ==")
    for n in notes:
        typer.echo(f"  提示: {n}")
    for p in problems:
        typer.echo(f"  问题: {p}")
    if problems:
        raise typer.Exit(code=1)
    typer.echo("  环境可用。")


@app.command("make-demo")
def make_demo(
    out_dir: Path = typer.Option(Path("demo"), "--out", help="生成目录"),
    dpi: int = typer.Option(200, help="DPI（自测用，低一点更快）"),
) -> None:
    """生成一套自测数据：空白模板 PDF + 模拟"手写"扫描件。

    真实扫描件在验证前无法端到端跑通，这套合成数据用来验证代码链路本身
    （渲染/配准/差分/分组/OCR/导出）。模拟笔迹用随机贝塞尔曲线，形态接近
    手写但不是真手写，所以只能验证链路，不能评估识别准确率。
    """
    from .demogen import build_demo

    _setup_logging()
    info = build_demo(out_dir, dpi=dpi)
    typer.echo(f"已生成自测数据于 {out_dir.resolve()}")
    for k, v in info.items():
        typer.echo(f"  {k}: {v}")
    typer.echo("\n用法：")
    typer.echo(f"  homework-ocr run --template {out_dir/'template'} --input {out_dir/'input'} --output {out_dir/'output'}")


@app.command("gui")
def gui_cmd(
    template: Optional[Path] = typer.Option(None, "--template", "-t", help="模板目录（可省略，界面上选）"),
    input_path: Optional[Path] = typer.Option(None, "--input", "-i", help="学生 PDF 目录（可省略）"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="输出根目录（可省略）"),
    config_path: Optional[Path] = typer.Option(DEFAULT_CONFIG, "--config", "-c"),
    port: int = typer.Option(0, help="端口，0 = 用上次设置或默认 8000"),
    host: str = typer.Option("127.0.0.1", help="监听地址。不要改成 0.0.0.0，会把学生数据暴露到局域网。"),
    no_browser: bool = typer.Option(False, "--no-browser", help="不自动打开浏览器"),
    reset: bool = typer.Option(False, "--reset", help="清除已保存的选择，重新走引导"),
) -> None:
    """启动图形界面。直接运行即可，不需要任何参数。

    第一次打开会引导你选择模板和文件夹；之后记住上次的选择，一键启动。
    """
    from .gui import serve
    from .userdata import Settings

    _setup_logging()

    if reset:
        from .userdata import settings_path

        settings_path().unlink(missing_ok=True)
        typer.echo("已清除上次的设置，将重新引导。")

    settings = Settings.load()
    if port:
        settings.port = port
    if not settings.port:
        settings.port = 8000

    # 命令行给的参数用来新建/覆盖一个批次；没给就用上次记住的。
    tpl_dir = Path(template) if template else None
    inp_dir = Path(input_path) if input_path else None
    out_dir = Path(output) if output else None

    batch = settings.get_batch()

    if tpl_dir or inp_dir:
        # 显式给了参数 -> 视为要定义一个批次。
        name = batch.name if (batch and batch.template_path == tpl_dir) else "默认"
        if tpl_dir:
            try:
                _resolve_template(str(tpl_dir))  # 提前校验，坏模板当场报
            except (typer.BadParameter, FileNotFoundError, ValueError) as exc:
                typer.echo(f"模板不可用：{exc}", err=True)
                raise typer.Exit(code=1)
        if batch and batch.template == str(tpl_dir or "") and batch.input_dir == str(inp_dir or ""):
            pass  # 和当前批次一致，不必新建
        else:
            batch = settings.add_batch(name, str(tpl_dir) if tpl_dir else (batch.template if batch else ""),
                                       str(inp_dir) if inp_dir else (batch.input_dir if batch else ""))
        settings.save()

    if out_dir:
        settings.output_dir = str(out_dir)
        settings.save()
    elif not settings.output_dir and inp_dir:
        settings.output_dir = str(inp_dir.parent / "识别结果")
        settings.save()

    if inp_dir:
        Path(inp_dir).mkdir(parents=True, exist_ok=True)
    if settings.output_dir:
        Path(settings.output_dir).mkdir(parents=True, exist_ok=True)
    if batch and batch.input_dir:
        Path(batch.input_dir).mkdir(parents=True, exist_ok=True)

    cfg = _load_config(config_path)
    _apply_offline(cfg)

    root = Path(settings.output_dir) if settings.output_dir else None
    if batch and batch.template:
        typer.echo(f"批次: {batch.name}")
        typer.echo(f"模板: {batch.template}")
    if batch and batch.input_dir:
        typer.echo(f"作业: {batch.input_dir}")
    if root and batch:
        typer.echo(f"结果: {root / batch.id}")
    if not batch:
        typer.echo("还没有批次，启动后在页面上新建一个。")
    typer.echo("")
    typer.echo(f"界面地址: http://{host}:{settings.port}/")
    typer.echo("出站网络已封禁，数据不会离开本机。关闭窗口或按 Ctrl+C 退出。")

    if host not in {"127.0.0.1", "localhost"}:
        log.warning("监听地址是 %s 而非本机，这会让学生数据可被局域网访问。", host)

    serve(cfg, settings, batch=batch, output_root=root,
          host=host, port=settings.port, open_browser=not no_browser)


@app.command("export")
def export_cmd(
    output: Path = typer.Option(Path("output"), "--output", "-o", help="run 的输出根目录"),
    to: Optional[Path] = typer.Option(None, "--to", help="输出 xlsx 路径，缺省写到 output 根目录"),
    name: Optional[str] = typer.Option(None, "--name", help="输出文件名（不含扩展名）"),
) -> None:
    """把已有结果（含人工复核）导出为 Excel。"""
    from .excel import export_xlsx

    _setup_logging()
    root = Path(output)
    if not root.exists():
        raise typer.BadParameter(f"输出目录不存在: {root}")

    docs: list[dict] = []
    stores: dict = {}
    for run_dir in sorted(d for d in root.iterdir() if (d / "result.json").exists()):
        docs.append(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
        from .review import ReviewStore

        stores[run_dir.name] = ReviewStore.load(run_dir)

    if not docs:
        raise typer.BadParameter(f"{root} 下没有结果，请先执行 run")

    target = Path(to) if to else root / f"{name or 'homework_results'}.xlsx"
    export_xlsx(target, docs, stores)

    need_review = sum(1 for d in docs for p in d.get("pages", [])
                      for q in p.get("questions", []) if q.get("review_required"))
    typer.echo(f"已导出: {target}")
    typer.echo(f"  {len(docs)} 份作业, 其中 {need_review} 题标记为需复核")


@app.command("show")
def show_cmd(
    template: Path = typer.Option(..., "--template", "-t"),
    result: Path = typer.Option(..., "--result", "-r", help="result.json"),
) -> None:
    """按阅读顺序打印一份结果，并标出需要复核的题目。"""
    _setup_logging()
    data = json.loads(Path(result).read_text(encoding="utf-8"))
    typer.echo(f"{data['document']}  模板={data['template_id']}  状态={data['status']}")
    typer.echo(f"引擎: {data.get('engine')}")
    for page in data.get("pages", []):
        typer.echo(f"\n[第 {page['page']} 页] {page['status']}  "
                   f"align={page.get('alignment', {}).get('score', 0):.3f}")
        for q in page.get("questions", []):
            flag = "★需复核" if q["review_required"] else "        "
            typer.echo(f"  {flag} Q{q['question_id']}: {q['text'] or '(空)'}")
            for r in q.get("review_reasons", []):
                typer.echo(f"           - {r}")
    s = data.get("summary", {})
    typer.echo(f"\n共 {s.get('total_questions', 0)} 题，{s.get('review_required', 0)} 题需复核")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
