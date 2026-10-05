"""本地 GUI 服务。

关于隐私：只绑定 127.0.0.1，不对外监听，配合 offline_guard 封禁出站。
浏览器页面和数据都在本机，没有任何请求会离开这台电脑。
选 HTTP + 浏览器而不是原生窗口，是因为复核界面要并排显示手写 crop 和识别文本，
这种图像密集的界面用 HTML 写效率高得多、可访问性也好。

启动：
    homework-ocr gui --template templates/yuwen_g3_1 --input input/ --output output/
"""

from __future__ import annotations

import json
import logging
import threading
import traceback
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .. import __version__
from ..browse import list_dir, validate_template_dir
from ..config import Config
from ..excel import export_xlsx
from ..pipeline import DocumentResult, build_pipeline, fingerprint_config, iter_pdfs
from ..postprocess import stitch_lines
from ..review import ReviewEntry, ReviewStore
from ..template import Template
from ..userdata import Batch, Settings, default_workspace, find_templates, user_data_dir

log = logging.getLogger(__name__)

HERE = Path(__file__).parent
TEMPLATES_DIR = HERE / "templates"
STATIC_DIR = HERE / "static"

# 允许直接预览的中间产物白名单。杜绝 ../ 之类的路径穿越。
IMAGE_ALLOWLIST = {"aligned", "mask", "debug", "answer"}


def is_safe_child(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------- 批处理状态
@dataclass
class JobState:
    running: bool = False
    total: int = 0
    done: int = 0
    current: str = ""
    failed: list[str] = field(default_factory=list)
    error: str = ""
    finished_at: str = ""
    log: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        pct = int(self.done * 100 / self.total) if self.total else 0
        return {
            "running": self.running,
            "total": self.total,
            "done": self.done,
            "current": self.current,
            "failed": self.failed,
            "error": self.error,
            "pct": pct,
            "finished_at": self.finished_at,
            "log": self.log[-40:],
        }


class GuiContext:
    def __init__(
        self,
        cfg: Config,
        settings: "Settings",
        *,
        batch: "Batch | None" = None,
        output_root: Path | None = None,
    ):
        self.cfg = cfg
        self.settings = settings
        self.batch = batch
        self.output_root = output_root
        self.template: Template | None = None
        self.job = JobState()
        self.model_root = Path("models") if Path("models").exists() else None
        self._lock = threading.Lock()
        self._load_template()

    # ---------- 状态 ----------
    def _load_template(self) -> None:
        self.template = None
        if self.batch and self.batch.template_path:
            try:
                self.template = Template.load(self.batch.template_path)
            except (FileNotFoundError, ValueError, OSError) as exc:
                log.warning("模板加载失败: %s", exc)

    @property
    def input_dir(self) -> Path | None:
        return self.batch.input_path if self.batch else None

    @property
    def output_dir(self) -> Path | None:
        """当前批次的输出目录 = 输出根目录 / 批次名。

        输出根目录一律从 settings 现取，而不是缓存成实例属性——
        新建批次时 output_dir 才刚被写进 settings，缓存会拿到旧值。
        """
        root = Path(self.settings.output_dir) if self.settings.output_dir else None
        if root is None:
            return None
        if not self.batch:
            return root
        return root / self.batch.id

    @property
    def ready(self) -> bool:
        return (
            self.template is not None
            and self.input_dir is not None
            and self.input_dir.exists()
        )

    def apply_settings(self, s: "Settings") -> None:
        """按新设置重建上下文。切批次会换模板，所以要重新加载。"""
        self.settings = s
        self.batch = s.get_batch()
        self.output_root = Path(s.output_dir) if s.output_dir else None
        self._load_template()

    def use_batch(self, batch: "Batch") -> None:
        self.batch = batch
        self.settings.active_batch = batch.id
        self.settings.save()
        self._load_template()
        self.job = JobState()

    def require_ready(self) -> None:
        if not self.ready:
            raise HTTPException(status_code=409, detail="当前批次还没选好模板或作业文件夹")

    def run_dirs(self) -> list[Path]:
        out = self.output_dir
        if not out or not out.exists():
            return []
        return sorted(d for d in out.iterdir() if (d / "result.json").exists())


def create_app(
    cfg: Config,
    settings: "Settings",
    *,
    batch: "Batch | None" = None,
    output_root: Path | None = None,
    open_browser: bool = True,
) -> FastAPI:
    ctx = GuiContext(cfg, settings, batch=batch, output_root=output_root)
    app = FastAPI(title="homework_ocr", docs_url=None, redoc_url=None)
    app.state.ctx = ctx

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    def render(request: Request, name: str, **ctx_extra: Any) -> HTMLResponse:
        base = {
            "version": __version__,
            "template_id": ctx.template.template_id if ctx.template else "",
            "template_dpi": ctx.template.dpi if ctx.template else 0,
            "input_dir": str(ctx.input_dir) if ctx.input_dir else "",
            "output_dir": str(ctx.output_dir) if ctx.output_dir else "",
        }
        base.update(ctx_extra)
        return templates.TemplateResponse(request, name, base)

    # ------------------------------------------------------------ 批处理
    def _worker(pdfs: list[Path]) -> None:
        # 抓一个局部引用：start_run 建好的那个 job 对象。finally 里只改它，
        # 保证「跑完了」这个状态一定会被写回去（之前这里会中途换掉 ctx.job，
        # 一旦前面抛异常 running 就永远是 True，按钮再也点不动）。
        job = ctx.job
        job.log.append(f"开始处理 {len(pdfs)} 份作业…")
        try:
            pipe = build_pipeline(ctx.template, ctx.cfg, model_root=ctx.model_root)  # type: ignore[arg-type]
            pipe.ocr.warmup()
            for pdf in pdfs:
                job.current = pdf.stem
                outdir = ctx.output_dir / pdf.stem  # type: ignore[operator]
                try:
                    result = pipe.process_pdf(pdf, output_dir=outdir)
                    job.log.append(
                        f"{pdf.stem}: {result.total_questions} 题, {result.review_questions} 题需复核"
                    )
                except Exception as exc:  # 单份失败不中断整批
                    log.exception("处理失败: %s", pdf)
                    job.failed.append(pdf.stem)
                    job.log.append(f"{pdf.stem}: 失败 {type(exc).__name__}: {str(exc)[:120]}")
                finally:
                    job.done += 1
            if job.failed:
                job.log.append(f"结束：成功 {job.done - len(job.failed)} 份，失败 {len(job.failed)} 份。")
            else:
                job.log.append(f"结束：{job.done} 份全部完成。")
        except Exception as exc:  # pragma: no cover
            log.exception("批处理崩溃")
            job.error = f"{type(exc).__name__}: {str(exc)[:300]}"
            job.log.append(traceback.format_exc(limit=3))
        finally:
            job.running = False
            job.current = ""
            from datetime import datetime

            job.finished_at = datetime.now().isoformat(timespec="seconds")

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        if not ctx.settings.batches and not ctx.settings.template:
            return render(request, "setup.html", templates=find_templates(),
                          saved=asdict(ctx.settings), home=str(default_workspace()))
        return render(request, "index.html", job=ctx.job.as_dict(),
                      students=_student_overview(ctx),
                      batches=_batch_overview(ctx),
                      active=ctx.batch.id if ctx.batch else "",
                      inbox_count=_inbox_count(ctx),
                      ready=ctx.ready)

    # ------------------------------------------------------------ 引导 / 设置
    @app.get("/api/browse")
    def browse(path: Optional[str] = None, show_hidden: bool = False,
               files: bool = False):
        # files=1 = 选 PDF 文件时，连文件一起列出来
        return JSONResponse(list_dir(path, show_hidden, want_files=files))

    @app.get("/api/templates")
    def api_templates():
        return JSONResponse(find_templates())

    @app.get("/api/validate-template")
    def api_validate(path: str):
        return JSONResponse(validate_template_dir(path))

    @app.post("/api/create-template")
    async def api_create_template(request: Request):
        """从用户选中的空白模板 PDF 直接建模板，省掉命令行那一步。"""
        from ..template import create_from_pdf

        body = await request.json()
        raw = str(body.get("pdf") or "").strip()
        if not raw:
            return JSONResponse({"ok": False, "message": "请选择空白模板 PDF"}, status_code=400)

        pdf = Path(raw).expanduser()
        if not pdf.exists() or pdf.suffix.lower() != ".pdf":
            return JSONResponse({"ok": False, "message": "找不到这个 PDF"}, status_code=400)

        folder = Path(str(body.get("folder") or (default_workspace() / "templates" / pdf.stem)))
        folder.mkdir(parents=True, exist_ok=True)
        dest = folder / pdf.name
        if pdf.resolve() != dest.resolve():
            dest.write_bytes(pdf.read_bytes())

        tpl = create_from_pdf(dest, template_id=folder.name, dpi=int(body.get("dpi") or 300))
        tpl.save()
        return JSONResponse({
            "ok": True,
            "path": str(folder),
            "id": tpl.template_id,
            "pages": len(tpl.pages),
            "questions": 0,
            "message": "模板已创建。还差一步：标注每道题的答案区域。",
        })

    @app.post("/api/setup")
    async def api_setup(request: Request):
        """首次引导：建第一个批次。"""
        body = await request.json()

        tpl_raw = str(body.get("template") or "").strip()
        if not tpl_raw:
            return JSONResponse({"ok": False, "message": "请先选择模板"}, status_code=400)
        check = validate_template_dir(tpl_raw)
        if not check.get("ok"):
            return JSONResponse({"ok": False, "message": check.get("reason", "模板不可用")}, status_code=400)

        inp = str(body.get("input") or "").strip()
        if not inp or not Path(inp).expanduser().exists():
            return JSONResponse({"ok": False, "message": "请选择存放学生作业的文件夹"}, status_code=400)

        outp = str(body.get("output") or "").strip()
        if not outp:
            outp = str(Path(inp).expanduser().parent / "识别结果")
        Path(outp).expanduser().mkdir(parents=True, exist_ok=True)

        new_settings = Settings(
            output_dir=str(Path(outp).expanduser()),
            config_file=str(body.get("config_file") or ""),
            port=int(body.get("port") or ctx.settings.port),
            onboarded=True,
        )
        new_settings.add_batch(str(body.get("batch_name") or "第一次"), tpl_raw,
                               str(Path(inp).expanduser()))
        new_settings.save()
        ctx.apply_settings(new_settings)
        return JSONResponse({
            "ok": True,
            "ready": ctx.ready,
            "questions": check.get("questions", 0),
            "message": "" if check.get("questions") else
                      "这个模板还没标注答案区域，识别会按整页进行，准确率会明显下降。",
        })

    @app.post("/api/import")
    async def api_import(request: Request):
        """路线 A：浏览器读文件内容上传，后端不再需要本机路径。

        为什么需要这个：浏览器出于安全限制**拿不到**用户所选文件的绝对路径
        （`File` 对象里没有 `path` 属性，这是实测确认的）。所以在 Web 界面里
        让人去指定一个"服务器能访问的文件夹"是很别扭的补丁。

        正解是：浏览器用原生文件对话框选文件（用户体验正常），把**内容**传给
        后端。后端把文件存进自己管理的收件箱，再照常处理。
        代价是结果落在程序自管区而不是用户指定的目录——这正是路线 A 的取舍。
        """
        ctx.require_ready()
        batch = ctx.batch
        if batch is None:
            raise HTTPException(status_code=409, detail="没有当前批次")

        form = await request.form()
        files = form.getlist("files") or []

        inbox = Path(batch.inbox_dir) if batch.inbox_dir else (user_data_dir() / "inbox" / batch.id)
        inbox.mkdir(parents=True, exist_ok=True)

        saved: list[str] = []
        skipped: list[str] = []
        for item in files:
            raw = getattr(item, "read", None)
            if raw is None or not getattr(item, "filename", None):
                continue
            name = Path(str(item.filename)).name  # 只取文件名，丢掉任何路径成分
            if not name.lower().endswith(".pdf"):
                skipped.append(f"{name}（不是 PDF）")
                continue
            dest = inbox / name
            # 同名文件不覆盖，追加 (2)、(3)
            if dest.exists():
                stem, suf = dest.stem, dest.suffix
                i = 2
                while (inbox / f"{stem}({i}){suf}").exists():
                    i += 1
                dest = inbox / f"{stem}({i}){suf}"
            dest.write_bytes(await item.read())
            saved.append(dest.name)

        # 记下收件箱位置，之后每次启动都能找到这些文件
        if saved and not batch.inbox_dir:
            batch.inbox_dir = str(inbox)
            ctx.settings.save()

        existing = len(list(inbox.glob("*.pdf"))) if inbox.exists() else 0
        return JSONResponse({
            "ok": True,
            "saved": saved,
            "skipped": skipped,
            "inbox": str(inbox),
            "total": existing,
        })

    @app.get("/api/inbox")
    def api_inbox():
        batch = ctx.batch
        if batch is None or not batch.inbox_dir:
            return JSONResponse({"files": [], "inbox": ""})
        p = Path(batch.inbox_dir)
        files = sorted(x.name for x in p.glob("*.pdf")) if p.exists() else []
        return JSONResponse({"files": files, "inbox": str(p)})

    @app.delete("/api/inbox/{name}")
    def api_inbox_delete(name: str):
        batch = ctx.batch
        if batch is None or not batch.inbox_dir or "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="非法文件名")
        target = Path(batch.inbox_dir) / name
        if not is_safe_child(target, Path(batch.inbox_dir)) or not target.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        target.unlink()
        return JSONResponse({"ok": True})

    @app.get("/api/batches")
    def api_batches():
        return JSONResponse({"batches": _batch_overview(ctx),
                             "active": ctx.batch.id if ctx.batch else ""})

    @app.post("/api/batches")
    async def api_add_batch(request: Request):
        """新建批次：批次名 + 模板 + 作业文件夹。

        模板的选择粒度是「批次」——期中、期末、周测各建一个批次，
        同一份扫描件在不同批次里可以按不同模板处理，互不干扰。
        """
        body = await request.json()
        name = str(body.get("name") or "").strip()
        tpl_raw = str(body.get("template") or "").strip()
        inp = str(body.get("input") or "").strip()

        if not name:
            return JSONResponse({"ok": False, "message": "给这个批次起个名字，例如「期中考试」"},
                                status_code=400)
        if not tpl_raw:
            return JSONResponse({"ok": False, "message": "请选择这个批次用哪份模板"}, status_code=400)
        check = validate_template_dir(tpl_raw)
        if not check.get("ok"):
            return JSONResponse({"ok": False, "message": check.get("reason", "模板不可用")}, status_code=400)
        if not inp or not Path(inp).expanduser().exists():
            return JSONResponse({"ok": False, "message": "请选择这个批次的作业文件夹"}, status_code=400)

        s = ctx.settings
        if not s.output_dir:
            s.output_dir = str(Path(inp).expanduser().parent / "识别结果")
        batch = s.add_batch(name, tpl_raw, str(Path(inp).expanduser()))
        s.save()
        ctx.use_batch(batch)
        return JSONResponse({
            "ok": True,
            "batch": batch.to_dict(),
            "questions": check.get("questions", 0),
            "message": "" if check.get("questions") else "这个模板还没标注答案区域，准确率会下降。",
        })

    @app.post("/api/batches/{batch_id}/activate")
    def api_activate_batch(batch_id: str):
        b = ctx.settings.batch(batch_id)
        if not b:
            return JSONResponse({"ok": False, "message": "批次不存在"}, status_code=404)
        ctx.use_batch(b)
        return JSONResponse({"ok": True, "batch": b.to_dict()})

    @app.delete("/api/batches/{batch_id}")
    def api_delete_batch(batch_id: str):
        s = ctx.settings
        if not s.batch(batch_id):
            return JSONResponse({"ok": False, "message": "批次不存在"}, status_code=404)
        s.remove_batch(batch_id)
        s.save()
        ctx.apply_settings(s)
        nb = s.get_batch()
        if nb:
            ctx.use_batch(nb)
        return JSONResponse({"ok": True})

    @app.get("/api/settings")
    def api_settings():
        return JSONResponse(asdict(ctx.settings))

    @app.post("/api/run")
    def start_run(bg: BackgroundTasks):
        ctx.require_ready()
        # 先把活数清楚再"开跑"。空文件夹以前会装作启动成功、0/0 秒结束、
        # 日志一个字都没有，界面上看着就像按钮根本没反应。
        batch = ctx.batch
        sources = batch.source_dirs() if batch else []
        pdfs = iter_pdfs(sources)
        if not pdfs:
            return JSONResponse(
                {"ok": False,
                 "message": "还没有可识别的作业。"
                            f"上传几份学生作业的 PDF，或指定一个装着 PDF 的文件夹。"
                            f"（当前查找位置：{', '.join(str(s) for s in sources) or '未设置'}）"},
                status_code=400,
            )
        with ctx._lock:
            if ctx.job.running:
                return JSONResponse({"ok": False, "message": "已有批处理在运行"}, status_code=409)
            ctx.job = JobState(running=True, total=len(pdfs))
        bg.add_task(_worker, pdfs)
        return JSONResponse({"ok": True, "total": len(pdfs)})

    @app.get("/api/job")
    def job_status():
        return JSONResponse(ctx.job.as_dict())

    # ------------------------------------------------------------ 复核
    @app.get("/review/{stem}", response_class=HTMLResponse)
    def review_page(request: Request, stem: str, all: bool = Query(False, alias="all")):
        run_dir = _safe_run_dir(ctx, stem)
        data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        store = ReviewStore.load(run_dir)
        # ?all=1 = 抽查模式：列出全部题目，不只看待复核的。
        items = _review_items(data, store, only_pending=not all)
        return render(request, "review.html", stem=stem, items=items, doc=data,
                      stats=store.stats(), empty=not items, show_all=all)

    @app.get("/api/review/{stem}")
    def review_items(stem: str, only_pending: bool = Query(True)):
        run_dir = _safe_run_dir(ctx, stem)
        data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        store = ReviewStore.load(run_dir)
        return JSONResponse({
            "items": _review_items(data, store, only_pending=only_pending),
            "stats": store.stats(),
        })

    @app.post("/api/review/{stem}")
    async def save_review(stem: str, request: Request):
        run_dir = _safe_run_dir(ctx, stem)
        body = await request.json()
        store = ReviewStore.load(run_dir)
        page = int(body.get("page", 1))
        qid = str(body.get("question_id", ""))

        store.update(
            page, qid,
            corrected_text=body.get("text"),
            status=body.get("status"),
            note=body.get("note"),
            original_text=body.get("original_text"),
            confidence=body.get("confidence"),
            review_reasons=body.get("review_reasons"),
        )
        store.save()
        return JSONResponse({"ok": True, "stats": store.stats()})

    @app.post("/api/review/{stem}/bulk")
    async def bulk_review(stem: str, request: Request):
        """一键确认"识别无误"。只影响传入的题号列表。"""
        run_dir = _safe_run_dir(ctx, stem)
        body = await request.json()
        store = ReviewStore.load(run_dir)
        for item in body.get("items", []):
            store.update(
                int(item.get("page", 1)),
                str(item.get("question_id", "")),
                corrected_text=item.get("original_text", ""),
                status="accepted",
                original_text=item.get("original_text", ""),
                confidence=item.get("confidence"),
                review_reasons=item.get("review_reasons"),
            )
        store.save()
        return JSONResponse({"ok": True, "count": len(body.get("items", [])), "stats": store.stats()})

    # ------------------------------------------------------------ 产物
    @app.get("/api/image/{stem}/{name}")
    def image(stem: str, name: str):
        run_dir = _safe_run_dir(ctx, stem)
        if "/" in name or "\\" in name or ".." in name:
            raise HTTPException(status_code=400, detail="非法文件名")
        path = run_dir / name
        if not is_safe_child(path, run_dir) or not path.exists():
            raise HTTPException(status_code=404, detail="文件不存在")
        if not any(tag in name for tag in IMAGE_ALLOWLIST):
            raise HTTPException(status_code=403, detail="该文件类型不允许预览")
        return FileResponse(str(path))

    @app.get("/api/export/xlsx")
    def export():
        docs, stores = _collect_results(ctx)
        if not docs:
            raise HTTPException(status_code=400, detail="还没有可导出的结果，请先运行批处理")
        path = ctx.output_dir / f"homework_results_{ctx.template.template_id}.xlsx"
        export_xlsx(path, docs, stores)
        return FileResponse(str(path), filename=path.name,
                            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    @app.get("/api/result/{stem}")
    def raw_result(stem: str):
        run_dir = _safe_run_dir(ctx, stem)
        return JSONResponse(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))

    @app.get("/api/template.json")
    def template_json():
        if ctx.template is None:
            raise HTTPException(status_code=409, detail="尚未选择模板")
        return JSONResponse(ctx.template.to_dict())

    @app.get("/healthz")
    def healthz():
        return JSONResponse({
            "ok": True,
            "version": __version__,
            "ready": ctx.ready,
            "template": ctx.template.template_id if ctx.template else None,
        })

    if open_browser:
        _schedule_browser(8000, ctx)

    return app


def _schedule_browser(port: int, ctx: GuiContext) -> None:  # pragma: no cover
    import threading
    import webbrowser

    def _open() -> None:
        import time

        for _ in range(40):
            time.sleep(0.25)
            try:
                import urllib.request

                urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1).read()
                webbrowser.open(f"http://127.0.0.1:{port}/")
                return
            except Exception:
                continue

    threading.Thread(target=_open, daemon=True).start()
    _ = ctx


def _safe_run_dir(ctx: GuiContext, stem: str) -> Path:
    if "/" in stem or "\\" in stem or ".." in stem:
        raise HTTPException(status_code=400, detail="非法路径")
    run_dir = ctx.output_dir / stem
    if not is_safe_child(run_dir, ctx.output_dir):
        raise HTTPException(status_code=400, detail="非法路径")
    if not (run_dir / "result.json").exists():
        raise HTTPException(status_code=404, detail=f"找不到 {stem} 的结果")
    return run_dir


def _review_items(doc: dict[str, Any], store: ReviewStore, only_pending: bool = True) -> list[dict[str, Any]]:
    """组装复核界面用的题目列表。"""
    items: list[dict[str, Any]] = []
    for page in doc.get("pages", []):
        pno = int(page.get("page", 1))
        for q in page.get("questions", []):
            qid = str(q.get("question_id", ""))
            entry = store.get(pno, qid)
            resolved = bool(entry and entry.status in {"accepted", "corrected", "rejected"})
            need = bool(q.get("review_required"))
            if only_pending and (resolved or not need):
                continue

            crop_name = q.get("crop_path")
            mask_name = page.get("debug_paths", {}).get("mask", "")
            debug_name = page.get("debug_paths", {}).get("debug", "")

            items.append({
                "page": pno,
                "question_id": qid,
                "question_type": q.get("question_type", "text"),
                "detected": bool(q.get("detected_handwriting")),
                "confidence": float(q.get("confidence", 0.0)),
                "review_required": need,
                "review_reasons": list(q.get("review_reasons", []) or []),
                "original_text": q.get("text", "") or "",
                "text": (entry.final_text if entry and entry.final_text else q.get("text", "") or ""),
                "status": entry.status if entry else "pending",
                "note": entry.note if entry else "",
                "crop_url": f"/api/image/{_stem_of(doc)}/{crop_name}" if crop_name else "",
                "mask_url": f"/api/image/{_stem_of(doc)}/{mask_name}" if mask_name else "",
                "debug_url": f"/api/image/{_stem_of(doc)}/{debug_name}" if debug_name else "",
            })
    return items


def _stem_of(doc: dict[str, Any]) -> str:
    return Path(doc.get("document", "")).stem


def _inbox_count(ctx: GuiContext) -> int:
    """这个批次已经上传了几份作业。"""
    b = ctx.batch
    if not b or not b.inbox_dir:
        return 0
    p = Path(b.inbox_dir)
    return len(list(p.glob("*.pdf"))) if p.exists() else 0


def _batch_overview(ctx: GuiContext) -> list[dict[str, Any]]:
    """批次列表给首页用。带上每个批次的学生数，方便一眼看出进度。"""
    out: list[dict[str, Any]] = []
    root = Path(ctx.settings.output_dir) if ctx.settings.output_dir else None
    for b in ctx.settings.batches:
        run_dir = (root / b.id) if root else None
        students = 0
        if run_dir and run_dir.exists():
            students = sum(1 for d in run_dir.iterdir() if (d / "result.json").exists())
        out.append({
            **b.to_dict(),
            "valid": b.is_valid(),
            "students": students,
            "output_dir": str(run_dir) if run_dir else "",
            "active": bool(ctx.batch and ctx.batch.id == b.id),
        })
    return out


def _student_overview(ctx: GuiContext) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for run_dir in ctx.run_dirs():
        try:
            data = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        store = ReviewStore.load(run_dir)
        total = review = pages_ok = pages = 0
        for page in data.get("pages", []):
            pages += 1
            pages_ok += int(bool(page.get("alignment", {}).get("success")))
            for q in page.get("questions", []):
                total += 1
                if q.get("review_required"):
                    review += 1
        out.append({
            "stem": run_dir.name,
            "document": data.get("document", run_dir.name),
            "total": total,
            "review": review,
            "resolved": store.stats()["accepted"] + store.stats()["corrected"],
            "pages": pages,
            "pages_ok": pages_ok,
            "status": data.get("status", ""),
        })
    return out


def _collect_results(ctx: GuiContext) -> tuple[list[dict[str, Any]], dict[str, ReviewStore]]:
    docs: list[dict[str, Any]] = []
    stores: dict[str, ReviewStore] = {}
    for run_dir in ctx.run_dirs():
        try:
            docs.append(json.loads((run_dir / "result.json").read_text(encoding="utf-8")))
            stores[run_dir.name] = ReviewStore.load(run_dir)
        except (OSError, json.JSONDecodeError):
            continue
    return docs, stores


HEALTH_MARKER = "homework_ocr"


def _probe_running(port: int, host: str = "127.0.0.1", timeout: float = 1.5) -> dict | None:
    """问一句「这个端口上是不是已经有一个本程序在跑」。

    已经跑着的时候再双击启动脚本，朴素做法是直接 uvicorn 绑定失败，
    用户看到的是「启动失败，退出码 3」——完全没提真正的原因。
    先问一声，就能改成「已经在运行，这就帮你打开界面」。
    """
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://{host}:{port}/healthz", timeout=timeout) as r:
            data = json.loads(r.read().decode("utf-8", "replace"))
            return data if data.get("ok") else None
    except Exception:
        return None


def serve(
    cfg: Config,
    settings: "Settings",
    *,
    batch: "Batch | None" = None,
    output_root: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8000,
    open_browser: bool = True,
) -> None:  # pragma: no cover
    import uvicorn

    # 已经在跑就别再绑一次端口，直接把界面打开就行。
    if _probe_running(port, host):
        log.info("检测到端口 %d 上已有本程序在运行，直接打开界面", port)
        print()
        print(f"  程序已经在运行了（端口 {port}）。")
        print(f"  界面地址: http://{host}:{port}/")
        print("  如果想重新启动，先关掉正在运行的窗口。")
        print()
        if open_browser:
            _open_browser(f"http://{host}:{port}/")
        return

    app = create_app(cfg, settings, batch=batch, output_root=output_root,
                     open_browser=open_browser)
    try:
        uvicorn.run(app, host=host, port=port, log_level="warning")
    except OSError as exc:
        # 探测和真正绑定之间存在竞态：可能刚好被别人抢占了端口。
        if _probe_running(port, host):
            print()
            print(f"  程序已经在运行了（端口 {port}）。界面地址: http://{host}:{port}/")
            print()
            if open_browser:
                _open_browser(f"http://{host}:{port}/")
            return
        raise OSError(
            f"端口 {port} 被其它程序占用，且那不是本工具。"
            f"请换一个端口启动，例如：--port {port + 1}"
        ) from exc


def _open_browser(url: str) -> None:
    import webbrowser

    try:
        webbrowser.open(url)
    except Exception:  # pragma: no cover
        log.warning("无法自动打开浏览器，请手动访问 %s", url)


_ = (BackgroundTasks, DocumentResult, fingerprint_config, ReviewEntry, stitch_lines)
