"""自测数据生成器。

为什么需要它：验证端到端链路需要输入，而我手上没有真实的学生扫描件。

设计要点——模拟"手写"不能用随机涂鸦。随机笔画 OCR 读不出来，测到的只是
"代码没崩"，测不到"识别结果对不对"。这里的做法是：

    用真实中文字形渲染答案，但给每个字加随机旋转/倾斜/基线抖动/字距扰动，
    并用比印刷体更浅、带蓝调的"笔迹色"绘制。

这样字形本身可读，于是我们可以**断言 OCR 结果是否等于我们写入的原文**，
同时也让配准和差分面对真实的亚像素级扰动。能同时验证链路和结果正确性。

模拟的扫描退化包括：小角度旋转、透视形变、光照梯度、高斯噪声、
轻微模糊、JPEG 压缩。

注意：模拟数据只能验证代码链路和"识别是否等于注入的原文"，
不能替代真实学生作业上的准确率评估。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from pathlib import Path

import fitz
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

A4_PT = (595.0, 842.0)

_FONT_CANDIDATES_WIN = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
]
_FONT_CANDIDATES_MAC = [
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Medium.ttc",
    "/Library/Fonts/Arial Unicode.ttf",
]
_FONT_CANDIDATES_LINUX = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]


def _find_font() -> str | None:
    for c in _FONT_CANDIDATES_WIN + _FONT_CANDIDATES_MAC + _FONT_CANDIDATES_LINUX:
        if Path(c).exists():
            return c
    return None


@dataclass
class QuestionSpec:
    qid: str
    prompt: str
    # 答案区域（模板像素坐标）
    area: tuple[int, int, int, int]
    ground_truth: str
    # 区域内印刷的干扰内容（拼音/题目文字），会与手写重叠，用来测模板相减
    printed_noise: list[tuple[str, int, int]] = None  # type: ignore[assignment]
    line_height: int = 0
    font_size: int = 0
    is_box: bool = False


class DemoBuilder:
    def __init__(self, dpi: int = 200, seed: int = 7):
        self.dpi = dpi
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.font_path = _find_font()
        self.width = int(A4_PT[0] / 72 * dpi)
        self.height = int(A4_PT[1] / 72 * dpi)
        self.scale = dpi / 300.0  # 以 300 DPI 为基准的版面尺寸

    def s(self, v: float) -> int:
        return int(round(v * self.scale))

    def font(self, size: int) -> ImageFont.FreeTypeFont:
        size = max(10, int(size))
        if self.font_path:
            try:
                return ImageFont.truetype(self.font_path, size)
            except Exception:
                pass
        return ImageFont.load_default()

    # ---------------- 模板页 ----------------
    def build_template_page(self) -> tuple[Image.Image, list[QuestionSpec]]:
        img = Image.new("RGB", (self.width, self.height), (255, 255, 255))
        d = ImageDraw.Draw(img)

        margin = self.s(60)
        y = margin

        title_font = self.font(self.s(44))
        d.text((margin, y), "三年级语文 第一单元测试", font=title_font, fill=(20, 20, 20))
        y += self.s(70)
        d.line([(margin, y), (self.width - margin, y)], fill=(120, 120, 120), width=2)
        y += self.s(30)

        questions: list[QuestionSpec] = []

        # --- Q1 看拼音写词语（答案区与印刷拼音重叠）---
        body = self.font(self.s(30))
        d.text((margin, y), "1. 看拼音，写词语。", font=body, fill=(20, 20, 20))
        y += self.s(55)

        area_top = y
        area_bottom = y + self.s(120)
        pinyin = ["chūn fēng", "wēn nuǎn", "liǔ shù", "yáng guāng"]
        pinyin_font = self.font(self.s(20))
        slot_w = self.s(300)
        px = margin
        for i, p in enumerate(pinyin):
            d.text((px, y + self.s(4)), p, font=pinyin_font, fill=(70, 70, 90))
            d.line([(px, area_bottom), (px + slot_w - self.s(20), area_bottom)], fill=(150, 150, 150), width=1)
            px += slot_w
        questions.append(
            QuestionSpec(
                qid="1",
                prompt="看拼音写词语",
                area=(margin, area_top, self.width - margin, area_bottom),
                ground_truth="春风温暖柳树阳光",
                printed_noise=[(p, margin + i * self.s(300), y + self.s(4)) for i, p in enumerate(pinyin)],
                line_height=self.s(64),
                font_size=self.s(48),
            )
        )
        y = area_bottom + self.s(45)

        # --- Q2 阅读理解（多行短答）---
        d.text((margin, y), "2. 读短文，回答问题。", font=body, fill=(20, 20, 20))
        y += self.s(45)
        d.text((margin, y), "小明每天坚持读书，成绩进步很快。", font=body, fill=(20, 20, 20))
        y += self.s(50)

        area_top = y
        area_bottom = y + self.s(190)
        for i in range(3):
            ly = area_top + self.s(20) + i * self.s(60)
            d.line([(margin, ly), (self.width - margin, ly)], fill=(170, 170, 170), width=1)
        questions.append(
            QuestionSpec(
                qid="2",
                prompt="阅读理解",
                area=(margin, area_top, self.width - margin, area_bottom),
                # 故意写长到会自然折行 —— 这样才能验证
                # 「同一区域多行用换行、不同区域用空行」的分隔逻辑
                ground_truth=("因为小明每天坚持读书读的书越来越多所以他懂得的道理也越来越多"
                              "成绩进步很快同学们都说他是我们班读书最多的人"
                              "我也觉得他写的作文比去年进步了不少"),
                line_height=self.s(60),
                font_size=self.s(40),
            )
        )
        y = area_bottom + self.s(45)

        # --- Q3 看图写话（大作文框，多行）---
        d.text((margin, y), "3. 看图写话。", font=body, fill=(20, 20, 20))
        y += self.s(45)
        area_top = y
        area_bottom = y + self.s(260)
        d.rectangle([margin, area_top, self.width - margin, area_bottom], outline=(150, 150, 150), width=2)
        for i in range(1, 5):
            ly = area_top + i * self.s(52)
            d.line([(margin, ly), (self.width - margin, ly)], fill=(210, 210, 210), width=1)
        questions.append(
            QuestionSpec(
                qid="3",
                prompt="看图写话",
                area=(margin, area_top, self.width - margin, area_bottom),
                # 两段：第一段和第二段之间会有明显的行间空隙，用来验证空行分隔
                ground_truth=("春天来了公园里的花开了红的黄的百花争着开放香味飘得很远"
                              "小朋友们在草地上放风筝有的跑有的追笑得很开心"
                              "太阳出来了照得人暖洋洋的我和好朋友坐在树下吃点心"),
                line_height=self.s(52),
                font_size=self.s(38),
                is_box=True,
            )
        )

        d.text((margin, self.height - self.s(70)),
               f"班级__________  姓名__________  得分__________",
               font=self.font(self.s(26)), fill=(20, 20, 20))
        return img, questions

    # ---------------- 模拟手写 ----------------
    def draw_handwriting(self, base: Image.Image, spec: QuestionSpec) -> tuple[Image.Image, str]:
        """在答案区里画"手写"文字。

        返回 (新图, 实际写入的原文)。原文用于后续断言 OCR 结果。
        """
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        x1, y1, x2, y2 = spec.area
        font = self.font(spec.font_size)

        # 笔迹色：比印刷体浅、偏蓝，接近圆珠笔/铅笔
        ink = (35, 55, 110, 255)
        if spec.is_box:
            start_y = y1 + self.s(18)
            max_y = y2 - self.s(20)
        else:
            # 写在横线上方
            start_y = y2 - self.s(28)
            max_y = y2

        x = x1 + self.s(14)
        line_y = start_y
        written: list[str] = []

        for ch in spec.ground_truth:
            if x + spec.font_size > x2 - self.s(10):
                x = x1 + self.s(14)
                line_y += spec.line_height
                if line_y > max_y:
                    break

            # 每个字做随机扰动：旋转、垂直抖动、水平压缩
            jitter_x = self.rng.uniform(-2, 2) * self.scale
            jitter_y = self.rng.uniform(-3, 3) * self.scale
            angle = self.rng.uniform(-7, 7)
            squeeze = self.rng.uniform(0.90, 1.08)
            size = int(spec.font_size * self.rng.uniform(0.88, 1.05))

            glyph = Image.new("RGBA", (int(spec.font_size * 1.9), int(spec.font_size * 1.9)), (0, 0, 0, 0))
            gd = ImageDraw.Draw(glyph)
            gf = self.font(size)
            gd.text((glyph.width * 0.2, glyph.height * 0.2), ch, font=gf, fill=ink)
            glyph = glyph.rotate(angle, resample=Image.BICUBIC, expand=False)
            gw = max(1, int(glyph.width * squeeze))
            glyph = glyph.resize((gw, glyph.height), Image.BICUBIC)

            ox = int(x + jitter_x)
            oy = int(line_y + jitter_y)
            overlay.alpha_composite(glyph, (ox, oy))
            written.append(ch)
            x += size * self.rng.uniform(1.02, 1.22)

        out = base.convert("RGBA")
        out.alpha_composite(overlay)
        return out.convert("RGB"), "".join(written)

    # ---------------- 扫描退化 ----------------
    def degrade(self, img: Image.Image, rotate: float, perspective: float,
                brightness: float, gradient: float, noise: float, blur: float) -> Image.Image:
        arr = np.array(img.convert("RGB")).astype(np.float32)

        # 光照梯度（扫描仪/手机常见的不均匀光照）
        h, w = arr.shape[:2]
        ramp = np.linspace(1.0 - gradient, 1.0, w, dtype=np.float32)[None, :, None]
        arr = arr * ramp * brightness

        # 透视形变
        if abs(perspective) > 0.2:
            jitter = perspective * self.dpi / 300.0 * 18
            src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            dst = np.float32([
                [self.rng.uniform(-jitter, jitter), self.rng.uniform(-jitter, jitter)],
                [w + self.rng.uniform(-jitter, jitter), self.rng.uniform(-jitter, jitter)],
                [w + self.rng.uniform(-jitter, jitter), h + self.rng.uniform(-jitter, jitter)],
                [self.rng.uniform(-jitter, jitter), h + self.rng.uniform(-jitter, jitter)],
            ])
            matrix = _perspective_from(src, dst)
            arr = _warp(arr, matrix, (w, h))

        # 旋转
        if abs(rotate) > 0.05:
            arr = _rotate(arr, rotate, (w, h))

        # 噪声 + 模糊
        if noise > 0:
            arr = arr + self.np_rng.normal(0, noise * 255 * 0.05, arr.shape).astype(np.float32)
        arr = np.clip(arr, 0, 255).astype(np.uint8)

        out = Image.fromarray(arr)
        if blur > 0:
            out = out.filter(ImageFilter.GaussianBlur(radius=blur * self.dpi / 300.0 * 1.2))
        return out

    # ---------------- 输出 ----------------
    def save_pdf(self, img: Image.Image, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        doc = fitz.open()
        page = doc.new_page(width=A4_PT[0], height=A4_PT[1])
        tmp = path.parent / f"{path.stem}_tmp.png"
        img.save(tmp)
        page.insert_image(page.rect, filename=str(tmp))
        doc.save(str(path))
        doc.close()
        tmp.unlink(missing_ok=True)
        return path


def _perspective_from(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.getPerspectiveTransform(src, dst)


def _warp(arr: np.ndarray, matrix: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    import cv2

    return cv2.warpPerspective(arr, matrix, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def _rotate(arr: np.ndarray, angle: float, size: tuple[int, int]) -> np.ndarray:
    import cv2

    w, h = size
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(arr, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE,
                          borderValue=(255, 255, 255))


# 模拟的三种作答情况：全对、部分涂改、漏答
SCENARIOS = [
    {
        "name": "student_01_clean",
        "rotate": 1.2, "perspective": 0.35, "brightness": 0.98, "gradient": 0.10,
        "noise": 0.25, "blur": 0.15, "skip": set(), "scribble": set(),
        "desc": "字迹清晰、页面端正、无涂改",
    },
    {
        "name": "student_02_rough",
        "rotate": -2.6, "perspective": 0.75, "brightness": 1.03, "gradient": 0.22,
        "noise": 0.55, "blur": 0.35, "skip": {"3"}, "scribble": {"2"},
        "desc": "页面倾斜+光照不均，第3题漏答，第2题有涂改",
    },
    {
        "name": "student_03_upside_down",
        "rotate": 1.0, "perspective": 0.4, "brightness": 0.95, "gradient": 0.15,
        "noise": 0.35, "blur": 0.2, "skip": set(), "scribble": set(),
        "upside_down": True,
        "desc": "整页倒置扫描（考验方向探测）",
    },
]


def build_demo(out_dir: Path, dpi: int = 200, seed: int = 7) -> dict:
    """生成模板 PDF、template.json 和若干模拟学生扫描件。返回摘要信息。"""
    from .template import Question, Template, TemplatePage

    out_dir = Path(out_dir)
    b = DemoBuilder(dpi=dpi, seed=seed)
    if not b.font_path:
        raise RuntimeError(
            "系统里找不到中文字体，无法生成可读的自测数据。"
            "Windows 应有 C:/Windows/Fonts/msyh.ttc。"
        )

    tpl_img, specs = b.build_template_page()
    tpl_dir = out_dir / "template"
    tpl_dir.mkdir(parents=True, exist_ok=True)
    b.save_pdf(tpl_img, tpl_dir / "template.pdf")

    tpl = Template(
        template_id="demo_yuwen",
        dpi=dpi,
        pdf_path=str(tpl_dir / "template.pdf"),
        pages=[TemplatePage(
            page=1, width=b.width, height=b.height,
            questions=[Question(id=s.qid, answer_area=list(s.area),
                                type="text" if s.qid != "3" else "text",
                                expected_lines=3 if s.qid in {"2", "3"} else 1)
                       for s in specs],
        )],
        notes="make-demo 自动生成的自测模板",
    )
    tpl.save()

    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    truth: dict[str, str] = {}

    for sc in SCENARIOS:
        page = tpl_img.copy()
        written_all: list[str] = []
        for spec in specs:
            if spec.qid in sc["skip"]:
                continue
            page, written = b.draw_handwriting(page, spec)
            if spec.qid in sc["scribble"]:
                _draw_scribble(page, spec)
            written_all.append(written)
        truth[sc["name"]] = "|".join(written_all)

        img = b.degrade(page, sc["rotate"], sc["perspective"], sc["brightness"],
                        sc["gradient"], sc["noise"], sc["blur"])
        if sc.get("upside_down"):
            import cv2

            arr = cv2.rotate(np.array(img), cv2.ROTATE_180)
            img = Image.fromarray(arr)
        b.save_pdf(img, input_dir / f"{sc['name']}.pdf")

    truth_path = out_dir / "ground_truth.txt"
    truth_path.write_text(
        "\n".join(f"{k}\t{v}" for k, v in truth.items()), encoding="utf-8"
    )

    return {
        "模板": str(tpl_dir / "template.pdf"),
        "template.json": str(tpl_dir / "template.json"),
        "页面尺寸": f"{b.width}x{b.height} @{dpi} DPI",
        "题目数": len(specs),
        "学生扫描件": len(SCENARIOS),
        "标准答案": str(truth_path),
    }


def _draw_scribble(page: Image.Image, spec: QuestionSpec) -> None:
    """在答案区画一道涂改线，模拟学生划掉重写。"""
    import cv2

    d = ImageDraw.Draw(page)
    x1, y1, x2, y2 = spec.area
    y = y1 + int((y2 - y1) * 0.45)
    d.line([(x1 + b_pad(10), y), (x2 - b_pad(10), y + b_pad(6))], fill=(45, 60, 110), width=3)
    _ = cv2


def b_pad(v: int) -> int:
    return v
