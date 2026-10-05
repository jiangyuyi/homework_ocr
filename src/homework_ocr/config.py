"""配置加载与校验。

所有可调参数集中在 config.yaml。代码里不允许出现散落的魔法数字阈值，
调参时只改 YAML，不动代码。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, get_type_hints

import yaml

DEFAULT_CONFIG_PATH = Path("config.yaml")


class ConfigError(ValueError):
    """配置文件缺字段或类型不对。"""


@dataclass
class RenderConfig:
    dpi: int = 300
    #: 渲染后统一转灰度。差分阶段只需要灰度，保留彩色只会让阈值更难调。
    grayscale: bool = True


@dataclass
class AlignmentConfig:
    enabled: bool = True
    #: ORB 特征点上限。300 DPI A4 页面上印刷文字足够提供数千稳定特征点。
    orb_features: int = 5000
    #: Lowe ratio test。越低越严格，误匹配越少但好匹配也越少。
    ratio_test: float = 0.75
    #: RANSAC 内点阈值（像素）。300 DPI 下 3.0 相当于 0.25mm。
    ransac_threshold: float = 3.0
    #: 低于这些内点数直接判为配准失败——差分结果在这种情况下没有意义。
    min_matches: int = 40
    min_inliers: int = 30
    min_inlier_ratio: float = 0.5
    #: 二次精配准。Homography 之后常有 1~3px 残差，会在差分里变成鬼影。
    enable_ecc: bool = True
    ecc_iterations: int = 100
    ecc_epsilon: float = 1e-5
    #: 学生把整页转了 90/180/270 度时，先做一次方向探测。
    try_rotations: bool = True
    #: 配准（ORB 特征检测 + ECC 精配准）前把页面缩到长边不超过这个像素数。
    #:
    #: 配准只需要几何关系，不需要全分辨率。实测 300 DPI A4(8.7MPix)：
    #:   全分辨率 23.0 秒 / 分数 0.926
    #:   缩到 2400px   10.8 秒 / 分数 0.943   ← 更快且更准
    #:   缩到 1600px    4.8 秒 / 分数 0.871   ← 再快但质量明显下降
    #: 缩到 2400px 之后，配准耗时与 DPI 基本解耦，600 DPI 也不会再卡到 90 秒。
    feature_max_side: int = 2400
    #: 变换接近单位阵时跳过 warp，既快又避免插值把页面弄糊。
    identity_tolerance: float = 2.0
    #: 方向探测阶段只数匹配点，不做完整配准。
    probe_features: int = 1200
    probe_min_matches: int = 12


@dataclass
class DifferenceConfig:
    #: template_subtract = student AND NOT template（业务语义正确，默认）
    #: absdiff        = |student - template|（仅用于对照排查）
    #: none           = 不做差分，直接对整页做 OCR
    method: str = "template_subtract"
    #: 大核形态学估计背景再做除法，消除阴影和不均匀光照。
    illumination: bool = True
    illumination_kernel: int = 51
    adaptive_block_size: int = 31
    adaptive_c: int = 15
    #: 模板抑制的膨胀半径（像素）。技术方案第十二节的问题：学生笔画压在
    #: 横线上时会被模板减掉。默认 1 像素，保守；宁可残留鬼影也不要吃掉笔画。
    template_dilate_px: int = 1


@dataclass
class RoiConfig:
    #: ROI 外扩。学生经常写到框线外面去。
    padding: int = 20
    #: 模板里没定义任何题目 ROI 时，是否退化为整页处理。
    allow_full_page: bool = True


@dataclass
class NoiseFilterConfig:
    min_component_area: int = 8
    #: 超过 ROI 面积这个比例的连通域视为阴影/订书钉/污渍，直接丢弃。
    max_component_area_ratio: float = 0.35
    open_kernel: int = 2
    close_kernel: int = 3
    #: 细长线条（作业本横线、方格线残留）按长宽比剔除。
    max_aspect_ratio: float = 40.0
    #: 墨迹/外接框面积的下限。低于它说明是个空心框（作文框、表格边框），
    #: 而不是一行字——框的周长很长但填充率极低。
    min_fill_ratio: float = 0.05


@dataclass
class GroupingConfig:
    #: 水平方向膨胀把同一行的字粘成一个区域。
    dilation_width: int = 20
    dilation_height: int = 5
    #: 行间距超过 中位行高 × 该系数 时切分成两行。
    line_gap_ratio: float = 0.9
    #: 单个区域高度超过 中位高度 × 该系数 时按水平投影再切。
    tall_region_ratio: float = 2.2
    #: 区域高度低于 中位行高 × 该系数 时视为碎屑/线框残片。
    min_region_height_ratio: float = 0.4


@dataclass
class OcrConfig:
    #: rapidocr | paddleocr
    engine: str = "rapidocr"
    det_ocr_version: str = "PP-OCRv5"
    rec_ocr_version: str = "PP-OCRv5"
    #: ch = 简中+英+日+拼音，正合语文作业。
    lang: str = "ch"
    #: server 精度更高，mobile 更快。作业识别优先准确率。
    model_type: str = "server"
    device: str = "cpu"
    #: 0 = 由 onnxruntime 自行决定。
    threads: int = 0
    #: 整页检测时的输入边长上限。ROI 通常很小，设大反而更准。
    det_limit_side_len: int = 1536
    #: True  = V1：引擎自己做检测+识别（稳定，默认）
    #: False = V2：我们已按行切好 region，只让引擎做识别（更快更稳）
    use_det: bool = True
    #: 识别批大小。CPU 上 4~8 通常比 1 快不少。
    rec_batch: int = 6
    #: OCR 检出的文本行必须与手写 mask 重叠到这个比例，否则丢弃。
    #: 这一步是模板差分的关键补充：mask 残影被过滤掉，真实笔迹被保留。
    min_box_overlap: float = 0.15
    #: 识别阶段的最小保留置信度。低于它的先丢，再进入 review 判定。
    min_det_score: float = 0.3
    #: 超大 ROI（作文）切块识别时的边长。
    max_rec_side: int = 3200

@dataclass
class ReviewConfig:
    min_ocr_confidence: float = 0.75
    min_alignment_score: float = 0.7
    #: 单个答案被切成过多碎片，通常意味着 mask 碎了或字迹太淡。
    max_fragments: int = 6
    #: 涂改检测：某一行的墨迹面积远大于同区域其他行时提示复核。
    heavy_correction_ratio: float = 3.0


@dataclass
class OfflineConfig:
    #: 运行时封禁出站 socket，把"不联云"从承诺变成可验证的事实。
    enforce: bool = True
    #: 放行的本机地址（ONNXRuntime 可能用到本地共享内存）。
    allow_loopback: bool = True


@dataclass
class OutputConfig:
    save_debug: bool = True
    save_crops: bool = True
    save_aligned: bool = True
    save_mask: bool = True
    formats: list[str] = field(default_factory=lambda: ["json", "csv", "txt", "xlsx"])
    #: 同一文字块内的多行之间用什么分隔。中文直接相连即可（不需要空格），
    #: 否则一句会被横格线断成好几行，读起来像被劈开的碎片。
    line_joiner: str = ""
    #: 不同文字块（段落/字段）之间用什么分隔。
    block_joiner: str = "\n"
    #: 判定「换块」的阈值：两块之间那段空白里，扣掉手写和横格线之后，
    #: 剩下的印刷墨量（折算成「每单位宽度的等效印刷行高」，px）超过它就换块。
    #: 实测：中间有印刷指令的空隙是 15.7 / 26.0，普通行距最大 0.34，46 倍余量。
    block_foreign_ink_px: float = 3.0
    #: 同一行内水平间距超过这个倍数（中位区域高度的倍数）也算换块。
    #: 用来把并排的独立字段分开，例如班级「五(1)」和姓名「洪逸欣」。
    block_hgap_ratio: float = 2.5
    #: debug overlay 的缩放比例，原图太大时看不清框。
    overlay_scale: float = 0.4
    jpeg_quality: int = 88


@dataclass
class RunConfig:
    #: 进程池大小。1 表示串行，调试时更好定位问题。
    workers: int = 1
    #: 批处理时遇到单页失败是否继续。
    continue_on_error: bool = True


@dataclass
class Config:
    render: RenderConfig = field(default_factory=RenderConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    difference: DifferenceConfig = field(default_factory=DifferenceConfig)
    roi: RoiConfig = field(default_factory=RoiConfig)
    noise_filter: NoiseFilterConfig = field(default_factory=NoiseFilterConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    ocr: OcrConfig = field(default_factory=OcrConfig)
    review: ReviewConfig = field(default_factory=ReviewConfig)
    offline: OfflineConfig = field(default_factory=OfflineConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def load(cls, path: str | Path | None = None, overrides: Mapping[str, Any] | None = None) -> "Config":
        data: dict[str, Any] = {}
        if path is not None:
            p = Path(path)
            if p.exists():
                loaded = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                if not isinstance(loaded, dict):
                    raise ConfigError(f"{p} 顶层必须是 mapping")
                data = loaded
        if overrides:
            data = _deep_merge(data, dict(overrides))
        return _build(cls, data, prefix="")

    def to_dict(self) -> dict[str, Any]:
        return _asdict(self)

    def scaled_for_dpi(self, dpi: int) -> "Config":
        """按 DPI 换算像素类阈值，返回新配置（不动原对象）。

        为什么需要：min_component_area=8、dilation_width=20 这类阈值是在
        300 DPI 下调的。直接拿去跑 600 DPI，字迹变成 2 倍大，这些阈值相对
        变小——线框残渣清不掉、字会被切碎。反过来 200 DPI 又太大。

        这里让配置文件里始终写「300 DPI 基准值」，程序按实际 DPI 自动换算，
        换扫描仪分辨率就不用重新调参了。

        只缩放**页面像素**类参数；OCR 的模型输入尺寸（det_limit_side_len）
        不缩——那是模型内部的事，跟页面分辨率无关。
        """
        base = 300.0
        k = float(dpi) / base
        if abs(k - 1.0) < 1e-6:
            return self

        out = copy.deepcopy(self)

        # 面积按 k²，其余按 k
        out.noise_filter.min_component_area = max(1, int(round(self.noise_filter.min_component_area * k * k)))
        out.noise_filter.open_kernel = _odd_scaled(self.noise_filter.open_kernel, k)
        out.noise_filter.close_kernel = _odd_scaled(self.noise_filter.close_kernel, k)

        out.grouping.dilation_width = max(1, int(round(self.grouping.dilation_width * k)))
        out.grouping.dilation_height = max(1, int(round(self.grouping.dilation_height * k)))

        out.roi.padding = max(1, int(round(self.roi.padding * k)))

        out.difference.illumination_kernel = _odd_scaled(self.difference.illumination_kernel, k)
        out.difference.adaptive_block_size = _odd_scaled(self.difference.adaptive_block_size, k)

        out.render.dpi = int(dpi)
        return out


def _odd_scaled(value: int, k: float, minimum: int = 3) -> int:
    """核尺寸必须保持奇数（OpenCV 要求），缩放后向上取到奇数。"""
    v = max(minimum, int(round(value * k)))
    return v if v % 2 == 1 else v + 1


def _deep_merge(base: dict[str, Any], extra: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def _resolved_hints(cls: type) -> dict[str, Any]:
    """把 dataclass 的字段注解解析成真实类型。

    本文件用了 `from __future__ import annotations`，所以 dataclasses.fields()
    拿到的 f.type 全是字符串。不解析的话，嵌套的配置段（OfflineConfig 等）
    会被当成普通值原样塞进 dict，之后访问 cfg.offline.enforce 就炸。
    """
    try:
        return get_type_hints(cls)
    except Exception:  # pragma: no cover - 注解里有解析不了的东西
        return {}


def _build(cls: type, data: Mapping[str, Any], prefix: str) -> Any:
    """按 dataclass 字段递归构造，缺字段用默认值，未知字段直接报错。"""
    if not is_dataclass(cls):
        raise ConfigError(f"{prefix or 'config'} 不是合法的配置段")

    hints = _resolved_hints(cls)
    kwargs: dict[str, Any] = {}
    known = {f.name: f for f in fields(cls)}
    unknown = set(data) - set(known)
    if unknown:
        where = prefix or "config"
        raise ConfigError(f"{where} 含有未知配置项: {sorted(unknown)}")

    for name, f in known.items():
        if name not in data:
            continue
        value = data[name]
        child_prefix = f"{prefix}.{name}" if prefix else name
        target = hints.get(name, f.type)
        if isinstance(target, type) and is_dataclass(target):
            if not isinstance(value, Mapping):
                raise ConfigError(f"{child_prefix} 必须是一个配置段（mapping）")
            kwargs[name] = _build(target, value, prefix=child_prefix)
        else:
            kwargs[name] = _coerce(value, f, child_prefix)
    return cls(**kwargs)


def _coerce(value: Any, f: Any, where: str) -> Any:
    """YAML 不会写错类型，但手写 override 可能写错，这里统一转一次。"""
    target = f.type
    # dataclass 字段的 type 可能是字符串（from __future__ import annotations）。
    if isinstance(target, str):
        target = {
            "int": int,
            "float": float,
            "bool": bool,
            "str": str,
            "list[str]": list,
        }.get(target.split(" | ")[0], None)
    try:
        if target is int and isinstance(value, bool):
            raise TypeError("bool 不能当 int")
        if target is float:
            return float(value)
        if target is int:
            return int(value)
        if target is bool:
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "on"}
            return bool(value)
        if target is str:
            return str(value)
        if target is list:
            if not isinstance(value, (list, tuple)):
                raise TypeError("需要列表")
            return list(value)
        return value
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{where} 取值无效: {value!r} ({exc})") from exc


def _asdict(obj: Any) -> Any:
    if is_dataclass(obj):
        return {f.name: _asdict(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [_asdict(v) for v in obj]
    return obj


DEFAULT_CONFIG = Config()
