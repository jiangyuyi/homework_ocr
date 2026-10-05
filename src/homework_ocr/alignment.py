"""页面配准。

技术方案第五~八节。实现要点：

* ORB + BFMatcher + Lowe ratio + RANSAC findHomography
* ECC 二次精配准（欧氏运动模型），消掉 1~3px 残差——这一步直接决定差分
  里有没有"文字双影"
* 方向探测：学生整页旋转 90/180/270 度时，先用低成本匹配挑出正确方向
* 单位阵短路：变换本来就接近恒等时不做 warp，省时间也避免插值模糊
* 严格的成功判据：匹配数、内点数、内点率三项都不达标就判失败，
  宁可输出 alignment_failed，也不要把无意义的差分结果当成答案
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import AlignmentConfig

log = logging.getLogger(__name__)


@dataclass
class AlignResult:
    ok: bool
    score: float
    aligned: np.ndarray | None = None
    homography: np.ndarray | None = None
    good_matches: int = 0
    inliers: int = 0
    inlier_ratio: float = 0.0
    rotation_applied: int = 0
    method: str = "none"
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _match_count(gray_a: np.ndarray, gray_b: np.ndarray, n_features: int, ratio: float) -> int:
    """只数匹配点，不做配准。用于方向探测这种廉价判别。"""
    orb = cv2.ORB_create(nfeatures=n_features)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)
    if des_a is None or des_b is None or len(kp_a) < 8 or len(kp_b) < 8:
        return 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_a, des_b, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < ratio * n.distance]
    return len(good)


def _is_identity(h: np.ndarray, tolerance: float) -> bool:
    """变换是否接近恒等。用角点位移判断，比逐元素比矩阵稳。"""
    h = np.asarray(h, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(h)):
        return False
    try:
        h_inv = np.linalg.inv(h)
    except np.linalg.LinAlgError:
        return False
    corners = np.array([[0, 0], [1000, 0], [1000, 1400], [0, 1400]], dtype=np.float64).reshape(-1, 1, 2)
    src = cv2.perspectiveTransform(corners, h)
    dst = cv2.perspectiveTransform(corners, h_inv)
    if not (np.all(np.isfinite(src)) and np.all(np.isfinite(dst))):
        return False
    return float(np.max(np.abs(src - dst))) <= tolerance


def detect_rotation(student_gray: np.ndarray, template_gray: np.ndarray, cfg: AlignmentConfig) -> int:
    """返回应施加于学生页的顺时针旋转角（0/90/180/270）。"""
    if not cfg.try_rotations:
        return 0

    best_angle, best_count = 0, -1
    # 学生页与模板的长宽比先做一次便宜的判断，避免无谓的 4 次 ORB。
    s_landscape = student_gray.shape[1] > student_gray.shape[0]
    t_landscape = template_gray.shape[1] > template_gray.shape[0]
    candidates = [0, 90, 180, 270]
    if s_landscape != t_landscape:
        candidates = [90, 270] + [0, 180]

    for angle in candidates:
        if angle == 0:
            probe = student_gray
        else:
            probe = cv2.rotate(student_gray, {90: cv2.ROTATE_90_CLOCKWISE,
                                              180: cv2.ROTATE_180,
                                              270: cv2.ROTATE_90_COUNTERCLOCKWISE}[angle])
        if probe.shape != template_gray.shape:
            probe = cv2.resize(probe, (template_gray.shape[1], template_gray.shape[0]),
                               interpolation=cv2.INTER_AREA)
        count = _match_count(probe, template_gray, cfg.probe_features, cfg.ratio_test)
        if count > best_count:
            best_angle, best_count = angle, count
        if angle == 0 and count >= cfg.probe_min_matches * 4:
            break  # 原始方向已经非常明确，不再试

    if best_count < cfg.probe_min_matches and best_angle != 0:
        return 0  # 都没匹配上，说明不是单纯旋转问题，交给主流程报错
    if best_angle:
        log.info("检测到页面方向偏差，施加旋转 %d°（探测匹配点 %d）", best_angle, best_count)
    return best_angle


def _feature_scale(shape: tuple[int, int], max_side: int) -> tuple[float, float]:
    """算特征检测用的缩放系数。

    配准要的是几何关系，不是分辨率。在 35 MPix 上跑 ORB 实测要 91 秒/页，
    而 8.7 MPix 只要 22 秒——特征检测的耗时几乎完全跟着像素数走。
    所以先缩到固定尺寸做检测和匹配，再把变换换算回原图。
    之后配准耗时就与 DPI 解耦了。
    """
    h, w = shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return 1.0, 1.0
    k = max_side / float(longest)
    return k, k


def _rescale_homography(h_small: np.ndarray, sx: float, sy: float) -> np.ndarray:
    """把缩放图上的单应矩阵换算回原图：H_full = A⁻¹ · H_small · A。"""
    a = np.array([[sx, 0.0, 0.0], [0.0, sy, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    a_inv = np.linalg.inv(a)
    return a_inv @ h_small.astype(np.float64) @ a


def _orb(gray: np.ndarray, cfg: AlignmentConfig) -> tuple[tuple, object]:
    orb = cv2.ORB_create(nfeatures=cfg.orb_features)
    return orb.detectAndCompute(gray, None)


def align_page(student: np.ndarray, template: np.ndarray, cfg: AlignmentConfig) -> AlignResult:
    """把学生页配准到模板坐标系。

    返回的 aligned 图与 template 同尺寸，后续所有坐标都以此为准。
    """
    if not cfg.enabled:
        return AlignResult(ok=True, score=1.0, aligned=student, method="disabled",
                           reason="配准已在配置中关闭（仅适用于学生页与模板像素级一致的情况）")

    student_gray = to_gray(student)
    template_gray = to_gray(template)
    out_size = (template.shape[1], template.shape[0])

    rotation = detect_rotation(student_gray, template_gray, cfg)
    if rotation:
        student = cv2.rotate(student, {90: cv2.ROTATE_90_CLOCKWISE,
                                       180: cv2.ROTATE_180,
                                       270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
        student_gray = to_gray(student)

    # 尺寸不同先统一，否则 ORB 不在同一尺度上工作。
    if student.shape[:2] != template.shape[:2]:
        student = cv2.resize(student, out_size, interpolation=cv2.INTER_AREA)
        student_gray = cv2.resize(student_gray, out_size, interpolation=cv2.INTER_AREA)

    # 特征检测在缩放图上做，耗时与 DPI 解耦。
    sx, sy = _feature_scale(student_gray.shape, cfg.feature_max_side)
    if sx < 1.0:
        f_student = cv2.resize(student_gray, (int(student_gray.shape[1] * sx),
                                                int(student_gray.shape[0] * sy)),
                               interpolation=cv2.INTER_AREA)
        f_template = cv2.resize(template_gray, (int(template_gray.shape[1] * sx),
                                                 int(template_gray.shape[0] * sy)),
                                interpolation=cv2.INTER_AREA)
    else:
        f_student, f_template = student_gray, template_gray

    kp_student, des_student = _orb(f_student, cfg)
    kp_template, des_template = _orb(f_template, cfg)

    if des_student is None or des_template is None:
        return AlignResult(ok=False, score=0.0, rotation_applied=rotation,
                           reason="无法提取 ORB 特征（页面可能整体空白）")

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_student, des_template, k=2)
    good = [m for pair in knn if len(pair) == 2 for m, n in [pair] if m.distance < cfg.ratio_test * n.distance]

    if len(good) < 8:
        return AlignResult(ok=False, score=0.0, good_matches=len(good), rotation_applied=rotation,
                           reason=f"有效匹配点仅 {len(good)} 个，不足以估计透视变换")

    src = np.float32([kp_student[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_template[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    h_small, inlier_mask = cv2.findHomography(src, dst, cv2.RANSAC, cfg.ransac_threshold)
    if h_small is None or inlier_mask is None:
        return AlignResult(ok=False, score=0.0, good_matches=len(good), rotation_applied=rotation,
                           reason="RANSAC 未能求出单应矩阵（页面可能被遮挡或缺角）")

    homography = _rescale_homography(h_small, sx, sy) if sx < 1.0 else h_small

    inliers = int(inlier_mask.sum())
    inlier_ratio = inliers / max(1, len(good))
    score = round(inlier_ratio, 4)
    warnings: list[str] = []

    if len(good) < cfg.min_matches or inliers < cfg.min_inliers or inlier_ratio < cfg.min_inlier_ratio:
        return AlignResult(
            ok=False,
            score=score,
            homography=homography,
            good_matches=len(good),
            inliers=inliers,
            inlier_ratio=round(inlier_ratio, 4),
            rotation_applied=rotation,
            method="orb",
            reason=(
                f"配准未达标: 匹配 {len(good)}/<{cfg.min_matches}, "
                f"内点 {inliers}/<{cfg.min_inliers}, 内点率 {inlier_ratio:.2f}/<{cfg.min_inlier_ratio}"
            ),
        )

    method = "orb"
    identity_full = _is_identity(homography, cfg.identity_tolerance)
    if identity_full:
        aligned = student
        warnings.append("变换接近单位阵，已跳过 warp")
    else:
        aligned = cv2.warpPerspective(student, homography, out_size,
                                      flags=cv2.INTER_LINEAR,
                                      borderMode=cv2.BORDER_CONSTANT,
                                      borderValue=(255, 255, 255))

    if cfg.enable_ecc:
        # ECC 保持原语义：作用在「已配准的整页图」上，用 warpAffine 直接出结果。
        #
        # 试过把 ECC 也搬到缩放图上跑（为了省 600 DPI 的时间），结果更差：
        # 两个变换合成会引入误差，200 DPI 的回归从 8/8 掉到 4/8。
        # 所以这里只降采样 ORB 的特征检测，ECC 一个字都不改。
        # 代价是 600 DPI 的 ECC 仍然慢，但正确性优先。
        refined = _refine_ecc(aligned, template, cfg, warnings)
        if refined is not None:
            aligned = refined
            method = "orb+ecc"

    return AlignResult(
        ok=True,
        score=score,
        aligned=aligned,
        homography=homography,
        good_matches=len(good),
        inliers=inliers,
        inlier_ratio=round(inlier_ratio, 4),
        rotation_applied=rotation,
        method=method,
        warnings=warnings,
    )


def _refine_ecc(aligned: np.ndarray, template: np.ndarray, cfg: AlignmentConfig,
                warnings: list[str]) -> np.ndarray | None:
    """ECC 精配准，返回已精配好的整页图（与模板同尺寸）。

    用 MOTION_EUCLIDEAN 而不是 AFFINE：扫描件的残差主要是平移+微旋转，
    欧氏模型参数少、更稳；仿射自由度更高，反而容易把页面"拉"变形。
    ECC 对初值敏感，失败就保留 ORB 结果，不硬来。
    """
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, cfg.ecc_iterations, cfg.ecc_epsilon)
    try:
        _, warp = cv2.findTransformECC(
            to_gray(template), to_gray(aligned), warp, cv2.MOTION_EUCLIDEAN, criteria, None, 5
        )
    except cv2.error as exc:
        warnings.append(f"ECC 精配准跳过: {str(exc)[:80]}")
        return None
    if not np.all(np.isfinite(warp)):
        warnings.append("ECC 结果非有限值，已忽略")
        return None
    return cv2.warpAffine(
        aligned, warp, (template.shape[1], template.shape[0]),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255),
    )
