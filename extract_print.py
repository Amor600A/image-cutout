# -*- coding: utf-8 -*-
"""跨境 POD 印花提取 v1：确定性 CV 管线（不使用生成式模型）

流程：预处理 -> 局部背景建模 -> 油墨概率 -> 粗蒙版与封闭孔洞填充
      -> trimap -> alpha 精修 -> 色彩解污染 -> alpha 去杂点 -> 非生成式放大输出
"""

import argparse
import os
import time

import cv2
import numpy as np
from scipy import ndimage


def ellipse(radius):
    """生成椭圆结构元，半径为像素数"""
    size = 2 * radius + 1
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def read_image(path):
    """读取图像为 BGR float32，支持中文路径"""
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("无法读取图像: %s" % path)
    return img.astype(np.float32) / 255.0


def write_image(path, bgr):
    """写出 BGR 图像，支持中文路径"""
    ok, buf = cv2.imencode(".png", bgr)
    if not ok:
        raise IOError("图像编码失败: %s" % path)
    buf.tofile(path)


def build_terms(xn, yn, degree):
    """构造二维多项式基，xn/yn 已归一化到 [-1, 1]"""
    cols = []
    for total in range(degree + 1):
        for j in range(total + 1):
            i = total - j
            cols.append((xn ** i) * (yn ** j))
    return np.stack(cols, axis=1)


def robust_fit(values, terms, mask, iters=3, keep=0.05, max_samples=60000):
    """在 mask 内做鲁棒最小二乘拟合，抑制原本被误当背景的油墨像素"""
    rng = np.random.default_rng(0)
    weights = mask
    coef = None
    for _ in range(iters):
        idx = np.flatnonzero(weights)
        if idx.size > max_samples:
            idx = rng.choice(idx, max_samples, replace=False)
        coef, _, _, _ = np.linalg.lstsq(terms[idx], values[idx], rcond=None)
        resid = values - terms @ coef
        weights = mask & (np.abs(resid) < keep)
    return coef


def estimate_background(img):
    """用低频多项式拟合织物背景，覆盖褶皱造成的亮度不均"""
    h, w = img.shape[:2]
    lum = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    mx = img.max(axis=2)
    mn = img.min(axis=2)
    sat = (mx - mn) / np.maximum(mx, 1e-6)
    # 初始背景候选：亮且几乎无彩色，避免把油墨当背景
    bg_like = (lum > 0.78) & (sat < 0.10)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    xn = xs / max(w - 1, 1) * 2.0 - 1.0
    yn = ys / max(h - 1, 1) * 2.0 - 1.0
    terms = build_terms(xn.ravel(), yn.ravel(), 5)

    # 先用亮度拟合确定可信背景像素，再分通道拟合颜色
    coef_l = robust_fit(lum.ravel(), terms, bg_like.ravel())
    resid_l = lum.ravel() - terms @ coef_l
    trusted = bg_like.ravel() & (np.abs(resid_l) < 0.06)

    bg = np.empty_like(img)
    for c in range(3):
        coef = robust_fit(img[:, :, c].ravel(), terms, trusted)
        bg[:, :, c] = (terms @ coef).reshape(h, w)

    return np.clip(bg, 0.2, 1.0), trusted.reshape(h, w)


def ink_score(img, bg, sigma=1.0):
    """用相对背景的亮度偏离与色度偏离，计算油墨前景概率"""
    ratio = img / bg
    dark = np.clip(1.0 - ratio, 0.0, 1.0).max(axis=2)
    chroma = np.clip(ratio.max(axis=2) - ratio.min(axis=2), 0.0, 1.0)
    score = np.maximum(dark, chroma * 0.6)
    if sigma > 0:
        # 平滑让半调网点连成面，避免把每个网点当成独立杂点
        score = cv2.GaussianBlur(score, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return score


def remove_small_regions(mask, min_area):
    """去掉面积小于阈值的独立连通域"""
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, lab, np.arange(1, n + 1))
    keep = np.zeros(n + 1, dtype=bool)
    keep[1:] = sizes >= min_area
    return keep[lab]


def fill_enclosed(fg, max_hole_area):
    """填充被油墨轮廓完整包住的孔洞（白色字、白描边等白墨区域）"""
    bg = ~fg
    cross = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)
    lab, n = ndimage.label(bg, structure=cross)
    if n == 0:
        return fg, np.zeros_like(fg)
    border_ids = np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]]))
    outside = np.isin(lab, border_ids)

    holes = bg & ~outside
    hole_lab, hn = ndimage.label(holes, structure=cross)
    holes_keep = np.zeros_like(holes)
    if hn > 0:
        sizes = ndimage.sum(holes, hole_lab, np.arange(1, hn + 1))
        keep_ids = np.flatnonzero(sizes <= max_hole_area) + 1
        if keep_ids.size:
            holes_keep = np.isin(hole_lab, keep_ids)
    return fg | holes_keep, holes_keep


def guided_filter(guide, src, radius, eps):
    """彩色引导滤波，用于把粗蒙版对齐到真实颜色边缘"""
    ksize = (2 * radius + 1, 2 * radius + 1)
    mean_i = cv2.boxFilter(guide, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_p = cv2.boxFilter(src, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ip = cv2.boxFilter(guide * src[..., None], cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    corr_ii = cv2.boxFilter(guide * guide, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)

    cov_ip = corr_ip - mean_i * mean_p[..., None]
    var_i = corr_ii - mean_i * mean_i

    a = cov_ip / (var_i + eps)
    b = mean_p - (a * mean_i).sum(axis=2)

    mean_a = cv2.boxFilter(a, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    mean_b = cv2.boxFilter(b, cv2.CV_32F, ksize, normalize=True, borderType=cv2.BORDER_REFLECT)
    out = (mean_a * guide).sum(axis=2) + mean_b
    return np.clip(out, 0.0, 1.0)


def decontaminate(img, bg, alpha, alpha_min=0.18):
    """反解前景色，消除边缘混入的织物白色"""
    a = np.maximum(alpha, 1e-3)
    fg = (img - (1.0 - a[..., None]) * bg) / a[..., None]
    fg = np.clip(fg, 0.0, 1.0)

    low = alpha < alpha_min
    if low.any():
        idx = ndimage.distance_transform_edt(low, return_distances=False, return_indices=True)
        fg[low] = fg[idx[0][low], idx[1][low]]
    return fg


def clean_alpha(alpha, open_min, hole_max):
    """只在 alpha 域清理杂点与针孔，不触碰 RGB"""
    solid = alpha > 0.02
    solid = remove_small_regions(solid, open_min)
    alpha = np.where(solid, alpha, 0.0)

    holes = ~(alpha > 0.98)
    lab, n = ndimage.label(holes)
    if n > 0:
        sizes = ndimage.sum(holes, lab, np.arange(1, n + 1))
        small = np.flatnonzero(sizes <= hole_max) + 1
        if small.size:
            alpha[np.isin(lab, small)] = 1.0
    return np.clip(alpha, 0.0, 1.0)


def crop_to_alpha(fg, alpha, pad):
    """按 alpha 包围盒裁剪"""
    h, w = alpha.shape
    mask = alpha > 0.01
    ys, xs = np.where(mask)
    x0 = max(int(xs.min()) - pad, 0)
    x1 = min(int(xs.max()) + pad + 1, w)
    y0 = max(int(ys.min()) - pad, 0)
    y1 = min(int(ys.max()) + pad + 1, h)
    return fg[y0:y1, x0:x1], alpha[y0:y1, x0:x1]


def resize_output(fg, alpha, long_side, sharpen):
    """非生成式放大：RGB 走 Lanczos，alpha 走 Cubic，分别处理后合成"""
    h, w = alpha.shape
    scale = long_side / max(h, w)
    nw, nh = int(round(w * scale)), int(round(h * scale))

    a_up = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_CUBIC)
    a_up = np.clip(a_up, 0.0, 1.0)

    prem = fg * alpha[..., None]
    prem_up = cv2.resize(prem, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    prem_up = np.clip(prem_up, 0.0, 1.0)

    fg_up = prem_up / np.maximum(a_up, 1e-3)[..., None]
    fg_up = np.clip(fg_up, 0.0, 1.0)

    low = a_up < 0.05
    if low.any():
        idx = ndimage.distance_transform_edt(low, return_distances=False, return_indices=True)
        fg_up[low] = fg_up[idx[0][low], idx[1][low]]

    # 收紧 alpha 软化带，减少半透明毛边
    a_up = np.clip((a_up - 0.10) / 0.80, 0.0, 1.0)
    a_up = cv2.GaussianBlur(a_up, (0, 0), 0.6)

    if sharpen > 0:
        base = cv2.GaussianBlur(fg_up, (0, 0), 1.1)
        fg_up = np.clip(fg_up + sharpen * (fg_up - base), 0.0, 1.0)

    return fg_up, a_up


def save_debug(debug_dir, bg, score, mask, trimap, alpha, fg):
    """输出中间结果便于人工确认"""
    os.makedirs(debug_dir, exist_ok=True)
    write_image(os.path.join(debug_dir, "01_background.png"),
                np.clip(bg * 255.0, 0, 255).astype(np.uint8))
    write_image(os.path.join(debug_dir, "02_score.png"),
                np.clip(score * 255.0, 0, 255).astype(np.uint8))
    write_image(os.path.join(debug_dir, "03_mask.png"),
                (mask.astype(np.uint8) * 255)[..., None].repeat(3, axis=2))
    write_image(os.path.join(debug_dir, "04_trimap.png"), trimap)
    write_image(os.path.join(debug_dir, "05_alpha.png"),
                np.clip(alpha * 255.0, 0, 255).astype(np.uint8)[..., None].repeat(3, axis=2))

    dark = np.full_like(fg, 0.18)
    comp_dark = fg * alpha[..., None] + dark * (1.0 - alpha[..., None])
    write_image(os.path.join(debug_dir, "06_check_dark.png"),
                (np.clip(comp_dark, 0, 1) * 255).astype(np.uint8))

    mag = np.zeros_like(fg)
    mag[:, :, 0] = 1.0
    mag[:, :, 2] = 1.0
    comp_mag = fg * alpha[..., None] + mag * (1.0 - alpha[..., None])
    write_image(os.path.join(debug_dir, "07_check_magenta.png"),
                (np.clip(comp_mag, 0, 1) * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description="POD 印花提取（确定性管线）")
    parser.add_argument("input", help="输入实拍图")
    parser.add_argument("-o", "--output", default=os.path.join("output", "002_out.png"), help="输出 RGBA PNG")
    parser.add_argument("--debug-dir", default="debug", help="中间结果目录")
    parser.add_argument("--long-side", type=int, default=1024, help="输出长边像素")
    parser.add_argument("--score-sigma", type=float, default=1.0)
    parser.add_argument("--thresh-lo", type=float, default=0.04)
    parser.add_argument("--thresh-hi", type=float, default=0.15)
    parser.add_argument("--band-width", type=int, default=2, help="未知带半径")
    parser.add_argument("--min-area", type=int, default=8)
    parser.add_argument("--max-hole-ratio", type=float, default=0.25, help="封闭孔洞填充面积上限占全图比例")
    parser.add_argument("--speck-min", type=int, default=12)
    parser.add_argument("--pinhole-max", type=int, default=8)
    parser.add_argument("--sharpen", type=float, default=0.3)
    parser.add_argument("--pad", type=int, default=8)
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    t0 = time.time()
    img = read_image(args.input)
    h, w = img.shape[:2]
    print("[1/8] input %dx%d" % (w, h))

    bg, trusted = estimate_background(img)
    print("[2/8] background estimated, trusted bg ratio %.3f" % trusted.mean())

    score = ink_score(img, bg, sigma=args.score_sigma)
    score_raw = ink_score(img, bg, sigma=0)
    q = np.percentile(score_raw, [50, 70, 80, 90, 95, 99])
    print("[3/8] score percentiles " + " ".join("%.3f" % v for v in q))

    fg = score > args.thresh_hi
    fg = cv2.morphologyEx(fg.astype(np.uint8), cv2.MORPH_CLOSE, ellipse(1)).astype(bool)
    fg = remove_small_regions(fg, args.min_area)
    print("[4/8] coarse mask ratio %.3f" % fg.mean())

    max_hole_area = int(args.max_hole_ratio * h * w)
    filled, holes = fill_enclosed(fg, max_hole_area)
    print("[5/8] holes filled ratio %.4f" % holes.mean())

    known_fg = cv2.erode(filled.astype(np.uint8), ellipse(1)).astype(bool)
    known_bg = ~cv2.dilate(filled.astype(np.uint8), ellipse(2)).astype(bool)
    hole_core = cv2.erode(holes.astype(np.uint8), ellipse(1)).astype(bool)
    known_fg = known_fg | hole_core
    known_bg = known_bg & ~hole_core

    trimap = np.full((h, w, 3), 128, dtype=np.uint8)
    trimap[known_fg] = (255, 255, 255)
    trimap[known_bg] = (0, 0, 0)

    # 只在粗蒙版边界内外的一条窄带内用原始色差求解 alpha，避免实心区域被蒙版撑大
    fg_u8 = filled.astype(np.uint8)
    inner = cv2.erode(fg_u8, ellipse(args.band_width)).astype(bool)
    outer = cv2.dilate(fg_u8, ellipse(args.band_width)).astype(bool)
    band = outer & ~inner

    # 用最近的高置信油墨色做反投影求 alpha：alpha 与观测颜色自洽，避免白边
    interior = cv2.erode(fg_u8, ellipse(2)).astype(bool) | hole_core
    idx = ndimage.distance_transform_edt(~interior, return_distances=False, return_indices=True)
    ink_ref = img[idx[0], idx[1]]
    vb = ink_ref - bg
    di = img - bg
    denom = (vb * vb).sum(axis=2) + 1e-4
    proj = np.clip((di * vb).sum(axis=2) / denom, 0.0, 1.0)
    # 白墨落在白布上时反投影不可解，直接沿用粗蒙版结果
    weak = (vb * vb).sum(axis=2) < 0.004
    proj[weak] = filled[weak]

    alpha = np.where(band, proj, filled.astype(np.float32))
    alpha = np.where(band, guided_filter(img, alpha, 1, 1e-5), alpha)
    alpha[hole_core] = 1.0
    alpha[~outer] = 0.0
    print("[6/8] alpha solved")

    alpha = clean_alpha(alpha, args.speck_min, args.pinhole_max)
    # 边界与低 alpha 像素直接取最近高置信油墨色，切断织物白色混入，根除白点
    fg_color = img.copy()
    swap = band | (alpha < 0.02)
    fg_color[swap] = ink_ref[swap]

    save_debug(args.debug_dir, bg, score, filled, trimap, alpha, fg_color)

    fg_c, alpha_c = crop_to_alpha(fg_color, alpha, args.pad)
    fg_up, a_up = resize_output(fg_c, alpha_c, args.long_side, args.sharpen)

    out = np.concatenate([fg_up, a_up[..., None]], axis=2)
    out_u8 = (np.clip(out, 0, 1) * 255).astype(np.uint8)
    write_image(args.output, out_u8)

    semis = ((alpha_c > 0.05) & (alpha_c < 0.95)).mean()
    print("[7/8] alpha semi-transparent ratio %.4f" % semis)
    print("[8/8] saved %s %dx%d, %.1fs" % (
        args.output, a_up.shape[1], a_up.shape[0], time.time() - t0))


if __name__ == "__main__":
    main()
