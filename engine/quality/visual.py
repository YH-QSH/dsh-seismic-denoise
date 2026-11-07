"""去噪前后视觉/图像域质量指标（无外部视觉模型）。

把地震剖面当成灰度图，计算：
- **SSIM**：整体结构相似度，低了说明去噪把同相轴结构改坏了；
- **边缘相关**：Sobel 梯度图的 Pearson，衡量同相轴边缘/弱层位是否还在；
- **弱信号能量比**：输入局部弱能量区里，输出还保留多少能量；
- **综合视觉分**：上述三项加权，0–1。

这些指标是"看图"的可计算代理，不替代人工/多模态复核，但能自动挡住
"能量掉一半、弱波被抹掉"的过度去噪。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_filter, sobel, uniform_filter


def _semblance(a: np.ndarray, wt: int, wx: int) -> np.ndarray:
    """局部横向相干性（0–1）：弱区里只有相干同相轴才算“有效弱信号”。"""
    a = np.asarray(a, dtype=np.float64)
    size = (max(1, int(wt)), max(1, int(wx)))
    mean = uniform_filter(a, size=size, mode="nearest")
    mean_sq = uniform_filter(a * a, size=size, mode="nearest")
    return np.clip((mean * mean) / np.maximum(mean_sq, 1e-30), 0.0, 1.0)


def _finite(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def _norm_image(a: np.ndarray) -> np.ndarray:
    """按 99 分位归一化到 0–1，避免单个强道把整张图压黑。"""
    v = np.abs(a)
    scale = float(np.percentile(v, 99.0)) if v.size else 0.0
    if not math.isfinite(scale) or scale <= 0:
        scale = float(np.max(v)) or 1.0
    return np.clip(v / scale, 0.0, 1.0)


def _downsample(a: np.ndarray, max_side: int = 480) -> tuple[np.ndarray, int]:
    ns, ntr = a.shape
    step = max(1, int(math.ceil(max(ns, ntr) / max_side)))
    return a[::step, ::step], step


def _ssim(a: np.ndarray, b: np.ndarray) -> float:
    """简化 SSIM（高斯窗，常量 0.01/0.03）。"""
    if a.shape != b.shape or a.size == 0:
        return 0.0
    try:
        sigma = 1.5
        mu_a = gaussian_filter(a, sigma=sigma, mode="nearest")
        mu_b = gaussian_filter(b, sigma=sigma, mode="nearest")
        va = gaussian_filter(a * a, sigma=sigma, mode="nearest") - mu_a * mu_a
        vb = gaussian_filter(b * b, sigma=sigma, mode="nearest") - mu_b * mu_b
        cov = gaussian_filter(a * b, sigma=sigma, mode="nearest") - mu_a * mu_b
        c1, c2 = 0.01 ** 2, 0.03 ** 2
        num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
        den = (mu_a * mu_a + mu_b * mu_b + c1) * (va + vb + c2)
        return float(np.mean(num / np.maximum(den, 1e-12)))
    except Exception:
        return 0.0


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.size < 4:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den <= 1e-30:
        return 0.0
    return float(np.clip(np.sum(a * b) / den, -1.0, 1.0))


def visual_qc(noisy, denoised, dt: float | None = None) -> dict:
    """计算视觉代理指标；返回可直接塞进 JSON 的 dict。"""
    x = _finite(noisy)
    y = _finite(denoised)
    if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 4:
        return {"available": False, "reason": "形状不匹配或数据太短"}

    # 弱区用输入局部包络的低分位定义；和 fidelity 后处理保持同一口径。
    try:
        dt_v = float(dt) if dt is not None else 0.001
    except (TypeError, ValueError):
        dt_v = 0.001
    if not math.isfinite(dt_v) or dt_v <= 0:
        dt_v = 0.001
    wt = max(3, int(round(0.064 / dt_v)))
    if wt % 2 == 0:
        wt += 1
    wx = max(3, min(5, x.shape[1] if x.shape[1] % 2 else x.shape[1] - 1))
    env = np.sqrt(np.maximum(
        uniform_filter(x * x, size=(wt, wx), mode="nearest"), 0.0
    ))
    coh = _semblance(x, wt, wx)
    # 弱区必须相对化：绝对相干阈值在 VSP 上会把所有点都排除。
    # 先在 valid 样点里取低能量 40%，再取其中相干最高的 75% 分位，
    # 保证至少能选出一批“弱而可能相干”的样点；真正纯噪声的相干值低，会被分位挡掉。
    valid = env > float(np.max(env, initial=0.0)) * 1e-6
    if bool(np.any(valid)):
        weak = valid & (env <= float(np.percentile(env[valid], 40.0)))
        if bool(np.any(weak)):
            coh_thr = float(np.percentile(coh[weak], 75.0))
            weak = weak & (coh >= coh_thr)
    else:
        weak = np.zeros_like(env, dtype=bool)
    e_in = float(np.mean(x[weak] ** 2)) if bool(np.any(weak)) else 0.0
    e_out = float(np.mean(y[weak] ** 2)) if bool(np.any(weak)) else 0.0
    weak_fraction = float(np.mean(weak))
    # 像素太少时比值没有统计意义（例如只有几道弱反射），直接不参与门禁。
    weak_ratio = (float(np.sqrt(e_out / e_in))
                  if e_in > 1e-30 and weak_fraction >= 5e-4 else None)

    a = _norm_image(x)
    b = _norm_image(y)
    a_s, step = _downsample(a, max_side=480)
    b_s, _ = _downsample(b, max_side=480)
    ssim = _ssim(a_s, b_s)
    ga = np.hypot(sobel(a_s, axis=0), sobel(a_s, axis=1))
    gb = np.hypot(sobel(b_s, axis=0), sobel(b_s, axis=1))
    edge = _pearson(ga, gb)
    weak_score = 0.5 if weak_ratio is None else float(min(1.0, weak_ratio))
    score = 0.45 * max(0.0, ssim) + 0.25 * max(0.0, edge) + 0.30 * weak_score

    return {
        "available": True,
        "ssim": round(float(ssim), 4),
        "edge_corr": round(float(edge), 4),
        "weak_signal_ratio": None if weak_ratio is None else round(float(weak_ratio), 4),
        "visual_score": round(float(np.clip(score, 0.0, 1.0)), 4),
        "weak_fraction": round(float(weak_fraction), 6),
        "downsample_step": int(step),
        "notes": ["SSIM/边缘为图像域代理；弱信号比为输入“弱且相干”区能量保留"],
    }
