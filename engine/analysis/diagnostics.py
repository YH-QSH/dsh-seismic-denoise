"""数据诊断：频谱、噪声类型打分、盲信噪比、相干性。

这里刻意不依赖质量评价模块，保证 formats/quality 并行开发时读取阶段也能跑；
若 quality 模块已就绪，来源匹配会优先用它更稳的盲 SNR。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d

from ..model import SeismicDataset


def _as_float(data: np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64)
    return np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)


def spectral_stats(data: np.ndarray, dt: float) -> dict:
    x = _as_float(data)
    ns = x.shape[0]
    if dt is None or dt <= 0 or ns < 16:
        return {"reliable": False, "bands": {}, "peak_hz": None, "line_score": 0.0}
    win = signal.windows.hann(ns, sym=False)[:, None]
    spec = np.fft.rfft((x - x.mean(axis=0, keepdims=True)) * win, axis=0)
    amp = np.abs(spec).mean(axis=1)
    power = amp ** 2
    f = np.fft.rfftfreq(ns, dt)
    smooth = uniform_filter1d(amp, max(1, min(31, len(amp) // 80 * 2 + 1)))
    peak_i = int(np.argmax(smooth[1:]) + 1) if len(smooth) > 2 else 0
    peak_hz = float(f[peak_i])
    total = float(power[1:].sum()) + 1e-15

    def band(lo: float, hi: float) -> float:
        m = (f >= lo) & (f < hi)
        return float(power[m].sum() / total)

    bands = {
        "0-5Hz": band(0, 5),
        "5-15Hz": band(5, 15),
        "15-30Hz": band(15, 30),
        "30-60Hz": band(30, 60),
        "60-120Hz": band(60, 120),
        "120-250Hz": band(120, 250),
        "250Hz+": band(250, max(f[-1] + 1, 251)),
    }

    # 工频：在 45–65 / 55–65 里找窄峰，和邻域中位数比
    def line_score(f0: float) -> float:
        m = (f >= f0 - 1.5) & (f <= f0 + 1.5)
        if not m.any():
            return 0.0
        peak = float(smooth[m].max())
        neigh = (f >= f0 - 12) & (f <= f0 + 12) & ~m
        base = float(np.median(smooth[neigh])) if neigh.any() else 0.0
        return max(0.0, min(1.0, (peak / (base + 1e-12) - 1.0) / 8.0))

    line50 = line_score(50.0)
    line60 = line_score(60.0)
    line_score_best = max(line50, line60)
    line_hz = 50.0 if line50 >= line60 else 60.0

    # 高频占比（>60Hz；采样率太低时退到 Nyquist 的 95%）
    nyq = 1.0 / (2.0 * dt)
    hf_lo = min(60.0, nyq * 0.95)
    hf_ratio = float(power[(f >= hf_lo)].sum() / total) if (f >= hf_lo).any() else 0.0
    lf_ratio = float(power[(f <= 8.0)].sum() / total) if (f <= 8.0).any() else 0.0

    return {
        "reliable": True,
        "peak_hz": round(peak_hz, 3),
        "bands": {k: round(v, 4) for k, v in bands.items()},
        "line_score": round(line_score_best, 4),
        "line_hz": line_hz if line_score_best > 0.05 else None,
        "hf_ratio": round(hf_ratio, 4),
        "lf_ratio": round(lf_ratio, 4),
        "nyquist_hz": round(nyq, 2),
        "n_samples": int(ns),
    }


def spike_stats(data: np.ndarray) -> dict:
    x = _as_float(data)
    if x.size == 0:
        return {"spike_ratio": 0.0, "kurtosis": 0.0}
    # 按“同一时刻的相邻道中值”找孤立尖峰；相干同相轴在相邻道同时出现，
    # 不会被误判成尖刺（旧版按单道时间中值会把强面波算成 29% 尖刺）。
    med = np.median(x, axis=1, keepdims=True)
    mad = np.median(np.abs(x - med), axis=1, keepdims=True) + 1e-12
    z = np.abs(x - med) / (1.4826 * mad)
    spike_ratio = float((z > 8.0).mean())
    # 峰值因子，尖刺多时明显偏高
    rms = float(np.sqrt(np.mean(x ** 2)) + 1e-12)
    kurt = float(np.mean((x / rms) ** 4) - 3.0)
    return {
        "spike_ratio": round(spike_ratio, 6),
        "kurtosis": round(kurt, 4),
        "clip_ratio": round(float((np.abs(x) > 0.999 * np.abs(x).max()).mean()), 6) if x.size else 0.0,
    }


def coherence_stats(data: np.ndarray) -> dict:
    """相邻道局部相干性；随机噪声会拉低它。"""
    x = _as_float(data)
    ns, ntr = x.shape
    if ntr < 2 or ns < 32:
        return {"adjacent_coherence": 0.0, "window_coherence": 0.0}
    w = min(128, ns)
    nw = ns // w
    if nw < 1:
        return {"adjacent_coherence": 0.0, "window_coherence": 0.0}
    xx = x[: nw * w].reshape(nw, w, ntr)
    xx = xx - xx.mean(axis=1, keepdims=True)
    a = xx[:, :, :-1].reshape(-1, w)
    b = xx[:, :, 1:].reshape(-1, w)
    num = np.sum(a * b, axis=1)
    den = np.sqrt(np.sum(a * a, axis=1) * np.sum(b * b, axis=1)) + 1e-12
    c = num / den
    c = c[np.isfinite(c)]
    if c.size == 0:
        return {"adjacent_coherence": 0.0, "window_coherence": 0.0}
    return {
        "adjacent_coherence": round(float(np.median(np.abs(c))), 4),
        "window_coherence": round(float(np.mean(np.abs(c) > 0.5)), 4),
    }


def blind_snr_quick(data: np.ndarray, dt: float | None) -> dict:
    """保守盲估计：短时窗能量的低分位当噪声、中高分位当信号+噪声。"""
    x = _as_float(data)
    ns, ntr = x.shape
    if ns < 64 or ntr < 1:
        return {"snr_db": None, "reliable": False, "method": "window-quantile"}
    w = min(256, max(32, int(0.064 / dt))) if dt else 64
    nw = ns // w
    if nw < 2:
        w = max(16, ns // 4)
        nw = ns // w
    e = (x[: nw * w].reshape(nw, w, ntr) ** 2).mean(axis=1).ravel()
    e = e[np.isfinite(e) & (e > 0)]
    if e.size < 4:
        return {"snr_db": None, "reliable": False, "method": "window-quantile"}
    noise = float(np.percentile(e, 10))
    high = float(np.percentile(e, 90))
    snr = 10.0 * math.log10((max(high - noise, 0.0) + 1e-15) / (noise + 1e-15))
    return {
        "snr_db": round(snr, 3),
        "noise_power": noise,
        "high_power": high,
        "reliable": bool(nw >= 4),
        "method": "window-quantile",
    }


def noise_profile(data: np.ndarray, dt: float | None, dz: float | None = None) -> dict:
    spec = spectral_stats(data, dt) if dt else {"reliable": False}
    spike = spike_stats(data)
    coh = coherence_stats(data)
    lf = float(spec.get("lf_ratio", 0.0))
    hf = float(spec.get("hf_ratio", 0.0))
    line = float(spec.get("line_score", 0.0))
    adj = float(coh.get("adjacent_coherence", 0.0))

    # 0–1 打分，只用于排序，不是概率
    # 面波：低频带 0–15Hz 为主，不需要强相干（真实面波常带频散、相邻道相干不稳定）
    low_band = lf + 0.8 * float(spec.get("bands", {}).get("5-15Hz", 0.0))
    surface = min(1.0, low_band * 1.8)
    random_n = min(1.0, hf * 3.0) * (0.65 + 0.35 * (1.0 - adj))
    # 只有稀疏、空间孤立的异常样点才算“尖刺”；强相干同相轴不算。
    # 占比 >5% 说明是成片强能量（面波/反射），不是尖刺。
    spike_ratio = float(spike.get("spike_ratio", 0.0))
    spike_n = 0.0 if spike_ratio > 0.05 else min(0.6, max(0.0, (spike_ratio - 0.002) / 0.01))
    random_n = max(random_n, spike_n)
    line_n = min(1.0, line * 2.5)
    # 散射/绕射：中频、横向不连贯；权重调低，避免随机噪声中频段被误判
    mid = float(spec.get("bands", {}).get("15-30Hz", 0.0)) + float(spec.get("bands", {}).get("30-60Hz", 0.0))
    scatter = min(1.0, mid * 0.7) * (0.35 + 0.65 * (1.0 - adj))
    # 折射：低频、相干、早起能量强（初至）；没有 f-k 线性度时给保守分
    early = 0.0
    try:
        e = x[: max(1, ns // 4)] ** 2
        early = float(e.mean() / (np.mean(x ** 2) + 1e-15))
    except Exception:
        early = 0.0
    refraction = min(1.0, low_band * 1.2) * (0.3 + 0.7 * adj) * (0.4 + 0.6 * min(1.0, early * 2.0))

    scores = {
        "面波": round(float(surface), 4),
        "随机噪声": round(float(random_n), 4),
        "工频干扰": round(float(line_n), 4),
        "散射绕射": round(float(scatter), 4),
        "折射波": round(float(refraction), 4),
    }
    evidence = []
    if spec.get("reliable"):
        evidence.append(f"主频约 {spec['peak_hz']}Hz，0–8Hz 能量占比 {lf:.1%}，>60Hz 占比 {hf:.1%}")
    if line > 0.15:
        evidence.append(f"检测到疑似 {spec.get('line_hz')}Hz 窄带峰（强度 {line:.2f}）")
    if spike.get("spike_ratio", 0) > 0.001:
        evidence.append(f"尖刺样点占比 {spike['spike_ratio']:.3%}，峰度 {spike.get('kurtosis')}")
    evidence.append(f"相邻道中位相干 {adj:.2f}")
    return {
        "scores": scores,
        "dominant": max(scores, key=scores.get),
        "evidence": evidence,
        "spectral": spec,
        "spike": spike,
        "coherence": coh,
    }


def diagnostic_summary(data: np.ndarray, dt: float | None, dz: float | None = None) -> dict:
    prof = noise_profile(data, dt, dz)
    snr = blind_snr_quick(data, dt)
    try:
        from ..quality.metrics import blind_snr as _bs
        q = _bs(data, dt, dz)
        if q.get("snr_db") is not None:
            snr = {**q, "fallback": False}
    except Exception:
        snr = {**snr, "fallback": True}
    return {
        "snr": snr,
        "noise": prof["scores"],
        "dominant_noise": prof["dominant"],
        "evidence": prof["evidence"],
        "spectral": prof["spectral"],
        "spike": prof["spike"],
        "coherence": prof["coherence"],
    }
