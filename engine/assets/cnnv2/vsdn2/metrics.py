"""定量评价指标：SNR / RMSE / 相关系数。

与既有 MATLAB 程序保持一致的三个口径：
    SNR  = 10*log10( sum(clean^2) / sum((x-clean)^2) )     单位 dB
    RMSE = sqrt( mean((x-clean)^2) )
    CC   = corr(clean(:), x(:))                            皮尔逊相关系数
另外提供 SNR 增益（去噪后 - 去噪前），这是论文里最常报的指标。
"""

from __future__ import annotations

from typing import Dict

import numpy as np


def snr_db(x: np.ndarray, ref: np.ndarray) -> float:
    """信噪比（dB）。ref 为干净参考。"""
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    num = float(np.sum(ref ** 2))
    den = float(np.sum((x - ref) ** 2))
    return float(10.0 * np.log10(num / (den + 1e-20) + 1e-20))


def rmse(x: np.ndarray, ref: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float(np.sqrt(np.mean((x - ref) ** 2)))


def cc(x: np.ndarray, ref: np.ndarray) -> float:
    """皮尔逊相关系数（展平后计算）。"""
    a = np.asarray(x, dtype=np.float64).ravel()
    b = np.asarray(ref, dtype=np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den < 1e-20:
        return 0.0
    return float(np.sum(a * b) / den)


def evaluate(x: np.ndarray, ref: np.ndarray) -> Dict[str, float]:
    """一次性算齐三个指标。"""
    return {
        "snr_db": snr_db(x, ref),
        "rmse": rmse(x, ref),
        "cc": cc(x, ref),
    }


def snr_gain(after: np.ndarray, before: np.ndarray, ref: np.ndarray) -> float:
    return snr_db(after, ref) - snr_db(before, ref)


def summarize(rows: list[Dict[str, float]]) -> Dict[str, float]:
    """对多剖面逐条指标求均值/标准差。

    rows 里每条含 snr_db / rmse / cc（以及可选的 snr_gain_db）。
    """
    out: Dict[str, float] = {}
    if not rows:
        return out
    keys = rows[0].keys()
    for k in keys:
        vals = np.array([r[k] for r in rows if k in r], dtype=np.float64)
        if vals.size == 0:
            continue
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    return out


def local_snr(x: np.ndarray, ref: np.ndarray, win: int = 64) -> np.ndarray:
    """滑动时窗 SNR 图（dB），用于看噪声压制是否均匀。"""
    x = np.asarray(x, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    nt = x.shape[0]
    win = min(win, nt)
    out = np.full(x.shape, np.nan)
    half = win // 2
    for i0 in range(0, nt, half):
        i1 = min(nt, i0 + win)
        if i1 - i0 < win // 2:
            break
        s = float(np.sum(ref[i0:i1] ** 2))
        e = float(np.sum((x[i0:i1] - ref[i0:i1]) ** 2))
        out[i0:i1] = 10.0 * np.log10(s / (e + 1e-20) + 1e-20)
    return out


# 指标中文名，画表时用
METRIC_LABELS = {
    "snr_db": "信噪比 SNR (dB)",
    "rmse": "均方根误差 RMSE",
    "cc": "相关系数 CC",
    "snr_gain_db": "SNR 增益 (dB)",
}
