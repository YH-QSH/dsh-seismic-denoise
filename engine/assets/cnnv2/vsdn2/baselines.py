"""传统去噪方法基线。

用来和 CNN 做同数据、同指标的公平对比。全部是纯 numpy/scipy 实现，
不需要 MATLAB。

方法对应关系（与既有 MATLAB 脚本一致）：
    fk_filter      <-> a3.m / quzao4.m 里的 FK 视速度滤波
    median_filter  <-> zhongzhi.m 中值滤波
    bandpass       <-> 小波/带通类去噪的"频率域"版本
    agc            <-> quzao 系列里的 AGC 预处理
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
from scipy import signal


# ==========================================================================
# FK（f-k 视速度滤波）
# ==========================================================================
def fk_filter(
    data: np.ndarray,
    dt: float,
    dz: float,
    vmin: float = 1200.0,
    taper: float = 400.0,
    reject_slow: bool = True,
) -> np.ndarray:
    """频率-波数域视速度滤波。

    视速度 ``v = |f / k|``。``reject_slow=True`` 时压制 ``v < vmin`` 的成分
    （管波、面波等低速线性同相轴），这正是 VSP 去噪中最主要的一步。

    Parameters
    ----------
    dt, dz : 时间/空间采样间隔（秒、米）
    vmin   : 视速度门限（m/s）
    taper  : 过渡带宽度（m/s），避免硬截断造成的空间假频
    """
    a = np.asarray(data, dtype=np.float64)
    nt, nz = a.shape

    F = np.fft.fftshift(np.fft.fft2(a))
    f = np.fft.fftshift(np.fft.fftfreq(nt, d=dt))      # Hz
    k = np.fft.fftshift(np.fft.fftfreq(nz, d=dz))      # 1/m
    Fg, Kg = np.meshgrid(f, k, indexing="ij")

    with np.errstate(divide="ignore", invalid="ignore"):
        vel = np.abs(Fg / Kg)
    vel[~np.isfinite(vel)] = np.inf

    ramp = np.clip((vel - (vmin - taper)) / max(2.0 * taper, 1e-9), 0.0, 1.0)
    ramp = 0.5 - 0.5 * np.cos(np.pi * ramp)            # 余弦过渡
    if not reject_slow:
        ramp = 1.0 - ramp
    # 垂直方向（k=0, 即水平同相轴）不压
    ramp[np.abs(Kg) < 1e-12] = 1.0

    out = np.real(np.fft.ifft2(np.fft.ifftshift(F * ramp)))
    return out.astype(np.float32)


def fk_spectrum(data: np.ndarray) -> np.ndarray:
    """对数幅度 f-k 谱（画图用）。"""
    F = np.fft.fftshift(np.fft.fft2(np.asarray(data, dtype=np.float64)))
    return np.log10(np.abs(F) + 1.0)


# ==========================================================================
# 中值滤波
# ==========================================================================
def median_filter(
    data: np.ndarray,
    kt: int = 5,
    kx: int = 1,
) -> np.ndarray:
    """二维中值滤波（默认沿时间方向 5 点、道方向 1 点，保幅性较好）。"""
    from scipy.ndimage import median_filter as _med

    a = np.asarray(data, dtype=np.float64)
    return _med(a, size=(max(1, kt), max(1, kx)), mode="reflect").astype(np.float32)


# ==========================================================================
# 带通滤波
# ==========================================================================
def bandpass(
    data: np.ndarray,
    dt: float,
    f_lo: float = 5.0,
    f_hi: float = 90.0,
    order: int = 4,
) -> np.ndarray:
    """零相位带通（沿时间方向）。用 sosfiltfilt 保证数值稳定。"""
    a = np.asarray(data, dtype=np.float64)
    fs = 1.0 / dt
    nyq = fs / 2.0
    f_hi = min(f_hi, 0.98 * nyq)
    f_lo = max(f_lo, 0.5)
    if f_lo >= f_hi:
        return a.astype(np.float32)
    sos = signal.butter(order, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
    return signal.sosfiltfilt(sos, a, axis=0).astype(np.float32)


# ==========================================================================
# AGC（振幅补偿，常作为去噪前处理）
# ==========================================================================
def agc(data: np.ndarray, window: int = 50, eps: float = 1e-6) -> np.ndarray:
    a = np.asarray(data, dtype=np.float64)
    k = np.ones(window) / window
    env = np.sqrt(np.maximum(signal.fftconvolve(a * a, k[:, None], mode="same"), eps))
    return (a / np.maximum(env, eps)).astype(np.float32)


# ==========================================================================
# 小波阈值（若装了 PyWavelets 则启用）
# ==========================================================================
def has_pywt() -> bool:
    try:
        import pywt  # noqa: F401

        return True
    except ImportError:
        return False


def wavelet_denoise(
    data: np.ndarray,
    level: int = 4,
    wavelet: str = "sym8",
    mode: str = "soft",
) -> Optional[np.ndarray]:
    """逐道小波阈值去噪；未安装 PyWavelets 时返回 None。"""
    try:
        import pywt
    except ImportError:
        return None

    a = np.asarray(data, dtype=np.float64)
    nt, nz = a.shape
    out = np.zeros_like(a)
    ml = max(1, min(level, int(np.floor(np.log2(nt))) - 1))
    for j in range(nz):
        tr = a[:, j]
        try:
            coeffs = pywt.wavedec(tr, wavelet, level=ml, mode="symmetric")
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            thr = sigma * np.sqrt(2.0 * np.log(nt))
            coeffs = [coeffs[0]] + [
                pywt.threshold(c, thr, mode=mode) for c in coeffs[1:]
            ]
            out[:, j] = pywt.waverec(coeffs, wavelet, mode="symmetric")[:nt]
        except Exception:
            out[:, j] = tr
    return out.astype(np.float32)


# ==========================================================================
# 统一接口
# ==========================================================================
def run_baselines(
    data: np.ndarray,
    dt: float,
    dz: float,
    vmin: float = 1200.0,
) -> Dict[str, np.ndarray]:
    """把常用基线一次性跑完，返回 {方法名: 去噪结果}。"""
    out: Dict[str, np.ndarray] = {}
    out["FK"] = fk_filter(data, dt, dz, vmin=vmin)
    out["中值滤波"] = median_filter(data, kt=5, kx=1)
    out["带通滤波"] = bandpass(data, dt, 5.0, 90.0)
    w = wavelet_denoise(data, level=4, wavelet="sym8")
    if w is not None:
        out["小波阈值"] = w
    return out
