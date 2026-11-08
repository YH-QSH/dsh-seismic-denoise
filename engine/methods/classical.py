"""经典地震去噪方法（numpy / scipy 实现，不依赖 torch）。

统一约定
--------
- 输入 ``data``: ``(采样点 ns, 道数 ntr)``，float32/float64 均可。
- ``dt`` 秒，``dz`` 米（只有 FK 需要真实道间距；缺失时跳过并给警告）。
- ``run_method`` 返回 ``(float32 数组, info)``；``info`` 至少含
  ``notes`` / ``params_used`` / ``warnings``。
- 所有方法**保持形状不变**，参数默认保守（宁可少去一点，不削光有效波）。

这些实现都是可解释的基线：CNN 不可用/显存不足时，规划器会退回到这里。
"""
from __future__ import annotations

import copy
import math

import numpy as np
from scipy import ndimage, signal

# --------------------------------------------------------------------------
# 方法元信息（available_methods 的唯一数据源）
# --------------------------------------------------------------------------
_METHODS: dict[str, dict] = {
    "bandpass": {
        "name": "带通滤波",
        "description": "零相位 Butterworth 带通，压制低频面波与高频随机噪声，保留有效波频带。",
        "targets": ["面波", "随机噪声", "工频干扰"],
        "params": {"f_lo": 5.0, "f_hi": 90.0, "order": 4},
    },
    "notch": {
        "name": "工频陷波",
        "description": "对工频基频及其谐波做窄带 IIR 陷波，Q 值可调，尽量不伤邻近频带。",
        "targets": ["工频干扰"],
        "params": {"freq": 50.0, "harmonics": 3, "q": 30.0},
    },
    "despike": {
        "name": "异常振幅压制（去尖峰）",
        "description": "用局部中值预测对比，按稳健标准差识别并替换尖峰样点。",
        "targets": ["随机噪声"],
        "params": {"window": 5, "threshold": 6.0},
    },
    "median": {
        "name": "中值滤波",
        "description": "时间-空间二维中值滤波，压制高频随机噪声与部分面波，保边能力较好。",
        "targets": ["面波", "随机噪声"],
        "params": {"kt": 5, "kx": 1},
    },
    "fk": {
        "name": "FK 视速度滤波",
        "description": "频率-波数域按视速度做余弦斜坡压制低速面波/折射波，需真实道间距 dz。",
        "targets": ["面波", "折射波", "散射绕射"],
        "params": {"vmin": 1200.0, "taper": 400.0, "reject_slow": True,
                   "f_lo": 2.0, "f_hi": None},
    },
    "fx_decon": {
        "name": "F-X 域预测滤波去噪",
        "description": "分频段用复数最小二乘预测滤波（Wiener 形式）增强可预测的相干同相轴，压制随机噪声。",
        "targets": ["随机噪声", "散射绕射"],
        "params": {"win": 9, "lags": 2, "alpha": 0.85, "eps": 0.001,
                   "f_lo": 4.0, "f_hi": 120.0},
    },
    "svd": {
        "name": "局部 SVD 秩截断",
        "description": "在时间-空间小窗口内做 SVD，只保留前若干奇异值重建，压制非相干噪声。",
        "targets": ["散射绕射", "面波"],
        "params": {"wt_ms": 256.0, "wx": 16, "rank": 2, "step": 0.5},
    },
    "spectral_sub": {
        "name": "谱减法",
        "description": "STFT 谱减：用低能量帧估计噪声谱，过减因子与谱底可调，减少音乐噪声。",
        "targets": ["随机噪声", "工频干扰"],
        "params": {"nperseg_ms": 128.0, "overlap": 0.75, "alpha": 1.5,
                   "floor": 0.05, "noise_percentile": 20.0,
                   "max_reduction_db": 20.0},
    },
    "wavelet": {
        "name": "小波阈值去噪",
        "description": "逐道小波分解，对细节系数做软阈值（通用阈值），压制高频随机噪声。",
        "targets": ["随机噪声"],
        "params": {"level": 4, "wavelet": "sym8", "threshold_scale": 1.0},
    },
}

# 参数取值范围（写进 params_schema，也用于钳制用户输入）
_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "bandpass": {"f_lo": (0.5, 240.0), "f_hi": (1.0, 480.0), "order": (1, 8)},
    "notch": {"freq": (10.0, 400.0), "harmonics": (1, 6), "q": (2.0, 100.0)},
    "despike": {"window": (3, 41), "threshold": (2.0, 20.0)},
    "median": {"kt": (1, 31), "kx": (1, 15)},
    "fk": {"vmin": (50.0, 20000.0), "taper": (20.0, 5000.0),
           "f_lo": (0.0, 240.0), "f_hi": (0.0, 480.0)},
    "fx_decon": {"win": (3, 101), "lags": (1, 8), "alpha": (0.0, 1.0),
                 "eps": (1e-10, 1.0), "f_lo": (0.0, 240.0), "f_hi": (0.0, 480.0)},
    "svd": {"wt_ms": (32.0, 2000.0), "wx": (2, 64), "rank": (1, 16),
            "step": (0.1, 1.0)},
    "spectral_sub": {"nperseg_ms": (32.0, 1024.0), "overlap": (0.0, 0.95),
                     "alpha": (0.2, 5.0), "floor": (0.0, 0.5),
                     "noise_percentile": (1.0, 60.0),
                     "max_reduction_db": (3.0, 60.0)},
    "wavelet": {"level": (1, 8), "threshold_scale": (0.1, 4.0)},
}


def _build_schema(key: str) -> dict:
    params = _METHODS[key]["params"]
    ranges = _RANGES.get(key, {})
    schema: dict[str, dict] = {}
    for name, default in params.items():
        if isinstance(default, bool):
            typ = "bool"
        elif isinstance(default, int):
            typ = "int"
        elif isinstance(default, float):
            typ = "float"
        elif default is None:
            typ = "float|null"
        elif isinstance(default, str):
            typ = "str"
        else:
            typ = "any"
        item: dict = {"type": typ, "default": default}
        if name in ranges:
            item["min"], item["max"] = ranges[name]
        schema[name] = item
    return schema


def available_methods() -> dict[str, dict]:
    """返回全部经典方法的能力说明（深拷贝，调用方可随意改）。"""
    out: dict[str, dict] = {}
    for key, meta in _METHODS.items():
        out[key] = {
            "key": key,
            "name": meta["name"],
            "description": meta["description"],
            "targets": list(meta["targets"]),
            "params": copy.deepcopy(meta["params"]),
            "params_schema": _build_schema(key),
        }
    return out


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
def _prepare(data, dt) -> tuple[np.ndarray, float, list[str]]:
    """校验输入并转 float64（不修改调用方数组）。"""
    warnings: list[str] = []
    a = np.asarray(data)
    if a.ndim != 2:
        raise ValueError(f"data 必须是二维数组 (采样点, 道数)，实际 shape={a.shape}")
    if a.size == 0:
        raise ValueError("data 为空数组")
    a = np.array(a, dtype=np.float64, copy=True)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
        warnings.append("输入含 NaN/Inf，已置 0 后处理")
    try:
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dt 无效（{dt!r}），必须为正的有限秒数") from exc
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt 无效（{dt!r}），必须为正的有限秒数")
    return a, dt, warnings


def _num(p: dict, key: str, default, lo=None, hi=None, warnings=None,
         integer: bool = False):
    """取参数并做类型/范围钳制；坏值回退默认值并记警告。"""
    v = p.get(key, default)
    try:
        v = int(v) if integer else float(v)
    except (TypeError, ValueError):
        if warnings is not None:
            warnings.append(f"参数 {key}={p.get(key)!r} 无效，改用默认 {default}")
        return default
    if integer:
        v = int(round(v))
    if not math.isfinite(float(v)):
        if warnings is not None:
            warnings.append(f"参数 {key} 非有限，改用默认 {default}")
        return default
    if lo is not None and v < lo:
        if warnings is not None:
            warnings.append(f"参数 {key}={v} 小于下限 {lo}，已钳制")
        v = lo
    if hi is not None and v > hi:
        if warnings is not None:
            warnings.append(f"参数 {key}={v} 大于上限 {hi}，已钳制")
        v = hi
    return v


def _zero_phase(sos: np.ndarray, a: np.ndarray, axis: int = 0) -> np.ndarray:
    """零相位滤波；数据太短时退化为前向-反向两次滤波。"""
    try:
        return signal.sosfiltfilt(sos, a, axis=axis)
    except ValueError:
        padlen = 3 * (2 * len(sos) + 1)
        if a.shape[axis] > padlen:
            raise
        # 短数据：因果滤波两次（正/反），相位近似抵消
        y = signal.sosfilt(sos, a, axis=axis)
        return signal.sosfilt(sos, np.flip(y, axis=axis), axis=axis)


def _finalize(out: np.ndarray, shape: tuple[int, int], notes: str, params_used: dict,
              warnings: list[str]) -> tuple[np.ndarray, dict]:
    out = np.asarray(out, dtype=np.float64)
    if out.shape != shape:
        # 理论上不该发生；兜底裁剪/补零，保证契约
        got = out.shape
        fixed = np.zeros(shape, dtype=np.float64)
        ns = min(shape[0], out.shape[0])
        ntr = min(shape[1], out.shape[1])
        fixed[:ns, :ntr] = out[:ns, :ntr]
        out = fixed
        warnings.append(f"方法输出形状 {got} 与输入 {shape} 不一致，已裁剪/补零")
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    info = {
        "notes": notes,
        "params_used": dict(params_used),
        "warnings": warnings,
    }
    return out.astype(np.float32), info


# --------------------------------------------------------------------------
# 各方法实现
# --------------------------------------------------------------------------
def _do_bandpass(a: np.ndarray, dt: float, p: dict, warnings: list[str]) -> tuple[np.ndarray, str]:
    fs = 1.0 / dt
    nyq = fs / 2.0
    f_lo = _num(p, "f_lo", 5.0, 0.5, 0.95 * nyq, warnings)
    f_hi = _num(p, "f_hi", 90.0, 1.0, 0.98 * nyq, warnings)
    # 高频采样（超声/实验室）沿用 5–90 Hz 会几乎滤掉全部信号；按 Nyquist 自适应。
    if nyq > 200.0 and f_lo <= 5.1 and f_hi <= 90.1:
        f_lo = max(1.0, nyq * 0.02)
        f_hi = min(0.98 * nyq, nyq * 0.45)
        if warnings is not None:
            warnings.append(f"高频采样：带通自适应为 {f_lo:.1f}–{f_hi:.1f} Hz")
    order = _num(p, "order", 4, 1, 8, warnings, integer=True)
    if f_lo >= f_hi * 0.9:
        warnings.append(f"带通上下限不合理（{f_lo}–{f_hi} Hz），跳过")
        return a.copy(), "带通参数不合理，未处理"
    sos = signal.butter(order, [f_lo / nyq, f_hi / nyq], btype="band", output="sos")
    out = _zero_phase(sos, a)
    return out, f"{order} 阶零相位带通 {f_lo:.1f}–{f_hi:.1f} Hz（Nyquist {nyq:.1f} Hz）"


def _do_notch(a: np.ndarray, dt: float, p: dict, warnings: list[str]) -> tuple[np.ndarray, str]:
    nyq = 0.5 / dt
    freq = _num(p, "freq", 50.0, 10.0, 0.95 * nyq, warnings)
    harmonics = _num(p, "harmonics", 3, 1, 6, warnings, integer=True)
    q = _num(p, "q", 30.0, 2.0, 100.0, warnings)
    fs = 1.0 / dt
    targets: list[float] = []
    for h in range(1, harmonics + 1):
        f = freq * h
        if 0.01 < f < 0.95 * nyq:
            targets.append(f)
    if not targets:
        warnings.append(f"陷波频率 {freq} Hz 及其谐波都超出可用频带，跳过")
        return a.copy(), "陷波频率超出频带，未处理"
    out = a
    for f in targets:
        b, aa = signal.iirnotch(f, q, fs=fs)
        if a.shape[0] > 3 * max(len(b), len(aa)):
            out = signal.filtfilt(b, aa, out, axis=0)
        else:
            out = signal.lfilter(b, aa, out, axis=0)
    return out, f"陷波 {', '.join(f'{f:.1f}' for f in targets)} Hz（Q={q:.0f}，{harmonics} 次谐波）"


def _do_despike(a: np.ndarray, dt: float, p: dict, warnings: list[str]) -> tuple[np.ndarray, str]:
    win = _num(p, "window", 5, 3, 41, warnings, integer=True)
    thr = _num(p, "threshold", 6.0, 2.0, 20.0, warnings)
    if win % 2 == 0:
        win += 1
    med = ndimage.median_filter(a, size=(win, 1), mode="reflect")
    res = a - med
    mad = 1.4826 * np.median(np.abs(res), axis=0, keepdims=True)
    std = np.std(a, axis=0, keepdims=True)
    scale = np.maximum(mad, std * 0.05)
    scale = np.maximum(scale, 1e-12)
    mask = np.abs(res) > thr * scale
    out = a.copy()
    out[mask] = med[mask]
    ratio = float(mask.mean())
    return out, f"去尖峰：窗口 {win}，阈值 {thr:.1f} 倍稳健标准差，替换 {ratio:.3%} 样点"


def _do_median(a: np.ndarray, dt: float, p: dict, warnings: list[str]) -> tuple[np.ndarray, str]:
    kt = _num(p, "kt", 5, 1, 31, warnings, integer=True)
    kx = _num(p, "kx", 1, 1, 15, warnings, integer=True)
    if kt == 1 and kx == 1:
        return a.copy(), "中值窗口为 1×1，等价于恒等，未处理"
    out = ndimage.median_filter(a, size=(kt, kx), mode="reflect")
    return out, f"{kt}×{kx}（时间×道）中值滤波"


def _do_fk(a: np.ndarray, dt: float, dz, p: dict,
           warnings: list[str]) -> tuple[np.ndarray, str]:
    ns, ntr = a.shape
    vmin = _num(p, "vmin", 1200.0, 50.0, 20000.0, warnings)
    taper = _num(p, "taper", 400.0, 20.0, 5000.0, warnings)
    reject_slow = bool(p.get("reject_slow", True))
    nyq = 0.5 / dt
    f_lo = _num(p, "f_lo", 2.0, 0.0, 0.9 * nyq, warnings)
    f_hi = p.get("f_hi")
    f_hi = 0.98 * nyq if f_hi is None else _num(p, "f_hi", 0.98 * nyq, 0.0, 0.98 * nyq, warnings)

    if ntr < 4:
        warnings.append(f"道数只有 {ntr}，FK 需要至少 4 道；跳过")
        return a.copy(), "道数不足，FK 未处理"
    try:
        dz = float(dz) if dz is not None else None
    except (TypeError, ValueError):
        dz = None
    if dz is None or not math.isfinite(dz) or dz <= 0:
        warnings.append("dz（道间距）缺失或无效：FK 需要真实道间距，本次跳过（返回原数据）")
        return a.copy(), "缺少有效 dz，FK 未处理"
    if ns * ntr > 12_000_000:
        warnings.append(f"数据规模 {ns}×{ntr} 过大（>1200 万样点），FK 避免占满内存，跳过")
        return a.copy(), "数据过大，FK 未处理"

    # 去均值，减少 f=0 泄漏；沿时间轴两端加余弦渐变，抑制 FFT 边界振铃
    x = a - a.mean(axis=0, keepdims=True)
    tw = int(min(max(8, ns // 40), ns // 4))
    if tw > 1:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(tw) / tw)
        x[:tw] *= ramp[:, None]
        x[-tw:] *= ramp[::-1, None]

    F = np.fft.fftshift(np.fft.fft2(x))
    f = np.fft.fftshift(np.fft.fftfreq(ns, d=dt))[:, None]
    k = np.fft.fftshift(np.fft.fftfreq(ntr, d=dz))[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        vel = np.abs(f / k)
    mask = np.ones((ns, ntr), dtype=np.float64)
    if reject_slow:
        ramp = 0.5 - 0.5 * np.cos(np.pi * np.clip((vel - vmin) / max(taper, 1e-6), 0.0, 1.0))
        ramp = np.nan_to_num(ramp, nan=1.0)
        # k≈0 的整列（近水平）保号，避免把水平同相轴一起削掉
        ramp[:, np.abs(k[0]) < 1e-9] = 1.0
        mask *= ramp
    # 频带外保护（余弦渐变）
    if f_lo > 0 or f_hi < 0.98 * nyq:
        fm = f[:, 0]
        band = ((fm >= f_lo) & (fm <= f_hi)).astype(np.float64)
        # 边缘 10% 做余弦过渡
        w = max(2, int(0.1 * max(1, np.count_nonzero(band))))
        band = ndimage.uniform_filter1d(band, size=max(3, w), mode="nearest")
        band = np.clip(band, 0.0, 1.0)[:, None]
        mask = 1.0 - band * (1.0 - mask)
    F *= mask
    out = np.real(np.fft.ifft2(np.fft.ifftshift(F)))
    kept = float(np.mean(mask))
    return out, (f"FK 视速度滤波：压制视速度 < {vmin:.0f} m/s 的成分"
                 f"（过渡带 {taper:.0f} m/s，dz={dz:.3f} m，保留能量约 {kept:.1%}）")


def _do_fx_decon(a: np.ndarray, dt: float, p: dict,
                 warnings: list[str]) -> tuple[np.ndarray, str]:
    """F-X 域复数最小二乘预测滤波（逐道用左右邻道预测，再按 alpha 混合）。"""
    ns, ntr = a.shape
    win = _num(p, "win", 9, 3, 101, warnings, integer=True)
    lags = _num(p, "lags", 2, 1, 8, warnings, integer=True)
    alpha = _num(p, "alpha", 0.85, 0.0, 1.0, warnings)
    eps = _num(p, "eps", 1e-3, 1e-10, 1.0, warnings)
    nyq = 0.5 / dt
    f_lo = _num(p, "f_lo", 4.0, 0.0, 0.9 * nyq, warnings)
    f_hi = p.get("f_hi")
    f_hi = 0.98 * nyq if f_hi is None else _num(p, "f_hi", 0.98 * nyq, 0.0, 0.98 * nyq, warnings)

    if ntr < 2 * lags + 3:
        warnings.append(f"道数只有 {ntr}，F-X 预测至少需要 {2 * lags + 3} 道；跳过")
        return a.copy(), "道数不足，F-X 预测未处理"
    if win > ntr:
        win = ntr if ntr % 2 == 1 else ntr - 1
    if win % 2 == 0:
        win += 1
    win = max(3, win)

    x = a - a.mean(axis=0, keepdims=True)
    A = np.fft.rfft(x, axis=0)                      # (nf, ntr)
    freqs = np.fft.rfftfreq(ns, d=dt)
    band = (freqs >= f_lo) & (freqs <= f_hi)
    idx = np.flatnonzero(band)
    if idx.size < 3:
        warnings.append(f"频带 {f_lo:.1f}–{f_hi:.1f} Hz 太窄，F-X 预测跳过")
        return a.copy(), "频带太窄，F-X 预测未处理"

    npar = 2 * lags
    # 左右邻道预测子（边界用复制填充，后续按到边界距离衰减 alpha）
    Ap = np.pad(A, ((0, 0), (lags, lags)), mode="edge")
    P_all = []
    for lag in range(1, lags + 1):
        P_all.append(Ap[:, lags - lag: lags - lag + ntr])
        P_all.append(Ap[:, lags + lag: lags + lag + ntr])
    P_all = np.stack(P_all, axis=1)                 # (nf, 2L, ntr)

    # 局部窗口的起止索引（沿道方向滑动）
    half = win // 2
    pos = np.arange(ntr)
    start = np.clip(pos - half, 0, ntr)
    end = np.clip(pos + half + 1, 0, ntr)

    def _local_sums(arr: np.ndarray) -> np.ndarray:
        """arr (nb, ntr) -> 局部窗口累加，形状 (nb, ntr)。"""
        c = np.concatenate([np.zeros((arr.shape[0], 1), dtype=np.complex128),
                            np.cumsum(arr, axis=1)], axis=1)
        return c[:, end] - c[:, start]

    A_out = A.copy()
    a_vec = float(alpha) * np.clip(
        np.minimum(1.0, (pos + 1) / (lags + 1)) *
        np.minimum(1.0, (ntr - pos) / (lags + 1)), 0.0, 1.0)
    eye = np.eye(npar, dtype=np.complex128)

    chunk = 64
    solve_fallback = False
    for c0 in range(0, idx.size, chunk):
        ii = idx[c0: c0 + chunk]
        Y = A[ii]                                    # (nc, ntr)
        P = P_all[ii]                                # (nc, 2L, ntr)
        nc = ii.size
        M = np.empty((nc, ntr, npar, npar), dtype=np.complex128)
        b = np.empty((nc, ntr, npar), dtype=np.complex128)
        for i in range(npar):
            b[:, :, i] = _local_sums(np.conj(P[:, i, :]) * Y)
            for j in range(i, npar):
                s = _local_sums(np.conj(P[:, i, :]) * P[:, j, :])
                M[:, :, i, j] = s
                M[:, :, j, i] = np.conj(s)
        diag = np.real(np.trace(M, axis1=2, axis2=3)) / npar
        reg = np.maximum(diag, 1e-30) * float(eps)
        M = M + reg[..., None, None] * eye
        try:
            W = np.linalg.solve(M, b[..., None])[..., 0]
        except np.linalg.LinAlgError:
            solve_fallback = True
            W = (np.linalg.pinv(M) @ b[..., None])[..., 0]
        pred = np.einsum("cnk,ckn->cn", W, P)        # (nc, ntr)
        A_out[ii] = Y + a_vec[None, :] * (pred - Y)

    if solve_fallback:
        warnings.append("部分频率点预测方程病态，已用伪逆求解")
    out = np.fft.irfft(A_out, n=ns, axis=0)
    return out, (f"F-X 预测滤波：窗宽 {win} 道、每侧 {lags} 个预测道、alpha={alpha:.2f}、"
                 f"频带 {f_lo:.1f}–{f_hi:.1f} Hz")


def _do_svd(a: np.ndarray, dt: float, p: dict,
            warnings: list[str]) -> tuple[np.ndarray, str]:
    ns, ntr = a.shape
    wt = _num(p, "wt_ms", 256.0, 32.0, 2000.0, warnings)
    wt = max(16, int(round(wt / 1000.0 / dt)))
    wx = _num(p, "wx", 16, 2, 64, warnings, integer=True)
    rank = _num(p, "rank", 2, 1, 16, warnings, integer=True)
    step_frac = _num(p, "step", 0.5, 0.1, 1.0, warnings)

    if ntr < 2:
        warnings.append("道数不足 2，SVD 无法做空间约束，跳过")
        return a.copy(), "道数不足，SVD 未处理"
    wt = min(wt, ns)
    wx = min(wx, ntr)
    rank = max(1, min(rank, wx - 1))
    step_t = max(1, int(round(wt * step_frac)))
    step_x = max(1, int(round(wx * step_frac)))
    if wt < 8 or wx < 2:
        warnings.append(f"窗口太小（{wt}×{wx}），SVD 跳过")
        return a.copy(), "窗口太小，SVD 未处理"

    w_t = np.hanning(max(3, wt))
    w_x = np.hanning(max(3, wx))
    w_t = np.maximum(w_t, 1e-3)
    w_x = np.maximum(w_x, 1e-3)
    wgt = np.outer(w_t, w_x)                         # (wt, wx)，与 rec.T 同向

    t_starts = list(range(0, ns - wt + 1, step_t))
    if t_starts[-1] != ns - wt:
        t_starts.append(ns - wt)
    x_starts = list(range(0, ntr - wx + 1, step_x))
    if x_starts[-1] != ntr - wx:
        x_starts.append(ntr - wx)

    num = np.zeros((ns, ntr), dtype=np.float64)
    den = np.zeros((ns, ntr), dtype=np.float64)
    for i0 in t_starts:
        for j0 in x_starts:
            patch = a[i0:i0 + wt, j0:j0 + wx]         # (wt, wx)
            U, S, Vt = np.linalg.svd(patch.T, full_matrices=False)
            rec = (U[:, :rank] * S[:rank]) @ Vt[:rank]  # (wx, wt)
            num[i0:i0 + wt, j0:j0 + wx] += rec.T * wgt
            den[i0:i0 + wt, j0:j0 + wx] += wgt
    out = np.divide(num, np.maximum(den, 1e-12))
    n_win = len(t_starts) * len(x_starts)
    return out, (f"局部 SVD 秩截断：窗口 {wt} 样点 × {wx} 道、保留 {rank} 个奇异值、"
                 f"共 {n_win} 个窗口")


def _do_spectral_sub(a: np.ndarray, dt: float, p: dict,
                     warnings: list[str]) -> tuple[np.ndarray, str]:
    ns, ntr = a.shape
    fs = 1.0 / dt
    nper = _num(p, "nperseg_ms", 128.0, 32.0, 1024.0, warnings)
    nper = int(round(nper / 1000.0 / dt))
    nper = max(16, min(nper, ns))
    overlap = _num(p, "overlap", 0.75, 0.0, 0.95, warnings)
    nover = int(round(nper * overlap))
    nover = min(nover, nper - 1)
    alpha = _num(p, "alpha", 1.5, 0.2, 5.0, warnings)
    floor = _num(p, "floor", 0.05, 0.0, 0.5, warnings)
    pct = _num(p, "noise_percentile", 20.0, 1.0, 60.0, warnings)
    max_red = _num(p, "max_reduction_db", 20.0, 3.0, 60.0, warnings)
    g_min = 10.0 ** (-max_red / 20.0)

    out = np.empty_like(a)
    smooth = 3
    for j in range(ntr):
        tr = a[:, j]
        f, t, Z = signal.stft(tr, fs=fs, nperseg=nper, noverlap=nover,
                              window="hann", boundary="zeros", padded=True)
        mag = np.abs(Z)
        frame_energy = np.sum(mag ** 2, axis=0)
        n_quiet = max(1, int(math.ceil(frame_energy.size * pct / 100.0)))
        quiet = np.argsort(frame_energy)[:n_quiet]
        noise = np.median(mag[:, quiet], axis=1)[:, None]
        clean = np.maximum(mag - alpha * noise, floor * mag)
        gain = np.minimum(clean / np.maximum(mag, 1e-30), 1.0)
        gain = np.maximum(gain, g_min)
        if smooth > 1:
            gain = ndimage.uniform_filter1d(gain, size=smooth, axis=0)
            gain = ndimage.uniform_filter1d(gain, size=smooth, axis=1)
            gain = np.clip(gain, g_min, 1.0)
        _, rec = signal.istft(Z * gain, fs=fs, nperseg=nper, noverlap=nover,
                              window="hann", boundary="zeros")
        if rec.size < ns:
            rec = np.pad(rec, (0, ns - rec.size))
        out[:, j] = rec[:ns]
    return out, (f"谱减：STFT {nper} 样点（{nper * dt * 1000:.0f} ms）、重叠 {nover}、"
                 f"过减 {alpha:.2f}、谱底 {floor:.2f}、最多压 {max_red:.0f} dB，"
                 f"噪声谱取低能量 {pct:.0f}% 帧")


def _do_wavelet(a: np.ndarray, dt: float, p: dict,
                warnings: list[str]) -> tuple[np.ndarray, str]:
    try:
        import pywt
    except Exception:
        warnings.append("未安装 pywt，小波去噪跳过（返回原数据）")
        return a.copy(), "pywt 不可用，小波去噪未处理"
    level = _num(p, "level", 4, 1, 8, warnings, integer=True)
    wavelet = str(p.get("wavelet", "sym8"))
    scale = _num(p, "threshold_scale", 1.0, 0.1, 4.0, warnings)
    ns, ntr = a.shape
    max_level = int(pywt.dwt_max_level(ns, pywt.Wavelet(wavelet).dec_len))
    if max_level < 1:
        warnings.append("数据太短，无法做小波分解；跳过")
        return a.copy(), "数据太短，小波去噪未处理"
    if level > max_level:
        warnings.append(f"小波层数 {level} 超过可分解层数 {max_level}，已降到 {max_level}")
        level = max_level
    out = np.empty_like(a)
    for j in range(ntr):
        try:
            coeffs = pywt.wavedec(a[:, j], wavelet, level=level, mode="symmetric")
        except Exception:
            out[:, j] = a[:, j]
            continue
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 if coeffs[-1].size else 0.0
        thr = scale * sigma * math.sqrt(2.0 * math.log(max(ns, 2)))
        coeffs[1:] = [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]
        rec = pywt.waverec(coeffs, wavelet, mode="symmetric")
        out[:, j] = rec[:ns] if rec.size >= ns else np.pad(rec, (0, ns - rec.size))
    return out, (f"{level} 层 {wavelet} 小波软阈值（通用阈值 × {scale:.2f}）")


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------
def run_method(key: str, data, dt, dz=None, params: dict | None = None):
    """执行一个经典去噪方法。

    参数
    ----
    key : available_methods() 里的键
    data : (采样点, 道数) 数组
    dt : 采样间隔（秒）
    dz : 道间距（米），FK 需要；其它方法可忽略/None
    params : 方法参数，缺省用默认值；非 dict 会报 ValueError

    返回
    ----
    (out_float32, info) ；info = {"notes", "params_used", "warnings"}
    """
    if not isinstance(key, str) or key not in _METHODS:
        raise ValueError(f"未知方法 {key!r}，可选：{', '.join(_METHODS)}")
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"params 必须是 dict 或 None，实际 {type(params).__name__}")
    a, dt, warnings = _prepare(data, dt)
    merged = copy.deepcopy(_METHODS[key]["params"])
    if params:
        for k, v in params.items():
            if k in merged:
                merged[k] = v
            else:
                warnings.append(f"方法 {key} 不认识参数 {k!r}，已忽略")
    p = merged
    shape = a.shape

    if key == "bandpass":
        out, note = _do_bandpass(a, dt, p, warnings)
    elif key == "notch":
        out, note = _do_notch(a, dt, p, warnings)
    elif key == "despike":
        out, note = _do_despike(a, dt, p, warnings)
    elif key == "median":
        out, note = _do_median(a, dt, p, warnings)
    elif key == "fk":
        out, note = _do_fk(a, dt, dz, p, warnings)
    elif key == "fx_decon":
        out, note = _do_fx_decon(a, dt, p, warnings)
    elif key == "svd":
        out, note = _do_svd(a, dt, p, warnings)
    elif key == "spectral_sub":
        out, note = _do_spectral_sub(a, dt, p, warnings)
    elif key == "wavelet":
        out, note = _do_wavelet(a, dt, p, warnings)
    else:  # pragma: no cover - 已在上方拦截
        raise ValueError(f"未知方法 {key!r}")

    # params_used 记录最终生效值（含钳制后的 int/float）
    return _finalize(out, shape, note, merged, warnings)
