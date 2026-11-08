"""去噪质量指标：盲信噪比 + 多项保真指标。

设计要点
--------
- **不假设有参考**：``blind_snr`` 用"局部窗口能量分位 + 高频段噪声外推"两路估计
  互相校验，给出 ``reliable`` 标记，说明这次估计能不能信。
- 所有指标都用 float，不给 NaN；算不出来给 ``None`` 并在 ``extra["notes"]`` 说明。
- 指标默认在"信号带"内统计（默认 5–90 Hz，按 Nyquist 自适应），因为行业门禁关心的是
  **有效波能量有没有被削**，而不是全频带能量（带通本来就会砍掉带外噪声）。
  全频带能量比另存 ``extra["energy_ratio_full"]``。
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

from ..model import QualityMetrics

# 局部窗口长度（秒）。太长看不出局部变化，太短统计不稳。
WIN_S = 0.256
# 计算盲信噪比时，窗口数低于这个值就认为不可靠。
MIN_FRAMES = 8
# 能量分位路要求窗口能量至少有这么大起伏，否则没有"安静段"可当噪声底。
Q_SPAN_MIN_DB = 6.0


# --------------------------------------------------------------------------
# 基础校验
# --------------------------------------------------------------------------
def _as_array(data, name: str) -> np.ndarray:
    """校验并转 float64；非有限值置 0（记在 extra 里由调用方说明）。"""
    a = np.asarray(data, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组 (采样点, 道数)，实际 shape={a.shape}")
    if a.size == 0 or a.shape[0] < 4 or a.shape[1] < 1:
        raise ValueError(f"{name} 尺寸太小，无法评价：shape={a.shape}")
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)
    return a


def _check_dt(dt) -> float:
    try:
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dt 无效（{dt!r}），无法计算频率相关指标") from exc
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt 无效（{dt!r}），必须为正的有限秒数")
    return dt


def signal_band(dt: float, dz: float | None = None) -> tuple[float, float]:
    """信号带（Hz）：默认 5–90 Hz，随 Nyquist 自适应收缩。

    这个带用于：``spectrum_corr``、``signal_leakage``、``energy_ratio``、``local_snr_gain_db``。
    """
    nyq = 0.5 / dt
    f_lo = max(4.0, 0.01 * nyq)
    f_hi = min(80.0, 0.50 * nyq)
    if f_hi <= f_lo * 1.2:
        f_lo = max(0.5, 0.02 * nyq)
        f_hi = 0.45 * nyq
    return float(f_lo), float(f_hi)


def _db(x: float, floor: float = 1e-20) -> float:
    return float(10.0 * math.log10(max(float(x), floor)))


def _zero_phase(sos: np.ndarray, a: np.ndarray, axis: int = 0) -> np.ndarray:
    """零相位滤波，短道时自动降级为前向-反向滤波。"""
    if sos is None or a.shape[axis] < 4:
        return a
    try:
        return signal.sosfiltfilt(sos, a, axis=axis)
    except Exception:
        y = signal.sosfilt(sos, a, axis=axis)
        return signal.sosfilt(sos, np.flip(y, axis=axis), axis=axis)


def _bandpass(a: np.ndarray, dt: float, f_lo: float, f_hi: float,
              order: int = 2) -> np.ndarray:
    nyq = 0.5 / dt
    lo = max(f_lo / nyq, 1e-4)
    hi = min(f_hi / nyq, 0.98)
    if lo >= hi:
        return a
    try:
        sos = signal.butter(order, [lo, hi], btype="band", output="sos")
    except Exception:
        return a
    return _zero_phase(sos, a)


# --------------------------------------------------------------------------
# 盲信噪比
# --------------------------------------------------------------------------
def _frame_power(a: np.ndarray, win: int) -> np.ndarray:
    """按时间切窗，返回每窗平均功率 (n_frames,)。"""
    ns, ntr = a.shape
    n = ns // win
    if n < 1:
        return np.asarray([float(np.mean(a.astype(np.float64) ** 2))])
    fr = a[: n * win].reshape(n, win, ntr)
    return np.mean(fr.astype(np.float64) ** 2, axis=(1, 2))


def blind_snr(data, dt, dz=None) -> dict:
    """盲信噪比估计（无参考）。

    两路独立估计：
    1. **局部窗口能量分位**：把时间轴切成 ~256 ms 的窗口，算每窗平均功率。
       安静窗口（10% 分位）近似噪声底，较强窗口（85% 分位）体现有效波；
       再用 ``总功率 - 噪声功率`` 得到信号功率。
    2. **高频段噪声外推**：随机噪声通常是宽带的，取高频段（≥62% Nyquist）
       的中位功率谱密度，按白噪声外推到全带，得到噪声总功率。

    两路在 12 dB 内一致时取几何平均并标 ``reliable=True``；否则只信能量分位一路并
    标 ``reliable=False``，让上层门禁把 SNR 相关检查降为软检查。

    返回
    ----
    {"snr_db", "signal_db", "noise_db", "reliable", "method", "details"}
    """
    a = _as_array(data, "data")
    dt = _check_dt(dt)
    ns, ntr = a.shape
    fs = 1.0 / dt
    nyq = fs / 2.0

    win = int(round(WIN_S / dt))
    win = max(16, min(win, max(16, ns)))
    if ns < 2 * win:
        # 数据太短，缩窗保证至少两窗
        win = max(8, ns // 2) if ns >= 16 else ns
    p_total = float(np.mean(a.astype(np.float64) ** 2))
    details = {
        "window_ms": round(win * dt * 1000.0, 3),
        "n_frames": int(ns // win),
        "n_traces": int(ntr),
        "band_hz": [round(_hif_lo(dt), 2), round(0.95 * nyq, 2)],
    }

    reasons: list[str] = []

    # ---- 1) 能量分位路 ----
    pw = _frame_power(a, win)
    n_frames = int(pw.size)
    details["n_frames"] = n_frames
    p_noise_q = None
    snr_q = None
    span_db = None
    q_valid = False
    if n_frames >= 2 and p_total > 0:
        p10 = float(np.percentile(pw, 10.0))
        p85 = float(np.percentile(pw, 85.0))
        span_db = _db(max(p85, 1e-30) / max(p10, 1e-30))
        details["frame_power_span_db"] = round(span_db, 3)
        p_noise_q = max(p10, p_total * 1e-9)
        p_sig_q = max(p_total - p_noise_q, p_noise_q * 1e-9)
        snr_q = _db(p_sig_q / p_noise_q)
        # 无论是否可信都记下分位估计：它是输入/输出之间唯一"两边都有定义"的
        # 估计器，门禁计算增益时要用同一条路线相减，避免估计器切换制造假增益。
        details["snr_quantile_db"] = round(snr_q, 3)
        details["noise_quantile_db"] = round(_db(p_noise_q), 3)
        q_valid = span_db >= Q_SPAN_MIN_DB
        details["quantile_valid"] = bool(q_valid)
        if not q_valid:
            reasons.append(
                f"窗口能量起伏仅 {span_db:.1f} dB（<{Q_SPAN_MIN_DB:.0f} dB），"
                "没有安静段，能量分位法估计的绝对信噪比不可信")

    # ---- 2) 高频噪声外推路 ----
    p_noise_hf = None
    snr_hf = None
    hf_ratio = None
    hf_valid = False
    if ns >= 16 and p_total > 0:
        spec = np.fft.rfft(a, axis=0)
        psd = np.mean(np.abs(spec) ** 2, axis=1) / float(fs)  # mean|X_k|^2 / fs
        freqs = np.fft.rfftfreq(ns, d=dt)
        lo = _hif_lo(dt)
        hi = 0.95 * nyq
        sel = (freqs >= lo) & (freqs <= hi)
        if int(sel.sum()) >= 3 and psd[sel].sum() > 0:
            dens = float(np.median(psd[sel]))
            # 平坦噪声外推：rfft 未归一化时，全带噪声功率 = dens * fs / ns。
            # 注意不是 dens * nyq——后者会随记录长度放大 ns/2 倍（旧实现在 4 s
            # 记录上把噪声底估高约 30 dB，直接导致盲 SNR 恒为负）。
            p_noise_hf = dens * fs / float(ns)
            hf_ratio = float(psd[sel].sum() / max(psd.sum(), 1e-30))
            if p_noise_hf > 0:
                p_sig_hf = max(p_total - p_noise_hf, p_noise_hf * 1e-9)
                snr_hf = _db(p_sig_hf / p_noise_hf)
                details["snr_hf_db"] = round(snr_hf, 3)
                details["noise_hf_db"] = round(_db(p_noise_hf), 3)
                details["hf_energy_ratio"] = round(hf_ratio, 4)
                # 高频段几乎没有能量时（例如已被低通），白噪声外推会严重低估
                # 噪声底、虚增信噪比，该路判为不可用。
                hf_valid = hf_ratio >= 0.005

    # ---- 融合 ----
    reliable = True
    if p_total <= 0:
        details["notes"] = "数据全零，无法估计信噪比"
        return {"snr_db": 0.0, "signal_db": -200.0, "noise_db": -200.0,
                "reliable": False, "method": "无法估计（数据全零）", "details": details}

    if n_frames < MIN_FRAMES:
        reliable = False
        reasons.append(f"时间窗只有 {n_frames} 个（<{MIN_FRAMES}），统计不稳")
    if ntr < 4:
        reliable = False
        reasons.append(f"道数只有 {ntr} 道，横向约束不足")
    if p_noise_hf is None or snr_hf is None:
        hf_valid = False
        reasons.append("高频段能量过低（可能已低通），无法外推噪声")
    elif not hf_valid:
        reasons.append(f"高频能量占比仅 {hf_ratio:.3%}，噪声外推不可用")
    details["hf_valid"] = bool(hf_valid)

    if q_valid and hf_valid:
        diff = abs(snr_q - snr_hf)
        details["estimator_gap_db"] = round(diff, 3)
        if diff > 12.0:
            reliable = False
            # 起伏大说明有安静段，分位法可信；起伏小说明分位法在把噪声当信号，
            # 此时高频外推更可信。
            if span_db is not None and span_db >= 12.0:
                reasons.append(f"两路估计相差 {diff:.1f} dB（>12 dB），取能量分位路")
                p_noise = p_noise_q
                used = "quantile"
            else:
                reasons.append(f"两路估计相差 {diff:.1f} dB（>12 dB），无安静段，取高频外推路")
                p_noise = p_noise_hf
                used = "hf"
        else:
            p_noise = math.sqrt(max(p_noise_q, 1e-30) * max(p_noise_hf, 1e-30))
            used = "quantile+hf"
    elif q_valid:
        p_noise = p_noise_q
        used = "quantile"
        reliable = False
        reasons.append("只有能量分位一路可用")
    elif hf_valid:
        p_noise = p_noise_hf
        used = "hf"
        reliable = False
        reasons.append("只有高频外推一路可用")
    elif p_noise_q is not None:
        p_noise = p_noise_q
        used = "quantile"
        reliable = False
        reasons.append("两路估计都不可靠，取能量分位路仅供参考")
    elif p_noise_hf is not None:
        p_noise = p_noise_hf
        used = "hf"
        reliable = False
        reasons.append("两路估计都不可靠，取高频外推路仅供参考")
    else:
        details["notes"] = "两路噪声估计均失败"
        return {"snr_db": 0.0, "signal_db": -200.0, "noise_db": -200.0,
                "reliable": False, "method": "无法估计", "details": details}

    p_noise = max(float(p_noise), 1e-30)
    p_sig = max(p_total - p_noise, p_noise * 1e-9)
    snr = _db(p_sig / p_noise)
    method = {
        "quantile": "局部窗口能量分位",
        "hf": "高频段噪声外推",
        "quantile+hf": "局部窗口能量分位 + 高频段噪声外推（几何平均）",
    }[used]
    if reasons:
        details["notes"] = "；".join(reasons)
    return {
        "snr_db": float(snr),
        "signal_db": float(_db(p_sig)),
        "noise_db": float(_db(p_noise)),
        "reliable": bool(reliable),
        "method": method,
        "details": details,
    }


def _hif_lo(dt: float) -> float:
    """高频噪声估计的起始频率：≥62% Nyquist，但至少 60 Hz。"""
    nyq = 0.5 / dt
    return float(min(max(60.0, 0.62 * nyq), 0.90 * nyq))


# --------------------------------------------------------------------------
# 局部相关（信号泄漏 / 残差相干性）
# --------------------------------------------------------------------------
def _local_corr(a: np.ndarray, b: np.ndarray, dt: float,
                band: tuple[float, float] | None = None) -> tuple[float, float, int]:
    """分窗相关系数。

    对每个（时间窗 × 道）算 ``corr(a, b)``，再按 b 在该窗的能量加权平均。
    返回 ``(加权均值, 95 分位, 有效样本数)``；样本不足返回 ``(0.0, 0.0, 0)``。
    """
    ns, ntr = a.shape
    win = int(round(WIN_S / dt))
    win = max(16, min(win, max(16, ns)))
    if ns < win:
        win = ns
    step = max(1, win // 2)

    if band is not None:
        a = _bandpass(a, dt, band[0], band[1])
        b = _bandpass(b, dt, band[0], band[1])

    vals: list[np.ndarray] = []
    wgts: list[np.ndarray] = []
    for s in range(0, max(1, ns - win + 1), step):
        x = a[s:s + win]
        y = b[s:s + win]
        x = x - x.mean(axis=0, keepdims=True)
        y = y - y.mean(axis=0, keepdims=True)
        nx = np.linalg.norm(x, axis=0)
        ny = np.linalg.norm(y, axis=0)
        valid = (nx > 1e-12) & (ny > 1e-12)
        c = np.zeros(ntr, dtype=np.float64)
        c[valid] = np.sum(x[:, valid] * y[:, valid], axis=0) / (nx[valid] * ny[valid])
        vals.append(np.clip(c, -1.0, 1.0))
        wgts.append(ny ** 2)
        if ns == win:
            break
    if not vals:
        return 0.0, 0.0, 0
    c_all = np.concatenate(vals)
    w_all = np.concatenate(wgts)
    w_sum = float(w_all.sum())
    if w_sum <= 0:
        return 0.0, 0.0, int(c_all.size)
    mean = float(np.sum(c_all * w_all) / w_sum)
    p95 = float(np.percentile(c_all, 95.0))
    return mean, p95, int(c_all.size)


def _mean_spectrum(a: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    spec = np.fft.rfft(a, axis=0)
    amp = np.mean(np.abs(spec), axis=1)
    freqs = np.fft.rfftfreq(a.shape[0], d=dt)
    return freqs, amp


def _octave_bands(f_lo: float, f_hi: float, per_octave: int = 3) -> list[tuple[float, float]]:
    """把 [f_lo, f_hi] 按 1/per_octave 倍频程切成对数带。"""
    bands: list[tuple[float, float]] = []
    lo = max(float(f_lo), 1e-3)
    ratio = 2.0 ** (1.0 / max(1, int(per_octave)))
    while lo < f_hi * 0.999 and len(bands) < 128:
        hi = min(lo * ratio, float(f_hi))
        if hi <= lo:
            break
        bands.append((lo, hi))
        lo = hi
    return bands


def _band_amplitudes(freqs: np.ndarray, amp: np.ndarray,
                     bands: list[tuple[float, float]]) -> np.ndarray:
    """每个倍频程带内的平均振幅。"""
    vals = []
    for lo, hi in bands:
        sel = (freqs >= lo) & (freqs < hi)
        if sel.any():
            vals.append(float(np.mean(amp[sel])))
    return np.asarray(vals, dtype=np.float64)


def _safe_pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64).ravel()
    y = np.asarray(y, dtype=np.float64).ravel()
    if x.size < 4 or x.size != y.size:
        return None
    x = x - x.mean()
    y = y - y.mean()
    den = float(np.linalg.norm(x) * np.linalg.norm(y))
    if den < 1e-30:
        return None
    return float(np.clip(np.dot(x, y) / den, -1.0, 1.0))


def _bootstrap_reference_gain(noisy: np.ndarray, denoised: np.ndarray,
                              reference: np.ndarray, *, n_boot: int = 400,
                              seed: int = 20261108) -> dict | None:
    """按道重采样估计参考增益的置信区间。

    参考增益是“输入误差 / 输出去噪误差”的全局比值；对道做 bootstrap
    可以回答“换一批道，这个增益还稳不稳”，比只看多个方法的离散度更贴近
    实际验收。返回 5%/95% 分位、标准差和样本数，供 UQ 层直接使用。
    """
    a = np.asarray(noisy, dtype=np.float64)
    d = np.asarray(denoised, dtype=np.float64)
    r = np.asarray(reference, dtype=np.float64)
    if a.shape != d.shape or a.shape != r.shape or a.ndim != 2:
        return None
    ntr = int(a.shape[1])
    if ntr < 20 or n_boot < 50:
        return None
    se_in = np.mean((a - r) ** 2, axis=0)
    se_out = np.mean((d - r) ** 2, axis=0)
    valid = np.isfinite(se_in) & np.isfinite(se_out)
    if int(valid.sum()) < 20:
        return None
    se_in = np.maximum(se_in[valid], 1e-30)
    se_out = np.maximum(se_out[valid], 1e-30)
    n = int(se_in.size)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(int(n_boot), n))
    mi = se_in[idx].mean(axis=1)
    mo = se_out[idx].mean(axis=1)
    gains = 10.0 * np.log10(mi / mo)
    gains = gains[np.isfinite(gains)]
    if gains.size < 50:
        return None
    lo, hi = np.percentile(gains, [5.0, 95.0])
    return {
        "low_db": round(float(lo), 3),
        "high_db": round(float(hi), 3),
        "std_db": round(float(np.std(gains, ddof=1)), 3),
        "n_boot": int(gains.size),
        "n_traces": n,
        "method": "trace_bootstrap",
    }


# --------------------------------------------------------------------------
# 主评价入口
# --------------------------------------------------------------------------
def evaluate(noisy: np.ndarray, denoised: np.ndarray, dt: float,
             dz: float | None = None,
             reference: np.ndarray | None = None) -> QualityMetrics:
    """去噪前后对比评价。

    参数
    ----
    noisy, denoised : (采样点 ns, 道数 ntr) 数组，形状必须一致
    dt : 采样间隔（秒）
    dz : 道间距（米），只用于记录，不参与计算
    reference : 可选的干净参考（合成数据/模型数据才有）

    返回 :class:`QualityMetrics`；算不出的项为 ``None``，原因写在 ``extra["notes"]``。
    """
    a = _as_array(noisy, "noisy")
    d = _as_array(denoised, "denoised")
    if a.shape != d.shape:
        raise ValueError(f"noisy 与 denoised 形状不一致：{a.shape} vs {d.shape}")
    dt = _check_dt(dt)
    ns, ntr = a.shape
    notes: list[str] = []
    extra: dict = {"dz": None if dz is None else float(dz),
                   "shape": [int(ns), int(ntr)], "window_ms": round(WIN_S * 1000.0, 1)}

    f_lo, f_hi = signal_band(dt)
    band = (f_lo, f_hi)
    extra["band_hz"] = [round(f_lo, 2), round(f_hi, 3)]
    extra["sampling_dt_ms"] = round(dt * 1000.0, 4)

    # ---- 盲信噪比 ----
    sni = blind_snr(a, dt, dz)
    sno = blind_snr(d, dt, dz)
    snr_in = sni["snr_db"]
    snr_out = sno["snr_db"]
    # 输入/输出必须用同一套估计器相减，否则会制造假增益/假损失。
    # 双可靠时用融合值；否则优先用两边都有的高频外推路，其次能量分位路。
    extra["snr_in_fused_db"] = float(snr_in)
    extra["snr_out_fused_db"] = float(snr_out)
    hf_in = sni["details"].get("snr_hf_db")
    hf_out = sno["details"].get("snr_hf_db")
    q_in = sni["details"].get("snr_quantile_db")
    q_out = sno["details"].get("snr_quantile_db")
    if sni["reliable"] and sno["reliable"]:
        extra["snr_gain_estimator"] = "fused"
    elif q_in is not None and q_out is not None:
        # 分位路在输入/输出两边都有定义，且对"噪声被削掉多少"最直接，
        # 因此优先用它做配对比较；绝对值的可信度由 reliable 标记另行说明。
        snr_in, snr_out = float(q_in), float(q_out)
        extra["snr_gain_estimator"] = "quantile（同估计器比较）"
        notes.append("盲信噪比非双可靠，输入/输出统一使用能量分位估计器比较")
    elif hf_in is not None and hf_out is not None:
        snr_in, snr_out = float(hf_in), float(hf_out)
        extra["snr_gain_estimator"] = "hf（同估计器比较）"
        notes.append("盲信噪比非双可靠，输入/输出统一使用高频段外推估计器比较")
    else:
        extra["snr_gain_estimator"] = "fused（估计器不一致，仅供参照）"
        notes.append("输入/输出没有共同可用的盲信噪比估计器，增益仅供参照")
    snr_gain = snr_out - snr_in
    extra["snr_in_reliable"] = bool(sni["reliable"])
    extra["snr_out_reliable"] = bool(sno["reliable"])
    extra["reliable"] = bool(sni["reliable"] and sno["reliable"])
    extra["snr_estimator_in"] = sni["method"]
    extra["snr_estimator_out"] = sno["method"]
    extra["blind_snr_in"] = sni["details"]
    extra["blind_snr_out"] = sno["details"]
    if not sni["reliable"] or not sno["reliable"]:
        notes.append("盲信噪比至少一路不可靠，SNR 相关指标仅供参考")

    # ---- 能量比（信号带内为主，全带另存） ----
    pa = _bandpass(a, dt, f_lo, f_hi)
    pd = _bandpass(d, dt, f_lo, f_hi)
    e_in = float(np.mean(pa.astype(np.float64) ** 2))
    e_out = float(np.mean(pd.astype(np.float64) ** 2))
    energy_ratio = None
    if e_in > 0:
        energy_ratio = float(e_out / e_in)
        extra["energy_ratio_full"] = float(np.mean(d ** 2) / max(np.mean(a ** 2), 1e-30))
    else:
        notes.append("信号带内输入能量为 0，能量比不可算")
        extra["energy_ratio_full"] = None

    # ---- 信号泄漏：被去除分量与去噪结果在信号带的局部相关 ----
    resid = a - d
    leakage, leak_p95, leak_n = _local_corr(resid, d, dt, band=band)
    extra["signal_leakage_p95"] = round(leak_p95, 4)
    extra["signal_leakage_samples"] = int(leak_n)
    if leak_n == 0:
        leakage = None
        notes.append("有效局部窗口不足，信号泄漏不可算")

    # ---- 频谱一致性（信号带振幅谱相关） ----
    # 直接用逐频点振幅谱做 Pearson 时，一根强工频线就能把相关系数拉到 0.8 以下，
    # 把合格的陷波误判成"改变了频谱"。改为 1/3 倍频程带平均后再相关：窄带干扰
    # 只影响一个带，既保留"整体谱形是否被破坏"的敏感性，又不冤枉窄带去噪。
    freqs, amp_a = _mean_spectrum(a, dt)
    _, amp_d = _mean_spectrum(d, dt)
    sel = (freqs >= f_lo) & (freqs <= f_hi)
    spectrum_corr = None
    if int(sel.sum()) >= 5:
        raw_corr = _safe_pearson(amp_a[sel], amp_d[sel])
        bands = _octave_bands(f_lo, f_hi, 3)
        amp_ba = _band_amplitudes(freqs, amp_a, bands)
        amp_bd = _band_amplitudes(freqs, amp_d, bands)
        if amp_ba.size >= 5:
            eps = max(float(np.max(amp_ba)), 1e-30) * 1e-9
            log_corr = _safe_pearson(np.log10(amp_ba + eps),
                                     np.log10(amp_bd + eps))
            amp_corr = _safe_pearson(amp_ba, amp_bd)
            # 取两者较小者：log 谱对整带被抹掉更敏感，振幅谱对整体比例更稳，
            # 两个视角都通过才算谱形保持。
            vals = [v for v in (log_corr, amp_corr) if v is not None]
            if vals:
                spectrum_corr = float(min(vals))
            extra["spectrum_corr_log_oct"] = None if log_corr is None else round(log_corr, 4)
            extra["spectrum_corr_amp_oct"] = None if amp_corr is None else round(amp_corr, 4)
            extra["spectrum_corr_raw"] = None if raw_corr is None else round(raw_corr, 4)
            extra["spectrum_corr_bands"] = int(amp_ba.size)
        else:
            spectrum_corr = raw_corr
        if spectrum_corr is None:
            notes.append("信号带振幅谱近似为零，频谱相关不可算")
    else:
        notes.append("信号带太窄，频谱相关不可算")

    # ---- 残差相干性（全带本地相关，衡量残留相干噪声） ----
    coh, coh_p95, coh_n = _local_corr(resid, d, dt, band=None)
    coherence_residual = None if coh_n == 0 else coh
    extra["coherence_residual_p95"] = round(coh_p95, 4)

    # ---- 削波 / 过冲 ----
    clip_ratio = 0.0
    peak_a = float(np.max(np.abs(a)))
    peak_d = float(np.max(np.abs(d)))
    if peak_a > 0:
        level = 0.999 * peak_a
        frac_in = float(np.mean(np.abs(a) >= level))
        frac_out = float(np.mean(np.abs(d) >= level))
        clip_ratio = float(max(0.0, frac_out - frac_in))
        extra["clip_ratio_input"] = round(frac_in, 6)
        extra["clip_level"] = round(level, 8)
        extra["peak_ratio"] = round(peak_d / peak_a, 6)
    else:
        notes.append("输入全零，削波比例不可算")

    # ---- 道级异常（坏道/条带） ----
    def _trace_rms_ratio(x: np.ndarray) -> float | None:
        rms = np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2, axis=0))
        med = float(np.median(rms))
        if not np.isfinite(med) or med <= 0:
            return None
        return round(float(np.max(rms) / med), 3)

    def _trace_window_ratio(x: np.ndarray, win: int = 64, half: int = 6) -> float | None:
        """滑窗能量的逐道局部异常：max_t(能量 / 邻道中位能量)。"""
        from scipy.ndimage import median_filter, uniform_filter1d

        a = np.asarray(x, dtype=np.float64)
        n_samples, n_tr = a.shape
        if n_tr < 3:
            return None
        win = max(8, min(int(win), n_samples))
        half = max(1, min(int(half), (n_tr - 1) // 2))
        energy = uniform_filter1d(a * a, size=win, axis=0, mode="nearest")
        local = median_filter(energy, size=(1, 2 * half + 1), mode="nearest")
        ratio = np.divide(energy, np.maximum(local, 1e-30))
        out = float(np.nanmax(ratio)) if np.isfinite(ratio).any() else None
        return None if out is None else round(out, 3)

    def _trace_resid_ratio(x: np.ndarray) -> float | None:
        """横向中值残差的逐道异常比：单道尖峰/垂直条带会远高于中位。"""
        from scipy.ndimage import median_filter

        a = np.asarray(x, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] < 3:
            return None
        lat = median_filter(a, size=(1, 3), mode="nearest")
        rms = np.sqrt(np.mean((a - lat) ** 2, axis=0))
        med = float(np.median(rms))
        if not np.isfinite(med) or med <= 0:
            return None
        return round(float(np.max(rms) / med), 3)

    extra["trace_rms_ratio_in"] = _trace_rms_ratio(a)
    extra["trace_rms_ratio_out"] = _trace_rms_ratio(d)
    extra["trace_window_ratio_in"] = _trace_window_ratio(a)
    extra["trace_window_ratio_out"] = _trace_window_ratio(d)
    extra["trace_resid_ratio_in"] = _trace_resid_ratio(a)
    extra["trace_resid_ratio_out"] = _trace_resid_ratio(d)

    # ---- 分窗局部 SNR 增益 ----
    local_gain = _local_snr_gain(a, d, dt)
    if local_gain is None:
        notes.append("数据太短，分窗局部 SNR 增益不可算")
    else:
        extra["local_snr_gain_p10_db"] = local_gain["p10"]
        extra["local_snr_gain_p90_db"] = local_gain["p90"]
        extra["local_snr_windows"] = local_gain["n"]
        local_gain = local_gain["median"]

    # ---- 相对参考（可选） ----
    ref_in = ref_out = ref_gain = None
    if reference is not None:
        r = np.asarray(reference, dtype=np.float64)
        if r.shape != a.shape:
            notes.append(f"参考数据形状 {r.shape} 与输入 {a.shape} 不一致，参考指标跳过")
        else:
            r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
            p_ref = float(np.mean(r ** 2))
            if p_ref <= 0:
                notes.append("参考数据能量为 0，参考指标跳过")
            else:
                floor = p_ref * 1e-12
                err_in = float(np.mean((a - r) ** 2))
                err_out = float(np.mean((d - r) ** 2))
                ref_in = float(10.0 * math.log10(p_ref / max(err_in, floor)))
                ref_out = float(10.0 * math.log10(p_ref / max(err_out, floor)))
                ref_gain = ref_out - ref_in
                extra["reference_power_db"] = round(_db(p_ref), 3)
                boot = _bootstrap_reference_gain(a, d, r)
                if boot is not None:
                    extra["reference_gain_bootstrap"] = boot

    if dz is not None:
        extra["dz"] = float(dz)

    # ---- 图像域视觉代理（SSIM/边缘/弱区能量） ----
    try:
        from .visual import visual_qc
        extra["visual"] = visual_qc(a, d, dt)
    except Exception as exc:  # 视觉指标失败不能影响主流程
        extra["visual"] = {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    extra["notes"] = notes
    return QualityMetrics(
        snr_in_db=float(snr_in),
        snr_out_db=float(snr_out),
        snr_gain_db=float(snr_gain),
        reference_snr_in_db=ref_in,
        reference_snr_out_db=ref_out,
        reference_gain_db=ref_gain,
        noise_before_db=float(sni["noise_db"]),
        noise_after_db=float(sno["noise_db"]),
        noise_reduction_db=float(sni["noise_db"] - sno["noise_db"]),
        energy_ratio=energy_ratio,
        signal_leakage=leakage,
        spectrum_corr=spectrum_corr,
        coherence_residual=coherence_residual,
        clip_ratio=clip_ratio,
        local_snr_gain_db=local_gain,
        estimator="盲信噪比：局部窗口能量分位（256 ms）+ 高频段噪声外推；"
                  "能量比/谱相关/泄漏在信号带 %.1f–%.1f Hz 内统计；"
                  "频谱相关为 1/3 倍频程带平均后 log 谱与线性谱相关的较小值"
                  % (f_lo, f_hi),
        extra=extra,
    )


def _window_snr(x: np.ndarray, dt: float) -> tuple[float | None, float | None]:
    """单个时间窗的两路信噪比估计 (分位路, 高频路)；不可用返回 None。

    局部窗太短，``blind_snr`` 的融合/选路会让输入与输出落到不同估计器上
    （例如带通把高频噪声压掉后输出只剩分位路），直接相减会出现 -20 dB 级
    假增益。因此这里把两路都显式算出来，由调用方强制用同一路比较。
    """
    ns = x.shape[0]
    if ns < 16:
        return None, None
    fs = 1.0 / dt
    nyq = fs / 2.0
    x64 = x.astype(np.float64)
    p_total = float(np.mean(x64 ** 2))
    if p_total <= 0:
        return None, None

    # 分位路
    q_snr = None
    win = int(round(WIN_S / dt))
    win = max(8, min(win, ns // 2))
    pw = _frame_power(x, win)
    if pw.size >= 2:
        p10 = float(np.percentile(pw, 10.0))
        p_sig = max(p_total - p10, 1e-30)
        q_snr = _db(p_sig / max(p10, 1e-30))

    # 高频外推路
    hf_snr = None
    spec = np.fft.rfft(x, axis=0)
    psd = np.mean(np.abs(spec) ** 2, axis=1) / float(fs)
    freqs = np.fft.rfftfreq(ns, d=dt)
    sel = (freqs >= _hif_lo(dt)) & (freqs <= 0.95 * nyq)
    if int(sel.sum()) >= 3 and psd[sel].sum() > 0:
        dens = float(np.median(psd[sel]))
        p_noise = dens * fs / float(ns)
        if p_noise > 0:
            hf_snr = _db(max(p_total - p_noise, 1e-30) / p_noise)
    return q_snr, hf_snr


def _local_snr_gain(a: np.ndarray, d: np.ndarray, dt: float,
                    max_windows: int = 24) -> dict | None:
    """在时间窗上估计盲信噪比，取输出减输入增益的中位数（抗个别坏窗）。

    优先用两窗都存在的同一路估计器（分位路优先，其次高频路）逐个相减，
    避免输入/输出因高频成分变化而切换估计器造成的假增益。单个窗的增益
    限制在 [-40, 40] dB，防止短窗统计噪声放大成极端值。
    """
    ns = a.shape[0]
    win = int(round(WIN_S / dt))
    win = max(16, min(win, max(16, ns)))
    if ns < 2 * win:
        return None
    starts = list(range(0, ns - win + 1, max(1, win // 2)))
    if len(starts) > max_windows:
        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)
        starts = [starts[i] for i in idx]
    gains_q: list[float] = []
    gains_hf: list[float] = []
    try:
        for s in starts:
            qx, hx = _window_snr(a[s:s + win], dt)
            qy, hy = _window_snr(d[s:s + win], dt)
            if qx is not None and qy is not None:
                gains_q.append(float(qy - qx))
            if hx is not None and hy is not None:
                gains_hf.append(float(hy - hx))
    except (ValueError, FloatingPointError):
        return None
    if gains_q:
        gains, used = gains_q, "quantile"
    elif gains_hf:
        gains, used = gains_hf, "hf"
    else:
        return None
    arr = np.clip(np.asarray(gains, dtype=np.float64), -40.0, 40.0)
    return {
        "median": float(np.median(arr)),
        "p10": round(float(np.percentile(arr, 10.0)), 3),
        "p90": round(float(np.percentile(arr, 90.0)), 3),
        "n": int(arr.size),
        "estimator": used,
    }
