"""多类噪声合成：每类噪声单独成场，可做分量监督。

几何支持两种（各占一半）：
  VSP    —— 接收点沿井深排列，直达波 + 上行反射 + 管波
  地面   —— 炮集，反射双曲线 + 面波扇形 + 折射初至

输出的 5 个噪声分量与 config.类别 一一对应。
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np

from .config import K, 合成参数


# ---------------------------------------------------------------- 基础
def 雷克(f0: float, dt: float, length: float | None = None) -> np.ndarray:
    if length is None:
        length = 1.6 / max(f0, 1e-6)
    n = max(int(round(length / dt)), 8)
    t = (np.arange(n) - (n - 1) / 2.0) * dt
    w = (1.0 - 2.0 * (np.pi * f0 * t) ** 2) * np.exp(-(np.pi * f0 * t) ** 2)
    w = w - w.mean()
    m = np.max(np.abs(w))
    return (w / m if m > 0 else w).astype(np.float32)


def 叠事件(data: np.ndarray, t0: np.ndarray, dt: float,
           w: np.ndarray, amp=1.0, gain=None) -> None:
    """按每道到达时间 t0 叠加子波；gain 可为逐道振幅数组。"""
    nt, nz = data.shape
    nw = w.size
    idx = np.rint(np.asarray(t0, np.float64) / dt).astype(np.int64)
    idx = np.broadcast_to(idx, (nz,))
    cols = np.arange(nz)
    g = np.ones(nz, np.float32) if gain is None else np.asarray(gain, np.float32)
    for j in range(nw):
        ii = idx + j
        ok = (ii >= 0) & (ii < nt)
        if not ok.any():
            continue
        data[ii[ok], cols[ok]] += float(w[j]) * amp * g[ok]


def 叠双曲线(data: np.ndarray, xc: int, t0: float, amp: float, dt: float,
            dx: float, v: float, w: np.ndarray, 孔径: int) -> None:
    nt, nz = data.shape
    nw = w.size
    for ix in range(max(0, xc - 孔径), min(nz, xc + 孔径 + 1)):
        x = (ix - xc) * dx
        t = float(np.sqrt(t0 * t0 + (x / v) ** 2))
        i = int(round(t / dt))
        if 0 <= i and i + nw <= nt:
            data[i:i + nw, ix] += amp * w


def 带限噪声(rng, nt, nz, dt, fc) -> np.ndarray:
    b = rng.standard_normal((nt, nz))
    B = np.fft.rfft(b, axis=0)
    f = np.fft.rfftfreq(nt, d=dt)
    nyq = 1.0 / (2.0 * dt)
    cut = fc * nyq
    taper = np.clip((cut - f) / (0.25 * cut + 1e-12), 0.0, 1.0)
    taper = 0.5 - 0.5 * np.cos(np.pi * taper)
    B *= taper[:, None]
    out = np.fft.irfft(B, n=nt, axis=0)
    s = np.sqrt(np.mean(out ** 2)) + 1e-12
    return out / s


def _u(rng, rngt, int_=False):
    lo, hi = rngt
    return int(rng.integers(int(lo), int(hi) + 1)) if int_ else float(rng.uniform(lo, hi))


def _面源VSP(rng, cfg, nt, nz, dt, dz):
    """复刻 MATLAB 里 shuju*.m 那一族：面源 VSP + 曲面层 + 低速面波 + 强衰减 + 弱噪声。

    和另外两种几何的关键差别是 dz 很大（20–30 m），同相轴每道移动几十个采样点，
    而且信号本身很弱——旧模型正是把这类数据整体当成"面波"抹掉的。
    """
    z = np.arange(nz) * dz
    k = np.arange(nz)
    v1 = _u(rng, cfg.面源_v1_range)
    f0 = _u(rng, cfg.f0_range)
    w = 雷克(f0, dt)
    decay = _u(rng, cfg.面源_衰减)

    src = (nz - 1) / 2.0 + rng.uniform(-4, 4)
    off = np.abs(k - src) * cfg.面源_源道距

    clean = np.zeros((nt, nz))
    parts = [np.zeros((nt, nz)) for _ in range(K)]

    # 直达波（面源双曲线）
    叠事件(clean, np.sqrt(z ** 2 + off ** 2) / v1, dt, w, 1.2)

    # 曲面层反射
    for ir in range(_u(rng, cfg.n_layers_range, True)):
        zr = float(rng.uniform(*cfg.面源_层深_range))
        vr = v1 * (1.0 + 0.3 * (ir + 1))
        curve = float(rng.uniform(*cfg.面源_起伏_amp)) * np.sin(
            2 * np.pi * k / nz * (0.6 + ir * 0.15))
        zrc = zr + curve
        tr = 2.0 * np.sqrt(zrc ** 2 + off ** 2) / vr + z / vr
        amp = _u(rng, cfg.layer_amp_range) * np.exp(-ir * 0.28)
        叠事件(clean, tr, dt, w, amp)

    # 低速面波（扇形）
    vg = _u(rng, cfg.面源_面波v)
    fg = _u(rng, cfg.面源_面波f)
    wg = 雷克(fg, dt, length=2.2 / max(fg, 1e-6))
    叠事件(parts[0], 0.12 + off / vg, dt, wg, _u(rng, cfg.面源_面波amp))

    # 散射点
    lo, hi = cfg.散射_amp_range
    for _ in range(_u(rng, cfg.散射数_range, True)):
        xc = int(rng.integers(1, nz - 1))
        t0 = float(rng.uniform(0.05, 0.9)) * nt * dt
        amp = float(rng.uniform(lo, hi)) * float(rng.choice([-1.0, 1.0]))
        叠双曲线(parts[3], xc, t0, amp, dt, dz, v1 * 0.6, w,
                 int(rng.integers(3, 9)))

    # 强衰减
    dec = np.exp(-decay * (np.arange(nt) * dt))[:, None]
    clean *= dec
    for i in (0, 3):
        parts[i] *= dec

    # 弱随机噪声：目标 SNR 18~40 dB（这类数据本来就很干净）
    e_sig = float(np.sum(clean ** 2)) + 1e-20
    e_noise = e_sig / (10.0 ** (_u(rng, (18.0, 40.0)) / 10.0))
    white = rng.standard_normal((nt, nz))
    white /= np.sqrt(np.mean(white ** 2)) + 1e-12
    white *= np.sqrt(0.7 * e_noise / white.size)
    band = 带限噪声(rng, nt, nz, dt, _u(rng, cfg.带限截止))
    band *= np.sqrt(0.3 * e_noise / max(band.size, 1))
    parts[1] = white + band

    noisy = clean.copy()
    for p in parts:
        noisy += p

    e_tot = float(np.sum((noisy - clean) ** 2)) + 1e-20
    frac = np.array([float(np.sum(p ** 2)) / e_tot for p in parts], np.float32)
    return {
        "clean": clean.astype(np.float32),
        "noisy": noisy.astype(np.float32),
        "parts": np.stack([p.astype(np.float32) for p in parts]),
        "snr_in_db": float(10.0 * np.log10(e_sig / e_tot)),
        "label": (frac > 0.05).astype(np.float32),
        "frac": frac,
        "geo": "面源VSP",
        "dz": float(dz),
        "vp": float(v1),
        "f0": float(f0),
    }


# ---------------------------------------------------------------- 主函数
def 生成(cfg: 合成参数, seed: int) -> Dict[str, Any]:
    """随机生成一条剖面。

    采样间隔 dt **不是随便取的**：先抽道间距 dz 和几何，估算这张图最晚的同相轴
    什么时候到，再反推 dt，保证记录窗（nt*dt）装得下。否则 dt 抽小了，
    深部反射和远偏移的面波会整个掉出记录，生成出全零的"干净波场"。
    """
    rng = np.random.default_rng(seed)
    nt, nz = cfg.nt, cfg.nz
    dt = cfg.dt                       # 训练固定在规范网格（2 ms / 5 m）
    dz = _u(rng, cfg.dz_range)
    vp = _u(rng, cfg.vp_range)

    u = rng.random()
    面源 = u < cfg.面源VSP概率
    地面 = (not 面源) and (u < cfg.面源VSP概率 + cfg.地面概率)

    if 面源:
        out = _面源VSP(rng, cfg, nt, nz, dt, dz)
        out["dt"] = dt
        return out

    x = np.arange(nz) * dz
    t = np.arange(nt) * dt

    vp = _u(rng, cfg.vp_range)
    f0 = _u(rng, cfg.f0_range)
    w = 雷克(f0, dt)
    decay = _u(rng, cfg.decay_range)

    clean = np.zeros((nt, nz), np.float64)
    parts = [np.zeros((nt, nz), np.float64) for _ in range(K)]  # 面波 随机 工频 散射 折射

    # ---------------- 有效波 ----------------
    n_lay = _u(rng, cfg.n_layers_range, True)
    if 地面:
        # 炮集：反射双曲线
        for k in range(n_lay):
            zr = float(rng.uniform(0.15, 1.0)) * dz * nz * 0.5
            t0 = 2.0 * zr / vp
            amp = _u(rng, cfg.layer_amp_range) * np.exp(-0.3 * k)
            for ix in range(nz):
                tt = float(np.sqrt(t0 * t0 + (x[ix] / vp) ** 2))
                i = int(round(tt / dt))
                if 0 <= i and i + w.size <= nt:
                    clean[i:i + w.size, ix] += amp * w * (1.0 / (1.0 + x[ix] / 1500.0))
        # 初至直达（很弱，只是让图上有东西可循）
        叠事件(clean, x / vp, dt, w, 0.15)
    else:
        # VSP：下行直达 + 上行反射（收敛成 V 形）
        叠事件(clean, x / vp, dt, w, 1.0)
        for k in range(n_lay):
            zr = float(rng.uniform(0.15, 1.0)) * dz * (nz - 1)
            m = x < zr
            if not m.any():
                continue
            tmp = np.zeros_like(clean)
            叠事件(tmp, (2.0 * zr - x) / vp, dt, w, _u(rng, cfg.layer_amp_range) * np.exp(-0.3 * k))
            clean += tmp * m[None, :]

    clean *= (1.0 / (1.0 + x / cfg.spreading_z0))[None, :]
    clean *= np.exp(-decay * t)[:, None]

    # ---------------- 0 面波 / 管波 ----------------
    if rng.random() < cfg.面波_存在概率:
        vg = _u(rng, cfg.面波_v_range)
        fg = max(cfg.面波_f_lo, f0 * _u(rng, cfg.面波_f_ratio))
        wg = 雷克(fg, dt, length=2.2 / max(fg, 1e-6))
        amp = _u(rng, cfg.面波_amp_range)
        if 地面:
            tau = x / vg + float(rng.uniform(0.0, 0.08))
            gain = 1.0 / (1.0 + x / 800.0)     # 面波随偏移距衰减
        else:
            tau = x / vg
            gain = None
        叠事件(parts[0], tau, dt, wg, amp, gain)

    # ---------------- 3 散射 / 绕射 ----------------
    lo, hi = cfg.散射_amp_range
    for _ in range(_u(rng, cfg.散射数_range, True)):
        xc = int(rng.integers(1, nz - 1))
        t0 = float(rng.uniform(0.05, 0.9)) * nt * dt
        amp = float(rng.uniform(lo, hi)) * float(rng.choice([-1.0, 1.0]))
        ap = int(rng.integers(cfg.散射_孔径[0], cfg.散射_孔径[1] + 1))
        v = vp * float(rng.uniform(0.25, 0.75))
        叠双曲线(parts[3], xc, t0, amp, dt, dz, v, w, ap)

    # ---------------- 4 折射波（仅地面） ----------------
    if 地面 and rng.random() < cfg.折射_存在概率:
        v2 = _u(rng, cfg.折射_v_range)
        t0 = float(rng.uniform(0.02, 0.15))
        amp = _u(rng, cfg.折射_amp_range)
        w2 = 雷克(f0 * 0.6, dt)
        叠事件(parts[4], t0 + x / v2, dt, w2, amp, 1.0 / (1.0 + x / 2000.0))

    # ---------------- 2 工频 ----------------
    if rng.random() < cfg.工频_存在概率:
        fm = _u(rng, cfg.工频_f_range)
        ph = float(rng.uniform(0, 2 * np.pi))
        a = _u(rng, cfg.工频_amp_range)
        mono = np.sin(2 * np.pi * fm * t + ph)[:, None] * np.ones((1, nz))
        mono += 0.35 * np.sin(2 * np.pi * 2 * fm * t + 1.7 * ph)[:, None]  # 二次谐波
        parts[2] = mono / (np.sqrt(np.mean(mono ** 2)) + 1e-12) * a

    # ---------------- 1 随机 ----------------
    white = rng.standard_normal((nt, nz))
    white /= np.sqrt(np.mean(white ** 2)) + 1e-12
    band = 带限噪声(rng, nt, nz, dt, _u(rng, cfg.带限截止))
    wr = _u(rng, cfg.随机_白比)
    parts[1] = wr * white + (1 - wr) * band

    # ---------------- 按目标信噪比装配 ----------------
    e_sig = float(np.sum(clean ** 2)) + 1e-20
    e_target = e_sig / (10.0 ** (_u(rng, cfg.snr_range) / 10.0))

    # 相干类（面波/散射/折射）占 coherent_share，其余给随机+工频
    coh_idx = [0, 3, 4]
    e_coh_nat = sum(float(np.sum(parts[i] ** 2)) for i in coh_idx)
    e_coh_tgt = e_target * _u(rng, cfg.相干占比)
    if e_coh_nat > 1e-20:
        s = np.sqrt(e_coh_tgt / e_coh_nat)
        for i in coh_idx:
            parts[i] *= s
    e_rest = max(e_target - e_coh_tgt, 0.0)
    e_rand_nat = float(np.sum(parts[1] ** 2))
    e_mono_nat = float(np.sum(parts[2] ** 2))
    denom = e_rand_nat + e_mono_nat
    if denom > 1e-20:
        s = np.sqrt(e_rest / denom)
        parts[1] *= s
        parts[2] *= s

    noisy = clean.copy()
    for p in parts:
        noisy += p

    e_noise = float(np.sum((noisy - clean) ** 2)) + 1e-20
    snr_in = 10.0 * np.log10(e_sig / e_noise)

    # 分类标签：该噪声能量占噪声总能量 > 2% 记为存在
    frac = np.array([float(np.sum(p ** 2)) / e_noise for p in parts], np.float32)
    label = (frac > 0.05).astype(np.float32)

    return {
        "clean": clean.astype(np.float32),
        "noisy": noisy.astype(np.float32),
        "parts": np.stack([p.astype(np.float32) for p in parts]),  # (K, nt, nz)
        "snr_in_db": float(snr_in),
        "label": label,
        "frac": frac,
        "geo": "地面" if 地面 else "VSP",
        "dz": float(dz),
        "dt": float(dt),
        "vp": float(vp),
        "f0": float(f0),
    }
