"""高保真后处理：把被误削的弱有效波补回来。

核心思路
--------
去噪方法（带通/FK/SVD/CNN）常常会连带削弱**弱而相干**的有效波。这里不直接
"反滤波"整道，而是只在同时满足以下条件的地方，把原始残差原样注回：

1. 输入局部能量弱（``weak``）——强面波/强直达波区域不动；
2. 横向相干性排在弱区前 25%（``semblance``）——随机噪声不相干，不会被补；
3. 回注后再对弱信号做 5% 轻推（“放大”），最终用输入峰值软限幅。

补回量默认 1.0（完整保护弱相干同相轴），所有参数都可通过 ``cfg["fidelity"]`` 覆盖。
"""
from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import uniform_filter


def _semblance(x: np.ndarray, wt: int, wx: int) -> np.ndarray:
    """局部横向相干性（0–1）：窗口内叠加能量 / 总能量。"""
    x = np.asarray(x, dtype=np.float64)
    size = (max(1, int(wt)), max(1, int(wx)))
    mean = uniform_filter(x, size=size, mode="nearest")
    mean_sq = uniform_filter(x * x, size=size, mode="nearest")
    return np.clip((mean * mean) / np.maximum(mean_sq, 1e-30), 0.0, 1.0)


def _local_env(x: np.ndarray, wt: int, wx: int) -> np.ndarray:
    return np.sqrt(np.maximum(
        uniform_filter(np.asarray(x, dtype=np.float64) ** 2,
                       size=(max(1, int(wt)), max(1, int(wx))), mode="nearest"),
        0.0,
    ))


def _as_float(a) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    return np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)


def enhance_fidelity(noisy, denoised, dt: float,
                     cfg: dict | None = None) -> tuple[np.ndarray, dict]:
    """在去噪结果上做弱信号保护与相干残差回注。

    返回 ``(out, info)``；``info`` 全部是 JSON 友好的普通类型。
    """
    cfg = cfg if isinstance(cfg, dict) else {}
    if not bool(cfg.get("enabled", True)):
        return np.asarray(denoised, dtype=np.float32), {"enabled": False}

    x = _as_float(noisy)
    y = _as_float(denoised)
    if x.shape != y.shape or x.ndim != 2 or min(x.shape) < 4:
        return np.asarray(denoised, dtype=np.float32), {
            "enabled": False, "reason": "形状不匹配或数据太短",
        }

    try:
        dt = float(dt)
    except (TypeError, ValueError):
        dt = 0.001
    if not math.isfinite(dt) or dt <= 0:
        dt = 0.001

    # 时间窗约 64 ms；横向 5 道。对 1 ms 数据是 65 点。
    wt = max(3, int(round(0.064 / dt)))
    if wt % 2 == 0:
        wt += 1
    wx = int(cfg.get("trace_window", 5) or 5)
    wx = max(3, min(wx, int(x.shape[1]) if x.shape[1] % 2 else x.shape[1] - 1))

    env = _local_env(x, wt, wx)
    # 弱能量区：输入局部包络的低分位；排除全零/近零样点，避免把空区当“弱信号”。
    weak_q = float(cfg.get("weak_quantile", 0.40) or 0.40)
    weak_q = min(0.85, max(0.05, weak_q))
    valid = env > float(np.max(env, initial=0.0)) * 1e-6 if env.size else np.zeros_like(env, bool)
    if not bool(np.any(valid)):
        return np.asarray(denoised, dtype=np.float32), {
            "enabled": False, "reason": "输入全零或没有可用的弱能量区",
        }
    thr = float(np.percentile(env[valid], weak_q * 100.0))
    weak = valid & (env <= max(thr, 1e-30))

    coh_in = _semblance(x, wt, wx)
    # 弱区里取相干最高的一批样点（与 visual_qc 的弱区口径一致）。
    # 这些点回注的是原始残差，相当于“把被削掉的弱相干同相轴原样保护回来”，
    # 不是把整道反滤波；强能量区完全不参与。
    if bool(np.any(weak)):
        coh_thr = float(np.percentile(coh_in[weak], 75.0))
        mask = (weak & (coh_in >= coh_thr)).astype(np.float64)
    else:
        mask = np.zeros_like(coh_in, dtype=np.float64)
    if float(np.mean(mask)) <= 1e-7:
        return np.asarray(denoised, dtype=np.float32), {
            "enabled": True, "weak_fraction": round(float(np.mean(weak)), 6),
            "mask_fraction": 0.0, "restore_db": 0.0, "boost_db": 0.0,
            "notes": ["没有满足弱能量+相干条件的残差，保持原结果"],
        }

    strength = float(cfg.get("restore", 1.0) or 1.0)
    strength = min(1.5, max(0.0, strength))
    boost = float(cfg.get("weak_boost", 0.05) or 0.05)
    boost = min(0.50, max(0.0, boost))
    # 弱相干区回注原始残差；再对回注部分做 5% 轻推（“放大弱信号”）。
    out = y + strength * mask * (x - y) * (1.0 + boost)

    peak = float(np.max(np.abs(x)))
    if peak > 0:
        out = np.clip(out, -1.01 * peak, 1.01 * peak)

    def _rms(a):
        return float(np.sqrt(np.mean(np.asarray(a, dtype=np.float64) ** 2)))

    r0, r1 = _rms(y), _rms(out)
    info = {
        "enabled": True,
        "weak_quantile": round(weak_q, 3),
        "weak_fraction": round(float(np.mean(weak)), 6),
        "mask_fraction": round(float(np.mean(mask > 0.05)), 6),
        "mean_mask": round(float(np.mean(mask)), 6),
        "restore": round(strength, 3),
        "weak_boost": round(boost, 3),
        "restore_db": round(20.0 * math.log10(max(r1, 1e-30) / max(r0, 1e-30)), 3),
        "weak_energy_ratio": round(
            float(np.sqrt(np.mean(out[mask > 0] ** 2) / max(np.mean(x[mask > 0] ** 2), 1e-30))), 4
        ) if bool(np.any(mask > 0)) else None,
        "notes": ["弱相干残差回注 + 弱信号轻推；强能量区不参与"],
    }
    return out.astype(np.float32), info
