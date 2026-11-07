"""来源匹配：从道头、文本头、几何与频谱判断数据领域，并决定是否需要去噪。"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..model import SeismicDataset, SourceAssessment
from ..utils import trace_offsets
from .diagnostics import diagnostic_summary


def _uniq(a: np.ndarray | None, tol: float = 0.0) -> int:
    if a is None:
        return 0
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return 0
    if tol > 0:
        a = np.round(a / tol) * tol
    return int(np.unique(a).size)


def _monotonic_fraction(a: np.ndarray | None) -> float:
    if a is None:
        return 0.0
    a = np.asarray(a, dtype=np.float64)
    if a.size < 3 or not np.isfinite(a).all():
        return 0.0
    d = np.diff(a)
    d = d[np.abs(d) > 1e-9]
    if d.size == 0:
        return 0.0
    return float(max(np.mean(d > 0), np.mean(d < 0)))


def _span_m(a: np.ndarray | None) -> float | None:
    if a is None:
        return None
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    span = float(a.max() - a.min())
    # 道头高程常见单位是厘米；跨度超过 1000 时按厘米处理
    if span > 1000:
        span /= 100.0
    return span


def _offset_stats(ds: SeismicDataset) -> dict:
    off = trace_offsets(ds)
    if off is None:
        return {"available": False}
    off = np.asarray(off, dtype=np.float64)
    off = off[np.isfinite(off)]
    if off.size == 0:
        return {"available": False}
    if np.median(off) > 10000:
        off = off / 100.0
    return {
        "available": True,
        "min_m": round(float(off.min()), 3),
        "max_m": round(float(off.max()), 3),
        "median_m": round(float(np.median(off)), 3),
    }


def _text(ds: SeismicDataset) -> str:
    return (ds.text_header or "").upper() + " " + Path(ds.path or "").name.upper()


def match_source(ds: SeismicDataset, cfg: dict | None = None) -> SourceAssessment:
    cfg = cfg or {}
    h = ds.headers
    text = _text(ds)
    ntr, ns = ds.n_traces, ds.n_samples
    dt_ms = None if ds.dt is None else ds.dt * 1000.0
    diag = diagnostic_summary(ds.data, ds.dt, ds.dz)
    snr_db = diag["snr"].get("snr_db")
    noise = diag["noise"]

    elev = h.values("receiver_elev")
    src_x, src_y = h.values("source_x"), h.values("source_y")
    cdp_x, cdp_y = h.values("cdp_x"), h.values("cdp_y")
    offs = _offset_stats(ds)

    elev_mono = _monotonic_fraction(elev)
    elev_span = _span_m(elev)
    n_src = max(_uniq(src_x, 1.0), _uniq(src_y, 1.0))
    n_cdp = max(_uniq(cdp_x, 1.0), _uniq(cdp_y, 1.0))
    n_src = min(n_src, ntr) if n_src else 0
    n_cdp = min(n_cdp, ntr) if n_cdp else 0

    evidence: list[str] = []
    warnings: list[str] = []
    domain = "未知"
    acquisition = "未知"
    geometry_type = "未知"
    confidence = 0.20

    # ---- 领域判别 ----
    das_kw = any(k in text for k in ("HDVS", "DAS", "FIBER", "光纤"))
    vsp_like = elev_mono > 0.75 and elev_span is not None and elev_span > 20 and n_src <= max(2, int(0.02 * ntr))
    marine_kw = any(k in text for k in ("MARINE", "STREAMER", "AIRGUN", "OCEAN", "海底", "拖缆"))
    passive_kw = any(k in text for k in ("PASSIVE", "MICROSEISMIC", "AMBIENT", "背景噪声", "微震"))
    lab_like = (dt_ms is not None and dt_ms < 0.02) or (ns < 512 and ntr < 128 and offs.get("max_m", 0) < 10)

    if das_kw or (vsp_like and ds.dz is not None and ds.dz <= 5 and ntr >= 200):
        domain = "DAS-VSP"
        acquisition = "井中 DAS 零偏移VSP" if n_src <= 1 else "井中 DAS 多偏移VSP"
        geometry_type = "井中接收阵列"
        confidence = 0.65 + (0.15 if das_kw else 0.0) + (0.1 if vsp_like else 0.0)
        evidence.append("文件名/文本头含 DAS/HDVS/光纤特征" if das_kw else "道间距很细且道数多，符合 DAS 井中观测")
        if vsp_like:
            evidence.append(
                f"接收高程随道号单调变化（{elev_mono:.0%}），跨度约 {elev_span:.1f}m → 井中接收"
            )
    elif marine_kw:
        domain = "海洋地震"
        acquisition = "海上拖缆/海底节点"
        geometry_type = "海上排列"
        confidence = 0.60
        evidence.append("文本头/文件名含 marine/streamer/airgun 等海上采集关键词")
    elif passive_kw:
        domain = "被动源/微震"
        acquisition = "连续背景噪声记录"
        geometry_type = "台阵/连续记录"
        confidence = 0.50
        evidence.append("文本头/文件名含 passive/microseismic/ambient 关键词")
    elif lab_like:
        domain = "实验室/超声"
        acquisition = "超声/小样实验"
        geometry_type = "小样/换能器阵列"
        confidence = 0.55
        evidence.append(f"采样间隔 {dt_ms}ms、{ns} 点 × {ntr} 道，符合超声/小样实验尺度")
    elif vsp_like:
        domain = "VSP"
        acquisition = "井中零偏移VSP" if n_src <= 1 else "井中多偏移VSP"
        geometry_type = "井中接收阵列"
        confidence = 0.60
        evidence.append(f"接收高程随道号单调变化（{elev_mono:.0%}），跨度约 {elev_span:.1f}m")
    elif n_src > 1 and n_cdp > 1:
        domain = "地面地震"
        acquisition = "2D/3D 地面反射" if n_cdp > 2 else "地面炮集"
        geometry_type = "地面排列"
        confidence = 0.50
        evidence.append(f"震源位置 {n_src} 个、CDP 位置 {n_cdp} 个，符合地面采集")
    else:
        domain = "未知"
        acquisition = "未知"
        geometry_type = "未知"
        confidence = 0.18
        warnings.append("道头与文本头不足以判定采集领域；将按数据本身特征选择保守去噪方案")

    if n_src == 1:
        evidence.append("source_x/source_y 只有一个位置 → 单炮/固定震源")
    elif n_src > 1:
        evidence.append(f"震源位置约 {n_src} 个")
    if offs.get("available"):
        evidence.append(
            f"偏移距 {offs['min_m']:.0f}–{offs['max_m']:.0f}m（中位 {offs['median_m']:.0f}m）"
        )
    evidence.append(
        f"{ns} 采样点 × {ntr} 道，dt={dt_ms:.3f}ms，"
        + (f"道间距≈{ds.dz:.2f}m" if ds.dz else "道间距未知")
    )
    evidence.append(f"主频约 {diag['spectral'].get('peak_hz')}Hz；噪声打分 {diag['dominant_noise']} 最高")

    # ---- 干净/需去噪 ----
    clean_snr = float(cfg.get("clean_snr_db", 28.0))
    clean_noise = float(cfg.get("clean_noise_score", 0.35))
    max_noise = max(noise.values()) if noise else 0.0
    reliable = bool(diag["snr"].get("reliable", False))
    clean_reasons: list[str] = []
    needs = True
    if snr_db is not None and reliable and snr_db >= clean_snr and max_noise < clean_noise:
        needs = False
        clean_reasons = [
            f"盲信噪比估计 {snr_db:.1f}dB ≥ 阈值 {clean_snr:.0f}dB",
            f"最高噪声类型打分 {max_noise:.2f} < {clean_noise:.2f}",
        ]
    elif snr_db is None:
        clean_reasons = ["无法可靠估计信噪比，按需要去噪处理（保守）"]

    # ---- 推荐方法 ----
    mapping = {
        "面波": ["fk", "bandpass", "median"],
        "随机噪声": ["fx_decon", "spectral_sub", "cnn"],
        "工频干扰": ["notch", "bandpass"],
        "散射绕射": ["fk", "svd", "cnn"],
        "折射波": ["fk", "median", "bandpass"],
    }
    ranked = sorted(noise.items(), key=lambda kv: kv[1], reverse=True)
    rec: list[str] = []
    for name, score in ranked:
        if score >= 0.25:
            for k in mapping.get(name, []):
                if k not in rec:
                    rec.append(k)
    if not rec:
        rec = ["bandpass", "cnn"]
    if domain in ("DAS-VSP", "VSP"):
        for k in ("fk", "cnn", "fx_decon"):
            if k not in rec:
                rec.append(k)
    rec = rec[:5]

    if not needs:
        rec = []

    return SourceAssessment(
        domain=domain,
        acquisition=acquisition,
        geometry_type=geometry_type,
        confidence=round(min(0.95, confidence), 3),
        evidence=evidence,
        geometry={
            "n_traces": ntr,
            "n_samples": ns,
            "dt_ms": None if dt_ms is None else round(dt_ms, 4),
            "dz_m": None if ds.dz is None else round(ds.dz, 4),
            "duration_s": None if ds.dt is None else round(ds.dt * ns, 4),
            "receiver_elevation_span_m": None if elev_span is None else round(elev_span, 2),
            "receiver_monotonic": round(elev_mono, 3),
            "source_positions": n_src,
            "cdp_positions": n_cdp,
            "offset": offs,
        },
        cleanliness={
            "snr_db": snr_db,
            "snr_reliable": reliable,
            "snr_method": diag["snr"].get("method"),
            "max_noise_score": round(max_noise, 3),
            "dominant_noise": diag["dominant_noise"],
            "spike_ratio": diag["spike"].get("spike_ratio"),
            "adjacent_coherence": diag["coherence"].get("adjacent_coherence"),
        },
        noise_types=noise,
        needs_denoise=needs,
        recommended=rec,
        clean_reasons=clean_reasons,
        warnings=warnings,
        notes="；".join(diag["evidence"]),
    )
