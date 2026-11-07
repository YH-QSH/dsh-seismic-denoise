"""读取阶段：读任意格式 → 缓存 → 道头表/图 → 数据集信息 JSON。"""
from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..model import SeismicDataset, SourceAssessment
from ..utils import Logger, ensure_dir, write_json
from .source import match_source

READ_DIR = "02_读取与诊断"


def _copy_original(input_path: Path, job_dir: Path, log: Callable[[str], None]) -> Path:
    dst = job_dir / "01_原始数据" / f"原始数据{input_path.suffix.lower()}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return dst
    try:
        shutil.copy2(input_path, dst)
        log(f"原始数据已复制：{dst.name}（{input_path.stat().st_size / 1e6:.1f} MB）")
    except Exception as e:
        log(f"复制原始数据失败（{e}），改为硬链接")
        try:
            import os
            os.link(input_path, dst)
        except Exception as e2:
            log(f"硬链接也失败（{e2}），成果文件夹将引用原路径")
            return input_path
    return dst


def estimate_dz_from_headers(ds: SeismicDataset, log: Callable[[str], None] | None = None) -> float | None:
    """兜底估计道间距：SEG-Y 里常见的 group_x / group_y / cdp_x 坐标差。

    只在数据集没有 dz 时使用；坐标单位按"米"处理，若中位差 >100 则当作厘米。
    """
    if ds.dz is not None:
        return float(ds.dz)
    for key in ("group_x", "group_y", "cdp_x", "cdp_y"):
        vals = ds.headers.values(key)
        if vals is None or vals.size < 3:
            continue
        v = np.asarray(vals, dtype=np.float64)
        diffs = np.abs(np.diff(v))
        diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
        if diffs.size < max(3, 0.5 * (v.size - 1)):
            continue
        step = float(np.median(diffs))
        if step > 100.0:
            step /= 100.0
        if 0.2 <= step <= 200.0:
            ds.dz = step
            ds.meta["dz_source"] = f"道头 {key} 坐标差中位数"
            ds.warnings = [w for w in ds.warnings if "无法从道头高程" not in str(w)]
            ds.warnings.append(f"道头未给 dz，已按 {key} 坐标差估计 dz={step:.3f} m")
            if log:
                log(f"道头未给道间距，按 {key} 估计 dz={step:.3f} m")
            return step
    return None


def _window_trace_anomaly(data: np.ndarray, win: int = 64, half: int = 6) -> np.ndarray:
    """逐道滑窗能量的局部异常比：max_t(能量 / 邻近道中位能量)。

    整道 RMS 会被强能量段的长尾稀释；坏道/垂直条带更可靠的特征是
    “某一小段时窗内能量远高于左右邻道”。返回每道的最大异常比。
    """
    from scipy.ndimage import median_filter, uniform_filter1d

    x = np.asarray(data, dtype=np.float64)
    n_samples, n_tr = x.shape
    if n_tr < 3:
        return np.zeros(n_tr, dtype=np.float64)
    win = max(8, min(int(win), n_samples))
    half = max(1, min(int(half), (n_tr - 1) // 2))
    energy = uniform_filter1d(x * x, size=win, axis=0, mode="nearest")
    local = median_filter(energy, size=(1, 2 * half + 1), mode="nearest")
    ratio = np.divide(energy, np.maximum(local, 1e-30))
    out = ratio.max(axis=0)
    out[~np.isfinite(out)] = 0.0
    return out


def repair_bad_traces(ds: SeismicDataset, ratio: float = 50.0,
                      max_fraction: float = 0.05,
                      log: Callable[[str], None] | None = None) -> list[int]:
    """检测并修复能量异常的道（坏道/强噪声道），返回 1-based 道号列表。

    同时看两种证据：整道 RMS 远高于中位数，或滑窗能量远高于邻道。
    连续坏道用邻近好道的中值重建；原始文件已在 01_原始数据 保留，
    修复记录写进数据集信息，不静默改数据。
    """
    log = log or Logger()
    data = np.asarray(ds.data)
    if data.ndim != 2 or data.shape[1] < 5:
        return []
    rms = np.sqrt(np.mean(data.astype(np.float64) ** 2, axis=0))
    med = float(np.median(rms))
    if not np.isfinite(med) or med <= 0:
        return []
    ntr = int(data.shape[1])
    # 近炮检道本身能量就高，不能只看全局中位数；再和局部中位数比一次，
    # 避免把整段高能量区（震源附近）误当坏道批量替换。
    from scipy.ndimage import median_filter
    local = median_filter(rms, size=min(11, ntr), mode="nearest")
    win_ratio = _window_trace_anomaly(data)
    # 横向中值残差：相干同相轴会被中值保留，单道尖峰/坏道残差极大。
    # 用于识别“整道 RMS 未到 50×、但相对邻道明显异常”的道（如 147/151）。
    lateral = median_filter(data.astype(np.float64), size=(1, 3), mode="nearest")
    resid_rms = np.sqrt(np.mean((data.astype(np.float64) - lateral) ** 2, axis=0))
    med_resid = float(np.median(resid_rms))
    resid_ratio = resid_rms / max(med_resid, 1e-30)
    bad = np.flatnonzero(
        ((rms > ratio * med) & (rms > 10.0 * np.maximum(local, 1e-30)))
        | ((rms > 10.0 * med) & (win_ratio > 20.0))
        | (resid_ratio > 200.0)
    )
    if bad.size == 0:
        return []
    if bad.size > max(1, int(max_fraction * ntr)):
        log(f"检测到 {bad.size} 个高能量道（>{ratio:g}×中位 RMS），超过 {max_fraction:.0%} 上限，暂不自动修复")
        return []
    good = np.setdiff1d(np.arange(ntr), bad)
    if good.size < 2:
        return []
    repaired: list[int] = []
    for i in bad:
        nearest = good[np.argsort(np.abs(good - i))[:8]]
        ds.data[:, int(i)] = np.median(ds.data[:, nearest], axis=1).astype(ds.data.dtype)
        repaired.append(int(i) + 1)
    ds.meta["preprocess"] = {
        "bad_traces_repaired": repaired,
        "method": f"整道 RMS/滑窗能量/横向中值残差异常；邻近好道中值重建（RMS>{ratio:g}×中位，滑窗>20×邻道，残差>200×中位）",
    }
    ds.warnings.append(f"已修复异常道：{repaired}")
    log(f"已修复异常道（1-based）：{repaired}")
    return repaired


def _read_with_fallback(path: Path, max_traces: int | None, log: Callable[[str], None]) -> SeismicDataset:
    try:
        from ..formats import read_any
        return read_any(path, max_traces=max_traces)
    except Exception as e:
        log(f"统一读取器失败：{e}；尝试旧版 vsdn2 读取器兜底")
        import sys
        assets = Path(__file__).resolve().parents[1] / "assets" / "cnnv2"
        sys.path.insert(0, str(assets))
        from vsdn2 import sgy  # type: ignore
        data, meta = sgy.read_segy(str(path), max_traces=max_traces)
        headers = sgy.read_trace_headers(str(path))
        from ..model import HeaderTable
        ht = HeaderTable()
        spec = {
            "trace_seq": ("道序号", 1, 4, "int32"),
            "receiver_elev": ("接收点高程", 41, 4, "int32"),
            "source_x": ("震源X", 73, 4, "int32"),
            "source_y": ("震源Y", 77, 4, "int32"),
            "ns": ("采样点数", 115, 2, "int16"),
            "dt": ("采样间隔μs", 117, 2, "int16"),
            "cdp_x": ("CDP X", 181, 4, "int32"),
            "cdp_y": ("CDP Y", 185, 4, "int32"),
        }
        for k, vals in headers.items():
            if k.startswith("_"):
                continue
            name, byte, size, kind = spec.get(k, (k, 1, 4, "int32"))
            ht.add(k, name, byte, size, kind, vals)
        dz = None
        try:
            dz = sgy.estimate_dz(str(path))
        except Exception:
            pass
        return SeismicDataset(
            path=str(path), format="segy", data=np.asarray(data, dtype=np.float32),
            dt=(meta.dt_us / 1e6 if getattr(meta, "dt_us", None) else None), dz=dz,
            headers=ht, text_header=getattr(meta, "text", "") or "",
            meta={"fallback": "vsdn2"}, confidence=0.8,
            warnings=["使用旧版 vsdn2 读取器兜底"],
        )


def write_header_csv(ds: SeismicDataset, path: Path) -> Path:
    rows = ds.headers.to_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("无道头字段\n", encoding="utf-8-sig")
        return path
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def _try_fig(fn, *args, **kwargs) -> Path | None:
    try:
        p = fn(*args, **kwargs)
        return Path(p) if p else None
    except Exception:
        return None


def read_stage(input_path: str | Path, job_dir: str | Path,
               max_traces: int | None = None,
               log: Callable[[str], None] | None = None) -> dict[str, Any]:
    log = log or Logger()
    input_path = Path(input_path)
    job_dir = Path(job_dir)
    ensure_dir(job_dir / READ_DIR)
    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件：{input_path}")

    log(f"读取文件：{input_path}")
    ds = _read_with_fallback(input_path, max_traces, log)
    log(f"读取完成：{ds.format}，{ds.n_traces} 道 × {ds.n_samples} 点，"
        f"dt={None if ds.dt is None else ds.dt * 1000:.4f}ms，"
        f"dz={None if ds.dz is None else round(ds.dz, 3)}m，置信度 {ds.confidence:.2f}")

    original = _copy_original(input_path, job_dir, log)
    estimate_dz_from_headers(ds, log=log)
    repaired: list[int] = []
    for _ in range(2):
        part = repair_bad_traces(ds, log=log)
        if not part:
            break
        repaired.extend(part)
    if repaired:
        repaired = sorted(set(repaired))
        if ds.meta.get("preprocess"):
            ds.meta["preprocess"]["bad_traces_repaired"] = repaired
    ds.save_cache(job_dir / READ_DIR)
    info = ds.to_info(include_headers=True)
    info["original_in_job"] = str(original)
    if ds.meta.get("dz_source"):
        info["dz_source"] = ds.meta["dz_source"]
    if repaired:
        info["preprocess"] = dict(ds.meta.get("preprocess") or {})
    # 缓存里的道头原值（save_cache 写的那份 JSON 会被这里覆盖，必须补回）
    info["_headers_raw"] = ds.headers.to_dict()
    write_json(job_dir / READ_DIR / "数据集信息.json", info)
    write_header_csv(ds, job_dir / READ_DIR / "道头表.csv")

    artifacts: dict[str, str] = {}
    figs = job_dir / READ_DIR
    try:
        from ..reporting.figures import (
            fig_data_overview, fig_fk, fig_header_analysis, fig_spectrum,
        )
        for key, fn, args in [
            ("overview", fig_data_overview, (ds.data, ds.dt, ds.dz, figs / "数据总览.png")),
            ("spectrum", fig_spectrum, (ds.data, ds.dt, figs / "频率谱.png")),
            ("fk", fig_fk, (ds.data, ds.dt, ds.dz, figs / "fk谱.png")),
            ("headers", fig_header_analysis, (ds, figs / "道头分析.png")),
        ]:
            p = _try_fig(fn, *args)
            if p:
                artifacts[key] = str(p)
            else:
                log(f"图 {key} 生成失败，已跳过")
    except Exception as e:
        log(f"绘图模块不可用（{e}），读取阶段先不生成图")

    msg = (
        f"已读取 {ds.n_traces} 道 × {ds.n_samples} 点，格式 {ds.format}，"
        f"读取置信度 {ds.confidence:.0%}；请查看道头表与数据总览图。"
    )
    if repaired:
        msg += f" 另已修复 {len(repaired)} 个异常能量道：{repaired}。"
    return {
        "stage": "read",
        "dataset": info,
        "repairs": repaired,
        "artifacts": artifacts,
        "message": msg,
    }


def load_dataset(job_dir: str | Path) -> SeismicDataset:
    job_dir = Path(job_dir)
    ds = SeismicDataset.load_cache(job_dir / READ_DIR)
    if Path(ds.path).exists() is False:
        # 缓存里 path 可能指向原文件；原文件移走也不影响后续处理
        pass
    return ds


def assess_stage(job_dir: str | Path, cfg: dict | None = None,
                 log: Callable[[str], None] | None = None) -> dict[str, Any]:
    log = log or Logger()
    job_dir = Path(job_dir)
    cfg = cfg or {}
    ds = load_dataset(job_dir)
    log("开始来源匹配与数据诊断")
    assessment = match_source(ds, cfg)
    write_json(job_dir / READ_DIR / "来源匹配.json", assessment)
    fig = None
    try:
        from ..reporting.figures import fig_source_assessment
        fig = _try_fig(fig_source_assessment, ds, assessment,
                       job_dir / READ_DIR / "来源匹配图.png")
    except Exception as e:
        log(f"来源匹配图生成失败：{e}")

    if assessment.needs_denoise:
        msg = (
            f"来源判断：{assessment.domain} / {assessment.acquisition}（置信度 "
            f"{assessment.confidence:.0%}）；主噪声：{max(assessment.noise_types, key=assessment.noise_types.get)}，"
            f"建议先试：{'、'.join(assessment.recommended[:3])}。需要去噪。"
        )
    else:
        msg = (
            f"来源判断：{assessment.domain} / {assessment.acquisition}（置信度 "
            f"{assessment.confidence:.0%}）；数据质量已较高，建议不去噪。"
        )
    log(msg)
    return {
        "stage": "assess",
        "source": assessment.to_info(),
        "needs_denoise": assessment.needs_denoise,
        "artifacts": {k: v for k, v in {"source_figure": None if fig is None else str(fig)}.items() if v},
        "message": msg,
    }
