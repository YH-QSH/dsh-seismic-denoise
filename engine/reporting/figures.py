"""报告图表生成。

契约（docs/接口约定.md §6）：
    setup_style()
    fig_data_overview / fig_header_analysis / fig_spectrum / fig_fk /
    fig_source_assessment / fig_compare / fig_metrics / fig_local_zoom

实现要点：
- 强制 matplotlib Agg 后端，适合插件无人值守运行；
- 自动寻找中文字体（微软雅黑/黑体/宋体…），找不到时退回 DejaVu 并改用英文
  标签，避免中文标题变成“豆腐块”；
- 画布固定 1600×1000、DPI 120，输出前按道/时间抽稀到 ≤1200×1200；
- 所有函数返回实际写出的 Path；参数非法或写文件失败时抛异常。
"""
from __future__ import annotations

import math
import textwrap
from pathlib import Path
from typing import Any

import numpy as np

import matplotlib

# 必须在 import pyplot 之前指定无界面后端（插件里没有 GUI）
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colormaps, font_manager  # noqa: E402

# --------------------------------------------------------------------------
# 全局样式与中文字体
# --------------------------------------------------------------------------
DPI = 120
FIG_W, FIG_H = 1600, 1000
FIGSIZE = (FIG_W / DPI, FIG_H / DPI)
MAX_DIM = 1200  # 绘图前最大抽稀尺寸

SEISMIC = colormaps["seismic"]  # 地震剖面统一色标（蓝-白-红）

_CJK_CANDIDATES = (
    "Microsoft YaHei",   # 微软雅黑（Windows 默认有）
    "SimHei",            # 黑体
    "SimSun",            # 宋体
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "DengXian",
)
_CJK_CHECKED = False
_CJK_NAME: str | None = None


def _ensure_cjk() -> str | None:
    """找一个可用的中文字体，返回字体名；没有则返回 None。"""
    global _CJK_CHECKED, _CJK_NAME
    if not _CJK_CHECKED:
        try:
            available = {f.name for f in font_manager.fontManager.ttflist}
        except Exception:
            available = set()
        for cand in _CJK_CANDIDATES:
            if cand in available:
                _CJK_NAME = cand
                break
        _CJK_CHECKED = True
    return _CJK_NAME


def cjk_available() -> bool:
    """当前环境是否有中文字体（供外部判断标题语言）。"""
    return _ensure_cjk() is not None


def L(zh: str, en: str) -> str:
    """有中文字体用中文标签，否则用英文，避免出现方块。"""
    return zh if cjk_available() else en


def _title(zh: str, en: str) -> str:
    """标题：找不到中文字体时直接换英文（而不是保留中文变方块）。"""
    return zh if cjk_available() else en


def setup_style() -> None:
    """设置全局绘图样式：Agg、1600×1000、DPI 120、中文字体、负号正常显示。"""
    font = _ensure_cjk()
    plt.rcParams.update({
        "figure.figsize": FIGSIZE,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "standard",   # 固定画布尺寸，不做紧裁剪
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
        "font.size": 10.5,
        "axes.titlesize": 12.5,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "axes.unicode_minus": False,   # 中文字体下负号正常
        "figure.autolayout": False,
    })
    if font:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]


# --------------------------------------------------------------------------
# 通用工具
# --------------------------------------------------------------------------
def _check_dt(dt: float | None) -> float:
    """校验采样间隔（秒），不合法直接抛 ValueError。"""
    try:
        value = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"采样间隔 dt 无效：{dt!r}") from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"采样间隔 dt 必须为正数，实际 {dt!r}")
    return value


def _check_dz(dz: float | None) -> float | None:
    """道间距（米）：无效或缺失时返回 None，图上退化为“道号”。"""
    if dz is None:
        return None
    try:
        value = float(dz)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) and value > 0 else None


def _as_2d(data: np.ndarray) -> np.ndarray:
    """统一成 (采样点, 道数)；一维当单道，三维 squeeze。"""
    arr = np.asarray(data)
    if arr.ndim == 1:
        arr = arr[:, None]
    elif arr.ndim > 2:
        arr = np.squeeze(arr)
        if arr.ndim == 1:
            arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError(f"数据必须是二维（采样点×道数），实际 shape={np.asarray(data).shape}")
    if arr.size == 0:
        raise ValueError("数据为空，无法绘图")
    return arr


def _thin(data: np.ndarray, max_dim: int = MAX_DIM) -> tuple[np.ndarray, int, int]:
    """抽稀到 ≤ max_dim × max_dim；返回 (数组, 行步长, 列步长)。"""
    arr = _as_2d(data)
    rs = max(1, int(math.ceil(arr.shape[0] / max_dim)))
    cs = max(1, int(math.ceil(arr.shape[1] / max_dim)))
    out = arr[::rs, ::cs]
    out = np.nan_to_num(
        np.asarray(out, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    return out, rs, cs


def _thin_pair(
    a: np.ndarray, b: np.ndarray, max_dim: int = MAX_DIM
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """两张同形状剖面按同一套步长抽稀，保证逐点可比。"""
    raw_a, raw_b = np.asarray(a), np.asarray(b)
    if raw_a.shape != raw_b.shape:
        raise ValueError(f"两张数据形状不一致：{raw_a.shape} vs {raw_b.shape}")
    arr_a, rs, cs = _thin(raw_a, max_dim)
    arr_b = _thin_steps(raw_b, rs, cs)
    return arr_a, arr_b, rs, cs


def _thin_steps(data: np.ndarray, rs: int, cs: int) -> np.ndarray:
    """按给定位长抽稀（用于残差等派生数组）。"""
    arr = _as_2d(data)[::rs, ::cs]
    return np.nan_to_num(
        np.asarray(arr, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )


def _clip_level(*arrays: np.ndarray, pct: float = 99.0) -> float:
    """取幅值分位数作为统一色标范围，避免个别坏点把图压成一片白。"""
    vals = []
    for arr in arrays:
        a = np.abs(np.asarray(arr, dtype=np.float64)).ravel()
        a = a[np.isfinite(a)]
        if a.size:
            vals.append(a)
    if not vals:
        return 1e-12
    merged = np.concatenate(vals)
    level = float(np.percentile(merged, pct))
    if not math.isfinite(level) or level <= 0:
        level = float(np.max(merged)) if merged.size else 1.0
    return max(level, 1e-12)


def _extent(ns: int, ntr: int, dt: float, dz: float | None) -> list[float]:
    """imshow extent：纵轴时间向下增大，横轴距离（m）或道号。"""
    xmax = ntr * dz if dz else float(ntr)
    return [0.0, float(xmax), ns * dt, 0.0]


def _show_section(
    ax,
    arr: np.ndarray,
    dt: float,
    dz: float | None,
    title: str,
    *,
    clip: float | None = None,
    show_cbar: bool = True,
) -> Any:
    """画一张地震剖面（时间向下）。"""
    arr = _as_2d(arr)
    ns, ntr = arr.shape
    vmax = _clip_level(arr) if clip is None else float(clip)
    im = ax.imshow(
        arr,
        cmap=SEISMIC,
        aspect="auto",
        extent=_extent(ns, ntr, dt, dz),
        vmin=-vmax,
        vmax=vmax,
        interpolation="nearest",
        origin="upper",
    )
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(L("距离 (m)", "Distance (m)") if dz else L("道号", "Trace"))
    ax.set_ylabel(L("时间 (s)", "Time (s)"))
    ax.grid(False)
    if show_cbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label(L("振幅", "Amplitude"), fontsize=9)
        cb.ax.tick_params(labelsize=8)
    return im


def _save(fig, out_png: str | Path) -> Path:
    """统一保存：建目录、tight_layout、按 DPI 落盘、关闭 figure。"""
    out = Path(out_png)
    try:
        out.parent.mkdir(parents=True, exist_ok=True)
        try:
            fig.tight_layout()
        except Exception:
            # 极少数布局（如极坐标）tight_layout 会失败，不影响保存
            pass
        fig.savefig(out, dpi=DPI, facecolor="white")
    finally:
        plt.close(fig)
    return out


def _fmt_pct(value: Any) -> str:
    """概率显示：0–1 当比例，>1 当百分数。"""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(v):
        return "—"
    return f"{v * 100:.0f}%" if v <= 1.0 else f"{v:.0f}%"


_NOISE_LABELS = {
    "面波": "Surface wave",
    "随机噪声": "Random noise",
    "工频": "Powerline",
    "工频干扰": "Powerline",
    "散射绕射": "Scattering/diffraction",
    "散射": "Scattering",
    "绕射": "Diffraction",
    "折射": "Refraction",
    "异常振幅": "Spikes",
    "低频": "Low frequency",
    "高频": "High frequency",
    "线性干扰": "Linear noise",
    "多次波": "Multiples",
    "未知": "Unknown",
}


def _noise_label(name: str) -> str:
    if cjk_available():
        return str(name)
    return _NOISE_LABELS.get(str(name), "Noise")


# --------------------------------------------------------------------------
# 1. 数据总览
# --------------------------------------------------------------------------
def fig_data_overview(
    data: np.ndarray,
    dt: float | None,
    dz: float | None,
    out_png: str | Path,
    title: str = "数据总览",
) -> Path:
    """数据总览：剖面子图 + 振幅分布 + 单道 RMS 能量 + 基本统计。

    参数
    ----
    data : (ns, ntr) float 数组
    dt   : 采样间隔（秒）
    dz   : 道间距（米，可为 None）
    out_png : 输出 PNG 路径
    """
    setup_style()
    dt_v = _check_dt(dt)
    dz_v = _check_dz(dz)
    arr, rs, cs = _thin(data)
    dt_s, dz_s = dt_v * rs, (dz_v * cs if dz_v else None)
    ns, ntr = arr.shape
    fig_title = _title(title, "Data overview")

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, width_ratios=[2.05, 1.0], hspace=0.38, wspace=0.20)
    ax_main = fig.add_subplot(gs[:, 0])
    _show_section(ax_main, arr, dt_s, dz_s, L("地震剖面", "Seismic section"))

    stats = [
        f"{L('采样点', 'Samples')}: {ns}   {L('道数', 'Traces')}: {ntr}",
        f"dt = {dt_v * 1000:.3f} ms   dz = {'—' if dz_v is None else f'{dz_v:.3f} m'}",
        f"{L('振幅范围', 'Amplitude')}: {float(np.min(arr)):.3g} ~ {float(np.max(arr)):.3g}",
        f"{L('99% 幅值', 'P99 amplitude')}: {np.percentile(np.abs(arr), 99):.3g}",
    ]
    ax_main.text(
        0.012, 0.985, "\n".join(stats), transform=ax_main.transAxes,
        va="top", ha="left", fontsize=9.5, color="0.12",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85,
                  edgecolor="0.75"),
    )

    ax_hist = fig.add_subplot(gs[0, 1])
    level = _clip_level(arr)
    sample = np.clip(arr.ravel(), -level, level)
    ax_hist.hist(sample, bins=90, color="#2E75B6", alpha=0.85)
    ax_hist.set_title(L("振幅分布", "Amplitude distribution"), fontsize=11)
    ax_hist.set_xlabel(L("振幅", "Amplitude"))
    ax_hist.set_ylabel(L("样本数", "Count"))
    ax_hist.grid(alpha=0.25)
    ax_hist.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax_rms = fig.add_subplot(gs[1, 1])
    rms = np.sqrt(np.mean(np.asarray(arr, dtype=np.float64) ** 2, axis=0))
    ax_rms.plot(np.arange(ntr), rms, color="#C0504D", lw=1.1)
    ax_rms.set_title(L("单道 RMS 能量", "Per-trace RMS"), fontsize=11)
    ax_rms.set_xlabel(L("道号", "Trace"))
    ax_rms.set_ylabel(L("RMS 振幅", "RMS amplitude"))
    ax_rms.grid(alpha=0.25)

    fig.suptitle(fig_title, fontsize=15, fontweight="bold")
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 2. 道头分析
# --------------------------------------------------------------------------
_HEADER_PRIORITY = (
    "trace_seq", "cdp", "ensemble", "inline", "crossline", "offset",
    "shotpoint", "receiver_elev", "source_elev", "source_depth",
    "receiver_depth", "cdp_x", "cdp_y",
)
_HEADER_CN = {
    "trace_seq": "道序号",
    "cdp": "CDP 号",
    "ensemble": "道集号",
    "inline": "Inline",
    "crossline": "Crossline",
    "offset": "偏移距",
    "shotpoint": "炮点号",
    "receiver_elev": "检波点高程",
    "source_elev": "炮点高程",
    "source_depth": "炮点深度",
    "receiver_depth": "检波点深度",
    "cdp_x": "CDP X",
    "cdp_y": "CDP Y",
}


def _unwrap_dataset(ds: Any) -> Any:
    """兼容 SeismicDataset / 整个 state dict / state["dataset"]。"""
    if isinstance(ds, dict):
        inner = ds.get("dataset")
        if isinstance(inner, dict):
            return inner
    return ds


def _header_fields(ds: Any) -> dict[str, Any]:
    """取出道头字段字典（兼容 HeaderTable 与 to_info 后的 dict）。"""
    ds = _unwrap_dataset(ds)
    headers = getattr(ds, "headers", None)
    if headers is None and isinstance(ds, dict):
        headers = ds.get("headers")
    if headers is None:
        return {}
    if hasattr(headers, "fields"):
        return dict(getattr(headers, "fields"))
    if isinstance(headers, dict):
        return dict(headers)
    return {}


def _header_values(field: Any) -> tuple[np.ndarray, str]:
    """从 HeaderField 或 to_info dict 中取数值序列与名称。"""
    if field is None:
        return np.zeros(0), ""
    vals, name = None, ""
    if hasattr(field, "values"):
        vals = getattr(field, "values")
        name = getattr(field, "name", "")
    elif isinstance(field, dict):
        vals = field.get("values")
        if vals is None:
            vals = field.get("values_sample")
        name = str(field.get("name") or "")
    try:
        arr = np.asarray(vals, dtype=np.float64).ravel()
    except (TypeError, ValueError):
        arr = np.zeros(0)
    arr = arr[np.isfinite(arr)]
    return arr, name


def fig_header_analysis(ds: Any, out_png: str | Path) -> Path:
    """关键道头随道号的变化（最多 6 个字段）。

    没有道头 / 字段无变化时也生成一张占位图并说明原因，方便报告统一插图。
    """
    setup_style()
    fields = _header_fields(ds)
    picked: list[tuple[str, np.ndarray]] = []
    for key in _HEADER_PRIORITY:
        field = fields.get(key)
        if field is None:
            continue
        vals, name = _header_values(field)
        if vals.size < 2 or float(np.nanmax(vals)) == float(np.nanmin(vals)):
            continue
        picked.append((name or key, vals))
        if len(picked) >= 6:
            break
    # 优先级字段都不满足时，再遍历其他有变化的字段
    if not picked:
        for key, field in fields.items():
            vals, name = _header_values(field)
            if vals.size < 2 or float(np.nanmax(vals)) == float(np.nanmin(vals)):
                continue
            picked.append((name or key, vals))
            if len(picked) >= 6:
                break

    fig = plt.figure(figsize=FIGSIZE)
    fig.suptitle(_title("道头分析", "Header analysis"), fontsize=15, fontweight="bold")
    if not picked:
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax.text(
            0.5, 0.5,
            _title("未发现可用的道头字段（或字段为常数）",
                   "No varying trace-header field is available"),
            ha="center", va="center", fontsize=16, color="0.4",
            transform=ax.transAxes,
        )
        return _save(fig, out_png)

    n = len(picked)
    ncols = 2 if n <= 2 else 3
    nrows = int(math.ceil(n / ncols))
    axes = fig.subplots(nrows, ncols, squeeze=False)
    for idx, (name, vals) in enumerate(picked):
        ax = axes[idx // ncols][idx % ncols]
        short = vals[:MAX_DIM]  # 道头只画前 1200 道，避免过密
        x = np.arange(short.size)
        ax.plot(x, short, color="#1F4E79", lw=1.0)
        zh = _HEADER_CN.get(name, name)
        ax.set_title(zh if cjk_available() else str(name), fontsize=10.5)
        ax.set_xlabel(L("道号", "Trace"))
        ax.set_ylabel(L("数值", "Value"))
        ax.grid(alpha=0.25)
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 3. 振幅谱
# --------------------------------------------------------------------------
def fig_spectrum(
    data: np.ndarray, dt: float | None, out_png: str | Path, title: str = "振幅谱"
) -> Path:
    """平均振幅谱（dB），叠加 10–90% 分位带，并标出 5–80 Hz 信号带。"""
    setup_style()
    dt_v = _check_dt(dt)
    arr, rs, _ = _thin(data)
    dt_s = dt_v * rs
    ns, _ntr = arr.shape
    freqs = np.fft.rfftfreq(ns, d=dt_s)
    specs = np.abs(np.fft.rfft(np.asarray(arr, dtype=np.float64), axis=0))
    peak = np.max(specs, axis=0, keepdims=True)
    peak = np.where(peak > 0, peak, 1.0)
    per_trace_db = 20.0 * np.log10(specs / peak + 1e-12)
    median_db = np.median(per_trace_db, axis=1)
    lo_db = np.percentile(per_trace_db, 10, axis=1)
    hi_db = np.percentile(per_trace_db, 90, axis=1)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.fill_between(freqs, lo_db, hi_db, color="#9DC3E6", alpha=0.45,
                    label=L("10–90% 分位", "10–90% band"))
    ax.plot(freqs, median_db, color="#C00000", lw=1.8,
            label=L("中位振幅谱", "Median spectrum"))
    nyquist = 0.5 / dt_s
    band_lo, band_hi = 5.0, min(80.0, nyquist)
    if band_hi > band_lo:
        ax.axvspan(band_lo, band_hi, color="#FFD966", alpha=0.18,
                   label=L("5–80 Hz 信号带", "5–80 Hz signal band"))
    ax.set_xlim(0.0, nyquist)
    ax.set_ylim(max(-80.0, float(np.nanpercentile(lo_db, 1)) - 5.0), 3.0)
    ax.set_title(_title(title, "Amplitude spectrum"), fontsize=15, fontweight="bold")
    ax.set_xlabel(L("频率 (Hz)", "Frequency (Hz)"))
    ax.set_ylabel(L("相对振幅 (dB)", "Relative amplitude (dB)"))
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 4. F-K 谱
# --------------------------------------------------------------------------
def fig_fk(
    data: np.ndarray,
    dt: float | None,
    dz: float | None,
    out_png: str | Path,
    title: str = "F-K 谱",
) -> Path:
    """二维 F-K 振幅谱（dB）。没有道间距时横轴用 1/道。"""
    setup_style()
    dt_v = _check_dt(dt)
    dz_v = _check_dz(dz)
    arr, rs, cs = _thin(data)
    dt_s = dt_v * rs
    dz_s = dz_v * cs if dz_v else None
    ns, ntr = arr.shape

    spectrum = np.fft.fftshift(np.fft.fft2(np.asarray(arr, dtype=np.float64)))
    amp = np.abs(spectrum)
    vmax = float(amp.max()) if amp.size else 0.0
    amp_db = 20.0 * np.log10(amp / (vmax if vmax > 0 else 1.0) + 1e-12)
    freqs = np.fft.fftshift(np.fft.fftfreq(ns, d=dt_s))
    spacing = dz_s if dz_s else 1.0
    wavenumbers = np.fft.fftshift(np.fft.fftfreq(ntr, d=spacing))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    vmin = max(float(amp_db.max()) - 60.0, -120.0)
    im = ax.imshow(
        amp_db,
        origin="lower",
        aspect="auto",
        extent=[float(wavenumbers[0]), float(wavenumbers[-1]),
                float(freqs[0]), float(freqs[-1])],
        cmap="magma",
        vmin=vmin,
        vmax=float(amp_db.max()) if amp_db.size else 0.0,
        interpolation="nearest",
    )
    ax.set_title(_title(title, "F-K spectrum"), fontsize=15, fontweight="bold")
    ax.set_xlabel(L("波数 (1/m)", "Wavenumber (1/m)") if dz_s
                  else L("波数 (1/道)", "Wavenumber (1/trace)"))
    ax.set_ylabel(L("频率 (Hz)", "Frequency (Hz)"))
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cb.set_label(L("相对振幅 (dB)", "Relative amplitude (dB)"))
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 5. 来源匹配
# --------------------------------------------------------------------------
def _assessment_mapping(ds: Any, assessment: Any) -> dict:
    """把 SourceAssessment / dict / 整个 state 统一成字典。"""
    if assessment is None:
        if isinstance(ds, dict):
            assessment = ds.get("source") or ds
        else:
            assessment = getattr(ds, "source", None) or {}
    if hasattr(assessment, "to_info"):
        assessment = assessment.to_info()
    return dict(assessment) if isinstance(assessment, dict) else {}


def fig_source_assessment(ds: Any, assessment: Any, out_png: str | Path) -> Path:
    """来源匹配图：噪声类型概率条 + 领域/证据/置信度信息面板。"""
    setup_style()
    data = _assessment_mapping(ds, assessment)
    ds_obj = _unwrap_dataset(ds)

    noise = data.get("noise_types") or {}
    if not isinstance(noise, dict):
        noise = {}
    items = sorted(
        [(str(k), v) for k, v in noise.items()],
        key=lambda kv: float(kv[1]) if isinstance(kv[1], (int, float)) else 0.0,
        reverse=True,
    )[:10]

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18], wspace=0.28)
    ax_bar = fig.add_subplot(gs[0, 0])
    if items:
        labels = [_noise_label(k) for k, _ in items][::-1]
        probs = []
        for _, value in items[::-1]:
            try:
                probs.append(float(value))
            except (TypeError, ValueError):
                probs.append(0.0)
        display = [p * 100 if p <= 1.0 else p for p in probs]
        colors = plt.cm.YlOrRd(np.linspace(0.35, 0.85, len(display)))
        bars = ax_bar.barh(np.arange(len(display)), display, color=colors,
                           edgecolor="0.4", linewidth=0.5)
        ax_bar.set_yticks(np.arange(len(display)))
        ax_bar.set_yticklabels(labels, fontsize=10)
        ax_bar.set_xlim(0, max(105.0, max(display) * 1.12))
        ax_bar.set_xlabel(L("概率 / 强度 (%)", "Probability / strength (%)"))
        ax_bar.set_title(L("噪声类型概率", "Noise-type probability"),
                         fontsize=12, fontweight="bold")
        ax_bar.grid(axis="x", alpha=0.25)
        for bar, value in zip(bars, display):
            ax_bar.text(bar.get_width() + 1.0, bar.get_y() + bar.get_height() / 2,
                        f"{value:.0f}%", va="center", fontsize=9)
    else:
        ax_bar.axis("off")
        ax_bar.text(0.5, 0.5, _title("未识别出明显的噪声类型",
                                    "No obvious noise type detected"),
                    ha="center", va="center", transform=ax_bar.transAxes,
                    fontsize=14, color="0.4")

    ax_info = fig.add_subplot(gs[0, 1])
    ax_info.axis("off")
    lines: list[str] = []
    lines.append(f"{L('领域', 'Domain')}：{data.get('domain', '未知')}")
    lines.append(f"{L('采集方式', 'Acquisition')}：{data.get('acquisition', '未知')}")
    lines.append(f"{L('观测系统', 'Geometry')}：{data.get('geometry_type', '未知')}")
    try:
        conf = float(data.get("confidence", 0.0))
    except (TypeError, ValueError):
        conf = 0.0
    lines.append(f"{L('匹配置信度', 'Confidence')}：{conf:.0%}")
    lines.append(f"{L('是否需要去噪', 'Needs denoising')}："
                 f"{L('是', 'Yes') if data.get('needs_denoise', True) else L('否', 'No')}")
    if ds_obj is not None:
        ntr = getattr(ds_obj, "n_traces", None)
        ns = getattr(ds_obj, "n_samples", None)
        if ntr is None and isinstance(ds_obj, dict):
            ntr, ns = ds_obj.get("n_traces"), ds_obj.get("n_samples")
        if ns or ntr:
            lines.append(f"{L('数据规模', 'Size')}：{ns or '—'} × {ntr or '—'}")
    lines.append("")

    evidence = data.get("evidence") or []
    if evidence:
        lines.append(L("判定证据：", "Evidence:"))
        for item in evidence[:6]:
            wrapped = textwrap.wrap(str(item), width=34)
            lines.extend(f"  • {w}" if i == 0 else f"    {w}"
                         for i, w in enumerate(wrapped))
    clean_reasons = data.get("clean_reasons") or []
    if clean_reasons:
        lines.append("")
        lines.append(L("判定干净的依据：", "Clean reasons:"))
        for item in clean_reasons[:5]:
            wrapped = textwrap.wrap(str(item), width=34)
            lines.extend(f"  • {w}" if i == 0 else f"    {w}"
                         for i, w in enumerate(wrapped))
    notes = data.get("notes")
    if notes:
        lines.append("")
        lines.append(L("补充说明：", "Notes:"))
        lines.extend(textwrap.wrap(str(notes), width=38))

    ax_info.text(
        0.0, 0.99, "\n".join(lines), transform=ax_info.transAxes,
        va="top", ha="left", fontsize=10.2, linespacing=1.55,
    )
    ax_info.set_title(L("来源匹配结论", "Source assessment"),
                      fontsize=12, fontweight="bold")
    fig.suptitle(_title("来源匹配与噪声类型", "Source assessment & noise types"),
                 fontsize=15, fontweight="bold")
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 6. 去噪前后对比
# --------------------------------------------------------------------------
def fig_compare(
    noisy: np.ndarray,
    denoised: np.ndarray,
    dt: float | None,
    dz: float | None,
    out_png: str | Path,
    title: str = "去噪前后",
    residual: np.ndarray | None = None,
) -> Path:
    """去噪前后对比：含噪 / 去噪后 / 残差 / 单道波形。"""
    setup_style()
    dt_v = _check_dt(dt)
    dz_v = _check_dz(dz)
    arr_n, arr_d, rs, cs = _thin_pair(noisy, denoised)
    dt_s = dt_v * rs
    dz_s = dz_v * cs if dz_v else None
    ns, ntr = arr_n.shape

    if residual is None:
        arr_r = np.asarray(noisy, dtype=np.float64) - np.asarray(denoised, dtype=np.float64)
    else:
        raw_r = np.asarray(residual)
        if raw_r.shape != np.asarray(noisy).shape:
            raise ValueError(f"残差形状 {raw_r.shape} 与输入 {np.asarray(noisy).shape} 不一致")
        arr_r = raw_r
    arr_r = _thin_steps(arr_r, rs, cs)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, hspace=0.34, wspace=0.16)
    clip_main = _clip_level(arr_n, arr_d)
    clip_res = _clip_level(arr_r)

    ax_a = fig.add_subplot(gs[0, 0])
    _show_section(ax_a, arr_n, dt_s, dz_s, f"(a) {L('含噪输入', 'Noisy input')}",
                  clip=clip_main)
    ax_b = fig.add_subplot(gs[0, 1])
    _show_section(ax_b, arr_d, dt_s, dz_s, f"(b) {L('去噪结果', 'Denoised')}",
                  clip=clip_main)
    ax_c = fig.add_subplot(gs[1, 0])
    _show_section(ax_c, arr_r, dt_s, dz_s, f"(c) {L('残差（去噪差分）', 'Residual')}",
                  clip=clip_res)

    ax_d = fig.add_subplot(gs[1, 1])
    idx = ntr // 2
    t = np.arange(ns) * dt_s
    ax_d.plot(arr_n[:, idx], t, color="0.55", lw=0.9,
              label=L("含噪", "Noisy"))
    ax_d.plot(arr_d[:, idx], t, color="#C00000", lw=1.0,
              label=L("去噪后", "Denoised"))
    ax_d.plot(arr_r[:, idx], t, color="#2E75B6", lw=0.8, alpha=0.8,
              label=L("残差", "Residual"))
    ax_d.set_ylim(t.max() if t.size else 1.0, 0.0)
    ax_d.set_title(f"(d) {L('单道波形对比', 'Single-trace comparison')}", fontsize=12)
    ax_d.set_xlabel(L("振幅", "Amplitude"))
    ax_d.set_ylabel(L("时间 (s)", "Time (s)"))
    ax_d.grid(alpha=0.25)
    ax_d.legend(loc="lower right", fontsize=8.5)

    fig.suptitle(_title(title, "Before / after denoising"),
                 fontsize=15, fontweight="bold")
    return _save(fig, out_png)


def fig_residual(
    noisy: np.ndarray,
    denoised: np.ndarray,
    dt: float | None,
    dz: float | None,
    out_png: str | Path,
    title: str = "去噪差分",
) -> Path:
    """额外的差分图（不在契约里，供流水线生成“去噪差分.png”）：残差剖面 + 逐道 RMS。"""
    setup_style()
    dt_v = _check_dt(dt)
    dz_v = _check_dz(dz)
    arr_n, arr_d, rs, cs = _thin_pair(noisy, denoised)
    dt_s = dt_v * rs
    dz_s = dz_v * cs if dz_v else None
    arr_r = _thin_steps(
        np.asarray(noisy, dtype=np.float64) - np.asarray(denoised, dtype=np.float64),
        rs, cs,
    )

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 1.0], wspace=0.22)
    ax_a = fig.add_subplot(gs[0, 0])
    _show_section(ax_a, arr_r, dt_s, dz_s,
                  L("残差（含噪 − 去噪）", "Residual (noisy − denoised)"),
                  clip=_clip_level(arr_r))
    ax_b = fig.add_subplot(gs[0, 1])
    rms_n = np.sqrt(np.mean(np.asarray(arr_n, dtype=np.float64) ** 2, axis=0))
    rms_d = np.sqrt(np.mean(np.asarray(arr_d, dtype=np.float64) ** 2, axis=0))
    ax_b.plot(np.arange(arr_n.shape[1]), rms_n, color="0.55", lw=1.0,
              label=L("含噪 RMS", "Noisy RMS"))
    ax_b.plot(np.arange(arr_d.shape[1]), rms_d, color="#C00000", lw=1.0,
              label=L("去噪后 RMS", "Denoised RMS"))
    ax_b.set_title(L("逐道 RMS 能量对比", "Per-trace RMS"), fontsize=12)
    ax_b.set_xlabel(L("道号", "Trace"))
    ax_b.set_ylabel(L("RMS 振幅", "RMS amplitude"))
    ax_b.grid(alpha=0.25)
    ax_b.legend()
    fig.suptitle(_title(title, "Denoising difference"),
                 fontsize=15, fontweight="bold")
    return _save(fig, out_png)


def fig_visual_qc(
    noisy: np.ndarray,
    denoised: np.ndarray,
    dt: float | None,
    dz: float | None,
    out_png: str | Path,
    visual: dict | None = None,
    title: str = "视觉对比与弱信号保留",
) -> Path:
    """高保真视觉复核：输入 / 输出 / 残差三面板 + 视觉代理指标。"""
    setup_style()
    dt_v = _check_dt(dt)
    dz_v = _check_dz(dz)
    arr_n, arr_d, rs, cs = _thin_pair(noisy, denoised)
    dt_s = dt_v * rs
    dz_s = dz_v * cs if dz_v else None
    arr_r = _thin_steps(
        np.asarray(noisy, dtype=np.float64) - np.asarray(denoised, dtype=np.float64),
        rs, cs,
    )
    clip_main = _clip_level(arr_n, arr_d)
    clip_res = _clip_level(arr_r)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 3, wspace=0.34)
    ax_a = fig.add_subplot(gs[0, 0])
    _show_section(ax_a, arr_n, dt_s, dz_s, f"(a) {L('含噪输入', 'Noisy')}",
                  clip=clip_main)
    ax_b = fig.add_subplot(gs[0, 1])
    _show_section(ax_b, arr_d, dt_s, dz_s, f"(b) {L('去噪结果', 'Denoised')}",
                  clip=clip_main)
    ax_c = fig.add_subplot(gs[0, 2])
    _show_section(ax_c, arr_r, dt_s, dz_s, f"(c) {L('残差（被去除）', 'Residual')}",
                  clip=clip_res)

    v = visual if isinstance(visual, dict) else {}
    if v.get("available"):
        weak_ratio = v.get("weak_signal_ratio")
        weak_txt = L("—", "—") if weak_ratio is None else f"{float(weak_ratio):.1%}"
        txt = (
            f"{L('SSIM', 'SSIM')} {float(v.get('ssim') or 0):.3f}   "
            f"{L('边缘相关', 'Edge corr')} {float(v.get('edge_corr') or 0):.3f}   "
            f"{L('弱区能量保留', 'Weak energy')} {weak_txt}   "
            f"{L('视觉综合分', 'Visual score')} {float(v.get('visual_score') or 0):.3f}"
        )
    else:
        txt = L("视觉指标不可用", "Visual metrics unavailable")
    fig.text(0.5, 0.025, txt, ha="center", va="bottom", fontsize=10.5,
             color="#333333", bbox=dict(boxstyle="round,pad=0.35", fc="#F2F2F2", ec="#BFBFBF"))

    fig.suptitle(_title(title, "Visual QC & weak-signal preservation"),
                 fontsize=15, fontweight="bold")
    fig.subplots_adjust(bottom=0.14)
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 7. 质量指标
# --------------------------------------------------------------------------
_METRIC_FIELDS = (
    ("snr_in_db", "去噪前 SNR (dB)", "Input SNR (dB)"),
    ("snr_out_db", "去噪后 SNR (dB)", "Output SNR (dB)"),
    ("snr_gain_db", "SNR 增益 (dB)", "SNR gain (dB)"),
    ("noise_reduction_db", "噪声降低 (dB)", "Noise reduction (dB)"),
    ("energy_ratio", "能量保持比", "Energy ratio"),
    ("signal_leakage", "信号泄漏", "Signal leakage"),
    ("spectrum_corr", "频谱相关", "Spectrum correlation"),
    ("coherence_residual", "残差相干性", "Coherence residual"),
    ("clip_ratio", "削顶比例", "Clipping ratio"),
    ("local_snr_gain_db", "局部 SNR 增益 (dB)", "Local SNR gain (dB)"),
)


def _metrics_payload(metrics: Any) -> tuple[dict, dict | None]:
    """兼容 QualityMetrics / MethodResult.to_info() / state["best"]。"""
    gate = None
    if hasattr(metrics, "to_info"):
        obj = metrics.to_info()
        return (obj if isinstance(obj, dict) else {}), None
    if isinstance(metrics, dict):
        if "metrics" in metrics and isinstance(metrics["metrics"], dict):
            gate = metrics.get("gate")
            return dict(metrics["metrics"]), gate
        if "best" in metrics and isinstance(metrics["best"], dict):
            best = metrics["best"]
            gate = best.get("gate")
            return dict(best.get("metrics") or {}), gate
        return dict(metrics), gate
    return {}, None


def _num(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def fig_metrics(metrics: Any, out_png: str | Path, title: str = "质量指标") -> Path:
    """质量指标图：SNR 柱状图 + 比例指标 + 指标清单 + 门禁检查清单。"""
    setup_style()
    data, gate = _metrics_payload(metrics)
    gate = gate if isinstance(gate, dict) else {}

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(2, 2, hspace=0.40, wspace=0.30)

    # (a) SNR 三项
    ax_snr = fig.add_subplot(gs[0, 0])
    keys = ["snr_in_db", "snr_out_db", "snr_gain_db"]
    labels = [L("去噪前", "Input"), L("去噪后", "Output"), L("增益", "Gain")]
    values = [_num(data.get(k)) for k in keys]
    bars = ax_snr.bar(labels, [0 if v is None else v for v in values],
                      color=["#8EA9DB", "#548235", "#C00000"])
    for bar, value in zip(bars, values):
        if value is not None:
            ax_snr.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{value:.2f}", ha="center",
                        va="bottom" if value >= 0 else "top", fontsize=9)
    ax_snr.axhline(0.0, color="0.4", lw=0.8)
    ax_snr.set_title(L("信噪比", "SNR (dB)"), fontsize=12)
    ax_snr.set_ylabel("dB")
    ax_snr.grid(axis="y", alpha=0.25)

    # (b) 比例指标与门禁参考线
    ax_ratio = fig.add_subplot(gs[0, 1])
    ratio_keys = ["energy_ratio", "signal_leakage", "spectrum_corr"]
    ratio_labels = [L("能量保持比", "Energy ratio"), L("信号泄漏", "Leakage"),
                    L("频谱相关", "Spectrum corr")]
    ratio_vals = [_num(data.get(k)) for k in ratio_keys]
    bars = ax_ratio.bar(ratio_labels, [0 if v is None else v for v in ratio_vals],
                        color=["#7F7F7F", "#ED7D31", "#4472C4"])
    thresholds = [1.0, 0.40, 0.80]
    for i, (bar, value, thr) in enumerate(zip(bars, ratio_vals, thresholds)):
        if value is not None:
            ax_ratio.text(bar.get_x() + bar.get_width() / 2, max(bar.get_height(), 0.0),
                          f"{value:.3f}", ha="center", va="bottom", fontsize=9)
        ax_ratio.plot([i - 0.4, i + 0.4], [thr, thr], color="#C00000",
                      lw=1.2, ls="--")
    ax_ratio.set_ylim(0.0, 1.35)
    ax_ratio.set_title(L("比例指标（红线为门禁参考）", "Ratio metrics (red = gate)"),
                       fontsize=12)
    ax_ratio.grid(axis="y", alpha=0.25)

    # (c) 指标清单
    ax_list = fig.add_subplot(gs[1, 0])
    ax_list.axis("off")
    lines = []
    for key, zh, en in _METRIC_FIELDS:
        value = data.get(key)
        value_txt = "—" if value is None else (
            f"{value:.4g}" if key in ("energy_ratio", "signal_leakage",
                                      "spectrum_corr", "coherence_residual",
                                      "clip_ratio")
            else f"{float(value):.2f}"
        ) if _num(value) is not None else "—"
        lines.append(f"{L(zh, en)}: {value_txt}")
    estimator = data.get("estimator")
    if estimator:
        lines.append(f"{L('估计方法', 'Estimator')}: {estimator}")
    ax_list.text(0.0, 0.98, "\n".join(lines), transform=ax_list.transAxes,
                 va="top", ha="left", fontsize=10.2, linespacing=1.6)
    ax_list.set_title(L("指标数值", "Metric values"), fontsize=12)

    # (d) 门禁检查
    ax_gate = fig.add_subplot(gs[1, 1])
    ax_gate.axis("off")
    if gate:
        passed = bool(gate.get("passed"))
        head = (L("门禁：通过", "Gate: PASSED") if passed
                else L("门禁：未通过", "Gate: FAILED"))
        gate_lines = [head, f"{L('得分', 'Score')}: {_num(gate.get('score')) or 0:.1f}"]
        for check in (gate.get("checks") or [])[:9]:
            if not isinstance(check, dict):
                continue
            mark = L("通过", "PASS") if check.get("passed") else L("未通过", "FAIL")
            gate_lines.append(
                f"[{mark}] {check.get('name', '')} = "
                f"{check.get('value')}  ({check.get('threshold', '')})"
            )
        if gate.get("failures"):
            gate_lines.append("")
            gate_lines.append(L("失败原因：", "Failures:"))
            for item in gate["failures"][:4]:
                gate_lines.extend(textwrap.wrap(f"  • {item}", width=44))
        ax_gate.text(0.0, 0.98, "\n".join(gate_lines), transform=ax_gate.transAxes,
                     va="top", ha="left", fontsize=9.6, linespacing=1.55,
                     color="#C00000" if not passed else "#375623")
    else:
        ax_gate.text(0.0, 0.98,
                     L("未提供门禁结果。\n默认参考：SNR 增益 ≥ 3 dB、能量保持比 0.30–1.05、\n"
                       "信号泄漏 ≤ 0.40、频谱相关 ≥ 0.80、削顶比例 ≤ 1e-3。",
                       "No gate result provided.\nDefaults: SNR gain ≥ 3 dB, energy ratio 0.30–1.05,\n"
                       "leakage ≤ 0.40, spectrum corr ≥ 0.80, clip ratio ≤ 1e-3."),
                     transform=ax_gate.transAxes, va="top", ha="left",
                     fontsize=9.6, color="0.35", linespacing=1.5)
    ax_gate.set_title(L("门禁检查", "Gate checks"), fontsize=12)

    suffix = ""
    if gate:
        suffix = L("（通过门禁）", " (gate passed)") if gate.get("passed") else L(
            "（未通过门禁）", " (gate failed)")
    fig.suptitle(_title(title, "Quality metrics") + suffix,
                 fontsize=15, fontweight="bold")
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 8. 局部放大
# --------------------------------------------------------------------------
def fig_local_zoom(
    noisy: np.ndarray,
    denoised: np.ndarray,
    dt: float | None,
    dz: float | None,
    out_png: str | Path,
) -> Path:
    """局部放大：自动选能量最强的时间窗与道窗，对比含噪/去噪/残差。"""
    setup_style()
    dt_v = _check_dt(dt)
    dz_v = _check_dz(dz)
    arr_n, arr_d, rs, cs = _thin_pair(noisy, denoised)
    dt_s = dt_v * rs
    dz_s = dz_v * cs if dz_v else None
    ns, ntr = arr_n.shape
    arr_r = _thin_steps(
        np.asarray(noisy, dtype=np.float64) - np.asarray(denoised, dtype=np.float64),
        rs, cs,
    )

    # 时间窗：对整幅能量做平滑，取能量最强的窗口
    energy = np.mean(np.asarray(arr_n, dtype=np.float64) ** 2, axis=1)
    kernel = max(3, min(31, energy.size // 10 * 2 + 1))
    smooth = np.convolve(energy, np.ones(kernel) / kernel, mode="same")
    win_t = int(min(ns, max(64, min(600, ns // 3))))
    center = int(np.argmax(smooth)) if smooth.size else ns // 2
    t0 = int(np.clip(center - win_t // 2, 0, max(0, ns - win_t)))
    t1 = t0 + win_t

    # 道窗：以能量最强的道为中心，最多 80 道
    trace_rms = np.mean(np.asarray(arr_n, dtype=np.float64) ** 2, axis=0)
    win_x = int(min(ntr, 80))
    tcenter = int(np.argmax(trace_rms)) if trace_rms.size else ntr // 2
    x0 = int(np.clip(tcenter - win_x // 2, 0, max(0, ntr - win_x)))
    x1 = x0 + win_x

    sub_n = arr_n[t0:t1, x0:x1]
    sub_d = arr_d[t0:t1, x0:x1]
    sub_r = arr_r[t0:t1, x0:x1]
    clip_main = _clip_level(sub_n, sub_d)

    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, 3, wspace=0.20)
    tag = f"{L('时间', 't')} {t0 * dt_s:.2f}–{t1 * dt_s:.2f} s, " \
          f"{L('道', 'trace')} {x0 + 1}–{x1}"
    ax_a = fig.add_subplot(gs[0, 0])
    _show_section(ax_a, sub_n, dt_s, dz_s,
                  f"(a) {L('含噪放大', 'Noisy zoom')}\n{tag}", clip=clip_main)
    ax_b = fig.add_subplot(gs[0, 1])
    _show_section(ax_b, sub_d, dt_s, dz_s,
                  f"(b) {L('去噪后放大', 'Denoised zoom')}\n{tag}", clip=clip_main)
    ax_c = fig.add_subplot(gs[0, 2])
    _show_section(ax_c, sub_r, dt_s, dz_s,
                  f"(c) {L('残差放大', 'Residual zoom')}", clip=_clip_level(sub_r))
    fig.suptitle(_title("局部放大对比", "Local zoom comparison"),
                 fontsize=15, fontweight="bold")
    return _save(fig, out_png)


# --------------------------------------------------------------------------
# 额外：一键生成报告所需图件（不在契约里，但流水线可以直接用）
# --------------------------------------------------------------------------
def generate_report_figures(
    job_dir: str | Path,
    noisy: np.ndarray,
    denoised: np.ndarray,
    dt: float | None,
    dz: float | None,
    *,
    ds: Any = None,
    assessment: Any = None,
    metrics: Any = None,
    state: dict | None = None,
) -> dict[str, Path]:
    """按约定目录一次性生成 02/03/04 三个文件夹的报告图，返回 {名称: 路径}。

    这是给 Lead 的 pipeline 用的便捷函数（契约之外新增，不影响契约函数）。
    """
    job = Path(job_dir)
    diag = job / "02_读取与诊断"
    den = job / "03_去噪结果"
    quality = job / "04_质量评价"
    for folder in (diag, den, quality):
        folder.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    out["overview"] = fig_data_overview(noisy, dt, dz, diag / "数据总览.png")
    out["spectrum"] = fig_spectrum(noisy, dt, diag / "频率谱.png")

    if ds is not None:
        try:
            out["header"] = fig_header_analysis(ds, diag / "道头分析.png")
        except Exception:
            pass

    source = assessment
    if source is None and isinstance(state, dict):
        source = state.get("source")
    if source is not None:
        try:
            out["source"] = fig_source_assessment(ds or {}, source,
                                                  diag / "来源匹配图.png")
        except Exception:
            pass

    out["fk"] = fig_fk(noisy, dt, dz, diag / "fk谱.png")
    out["compare"] = fig_compare(noisy, denoised, dt, dz,
                                 den / "去噪后剖面.png", title="去噪前后")
    out["diff"] = fig_residual(noisy, denoised, dt, dz, den / "去噪差分.png")
    if metrics is None and isinstance(state, dict):
        metrics = (state.get("best") or {}).get("metrics")
    out["metrics"] = fig_metrics(metrics or {}, quality / "指标对比.png")
    out["zoom"] = fig_local_zoom(noisy, denoised, dt, dz,
                                 quality / "局部放大对比.png")
    return out


__all__ = [
    "setup_style",
    "fig_data_overview",
    "fig_header_analysis",
    "fig_spectrum",
    "fig_fk",
    "fig_source_assessment",
    "fig_compare",
    "fig_metrics",
    "fig_local_zoom",
    # 契约外补充
    "fig_residual",
    "generate_report_figures",
    "cjk_available",
]
