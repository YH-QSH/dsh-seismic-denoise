"""绘图：中文字体、地震剖面配色、对比图与指标图。

统一使用 ``seismic`` 蓝白红配色（与论文常见的红蓝剖面一致），
并自动挑选系统中可用的中文字体，避免出现"□□□"方块。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # 无 GUI 环境
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.font_manager import FontProperties, findfont

from . import metrics as M

# --------------------------------------------------------------------------
# 中文字体
# --------------------------------------------------------------------------
_CJK_CANDIDATES = [
    "Microsoft YaHei",
    "SimHei",
    "DengXian",
    "SimSun",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
]

_CJK_OK: Optional[bool] = None


def setup_chinese_font(verbose: bool = False) -> bool:
    """设置 matplotlib 中文字体，返回是否成功。"""
    global _CJK_OK
    available = {f.name for f in matplotlib.font_manager.fontManager.ttflist}
    picked = [n for n in _CJK_CANDIDATES if n in available]
    if picked:
        plt.rcParams["font.sans-serif"] = picked + ["DejaVu Sans"]
        plt.rcParams["font.family"] = "sans-serif"
        _CJK_OK = True
    else:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
        _CJK_OK = False
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["savefig.dpi"] = 160
    plt.rcParams["savefig.bbox"] = "tight"
    if verbose:
        print("中文字体:", picked or "未找到，将使用英文标签")
    return bool(_CJK_OK)


def cjk_available() -> bool:
    if _CJK_OK is None:
        setup_chinese_font()
    return bool(_CJK_OK)


def L(zh: str, en: str) -> str:
    """按字体可用性返回中文或英文标签。"""
    return zh if cjk_available() else en


# --------------------------------------------------------------------------
# 配色与坐标
# --------------------------------------------------------------------------
SEISMIC = LinearSegmentedColormap.from_list(
    "seismic_custom",
    ["#1a2f8f", "#4a6fd6", "#9db4ea", "#f7f7f7", "#f2b0a6", "#dd5f4a", "#a01010"],
)


def _axes(data: np.ndarray, dt: float, dz: float):
    """返回 imagesc 的 extent 与坐标标签。"""
    nt, nz = data.shape
    # 以"道号/深度"为横轴，时间为纵轴且向下增大（与地震剖面惯例一致）
    extent = [0, nz, nt * dt, 0]
    return extent


def clip_level(*arrays: np.ndarray, pct: float = 98.0) -> float:
    vals = np.concatenate([np.abs(np.asarray(a, dtype=np.float64)).ravel() for a in arrays])
    return float(np.percentile(vals, pct)) + 1e-12


def plot_section(
    ax,
    data: np.ndarray,
    dt: float = 0.002,
    dz: float = 5.0,
    title: str = "",
    clip: Optional[float] = None,
    cmap=None,
    show_colorbar: bool = False,
    cbar_label: str = "振幅",
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    fontsize: int = 10,
):
    nt, nz = data.shape
    if clip is None:
        clip = clip_level(data)
    im = ax.imshow(
        np.asarray(data, dtype=np.float32),
        cmap=cmap or SEISMIC,
        aspect="auto",
        extent=[0, nz, nt * dt, 0],
        vmin=-clip,
        vmax=clip,
        interpolation="nearest",
    )
    ax.set_title(title, fontsize=fontsize + 1)
    ax.set_xlabel(xlabel if xlabel is not None else L("道号", "Trace"), fontsize=fontsize)
    ax.set_ylabel(ylabel if ylabel is not None else L("时间 (s)", "Time (s)"), fontsize=fontsize)
    ax.tick_params(labelsize=fontsize - 1)
    if show_colorbar:
        cb = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cb.set_label(cbar_label, fontsize=fontsize)
        cb.ax.tick_params(labelsize=fontsize - 2)
    return im


# --------------------------------------------------------------------------
# 主图：多方法对比
# --------------------------------------------------------------------------
def figure_compare(
    noisy: np.ndarray,
    clean: np.ndarray,
    results: Dict[str, np.ndarray],
    out_path: str | Path,
    dt: float = 0.002,
    dz: float = 5.0,
    suptitle: str = "",
    max_panels: int = 6,
) -> Path:
    """把 (含噪 / 干净 / 各方法) 画成一张对比图，共用同一色标。"""
    setup_chinese_font()
    items = [("(a) " + L("含噪输入", "Noisy input"), noisy, "")]
    names = [k for k in results if k != "干净参考"]
    if "干净参考" in results:
        items.append(("(b) " + L("干净参考", "Clean reference"), results["干净参考"], ""))
    for i, k in enumerate(names[: max_panels - len(items)]):
        tag = f"({chr(ord('a') + len(items))}) " if True else ""
        items.append((f"{tag}{k}", results[k], k))

    # 重新编号 (a)(b)(c)...
    items = [(f"({chr(ord('a') + i)}) " + t.split(") ", 1)[-1], d, k)
             for i, (t, d, k) in enumerate(items)]

    n = len(items)
    ncol = 3 if n > 4 else 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 4.2 * nrow), squeeze=False)

    clip = clip_level(noisy, clean, *results.values())
    for i, (title, data, _key) in enumerate(items):
        ax = axes[i // ncol][i % ncol]
        plot_section(ax, data, dt, dz, title, clip=clip, show_colorbar=True)
    for j in range(n, nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    if suptitle:
        fig.suptitle(suptitle, fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 指标条形图
# --------------------------------------------------------------------------
def figure_metrics(
    table: Dict[str, Dict[str, float]],
    out_path: str | Path,
    suptitle: str = "",
) -> Path:
    """table: {方法名: {"snr_db":..,"rmse":..,"cc":..}}"""
    setup_chinese_font()
    names = list(table.keys())
    keys = ["snr_db", "rmse", "cc"]
    labels = [M.METRIC_LABELS[k] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(names)))
    for ax, k, lab in zip(axes, keys, labels):
        vals = [table[n].get(k, np.nan) for n in names]
        bars = ax.bar(range(len(names)), vals, color=colors)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
        ax.set_title(lab, fontsize=11)
        ax.grid(axis="y", alpha=0.3)
        for b, v in zip(bars, vals):
            if np.isfinite(v):
                ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}",
                        ha="center", va="bottom", fontsize=8)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 训练曲线
# --------------------------------------------------------------------------
def figure_train_curves(history: Dict[str, list], out_path: str | Path) -> Path:
    setup_chinese_font()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ep = history.get("epoch", list(range(len(history.get("train_loss", [])))))

    ax = axes[0]
    if "train_loss" in history:
        ax.plot(ep, history["train_loss"], label=L("训练损失", "train loss"), lw=1.6)
    if "val_loss" in history:
        ax.plot(ep, history["val_loss"], label=L("验证损失", "val loss"), lw=1.6)
    ax.set_xlabel(L("epoch", "epoch"))
    ax.set_ylabel("loss")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1]
    if "val_snr_db" in history:
        ax.plot(ep, history["val_snr_db"], "o-", ms=3, label=L("去噪后 SNR", "val SNR"), lw=1.6)
    if "val_snr_in_db" in history:
        ax.plot(ep, history["val_snr_in_db"], "s--", ms=3,
                label=L("去噪前 SNR", "input SNR"), lw=1.4)
    ax.set_xlabel(L("epoch", "epoch"))
    ax.set_ylabel(L("信噪比 (dB)", "SNR (dB)"))
    ax.grid(alpha=0.3)
    ax.legend()

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# f-k 谱对比
# --------------------------------------------------------------------------
def figure_fk(
    noisy: np.ndarray,
    clean: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    suptitle: str = "",
) -> Path:
    from .baselines import fk_spectrum

    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    specs = [
        (L("含噪 f-k 谱", "noisy f-k"), fk_spectrum(noisy)),
        (L("去噪后 f-k 谱", "denoised f-k"), fk_spectrum(pred)),
        (L("干净 f-k 谱", "clean f-k"), fk_spectrum(clean)),
    ]
    vmax = max(float(np.percentile(s[1], 99.5)) for s in specs)
    for ax, (t, s) in zip(axes, specs):
        im = ax.imshow(s, cmap="magma", aspect="auto", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(t, fontsize=11)
        ax.set_xlabel(L("波数", "wavenumber"))
        ax.set_ylabel(L("频率", "frequency"))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


# --------------------------------------------------------------------------
# 单道波形对比
# --------------------------------------------------------------------------
def figure_traces(
    noisy: np.ndarray,
    clean: np.ndarray,
    pred: np.ndarray,
    out_path: str | Path,
    dt: float = 0.002,
    trace_idx: Optional[int] = None,
    color: str = "#c0392b",
    suptitle: str = "",
) -> Path:
    setup_chinese_font()
    nt, nz = noisy.shape
    if trace_idx is None:
        trace_idx = nz // 2
    t = np.arange(nt) * dt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    ax = axes[0]
    ax.plot(t, noisy[:, trace_idx], color="0.55", lw=0.9, label=L("含噪", "noisy"))
    ax.plot(t, clean[:, trace_idx], color="0.1", lw=1.1, label=L("干净", "clean"))
    ax.plot(t, pred[:, trace_idx], color=color, lw=1.0, label=L("CNN 去噪", "CNN"))
    ax.set_xlabel(L("时间 (s)", "Time (s)"))
    ax.set_ylabel(L("振幅", "Amplitude"))
    ax.set_title(L(f"第 {trace_idx} 道波形", f"Trace {trace_idx}"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    err_n = noisy[:, trace_idx] - clean[:, trace_idx]
    err_p = pred[:, trace_idx] - clean[:, trace_idx]
    ax.plot(t, err_n, color="0.55", lw=0.9, label=L("含噪误差", "noisy error"))
    ax.plot(t, err_p, color=color, lw=0.9, label=L("去噪后误差", "residual"))
    ax.set_xlabel(L("时间 (s)", "Time (s)"))
    ax.set_ylabel(L("误差", "Error"))
    ax.set_title(L("残差对比", "Residual comparison"))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    if suptitle:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
