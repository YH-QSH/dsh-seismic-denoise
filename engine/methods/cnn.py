"""CNN 去噪：噪声类型分类 → 按类别门控 → 分量分离网络重建。

复用 ``engine/assets/cnnv2`` 下已训练好的权重与网络结构：
- 分类器：5 类噪声的多标签概率（面波/随机噪声/工频干扰/散射绕射/折射波）；
- 分离网络：输出 ``[干净波场, 各噪声分量]``（K+1 通道）。

流程：先对整段剖面做滑窗分类得到类别概率 ``p``，按 ``w_k = (p_k > 阈值) * p_k``
门控各噪声通道，再按 ``clean = noisy - alpha * Σ w_k * noise_k`` 重建。
``alpha`` 控制去噪强度（默认 0.65，保守）；``gate=False`` 时等价于全部通道权重 1。

异常处理：CUDA 推理失败自动降级 CPU 重试一次；CPU 仍失败抛 ``RuntimeError``。
"""
from __future__ import annotations

import copy
import math
from pathlib import Path

import numpy as np

from ..utils import resample_to

_ASSET_DIR = Path(__file__).resolve().parent.parent / "assets" / "cnnv2"
_WEIGHT_DIR = _ASSET_DIR / "权重"
_SEP_PATH = _WEIGHT_DIR / "分离网络.pt"
_CLS_PATH = _WEIGHT_DIR / "分类器.pt"

_DEFAULTS: dict = {
    "alpha": 0.65,          # 去噪强度 0–1
    "gate": True,           # 是否用分类概率门控
    "gate_threshold": 0.5,  # 概率超过该值才启用对应类别
    "device": "auto",       # auto/cpu/cuda
    "resample": False,      # 是否先重采样到训练规范网格（2 ms / 5 m）
    "max_traces": 512,      # 单次送网络的最大道数（大剖面分块）
    "patch_t": 256,         # 时间方向 patch
    "patch_x": 64,          # 道方向 patch
    "stride_t": None,       # 默认 patch_t//2
    "stride_x": None,       # 默认 patch_x//2
    "batch_sep": 8,
    "batch_cls": 16,
}

_MODEL_CACHE: dict[str, tuple] = {}
_CKPT_INFO: dict[str, dict] = {}


# --------------------------------------------------------------------------
# 可用性 / 设备 / 权重
# --------------------------------------------------------------------------
def cnn_available() -> bool:
    """权重齐全且 torch 可导入时返回 True（不加载模型，不占显存）。"""
    if not (_SEP_PATH.is_file() and _CLS_PATH.is_file()):
        return False
    try:
        import torch  # noqa: F401
    except Exception:
        return False
    return True


def _resolve_device(want, warnings: list[str]) -> str:
    want = str(want or "auto").lower()
    try:
        import torch
    except Exception:
        return "cpu"
    if want == "cuda":
        if torch.cuda.is_available():
            return "cuda"
        warnings.append("请求 CUDA 但当前不可用，已降级 CPU")
        return "cpu"
    if want in ("auto", ""):
        return "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cpu":
        return "cpu"
    warnings.append(f"未知 device={want!r}，按 auto 处理")
    return "cuda" if torch.cuda.is_available() else "cpu"


def _prepare(data, dt) -> tuple[np.ndarray, float, list[str]]:
    warnings: list[str] = []
    a = np.asarray(data)
    if a.ndim != 2:
        raise ValueError(f"data 必须是二维数组 (采样点, 道数)，实际 shape={a.shape}")
    if a.size == 0:
        raise ValueError("data 为空数组")
    a = np.array(a, dtype=np.float32, copy=True)
    if not np.isfinite(a).all():
        a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        warnings.append("输入含 NaN/Inf，已置 0 后处理")
    try:
        dt = float(dt)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"dt 无效（{dt!r}），必须为正的有限秒数") from exc
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt 无效（{dt!r}），必须为正的有限秒数")
    return a, dt, warnings


def _model_config(ckpt: dict):
    """从 checkpoint 的 'config' 里取模型结构参数，缺省回退默认。"""
    from ..assets.cnnv2.vsdn2 import config as C
    raw = ckpt.get("config") or {}
    if not isinstance(raw, dict):
        return C.模型参数()
    try:
        return C.模型参数(**raw.get("模型", {}))
    except TypeError:
        return C.模型参数()


def _load_models(device: str):
    """加载并缓存（按设备分开缓存，CPU 回退时不重复读盘之外的初始化）。"""
    from ..assets.cnnv2.vsdn2 import models as M
    import torch

    key = str(device)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    sep_ck = torch.load(str(_SEP_PATH), map_location="cpu", weights_only=False)
    cls_ck = torch.load(str(_CLS_PATH), map_location="cpu", weights_only=False)
    sep = M.建模型(_model_config(sep_ck), "分离")
    sep.load_state_dict(sep_ck["model"])
    sep.eval()
    sep.to(device)
    cls = M.建模型(_model_config(cls_ck), "分类")
    cls.load_state_dict(cls_ck["model"])
    cls.eval()
    cls.to(device)
    _MODEL_CACHE[key] = (sep, cls)
    _CKPT_INFO[key] = {
        "separator": {k: sep_ck.get(k) for k in ("epoch", "val_snr", "val_cc", "snr_in")
                      if sep_ck.get(k) is not None},
        "classifier": {k: cls_ck.get(k) for k in ("f1", "acc") if cls_ck.get(k) is not None},
    }
    return sep, cls


def _patch_stride(p: dict, patch: tuple[int, int]) -> tuple[int, int]:
    st = p.get("stride_t")
    sx = p.get("stride_x")
    st = int(st) if st else max(1, patch[0] // 2)
    sx = int(sx) if sx else max(1, patch[1] // 2)
    return st, sx


# --------------------------------------------------------------------------
# 分块拼接
# --------------------------------------------------------------------------
def _chunk_ranges(ntr: int, max_traces: int, overlap: int = 32) -> list[tuple]:
    """返回 [(start, end, 权重数组)]；权重用于重叠区交叉淡化。"""
    if max_traces is None or max_traces <= 0 or ntr <= max_traces:
        return [(0, ntr, np.ones(ntr, dtype=np.float64))]
    max_traces = int(max_traces)
    ov = int(min(max(1, overlap), max(1, max_traces // 4)))
    ranges: list[list[int]] = []
    s = 0
    while True:
        e = min(ntr, s + max_traces)
        ranges.append([s, e])
        if e >= ntr:
            break
        s = e - ov
    out: list[tuple] = []
    last = len(ranges) - 1
    for i, (s, e) in enumerate(ranges):
        n = e - s
        w = np.ones(n, dtype=np.float64)
        if i > 0:
            w[:ov] = np.linspace(0.0, 1.0, ov, endpoint=False)
        if i < last:
            w[-ov:] = np.linspace(1.0, 0.0, ov, endpoint=False)
        out.append((s, e, w))
    return out


# --------------------------------------------------------------------------
# 推理
# --------------------------------------------------------------------------
def _infer(a: np.ndarray, p: dict, device: str, warnings: list[str],
           notes: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """分类 + 分块分离；返回 (class_probs(K,), gating(K,), 去噪结果)。

    只做推理，不处理重采样的回映射。
    """
    from ..assets.cnnv2.vsdn2 import tiling as T
    from ..assets.cnnv2.vsdn2 import config as C

    sep, cls = _load_models(device)
    patch = (max(8, int(p["patch_t"])), max(4, int(p["patch_x"])))
    stride = _patch_stride(p, patch)
    ntr = a.shape[1]
    ranges = _chunk_ranges(ntr, int(p["max_traces"]))
    alpha = float(np.clip(float(p["alpha"]), 0.0, 1.0))
    gate = bool(p["gate"])
    gate_thr = float(np.clip(float(p["gate_threshold"]), 0.0, 1.0))

    # ---- 第一遍：分类（分块加权平均） ----
    probs_acc = np.zeros(C.K, dtype=np.float64)
    w_acc = 0.0
    for s, e, w in ranges:
        pr = T.分类剖面(cls, a[:, s:e], patch=patch, stride=stride,
                        device=device, batch=int(p["batch_cls"]))
        cw = float(w.sum())
        probs_acc += np.asarray(pr, dtype=np.float64) * cw
        w_acc += cw
    probs = probs_acc / max(w_acc, 1e-12)

    # ---- 门控权重 ----
    if gate:
        gw = (probs > gate_thr).astype(np.float64) * probs
    else:
        gw = np.ones(C.K, dtype=np.float64)

    # ---- 第二遍：分块分离 + 交叉淡化拼接 ----
    acc = np.zeros_like(a, dtype=np.float64)
    wsum = np.zeros(ntr, dtype=np.float64)
    for s, e, w in ranges:
        pred = T.分离剖面(sep, a[:, s:e], patch=patch, stride=stride,
                          device=device, batch=int(p["batch_sep"]))
        removed = np.tensordot(gw, np.asarray(pred[1:], dtype=np.float64),
                               axes=(0, 0))  # Σ w_k * n_k
        chunk = a[:, s:e].astype(np.float64) - alpha * removed
        acc[:, s:e] += chunk * w[None, :]
        wsum[s:e] += w
    clean = acc / np.maximum(wsum, 1e-12)[None, :]

    if len(ranges) > 1:
        notes.append(f"剖面分 {len(ranges)} 块推理（每块 ≤{int(p['max_traces'])} 道，"
                     f"重叠区线性交叉淡化）")
    if gate and float(gw.sum()) <= 0:
        warnings.append("分类器未检出任何已知噪声类型（概率均未超过阈值），本次未去除成分；"
                        "可尝试 gate=False 或换方法")
    return probs.astype(np.float64), gw, clean


def _fit_shape(a: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """裁剪/补零到目标形状（重采样后尺寸可能差一两个样点）。"""
    if a.shape == shape:
        return a
    out = np.zeros(shape, dtype=np.float64)
    ns = min(shape[0], a.shape[0])
    ntr = min(shape[1], a.shape[1])
    out[:ns, :ntr] = a[:ns, :ntr]
    return out


def run_cnn(data, dt, dz=None, params: dict | None = None):
    """CNN 去噪入口（契约同名）。

    返回
    ----
    (out_float32, info)；info 含 notes / params_used / warnings /
    ``class_probs`` / ``gating_weights`` / ``alpha`` / ``device`` / ``energy_ratio``。
    """
    if params is not None and not isinstance(params, dict):
        raise ValueError(f"params 必须是 dict 或 None，实际 {type(params).__name__}")
    a, dt, warnings = _prepare(data, dt)
    notes: list[str] = []
    merged = copy.deepcopy(_DEFAULTS)
    if params:
        for k, v in params.items():
            if k in merged:
                merged[k] = v
            else:
                warnings.append(f"run_cnn 不认识参数 {k!r}，已忽略")
    if not cnn_available():
        raise RuntimeError(
            "CNN 不可用：需要 torch 以及权重文件 "
            f"{_SEP_PATH.name}/{_CLS_PATH.name}（目录 {_WEIGHT_DIR}）")
    try:
        import torch
    except Exception as exc:  # pragma: no cover - cnn_available 已检查
        raise RuntimeError(f"CNN 不可用：torch 导入失败（{exc}）") from exc

    alpha = float(np.clip(float(merged["alpha"]), 0.0, 1.0))
    merged["alpha"] = alpha
    device = _resolve_device(merged.get("device", "auto"), warnings)
    merged["device"] = device
    shape = a.shape
    ns, ntr = shape

    # ---- 可选：重采样到训练规范网格 ----
    resample_on = bool(merged.get("resample", False))
    if resample_on:
        dz_val = None
        try:
            dz_val = float(dz) if dz is not None else None
        except (TypeError, ValueError):
            dz_val = None
        if dz_val is None or not math.isfinite(dz_val) or dz_val <= 0:
            warnings.append("resample=True 但 dz 缺失/无效，本次不重采样直接推理")
            resample_on = False
        else:
            from ..assets.cnnv2.vsdn2 import config as C
            ai = resample_to(a, dt, dz_val, C.REF_DT, C.REF_DZ)
            notes.append(f"已重采样到规范网格 {C.REF_DT * 1000:.1f} ms / {C.REF_DZ:.1f} m "
                         f"（原 {dt * 1000:.2f} ms / {dz_val:.2f} m）")
            if ai.shape[0] < 32 or ai.shape[1] < 4:
                raise RuntimeError(f"重采样后尺寸过小：{ai.shape}")
    else:
        ai = a

    # ---- 推理（CUDA 失败自动降级 CPU 一次） ----
    try:
        try:
            probs, gw, clean = _infer(ai, merged, device, warnings, notes)
        except Exception as exc:
            if device == "cuda":
                warnings.append(f"CUDA 推理失败（{type(exc).__name__}: {exc}），已降级 CPU 重试")
                device = "cpu"
                merged["device"] = "cpu"
                probs, gw, clean = _infer(ai, merged, device, warnings, notes)
            else:
                raise
    except Exception as exc:
        raise RuntimeError(f"CNN 推理失败：{type(exc).__name__}: {exc}") from exc

    # ---- 若重采样：把"去除分量"插值回原网格再相减，避免二次插值改变有效波 ----
    if resample_on:
        from ..assets.cnnv2.vsdn2 import config as C
        removed = ai.astype(np.float64) - clean
        removed_back = resample_to(removed, C.REF_DT, C.REF_DZ, dt,
                                   float(dz) if dz is not None else None)
        removed_back = _fit_shape(np.asarray(removed_back, dtype=np.float64), shape)
        out = a.astype(np.float64) - removed_back
        denom = float(np.mean(ai.astype(np.float64) ** 2))
    else:
        out = clean
        denom = float(np.mean(a.astype(np.float64) ** 2))

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out = _fit_shape(out, shape)
    energy_ratio = float(np.mean(out.astype(np.float64) ** 2) / max(denom, 1e-30))
    if denom <= 0:
        warnings.append("输入全零，energy_ratio 记 0")

    try:
        from ..assets.cnnv2.vsdn2 import config as C
        classes = list(C.类别)
    except Exception:  # pragma: no cover
        classes = [f"class_{i}" for i in range(len(probs))]

    info = {
        "notes": "CNN 分类 + 门控分离："
                 + "；".join(notes + [
                     f"{len(classes)} 类噪声概率分类，"
                     f"门控{'开启' if merged['gate'] else '关闭'}，alpha={alpha:.2f}，"
                     f"device={device}，patch={merged['patch_t']}×{merged['patch_x']}"]),
        "params_used": dict(merged),
        "warnings": warnings,
        "class_probs": {c: round(float(v), 4) for c, v in zip(classes, probs)},
        "gating_weights": {c: round(float(v), 4) for c, v in zip(classes, gw)},
        "alpha": alpha,
        "device": device,
        "energy_ratio": round(float(energy_ratio), 4),
        "resampled": bool(resample_on),
        "checkpoint": copy.deepcopy(_CKPT_INFO.get(device, {})),
    }
    return np.asarray(out, dtype=np.float32), info
