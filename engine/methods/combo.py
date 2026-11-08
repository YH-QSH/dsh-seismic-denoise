"""组合去噪：把多个方法按顺序串成一条链。

设计约束：
- 每个阶段都接收上一步的完整 ``(ns, ntr)`` 剖面，不改变采样/道距；
- 阶段之间不重复做 fidelity / 门禁，整条链输出后由 runner 统一评价；
- 组合本身仍是一个 ``MethodResult``，报告里能列出每个阶段和方法参数。

链的表示（放在 ``MethodSpec.params`` 里）::

    {"chain": [
        {"key": "bandpass", "params": {"f_lo": 5, "f_hi": 80}},
        {"key": "fk", "params": {"vmin": 1200}},
    ]}
"""
from __future__ import annotations

import numpy as np

from .classical import run_method

_LABELS = {
    "bandpass": "带通",
    "notch": "陷波",
    "despike": "去尖峰",
    "median": "中值",
    "fk": "FK",
    "fx_decon": "F-X",
    "svd": "SVD",
    "spectral_sub": "谱减",
    "wavelet": "小波",
    "cnn": "CNN",
}

_ALLOWED = set(_LABELS)


def _label(key: str) -> str:
    return _LABELS.get(key, key)


def validate_chain(chain) -> list[dict]:
    """校验并规范化链；非法时抛 ValueError。"""
    if not isinstance(chain, (list, tuple)) or len(chain) < 2:
        raise ValueError("组合去噪至少需要两个阶段")
    out: list[dict] = []
    for i, stage in enumerate(chain):
        if not isinstance(stage, dict):
            raise ValueError(f"第 {i + 1} 个阶段必须是 dict")
        key = str(stage.get("key") or "").strip()
        if key not in _ALLOWED:
            raise ValueError(f"第 {i + 1} 个阶段方法 {key!r} 不可用于组合")
        params = stage.get("params")
        if params is not None and not isinstance(params, dict):
            raise ValueError(f"第 {i + 1} 个阶段参数必须是 dict")
        out.append({"key": key, "params": dict(params or {})})
    keys = [s["key"] for s in out]
    if len(set(keys)) != len(keys):
        raise ValueError("组合链里同一方法不能重复出现")
    return out


def combo_label(chain) -> str:
    try:
        return " → ".join(_label(str(s.get("key"))) for s in chain)
    except Exception:
        return "组合"


def run_chain(data, dt: float, dz: float | None, chain) -> tuple[np.ndarray, dict]:
    """顺序执行链，返回 ``(out_float32, info)``。异常向上抛给 runner。"""
    stages = validate_chain(chain)
    cur = np.asarray(data, dtype=np.float32)
    if cur.ndim != 2:
        raise ValueError(f"组合去噪要求二维剖面，实际 shape={cur.shape}")
    stage_infos: list[dict] = []
    warnings: list[str] = []
    for i, stage in enumerate(stages):
        key = stage["key"]
        params = dict(stage.get("params") or {})
        if key == "cnn":
            from .cnn import run_cnn
            nxt, info = run_cnn(cur, dt, dz, params)
        else:
            nxt, info = run_method(key, cur, dt, dz, params)
        arr = np.asarray(nxt, dtype=np.float32)
        if arr.shape != cur.shape:
            raise ValueError(f"阶段 {key} 输出 shape={arr.shape} 与输入 {cur.shape} 不一致")
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"阶段 {key} 输出包含非有限值")
        cur = arr
        info = dict(info or {})
        stage_infos.append({
            "index": i + 1,
            "key": key,
            "label": _label(key),
            "params": params,
            "notes": str(info.get("notes") or ""),
            "warnings": list(info.get("warnings") or []),
        })
        warnings.extend(str(w) for w in (info.get("warnings") or []))

    notes = "组合去噪：" + " → ".join(f"{s['label']}" for s in stage_infos)
    return cur.astype(np.float32, copy=False), {
        "notes": notes,
        "stages": stage_infos,
        "params_used": {"chain": stages},
        "warnings": warnings,
    }
