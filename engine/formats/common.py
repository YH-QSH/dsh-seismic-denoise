"""格式层公共小工具：单位归一化、数组方向整理、道间距估计。

约定（与 docs/接口约定.md 一致）：
- 数据矩阵形状 ``(ns, ntr)``：ns = 采样点数（时间方向），ntr = 道数；
- ``dt`` 单位秒，``dz`` 单位米；
- 所有对外数值都用普通 Python float / int，方便 JSON 序列化。
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# 数值小工具
# --------------------------------------------------------------------------
def clamp01(x: Any) -> float:
    """把数值夹到 [0, 1]，NaN 当 0。"""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return float(min(1.0, max(0.0, v)))


def human_size(n: Any) -> str:
    """字节数转可读字符串。"""
    try:
        v = float(n)
    except (TypeError, ValueError):
        return "0 B"
    if v < 1024:
        return f"{int(v)} B"
    for unit in ("KB", "MB", "GB", "TB"):
        v /= 1024.0
        if v < 1024 or unit == "TB":
            return f"{v:.1f} {unit}"
    return f"{v:.1f} TB"


def _first_scalar(value: Any) -> float | None:
    """从标量 / 长度 1 的数组里取出一个 float；取不到返回 None。"""
    try:
        arr = np.asarray(value)
        if arr.size != 1:
            return None
        return float(arr.ravel()[0])
    except (TypeError, ValueError):
        return None


def normalize_dt(value: Any) -> tuple[float | None, str]:
    """把各种单位的采样间隔归一化成**秒**。

    现实里 dt 的单位五花八门：SEG-Y 用微秒，MAT 的 header.dt 常用微秒，
    NPZ/HDF5 里可能是秒或毫秒。这里按量级判断，返回 ``(dt_秒, 单位说明)``：

    - ``< 0.5``  → 秒
    - ``0.5 ~ 50`` → 毫秒
    - ``>= 50``  → 微秒

    常规地震采样率 0.5–20 ms，三种单位落在这三个区间里互不重叠，够用。
    判定不出来（非正数 / 非有限值 / 过大）返回 ``(None, "")``。
    """
    v = _first_scalar(value)
    if v is None or not math.isfinite(v) or v <= 0:
        return None, ""
    if v < 0.5:
        return v, "秒"
    if v < 50.0:
        return v / 1000.0, "毫秒"
    if v <= 1_000_000.0:
        return v / 1e6, "微秒"
    return None, ""


# --------------------------------------------------------------------------
# 二维数组整理
# --------------------------------------------------------------------------
def materialize_array(
    arr: Any,
    max_traces: int | None = None,
    warnings: list[str] | None = None,
    ntr_hint: int | None = None,
    name: str = "数组",
) -> tuple[np.ndarray, bool, list[str]]:
    """把任意二维数值数组整理成 ``(ns, ntr)`` float32 连续数组。

    方向判定（地震数据几乎总是时间点多于道数）：

    1. 有 ``ntr_hint``（比如来自道头 / 元数据）时以它为准；
    2. 否则 ``行数 < 列数`` 视为 ``(ntr, ns)`` 并转置。

    返回 ``(data, transposed, warnings)``。``max_traces`` 在*转置之后*按道截断，
    先截断再复制，避免大数组白占内存。
    """
    warnings = warnings if warnings is not None else []
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError(f"{name}必须是二维，实际 shape={a.shape}")
    if a.size == 0:
        raise ValueError(f"{name}是空数组，shape={a.shape}")
    if not np.issubdtype(a.dtype, np.number):
        raise ValueError(f"{name}不是数值类型（dtype={a.dtype}）")

    rows, cols = int(a.shape[0]), int(a.shape[1])
    transposed = False
    hint = int(ntr_hint) if ntr_hint else 0
    if hint > 1:
        if rows == hint and cols != hint:
            transposed = True
        elif cols == hint:
            transposed = False
        else:
            transposed = rows < cols
    else:
        transposed = rows < cols

    view = a.T if transposed else a
    if max_traces is not None and view.shape[1] > int(max_traces):
        warnings.append(
            f"按 max_traces={int(max_traces)} 截断：原 {view.shape[1]} 道，仅保留前 {int(max_traces)} 道"
        )
        view = view[:, : int(max_traces)]

    data = np.ascontiguousarray(view, dtype=np.float32)
    if transposed:
        # 记一笔方向判定，报告里可以说明；不算警告
        pass
    return data, transposed, warnings


# --------------------------------------------------------------------------
# 道间距估计
# --------------------------------------------------------------------------
def _apply_elev_scalar(values: np.ndarray, scalar_values: Any) -> tuple[np.ndarray, float | None]:
    """按 SEG-Y 惯例应用高程/深度标定因子。

    正数表示乘，负数表示除。因子明显不合理（0 / 非有限 / 过快发散）时忽略。
    """
    scalar = _first_scalar(np.median(np.asarray(scalar_values, dtype=np.float64))) \
        if scalar_values is not None else None
    # 上面的写法只对长度 1 的数组可靠，下面改用过滤后的中位数更稳
    try:
        s = np.asarray(scalar_values, dtype=np.float64).ravel()
        s = s[np.isfinite(s) & (s != 0)]
        scalar = float(np.median(s)) if s.size else None
    except (TypeError, ValueError):
        scalar = None
    if scalar is None or not math.isfinite(scalar) or abs(scalar) < 1e-9:
        return values, None
    if abs(scalar) > 1e6:
        return values, None
    scaled = values * scalar if scalar > 0 else values / abs(scalar)
    return scaled, scalar


def estimate_dz_from_series(
    values: Any,
    scalar_values: Any = None,
    label: str = "接收点高程",
) -> tuple[float | None, dict]:
    """从逐道高程 / 深度序列估计道间距 dz（米）。

    步骤：

    1. 应用高程标定因子（若提供）；
    2. 取相邻道差值的绝对值；中位数 > 100 时按“单位是厘米”再除 100
       （SERCEL 等系统常见写法，接口约定也要求这个兜底）；
    3. 差值足够多时取**众数**（1 mm 分辨率）作 dz——真实资料里局部会有
       缺失道、变间距，众数比中位数更接近标称道间距；
    4. 做合理性检查（0.05–200 m），并给出分布离散度信息供写警告。

    返回 ``(dz 或 None, info)``；info 里含所用方法与分布统计。
    """
    v = np.asarray(values, dtype=np.float64).ravel()
    v = v[np.isfinite(v)]
    if v.size < 3 or np.unique(v).size < 2:
        return None, {}

    scaled, scalar_used = _apply_elev_scalar(v, scalar_values)
    diffs = np.abs(np.diff(scaled))
    diffs = diffs[np.isfinite(diffs) & (diffs > 1e-9)]
    if diffs.size < 2:
        return None, {}

    med = float(np.median(diffs))
    if med <= 0:
        return None, {}

    unit = "米"
    if med > 100.0:
        # 相邻道高程差超过 100 m 不现实，几乎可以肯定是厘米
        diffs = diffs / 100.0
        med = med / 100.0
        unit = "厘米→米"

    support = 0.0
    if diffs.size >= 6:
        q = np.round(diffs, 3)
        uniq, counts = np.unique(q, return_counts=True)
        idx = int(np.argmax(counts))
        mode = float(uniq[idx])
        support = float(counts[idx]) / float(diffs.size)
    else:
        mode = med

    dz = mode if support >= 0.15 else med
    if not (0.05 <= dz <= 200.0):
        return None, {}

    p5, p50, p95 = (float(x) for x in np.percentile(diffs, [5, 50, 95]))
    spread = (p95 - p5) / dz if dz else 0.0
    info = {
        "method": f"{label}逐道差",
        "unit": unit,
        "scalar": scalar_used,
        "n_points": int(v.size),
        "n_diffs": int(diffs.size),
        "median": round(float(p50), 4),
        "p5": round(p5, 4),
        "p95": round(p95, 4),
        "mode": round(float(mode), 4),
        "mode_support": round(support, 3),
        "spread": round(float(spread), 3),
        "uniform": bool(spread <= 0.35),
    }
    return float(dz), info


def dz_irregular_warning(dz: float | None, info: dict, label: str = "接收点高程") -> str | None:
    """道间距明显不均匀时给一条人话警告；均匀返回 None。"""
    if dz is None or not info or info.get("uniform", True):
        return None
    return (
        f"{label}逐道差分布不均匀：中位 {info.get('median')} m、5%–95% 分位 "
        f"{info.get('p5')}–{info.get('p95')} m，dz 取众数 {dz:.3f} m"
        "（可能存在缺失道/变间距，走时换算时注意）"
    )
