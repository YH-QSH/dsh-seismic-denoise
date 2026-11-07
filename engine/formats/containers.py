"""容器格式读取：MAT / NPY / NPZ / HDF5。

这些格式没有 SEG-Y 那样的公共道头标准，所以策略是“尽量找、找不到就给默认”：

* MAT：优先识别 ``data + header + trace_headers`` 嵌套结构（vsdn2 等工具
  导出的 MAT 就是这种），识别不到就遍历变量挑最像地震数据的二维数组；
* NPY：取唯一的二维数组；
* NPZ：取最大的二维数组，并顺带找 ``dt`` / ``dz`` 等同义标量；
* HDF5：递归找最大的二维数值数据集，并从数据集/父组/根组的属性里找 dt/dz。

所有读取结果都统一整理成 ``(ns, ntr)`` float32，dt 秒、dz 米。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..model import HeaderTable, SeismicDataset
from ..utils import clean_array
from .common import (
    clamp01,
    dz_irregular_warning,
    estimate_dz_from_series,
    materialize_array,
    normalize_dt,
)

# dt / dz 的常见键名（统一小写比较）
_DT_KEYS = (
    "dt", "dt_s", "dt_sec", "dt_ms", "dt_us", "dt_usec",
    "sample_interval", "sampleinterval", "sampling_interval", "samplinginterval",
    "sample_interval_s", "sample_interval_ms", "sample_interval_us", "sample_dt",
)
_DZ_KEYS = (
    "dz", "dz_m", "spacing", "trace_spacing", "tracespacing",
    "group_interval", "groupinterval", "receiver_spacing", "receiverspacing",
    "station_spacing", "stationinterval", "receiver_interval", "spatial_interval",
)

# MAT trace_headers -> 公共道头字段（按优先级尝试的字段名）
_MAT_HEADER_MAP: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("receiver_elev", ("ReceiverGroupElevation", "ReceiverElevation", "receiver_elev",
                       "RecGroupElev", "GroupElevation", "elev")),
    ("source_elev", ("SourceElevation", "source_elev", "ShotElevation", "SourceDepth")),
    ("receiver_depth", ("ReceiverGroupDepth", "ReceiverDepth", "receiver_depth")),
    ("offset", ("offset", "Offset", "SourceToReceiverDistance", "SourceReceiverDistance")),
    ("source_x", ("SourceX", "source_x", "SourceCoordinateX", "ShotX")),
    ("source_y", ("SourceY", "source_y", "SourceCoordinateY", "ShotY")),
    ("group_x", ("GroupX", "group_x", "GroupCoordinateX", "ReceiverX")),
    ("group_y", ("GroupY", "group_y", "GroupCoordinateY", "ReceiverY")),
    ("cdp", ("CDP", "cdp", "EnsembleNumber", "Ensemble")),
    ("shotpoint", ("ShotPoint", "shotpoint", "Shotpoint", "EnergySourcePoint")),
)


# --------------------------------------------------------------------------
# 公共：组装 SeismicDataset
# --------------------------------------------------------------------------
def _max_abs_chunked(arr: np.ndarray, step: int = 4_000_000) -> float:
    """分块求 max|值|，避免对大数组整块取 abs 多占一份内存。"""
    flat = arr.reshape(-1)
    peak = 0.0
    for i in range(0, flat.size, step):
        chunk = np.abs(flat[i:i + step])
        value = float(chunk.max())
        if value > peak:
            peak = value
    return peak


def _build(
    path: Path,
    fmt: str,
    arr: Any,
    *,
    dt: float | None,
    dz: float | None,
    headers: HeaderTable | None = None,
    meta: dict | None = None,
    confidence: float = 0.6,
    warnings: list[str] | None = None,
    max_traces: int | None = None,
    text: str = "",
    ntr_hint: int | None = None,
    note_missing: bool = True,
) -> SeismicDataset:
    """统一整理数组方向、清理非有限值、补齐缺失元数据警告。"""
    warnings = list(warnings or [])
    meta = dict(meta or {})
    data, transposed, warnings = materialize_array(
        arr, max_traces, warnings, ntr_hint=ntr_hint, name="数据矩阵"
    )
    meta["transposed"] = bool(transposed)

    if dt is None and note_missing:
        warnings.append("未找到采样间隔 dt（秒），频域类方法需要显式指定 dt")
    if dz is None and note_missing:
        warnings.append("未找到道间距 dz（米），走时/空间换算会受限")

    data, clean_warnings = clean_array(data)
    warnings.extend(clean_warnings)
    if data.size:
        amp = _max_abs_chunked(data)
        if amp > 1e20:
            warnings.append(f"数据振幅异常大（max|值|={amp:.3g}），可能格式识别有误")
            confidence = min(confidence, 0.3)

    return SeismicDataset(
        path=str(path),
        format=fmt,
        data=data,
        dt=dt,
        dz=dz,
        headers=headers if headers is not None else HeaderTable(),
        text_header=text,
        meta=meta,
        confidence=clamp01(confidence),
        warnings=warnings,
        source_kind=fmt,
    )


# --------------------------------------------------------------------------
# NPY
# --------------------------------------------------------------------------
def _squeeze_to_2d(arr: np.ndarray, warnings: list[str]) -> np.ndarray:
    """1 维当单道，3 维尝试 squeeze 掉长度为 1 的轴。"""
    if arr.ndim == 1:
        warnings.append("一维数组按单道处理（形状改为 (ns, 1)）")
        return arr.reshape(-1, 1)
    if arr.ndim == 3:
        squeezed = np.squeeze(arr)
        if squeezed.ndim == 2:
            warnings.append(f"三维数组已压缩长度 1 的轴：{arr.shape} → {squeezed.shape}")
            return squeezed
        raise ValueError(f"不支持的三维数组 shape={arr.shape}")
    if arr.ndim != 2:
        raise ValueError(f"只支持一维/二维数组，实际 shape={arr.shape}")
    return arr


def read_npy(path: str | Path, max_traces: int | None = None) -> SeismicDataset:
    """读取 .npy；取其中唯一的二维数组。"""
    p = Path(path)
    warnings: list[str] = []
    arr = np.load(p, allow_pickle=False, mmap_mode="r")
    if not np.issubdtype(arr.dtype, np.number):
        raise ValueError(f"NPY 数组不是数值类型（dtype={arr.dtype}）")
    if np.iscomplexobj(arr):
        warnings.append("NPY 数组为复数，已取模值作为实地震数据")
        arr = np.abs(arr)
    arr = _squeeze_to_2d(arr, warnings)

    meta = {
        "file_size": int(p.stat().st_size),
        "stored_shape": list(arr.shape),
        "stored_dtype": str(arr.dtype),
    }
    return _build(
        p, "npy", arr, dt=None, dz=None, meta=meta,
        confidence=0.55, warnings=warnings, max_traces=max_traces,
    )


# --------------------------------------------------------------------------
# NPZ
# --------------------------------------------------------------------------
def _find_scalar(container: Any, keys: tuple[str, ...]) -> float | None:
    """在 dict / NpzFile / h5py attrs 里按候选键找标量。"""
    if container is None:
        return None
    items = None
    if hasattr(container, "files"):  # NpzFile
        items = [(k, container[k]) for k in container.files]
    elif hasattr(container, "keys"):  # dict / attrs
        items = [(k, container[k]) for k in container.keys()]
    if not items:
        return None
    wanted = {k.lower() for k in keys}
    for name, value in items:
        base = str(name).rsplit("/", 1)[-1].lower()
        if base not in wanted:
            continue
        try:
            arr = np.asarray(value).ravel()
            if arr.size == 1:
                return float(arr[0])
        except (TypeError, ValueError):
            continue
    return None


def read_npz(path: str | Path, max_traces: int | None = None) -> SeismicDataset:
    """读取 .npz；取最大的二维数值数组，并尝试读取 dt / dz。"""
    p = Path(path)
    warnings: list[str] = []
    with np.load(p, allow_pickle=False) as z:
        candidates = []
        for key in z.files:
            try:
                a = z[key]
            except (ValueError, OSError):
                continue
            if a.ndim != 2 or a.size == 0:
                continue
            if not np.issubdtype(a.dtype, np.number) or np.iscomplexobj(a):
                continue
            candidates.append((int(a.size), key, a))
        if not candidates:
            raise ValueError(f"NPZ 里没有找到二维数值数组（键：{list(z.files)}）")
        candidates.sort(key=lambda item: -item[0])
        _size, key, arr = candidates[0]
        dt_raw = _find_scalar(z, _DT_KEYS)
        dz_raw = _find_scalar(z, _DZ_KEYS)
        all_keys = list(z.files)

    dt, dt_unit = normalize_dt(dt_raw)
    if dt is None and dt_raw is not None:
        warnings.append(f"NPZ 里的 dt={dt_raw} 无法判断单位，已忽略")
    dz = None
    dz_warn = None
    if dz_raw is not None:
        if dz_raw > 100.0:
            dz = dz_raw / 100.0
            dz_warn = f"NPZ 里的 dz={dz_raw} 数值像厘米，已按 /100 换算为 {dz:.3f} m"
        else:
            dz = float(dz_raw)
            if not (0.05 <= dz <= 200.0):
                warnings.append(f"NPZ 里的 dz={dz_raw} 不在合理范围（0.05–200 m），已忽略")
                dz = None
    if dz_warn:
        warnings.append(dz_warn)
    if dt is not None and dt_unit and dt_unit != "秒":
        warnings.append(f"NPZ 里的 dt 按{dt_unit}解释，已归一化为秒")

    meta = {
        "file_size": int(p.stat().st_size),
        "data_key": key,
        "keys": all_keys,
        "stored_shape": list(arr.shape),
        "stored_dtype": str(arr.dtype),
        "dt_raw": dt_raw,
        "dt_unit": dt_unit,
        "dz_raw": dz_raw,
    }
    confidence = 0.55 + (0.20 if dt is not None else 0.0) + (0.05 if dz is not None else 0.0)
    return _build(
        p, "npz", arr, dt=dt, dz=dz, meta=meta,
        confidence=confidence, warnings=warnings, max_traces=max_traces,
    )


# --------------------------------------------------------------------------
# MAT
# --------------------------------------------------------------------------
def _unwrap_mat(obj: Any) -> Any:
    """把 MATLAB 的 1x1 object 数组剥成内部对象。"""
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        return obj.ravel()[0]
    return obj


def _mat_fields(obj: Any) -> list[str]:
    if obj is None:
        return []
    names = getattr(obj, "_fieldnames", None)
    if names:
        return list(names)
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return list(obj.dtype.names)
    return []


def _mat_get(obj: Any, key: str) -> Any:
    """从 mat_struct / 结构化数组里按字段名取值。"""
    if obj is None:
        return None
    if isinstance(obj, np.ndarray) and obj.dtype.names and key in obj.dtype.names:
        return obj[key]
    try:
        if hasattr(obj, key):
            return getattr(obj, key)
    except (AttributeError, ValueError):
        pass
    return None


def _mat_to_2d(obj: Any) -> np.ndarray | None:
    """把 MAT 变量整理成二维数值数组；不是二维数值数组就返回 None。"""
    arr = _unwrap_mat(obj)
    if arr is None:
        return None
    try:
        a = np.asarray(arr)
    except (TypeError, ValueError):
        return None
    if a.ndim != 2 or a.size == 0:
        return None
    if a.dtype == object or not np.issubdtype(a.dtype, np.number):
        return None
    if np.iscomplexobj(a):
        a = np.abs(a)
    return a


def _mat_trace_values(obj: Any, names: tuple[str, ...]) -> np.ndarray | None:
    """从一个 trace_headers（struct 数组 / 单个 struct）里取逐道字段。"""
    obj = _unwrap_mat(obj)
    if obj is None:
        return None
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        for name in names:
            if name in obj.dtype.names:
                values = np.asarray(obj[name], dtype=np.float64).ravel()
                if values.size:
                    return values
        return None
    if isinstance(obj, np.ndarray) and obj.dtype == object:
        values = []
        for element in obj.ravel():
            element = _unwrap_mat(element)
            found = None
            for name in names:
                v = _mat_get(element, name)
                if v is not None:
                    try:
                        found = float(np.asarray(v).ravel()[0])
                    except (TypeError, ValueError, IndexError):
                        found = math.nan
                    break
            values.append(math.nan if found is None else found)
        arr = np.asarray(values, dtype=np.float64)
        return arr if np.isfinite(arr).any() else None
    for name in names:
        v = _mat_get(obj, name)
        if v is not None:
            arr = np.asarray(v, dtype=np.float64).ravel()
            if arr.size:
                return arr
    return None


def _mat_struct_candidate(obj: Any) -> dict | None:
    """识别 ``data + header + trace_headers`` 结构，返回候选信息。"""
    fields = _mat_fields(obj)
    if not fields:
        return None
    data = _mat_to_2d(_mat_get(obj, "data"))
    if data is None:
        return None

    dt = None
    dt_unit = ""
    dt_raw = None
    header = _unwrap_mat(_mat_get(obj, "header"))
    if header is not None:
        for key in ("dt", "dtOrig", "dt_orig", "DT", "sample_interval",
                    "sampling_interval", "dt_ms", "dt_us"):
            raw = _mat_get(header, key)
            if raw is None:
                continue
            # 有些 MAT 的 header 是 1xN 结构体数组，取第一个
            raw = np.asarray(raw).ravel()[0] if np.asarray(raw).size else None
            candidate, unit = normalize_dt(raw)
            if candidate is not None:
                dt, dt_unit, dt_raw = candidate, unit, raw
                break

    dz = None
    dz_info: dict = {}
    dz_warning = None
    headers = HeaderTable()
    trace_headers = _unwrap_mat(_mat_get(obj, "trace_headers"))
    if trace_headers is not None:
        for key, names in _MAT_HEADER_MAP:
            values = _mat_trace_values(trace_headers, names)
            if values is None:
                continue
            headers.add(key, key, 0, 4, "float", values, note="来自 MAT trace_headers")
            if key == "receiver_elev" and dz is None:
                dz, dz_info = estimate_dz_from_series(values, None, "MAT 接收点高程")
        if dz is None:
            depth_values = _mat_trace_values(trace_headers, ("ReceiverGroupDepth", "ReceiverDepth"))
            if depth_values is not None:
                dz, dz_info = estimate_dz_from_series(depth_values, None, "MAT 接收点深度")
        if dz is not None:
            dz_warning = dz_irregular_warning(dz, dz_info)

    return {
        "data": data,
        "dt": dt,
        "dt_unit": dt_unit,
        "dt_raw": dt_raw,
        "dz": dz,
        "dz_info": dz_info,
        "dz_warning": dz_warning,
        "headers": headers,
        "struct_name": fields,
    }


def read_mat(path: str | Path, max_traces: int | None = None) -> SeismicDataset:
    """读取 .mat（v5 用 scipy，v7.3 自动回退 HDF5）。"""
    import scipy.io

    p = Path(path)
    warnings: list[str] = []
    try:
        with p.open("rb") as f:
            head = f.read(512)
    except OSError:
        head = b""
    if b"\x89HDF\r\n\x1a\n" in head[:512]:
        warnings.append("MAT 为 v7.3（HDF5 内核），已改用 HDF5 读取器")
        return _read_hdf5_impl(p, fmt="mat", max_traces=max_traces, warnings=warnings)
    try:
        mat = scipy.io.loadmat(str(p), squeeze_me=False, struct_as_record=False)
    except NotImplementedError:
        warnings.append("MAT 为 v7.3（HDF5 格式），已改用 HDF5 读取器")
        return _read_hdf5_impl(p, fmt="mat", max_traces=max_traces, warnings=warnings)
    except ValueError as exc:
        # scipy 对 HDF5 内核的 .mat 有时只报 “Unknown mat file type”，一并按 v7.3 处理
        if "Unknown mat file type" in str(exc) or "v7.3" in str(exc):
            warnings.append("MAT 头不是 v5 格式（疑似 v7.3/HDF5），已改用 HDF5 读取器")
            return _read_hdf5_impl(p, fmt="mat", max_traces=max_traces, warnings=warnings)
        raise

    variables = {k: v for k, v in mat.items() if not k.startswith("__")}
    if not variables:
        raise ValueError("MAT 文件里没有可用变量")

    candidates: list[tuple[int, str, dict]] = []
    for name, value in variables.items():
        obj = _unwrap_mat(value)
        cand = _mat_struct_candidate(obj)
        if cand is not None:
            candidates.append((int(cand["data"].size), name, cand))
    candidates.sort(key=lambda item: -item[0])

    if candidates:
        _size, name, cand = candidates[0]
        data = cand["data"]
        headers = cand["headers"]
        meta = {
            "file_size": int(p.stat().st_size),
            "selected_variable": name,
            "struct_fields": cand["struct_name"],
            "stored_shape": list(data.shape),
            "stored_dtype": str(data.dtype),
            "dt_raw": cand["dt_raw"],
            "dt_unit": cand["dt_unit"],
            "dz_method": (cand["dz_info"] or {}).get("method"),
            "dz_info": cand["dz_info"],
            "header_fields": list(headers.fields.keys()),
        }
        if cand["dt"] is not None and cand["dt_unit"] and cand["dt_unit"] != "秒":
            warnings.append(f"MAT header.dt 按{cand['dt_unit']}解释，已归一化为秒")
        if cand["dz_warning"]:
            warnings.append(cand["dz_warning"])
        confidence = 0.80 + (0.05 if cand["dt"] is not None else 0.0)
        if len(candidates) > 1:
            warnings.append(
                f"MAT 里有 {len(candidates)} 个 data+header 结构，已选数据量最大的「{name}」"
            )
        return _build(
            p, "mat", data, dt=cand["dt"], dz=cand["dz"], headers=headers,
            meta=meta, confidence=confidence, warnings=warnings, max_traces=max_traces,
        )

    # ---- 兜底：遍历变量找最大二维数值数组 ----
    best: tuple[int, str, np.ndarray] | None = None
    for name, value in variables.items():
        arr = _mat_to_2d(value)
        if arr is None:
            continue
        if best is None or arr.size > best[0]:
            best = (int(arr.size), name, arr)
    if best is None:
        raise ValueError(f"MAT 里没有找到二维数值数据（变量：{list(variables)}）")

    _size, name, arr = best
    dt = None
    dt_unit = ""
    dt_raw = None
    for key in ("dt", "DT", "sample_interval", "sampling_interval", "dt_ms", "dt_us"):
        if key in variables:
            cand_dt, unit = normalize_dt(variables[key])
            if cand_dt is not None:
                dt, dt_unit, dt_raw = cand_dt, unit, variables[key]
                break
    if dt is not None and dt_unit and dt_unit != "秒":
        warnings.append(f"MAT 顶层变量 {key} 按{dt_unit}解释，已归一化为秒")
    warnings.append(
        f"MAT 没有 data+header 结构，已退化为选中最大的二维变量「{name}」，dt/dz 可能缺失"
    )
    meta = {
        "file_size": int(p.stat().st_size),
        "selected_variable": name,
        "variables": list(variables.keys()),
        "stored_shape": list(arr.shape),
        "stored_dtype": str(arr.dtype),
        "dt_raw": dt_raw,
        "dt_unit": dt_unit,
    }
    return _build(
        p, "mat", arr, dt=dt, dz=None, meta=meta,
        confidence=0.60, warnings=warnings, max_traces=max_traces,
    )


# --------------------------------------------------------------------------
# HDF5
# --------------------------------------------------------------------------
def _h5_attr(owners: list[Any], keys: tuple[str, ...]) -> tuple[float | None, str | None]:
    """在若干属性容器里按候选键找标量，返回 (值, 命中的键)。"""
    wanted = {k.lower() for k in keys}
    for attrs in owners:
        if attrs is None:
            continue
        try:
            names = list(attrs.keys())
        except (AttributeError, TypeError):
            continue
        for name in names:
            if str(name).lower() not in wanted:
                continue
            try:
                arr = np.asarray(attrs[name]).ravel()
                if arr.size == 1:
                    return float(arr[0]), str(name)
            except (TypeError, ValueError):
                continue
    return None, None


def _read_hdf5_impl(
    p: Path,
    fmt: str = "hdf5",
    max_traces: int | None = None,
    warnings: list[str] | None = None,
) -> SeismicDataset:
    """HDF5 读取核心（MAT v7.3 也走这里）。"""
    import h5py

    warnings = list(warnings or [])
    with h5py.File(str(p), "r") as f:
        found: list[tuple[int, str]] = []

        def visit(name: str, obj: Any) -> None:
            if not isinstance(obj, h5py.Dataset):
                return
            if obj.ndim != 2 or obj.size == 0:
                return
            if not np.issubdtype(obj.dtype, np.number) or np.issubdtype(obj.dtype, np.complexfloating):
                return
            found.append((int(obj.size), obj.name))

        f.visititems(visit)
        if not found:
            raise ValueError("HDF5 里没有找到二维数值数据集")
        found.sort(key=lambda item: -item[0])
        _size, ds_name = found[0]
        ds = f[ds_name]

        # ---- dt / dz：数据集属性 → 父组属性 → 根属性 ----
        owners: list[Any] = [ds.attrs, f.attrs]
        parts = ds_name.strip("/").split("/")[:-1]
        for i in range(len(parts), 0, -1):
            group_path = "/" + "/".join(parts[:i])
            if group_path in f:
                owners.append(f[group_path].attrs)

        dt_raw, dt_key = _h5_attr(owners, _DT_KEYS)
        dz_raw, dz_key = _h5_attr(owners, _DZ_KEYS)
        dt, dt_unit = normalize_dt(dt_raw)

        dz = None
        if dz_raw is not None:
            if dz_raw > 100.0:
                dz = dz_raw / 100.0
                warnings.append(f"HDF5 属性 {dz_key}={dz_raw} 数值像厘米，已 /100 换算为 {dz:.3f} m")
            else:
                dz = float(dz_raw)
                if not (0.05 <= dz <= 200.0):
                    warnings.append(f"HDF5 属性 {dz_key}={dz_raw} 不在合理范围（0.05–200 m），已忽略")
                    dz = None

        # ---- 读数据（先按方向截断，避免整块读进来再剪）----
        rows, cols = int(ds.shape[0]), int(ds.shape[1])
        transposed = rows < cols
        orig_ntr = rows if transposed else cols
        truncated = False
        if max_traces is not None and orig_ntr > int(max_traces):
            truncated = True
            if transposed:
                block = ds[: int(max_traces), :]
            else:
                block = ds[:, : int(max_traces)]
            ntr_hint = int(max_traces)
        else:
            block = ds[()]
            ntr_hint = orig_ntr
        stored_shape = [rows, cols]
        stored_dtype = str(ds.dtype)
        file_size = int(p.stat().st_size)

    arr = np.asarray(block)
    if truncated:
        warnings.append(
            f"按 max_traces={int(max_traces)} 截断：原始约 {orig_ntr} 道，"
            f"仅保留前 {int(max_traces)} 道"
        )
    meta = {
        "file_size": file_size,
        "dataset": ds_name,
        "datasets": [name for _size, name in found[:10]],
        "stored_shape": stored_shape,
        "stored_dtype": stored_dtype,
        "dt_raw": dt_raw,
        "dt_key": dt_key,
        "dt_unit": dt_unit,
        "dz_raw": dz_raw,
        "dz_key": dz_key,
    }
    if dt is not None and dt_unit and dt_unit != "秒":
        warnings.append(f"HDF5 属性 {dt_key} 按{dt_unit}解释，已归一化为秒")
    confidence = 0.65 + (0.20 if dt is not None else 0.0) + (0.05 if dz is not None else 0.0)
    return _build(
        p, fmt, arr, dt=dt, dz=dz, meta=meta,
        confidence=confidence, warnings=warnings, max_traces=None,  # 已在上面截断
        ntr_hint=ntr_hint,
    )


def read_hdf5(path: str | Path, max_traces: int | None = None) -> SeismicDataset:
    """读取 HDF5：递归找最大的二维数值数据集。"""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"文件不存在：{p}")
    return _read_hdf5_impl(p, fmt="hdf5", max_traces=max_traces)
