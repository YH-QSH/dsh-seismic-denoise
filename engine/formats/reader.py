"""统一读取入口：先识别格式，再分派到对应 reader；失败时给出清晰的尝试记录。"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from ..model import SeismicDataset
from .containers import read_hdf5, read_mat, read_npy, read_npz
from .detect import detect_format
from .raw import read_raw
from .segy import BASE_SIZE, probe_segy, read_segy

_READERS: dict[str, Callable[..., SeismicDataset]] = {
    "segy": read_segy,
    "mat": read_mat,
    "npy": read_npy,
    "npz": read_npz,
    "hdf5": read_hdf5,
    "raw": read_raw,
}

_ZIP_MAGIC = b"PK\x03\x04"


def _segy_probe_score(p: Path, size: int) -> float:
    if size < BASE_SIZE:
        return 0.0
    try:
        return float(probe_segy(p).get("score", 0.0))
    except (ValueError, OSError, KeyError, IndexError, TypeError):
        return 0.0


def _finalize(ds: SeismicDataset, detected: str, reader_name: str) -> SeismicDataset:
    """补齐公共字段：float32 二维、meta 记录识别/读取路径、警告去重。"""
    data = np.asarray(ds.data)
    if data.ndim != 2:
        raise ValueError(f"reader {reader_name} 返回的数据不是二维：shape={data.shape}")
    if data.dtype != np.float32 or not data.flags["C_CONTIGUOUS"]:
        ds.data = np.ascontiguousarray(data, dtype=np.float32)
    if not isinstance(ds.warnings, list):
        ds.warnings = list(ds.warnings or [])
    if not isinstance(ds.meta, dict):
        ds.meta = {}
    ds.meta.setdefault("detected_format", detected)
    ds.meta["read_by"] = reader_name
    ds.meta["package"] = "engine.formats"
    if detected not in ("unknown", reader_name):
        ds.warnings.append(f"格式识别为 {detected}，但实际由 {reader_name} 读取器解析，请核对数据")
    # 同一句警告只保留一次（分块读取时可能重复）
    deduped: list[str] = []
    for msg in ds.warnings:
        if msg not in deduped:
            deduped.append(msg)
    ds.warnings = deduped
    return ds


def read_any(path: str | Path, max_traces: int | None = None) -> SeismicDataset:
    """读取任意受支持格式的地震数据，返回统一 SeismicDataset。"""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"文件不存在或不是普通文件：{p}")
    size = int(p.stat().st_size)
    if size <= 0:
        raise ValueError(f"空文件：{p}")

    fmt = detect_format(p)
    attempts: list[str] = []
    chain: list[str] = []

    if fmt in _READERS:
        chain.append(fmt)
    else:
        with p.open("rb") as f:
            magic = f.read(4)
        if magic == _ZIP_MAGIC:
            raise ValueError(
                f"文件是 ZIP 压缩包但不含 .npy 成员，不是受支持的 npz：{p}"
            )
        attempts.append(f"格式识别: unknown（魔数与扩展名都无法确定）")
        # 仍按“像不像 SEG-Y”决定先试谁，最后 RAW 兜底
        if _segy_probe_score(p, size) >= 0.35:
            chain.append("segy")
        chain.append("raw")

    for name in chain:
        try:
            ds = _READERS[name](p, max_traces=max_traces)
        except Exception as exc:  # 记录后继续尝试下一种解释
            attempts.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if fmt == "unknown":
            ds.warnings.insert(
                0,
                "无法通过魔数/扩展名确定格式，已按内置试探读取；请核对结果（尤其 ns/dt）",
            )
        elif name != fmt:
            ds.warnings.insert(0, f"按 {fmt} 读取失败，已回退为 {name} 解析")
        return _finalize(ds, fmt, name)

    tried = "；".join(attempts) if attempts else f"没有可用的读取器（识别结果 {fmt}）"
    raise ValueError(f"无法读取文件 {p}（{size} 字节）。已尝试：{tried}")
