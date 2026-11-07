"""地震数据格式层：SEG-Y / MAT / NPY / NPZ / HDF5 / RAW 的统一识别与读取。

对外契约（见 docs/接口约定.md）：
- :func:`detect_format`     识别文件格式
- :func:`read_any`          统一读取入口
- :func:`read_segy_headers` 只解析 SEG-Y 道头
- :func:`analyze_binary`    任意二进制初判
"""
from __future__ import annotations

from .analyze import analyze_binary
from .common import clamp01, human_size, normalize_dt
from .containers import read_hdf5, read_mat, read_npy, read_npz
from .detect import detect_format
from .raw import read_raw, scan_candidates
from .reader import read_any
from .segy import (
    BASE_SIZE,
    BIN_SIZE,
    TEXT_SIZE,
    TRACE_HEADER_SIZE,
    ibm_to_ieee,
    ieee_to_ibm,
    probe_segy,
    read_segy,
    read_segy_headers,
    write_segy,
)

__all__ = [
    # 契约必须导出
    "detect_format",
    "read_any",
    "read_segy_headers",
    "analyze_binary",
    # 分格式读取器（pipeline/analysis 可能直接用）
    "read_segy",
    "read_mat",
    "read_npy",
    "read_npz",
    "read_hdf5",
    "read_raw",
    # 辅助
    "scan_candidates",
    "probe_segy",
    "write_segy",
    "ibm_to_ieee",
    "ieee_to_ibm",
    "normalize_dt",
    "clamp01",
    "human_size",
    # 常量
    "TEXT_SIZE",
    "BIN_SIZE",
    "BASE_SIZE",
    "TRACE_HEADER_SIZE",
]
