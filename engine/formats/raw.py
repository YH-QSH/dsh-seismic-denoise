"""裸二进制（RAW）读取：没有道头时靠扫描猜测 dtype / 端序 / 每道采样点数。

思路
----
1. 从文件头部读一小段（默认 ≤2 MB），分别按 大端/小端 × float32/float64/
   int16/int32 解释；
2. 对每种解释枚举候选 ns（常见采样点数以及“文件总样本数 / 常见道数”），
   把前缀缓冲区 reshape 成 ``(ntr, ns)``，给候选打分：

   - **相邻道相似度**（横向连续性）：地震道相邻道波形相似；
   - **时间方向自相关**（纵向连续性）：一维信号相邻点相关；
   - **零值比例 / 振幅量级**：排除把字节解释成天文数字或全零的方案；
   - **总样本数能否被 ns 整除**：道长应能铺满文件。

3. 取得分最高的候选读数据。由于没有头信息，置信度封顶 0.45，
   并始终给出“按裸二进制推断”的警告。
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from ..model import HeaderTable, SeismicDataset
from ..utils import clean_array
from .common import clamp01, human_size, materialize_array

# 常见每道采样点数（地震仪器/处理流程里高频出现的值）
_NS_BASES = (
    32, 40, 48, 50, 64, 80, 100, 120, 128, 160, 200, 240, 250, 256, 300, 320,
    400, 480, 500, 512, 600, 640, 750, 800, 1000, 1024, 1200, 1250, 1500, 1600,
    2000, 2048, 2500, 3000, 3200, 4000, 4096, 5000, 6000, 8000, 10000, 12000,
    16000, 20000,
)
# 常见道数（用来反推不常见的 ns）
_NTR_BASES = (
    10, 16, 20, 24, 25, 32, 40, 50, 64, 80, 100, 120, 128, 160, 200, 240, 250,
    300, 400, 500, 640, 800, 1000, 1200, 1600, 2000, 2500, 3000, 4000, 5000,
    8000, 10000,
)
# 候选样本类型：(dtype 字符, 名称)
_DTYPES = (
    ("f4", "float32"),
    ("f8", "float64"),
    ("i2", "int16"),
    ("i4", "int32"),
)
_MAX_NS = 200_000
_MIN_NS = 16


def _ns_candidates(total_samples: int) -> list[int]:
    """枚举可能的一维道长（每道采样点数）。"""
    if total_samples < _MIN_NS * 2:
        return []
    found: set[int] = set()
    for ns in _NS_BASES:
        if _MIN_NS <= ns <= _MAX_NS and total_samples % ns == 0 and total_samples // ns >= 2:
            found.add(int(ns))
    for ntr in _NTR_BASES:
        if ntr >= 2 and total_samples % ntr == 0:
            ns = total_samples // ntr
            if _MIN_NS <= ns <= _MAX_NS:
                found.add(int(ns))
    return sorted(found)


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    """两个向量的余弦相似度（调用前已去均值），异常时返回 0。"""
    if x.size == 0:
        return 0.0
    nx = float(np.linalg.norm(x))
    ny = float(np.linalg.norm(y))
    if not math.isfinite(nx) or not math.isfinite(ny) or nx <= 0.0 or ny <= 0.0:
        return 0.0
    value = float(np.dot(x, y) / (nx * ny))
    if not math.isfinite(value):
        return 0.0
    return max(-1.0, min(1.0, value))


def _score_block(block: np.ndarray, ns: int, total_samples: int) -> float:
    """给一个 ``(ntr, ns)`` 解释打分（0–1）。"""
    m = int(block.shape[0])
    if m < 2 or ns < _MIN_NS:
        return 0.0
    # 打分只看 ≤512 个时间点，控制大 ns 时的计算量（不影响量级判断）
    stride = max(1, ns // 512)
    # 用 float64 统计：float32 在极端误解释（如把 float64 当 float32）时会溢出；
    # 有些字节模式会被解释成 NaN/Inf，转 float64 时的 cast 警告这里静默，下面会显式判非有限
    with np.errstate(invalid="ignore", over="ignore"):
        v = np.asarray(block[:, ::stride], dtype=np.float64)
    finite_ratio = float(np.isfinite(v).mean())
    if finite_ratio < 0.90:
        return float(0.15 * finite_ratio)
    v = np.where(np.isfinite(v), v, 0.0)
    # 振幅量级：裸字节被解释成 1e±30 的天文数字时基本可以判负
    peak = float(np.max(np.abs(v))) if v.size else 0.0
    v = v - v.mean(axis=1, keepdims=True)
    scale = float(np.max(np.abs(v))) if v.size else 0.0
    if scale > 0.0:
        v = v / scale  # 归一化后做相关，避免 norm 溢出

    spatial = _corr(v[1:].ravel(), v[:-1].ravel())
    temporal = _corr(v[:, 1:].ravel(), v[:, :-1].ravel())
    zero_frac = float(np.mean(v == 0.0))
    amp_ok = 1.0 if 1e-12 <= peak <= 1e12 else 0.0
    exact = 1.0 if total_samples % ns == 0 else 0.25

    score = (
        0.40 * max(spatial, 0.0)
        + 0.25 * max(temporal, 0.0)
        + 0.15 * (1.0 - zero_frac)
        + 0.10 * amp_ok
        + 0.10 * exact
    )
    return clamp01(score)


def _dtype_label(dtype: np.dtype) -> str:
    name = "float32" if dtype.kind == "f" and dtype.itemsize == 4 else None
    if name is None:
        name = "float64" if dtype.kind == "f" else ("int16" if dtype.itemsize == 2 else "int32")
    endian = "大端" if dtype.byteorder == ">" else "小端"
    return f"{name}（{endian}）"


def scan_candidates(path: str | Path, max_bytes: int = 2_000_000, top: int = 8) -> list[dict]:
    """扫描 RAW 候选解释，返回按得分降序的候选列表。"""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"文件不存在：{p}")
    size = int(p.stat().st_size)
    if size < 64:
        return []
    nbytes = int(min(size, max(64, int(max_bytes))))
    nbytes -= nbytes % 8
    if nbytes < 64:
        return []
    with p.open("rb") as f:
        raw = f.read(nbytes)

    results: list[dict] = []
    for endian in (">", "<"):
        for code, label in _DTYPES:
            dtype = np.dtype(endian + code)
            bps = dtype.itemsize
            total = size // bps
            if total < _MIN_NS * 2:
                continue
            usable = (len(raw) // bps) * bps
            flat = np.frombuffer(raw[:usable], dtype=dtype)
            for ns in _ns_candidates(total):
                if ns > flat.size:
                    continue
                ntr = int(total // ns)
                if ntr < 2:
                    continue
                m = max(2, min(ntr, 64, int(flat.size // ns)))
                block = flat[: m * ns].reshape(m, ns)
                score = _score_block(block, ns, total)
                if score <= 0.0:
                    continue
                results.append({
                    "dtype": dtype.str,
                    "dtype_label": label if dtype.byteorder == "|" else _dtype_label(dtype),
                    "endian": "big" if endian == ">" else "little",
                    "bytes_per_sample": bps,
                    "ns": int(ns),
                    "ntr": int(ntr),
                    "score": round(float(score), 4),
                })
    results.sort(key=lambda item: (-item["score"], -item["ns"]))
    # 同一 dtype 下 ns 相邻的候选容易互为倍数，保留前 top 个即可
    return results[: max(1, int(top))]


def read_raw(path: str | Path, max_traces: int | None = None) -> SeismicDataset:
    """按裸二进制推断读取数据；置信度封顶 0.45。"""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"文件不存在：{p}")
    size = int(p.stat().st_size)
    if size < 64:
        raise ValueError(f"文件太小（{size} 字节），无法按 RAW 推断")

    candidates = scan_candidates(p, max_bytes=min(size, 2_000_000), top=5)
    if not candidates:
        raise ValueError("RAW 扫描没有找到可行的 dtype/ns 组合（文件可能被压缩或不是样本数据）")
    best = candidates[0]
    dtype = np.dtype(best["dtype"])
    bps = int(dtype.itemsize)
    ns = int(best["ns"])
    total = int(size // bps)
    ntr = int(total // ns)
    if ntr < 2:
        raise ValueError(f"RAW 候选 ns={ns} 只能解释出 {ntr} 道，放弃")

    warnings: list[str] = [
        f"按裸二进制推断：{best['dtype_label']}，ns={ns}，ntr={ntr}，"
        f"打分 {best['score']:.2f}（置信度低，建议显式提供 ns/dt 或使用带道头的 SEG-Y）",
    ]
    if size % (ns * bps) != 0:
        warnings.append(f"文件末尾有 {size % (ns * bps)} 字节不足一整道，已忽略")

    n_read = ntr
    if max_traces is not None and ntr > int(max_traces):
        n_read = int(max_traces)
        warnings.append(f"按 max_traces={n_read} 截断：共推断出 {ntr} 道，仅读取前 {n_read} 道")

    out = np.empty((ns, n_read), dtype=np.float32)
    per_trace_bytes = ns * bps
    chunk_traces = max(1, min(256, (16 * 1024 * 1024) // max(1, per_trace_bytes)))
    with p.open("rb") as f:
        for i0 in range(0, n_read, chunk_traces):
            k = min(chunk_traces, n_read - i0)
            f.seek(i0 * per_trace_bytes)
            buf = f.read(k * per_trace_bytes)
            if len(buf) < k * per_trace_bytes:
                warnings.append(f"读取第 {i0 + k} 道附近遇到文件末尾，缺失部分补 0")
                buf = buf + bytes(k * per_trace_bytes - len(buf))
            vals = np.frombuffer(buf, dtype=dtype).reshape(k, ns)
            out[:, i0:i0 + k] = vals.astype(np.float32).T
    data, clean_warnings = clean_array(out)
    warnings.extend(clean_warnings)

    confidence = clamp01(round(0.10 + 0.45 * float(best["score"]), 3))
    confidence = min(confidence, 0.45)

    meta = {
        "file_size": size,
        "file_size_text": human_size(size),
        "dtype": best["dtype"],
        "dtype_label": best["dtype_label"],
        "endian": best["endian"],
        "bytes_per_sample": bps,
        "ns": ns,
        "ntr_total": ntr,
        "ntr_read": n_read,
        "trace_major": True,
        "scan_score": best["score"],
        "candidates": candidates,
        "assumed": "traces contiguous, (ntr, ns) row-major",
    }
    return SeismicDataset(
        path=str(p),
        format="raw",
        data=data,
        dt=None,
        dz=None,
        headers=HeaderTable(),
        text_header="",
        meta=meta,
        confidence=confidence,
        warnings=warnings,
        source_kind="raw",
    )
