"""SEG-Y 读取实现：文本头 / 二进制头 / 扩展文本头 / 道头 / IBM 浮点 / 变长道。

设计要点
--------
* **道头字节一律 1-based**：与 SEG-Y 标准、接口约定一致；内部换算成
  0-based 只出现在 ``byte - 1`` 一处。
* 大小端自动判别：分别用大端/小端解析 400 字节二进制头，按
  “格式码合法 + ns/dt 合理 + 道长与文件大小对齐”打分选优。
* 文本头按 EBCDIC(cp500) / ASCII / latin-1 依次尝试，按可打印字符比例
  和 SEG-Y 标记（``C 1``、``CLIENT`` 等）打分选优。
* dt 优先取道头（μs → 秒），取不到再用二进制头 3217 字节处的值。
* dz 用接收点高程逐道差估计（>100 视为厘米），拿不到再用接收点深度。
* 大文件分块读样本，峰值内存约为数据本身 1.2 倍。
"""
from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from ..model import HeaderTable, SeismicDataset
from ..utils import clean_array
from .common import (
    clamp01,
    dz_irregular_warning,
    estimate_dz_from_series,
    human_size,
)

# 段大小（字节）
TEXT_SIZE = 3200
BIN_SIZE = 400
BASE_SIZE = TEXT_SIZE + BIN_SIZE
TRACE_HEADER_SIZE = 240

# 采样格式码 -> numpy dtype 字符（不含字节序）。契约要求 1/2/3/5/8，
# 另顺带支持 6/9/10/11/12/16，不额外增加复杂度。
_FMT_DTYPE: dict[int, str] = {
    1: "u4",   # IBM 32 位浮点（先按 uint32 取出再换算）
    2: "i4",
    3: "i2",
    5: "f4",
    6: "f8",
    8: "i1",
    9: "i8",
    10: "u4",
    11: "u2",
    12: "u8",
    16: "u1",
}
_FMT_BPS: dict[int, int] = {k: np.dtype(v).itemsize for k, v in _FMT_DTYPE.items()}
_FMT_NAME: dict[int, str] = {
    1: "4字节 IBM 浮点",
    2: "4字节整型",
    3: "2字节整型",
    5: "4字节 IEEE 浮点",
    6: "8字节 IEEE 浮点",
    8: "1字节整型",
    9: "8字节整型",
    10: "4字节无符号整型",
    11: "2字节无符号整型",
    12: "8字节无符号整型",
    16: "1字节无符号整型",
}

# 需要解析的公共道头字段：(key, 中文名, 1-based 起始字节, 字节数, kind, 备注)
_HEADER_FIELDS: tuple[tuple[str, str, int, int, str, str], ...] = (
    ("trace_seq", "道序号（线内）", 1, 4, "int32", "SEG-Y 标准字节 1–4"),
    ("trace_seq_file", "道序号（文件内）", 5, 4, "int32", "SEG-Y 标准字节 5–8"),
    ("energy_source", "震源点号", 9, 4, "int32", "字节 9–12"),
    ("shotpoint", "炮点号", 17, 4, "int32", "字节 17–20"),
    ("ensemble", "道集号/炮点号", 21, 4, "int32", "字节 21–24"),
    ("cdp", "CDP 号", 21, 4, "int32", "与 ensemble 同为字节 21–24"),
    ("offset", "炮检距", 37, 4, "int32", "字节 37–40"),
    ("receiver_elev", "接收点高程", 41, 4, "int32", "字节 41–44，受高程标定因子约束"),
    ("source_elev", "炮点高程", 45, 4, "int32", "字节 45–48"),
    ("source_depth", "炮点深度", 49, 4, "int32", "字节 49–52"),
    ("receiver_depth", "接收点深度", 53, 4, "int32", "字节 53–56"),
    ("elevation_scalar", "高程/深度标定因子", 69, 2, "int16", "字节 69–70，负数为除数"),
    ("coord_scalar", "坐标标定因子", 71, 2, "int16", "字节 71–72，负数为除数"),
    ("source_x", "炮点 X 坐标", 73, 4, "int32", "字节 73–76，已乘坐标标定因子"),
    ("source_y", "炮点 Y 坐标", 77, 4, "int32", "字节 77–80，已乘坐标标定因子"),
    ("group_x", "检波点 X 坐标", 81, 4, "int32", "字节 81–84，已乘坐标标定因子"),
    ("group_y", "检波点 Y 坐标", 85, 4, "int32", "字节 85–88，已乘坐标标定因子"),
    ("coord_unit", "坐标单位", 89, 2, "int16", "字节 89–90：1=米/英尺"),
    ("ns", "本道采样点数", 115, 2, "int16", "字节 115–116"),
    ("dt", "本道采样间隔", 117, 2, "int16", "字节 117–118，微秒"),
    ("cdp_x", "CDP X 坐标", 181, 4, "int32", "字节 181–184"),
    ("cdp_y", "CDP Y 坐标", 185, 4, "int32", "字节 185–188"),
    ("inline", "Inline 号", 189, 4, "int32", "字节 189–192"),
    ("crossline", "Crossline 号", 193, 4, "int32", "字节 193–196"),
)


# --------------------------------------------------------------------------
# IBM / IEEE 浮点换算（与 vsdn2.sgy 语义一致，向量化实现）
# --------------------------------------------------------------------------
def ibm_to_ieee(ibm: Any) -> np.ndarray:
    """IBM 32 位浮点 -> IEEE float32。

    位布局：1 位符号 + 7 位指数（base 16，偏移 64）+ 24 位尾数；
    值 = 符号 * 尾数 / 2^24 * 16^(指数 - 64) = 尾数 * 2^(4*(指数-64) - 24)。

    用 float32 的 ``ldexp`` 直接按 2 的幂缩放：中间量比 float64 版本小一半以上，
    且 24 位尾数在 float32 里可精确表示（< 2^24）。
    """
    u = np.asarray(ibm, dtype=np.uint32)
    mant = (u & 0x00FFFFFF).astype(np.float32)
    expo = ((u >> 24) & 0x7F).astype(np.int32)
    shift = (expo - 64) * 4 - 24
    with np.errstate(over="ignore", invalid="ignore", under="ignore"):
        out = np.ldexp(mant, shift)
        np.negative(out, out=out, where=(u & 0x80000000) != 0)
    return out.astype(np.float32, copy=False)


def ieee_to_ibm(values: Any) -> np.ndarray:
    """IEEE float32 -> IBM 32 位浮点（写合成 SEG-Y 用）。"""
    v = np.asarray(values, dtype=np.float64)
    shape = v.shape
    flat = v.ravel()
    out = np.zeros(flat.size, dtype=np.uint32)

    finite = np.isfinite(flat) & (flat != 0.0)
    x = np.abs(flat[finite])
    if x.size:
        sign = np.where(flat[finite] < 0.0, np.uint32(0x80000000), np.uint32(0))
        expo = np.floor(np.log(x) / math.log(16.0)).astype(np.int32) + 64
        mant = np.rint(x / np.power(16.0, (expo - 64).astype(np.float64)) * 2.0 ** 24).astype(np.int64)
        # 尾数进位：达到 2^24 时指数 +1、尾数右移 4 位
        carry = mant >= (1 << 24)
        expo = expo + carry.astype(np.int32)
        mant = np.where(carry, mant >> 4, mant)
        ok = (expo > 0) & (expo < 128) & (mant > 0)
        packed = sign | (expo.astype(np.uint32) << 24) | mant.astype(np.uint32)
        out[finite] = np.where(ok, packed, out[finite])
        # 上溢 -> IBM 可表示的最大值
        over = np.nonzero(finite)[0][expo >= 128]
        if over.size:
            out[over] = 0x7FFFFFFF
    return out.reshape(shape)


# --------------------------------------------------------------------------
# 小工具
# --------------------------------------------------------------------------
def _i16(buf: bytes | bytearray, off0: int, endian: str) -> int:
    """从缓冲区 off0（0-based）读一个 int16。"""
    chunk = bytes(buf[off0:off0 + 2])
    if len(chunk) < 2:
        raise ValueError(f"读取 int16 越界：offset={off0}, buffer={len(buf)}")
    return int(np.frombuffer(chunk, dtype=endian + "i2")[0])


def _plausible_ns(ns: int) -> bool:
    return 1 <= int(ns) <= 1_000_000


def _plausible_dt_us(dt: int) -> bool:
    return 1 <= int(dt) <= 1_000_000


def _max_abs_chunked(arr: np.ndarray, step: int = 4_000_000) -> float:
    """分块求 max|值|，避免为大数组多开一份 abs 副本。"""
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        return 0.0
    best = 0.0
    for i in range(0, flat.size, step):
        block = flat[i:i + step]
        if block.size:
            best = max(best, float(np.max(np.abs(block))))
    return best


def _finite_ratio_chunked(arr: np.ndarray, step: int = 4_000_000) -> float:
    """分块统计有限值比例，避免整块 bool 副本。"""
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        return 1.0
    total = int(flat.size)
    good = 0
    for i in range(0, total, step):
        block = flat[i:i + step]
        good += int(np.isfinite(block).sum())
    return good / total


def _decode_text(raw: bytes) -> tuple[str, str, dict]:
    """尝试 EBCDIC / ASCII / latin-1 解码 3200 字节文本头，返回最优结果。"""
    info: dict[str, Any] = {"printable": 0.0, "marker": False, "score": 0.0}
    if not raw:
        return "", "none", info
    best = ("", "none", dict(info))
    for enc in ("cp500", "ascii", "latin-1"):
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
        printable = sum(1 for ch in text if ch.isprintable() or ch in "\r\n\t") / max(1, len(text))
        head = text[:600]
        marker = any(m in head for m in ("C 1", "C01", "C  1", "C1 "))
        keyword = any(
            k in text.upper() for k in ("SEG-Y", "SEGY", "CLIENT", "WELL", "LINE", "SHOT", "END EBCDIC")
        )
        score = printable + (0.25 if marker else 0.0) + (0.10 if keyword else 0.0)
        cand_info = {"printable": round(printable, 4), "marker": bool(marker), "score": round(score, 4)}
        if score > best[2]["score"]:
            best = (text, enc, cand_info)
    return best


def _parse_binary_header(head: bytes, endian: str) -> dict:
    """解析二进制头；``head`` 必须是文件开头至少 3600 字节（0-based 偏移）。"""

    def i16(off: int) -> int:
        if len(head) < off + 2:
            return 0
        return int(np.frombuffer(head[off:off + 2], dtype=endian + "i2")[0])

    return {
        "job_id": i16(3200),
        "n_aux": i16(3214),
        "dt_us": i16(3216),      # 1-based 3217
        "ns": i16(3220),         # 1-based 3221
        "fmt": i16(3224),        # 1-based 3225
        "sorting": i16(3228),
        "measure": i16(3252),
        "n_ext": i16(3504),      # 1-based 3505
    }


def _bin_score(info: dict, file_size: int) -> float:
    """给二进制头解析结果打分（0–1），用于大小端判别。"""
    score = 0.0
    fmt = info.get("fmt", 0)
    ns = info.get("ns", 0)
    dt = info.get("dt_us", 0)
    n_ext = info.get("n_ext", 0)

    if fmt in _FMT_DTYPE:
        score += 0.35
    if _plausible_ns(ns):
        score += 0.25
    elif 1 <= ns <= 2_000_000:
        score += 0.10
    if 10 <= dt <= 100_000:
        score += 0.20
    elif dt == 0:
        score += 0.05

    if 0 <= n_ext <= 50:
        score += 0.05
        start = BASE_SIZE + n_ext * TEXT_SIZE
    else:
        start = BASE_SIZE

    bps = _FMT_BPS.get(fmt)
    if bps and _plausible_ns(ns):
        tlen = TRACE_HEADER_SIZE + ns * bps
        if start < file_size:
            n = (file_size - start) // tlen
            rem = (file_size - start) % tlen
            if n >= 1 and rem == 0:
                score += 0.15
            elif n >= 1 and rem <= 16:
                score += 0.08
    return round(clamp01(score), 4)


def _count_ext_text(path: Path, start: int, size: int) -> int:
    """n_ext = -1 时向下扫描扩展文本头，直到 EndText 标记（最多 50 块）。"""
    count = 0
    with path.open("rb") as f:
        for _ in range(50):
            pos = start + count * TEXT_SIZE
            if pos + TEXT_SIZE > size:
                break
            f.seek(pos)
            block = f.read(TEXT_SIZE)
            text, _enc, _info = _decode_text(block)
            count += 1
            if "endtext" in text.lower():
                break
    return count


# --------------------------------------------------------------------------
# 道头
# --------------------------------------------------------------------------
def _parse_trace_headers(buf: bytes, n: int, endian: str) -> HeaderTable:
    """把连续 n 个 240 字节道头解析成 HeaderTable（1-based 字节）。"""
    table = HeaderTable()
    if n <= 0:
        return table
    expected = n * TRACE_HEADER_SIZE
    if len(buf) < expected:
        raise ValueError(f"道头缓冲区不足：需要 {expected} 字节，实际 {len(buf)}")
    for key, name, byte, size, kind, note in _HEADER_FIELDS:
        code = "i4" if size == 4 else "i2"
        values = np.ndarray(
            (n,), dtype=endian + code, buffer=buf, offset=byte - 1, strides=(TRACE_HEADER_SIZE,)
        ).astype(np.float64)
        table.add(key, name, byte, size, kind, values, note=note)

    # 坐标按标定因子还原（仅当因子存在且合理时）
    coord_scalar = table.values("coord_scalar")
    if coord_scalar is not None and coord_scalar.size:
        s = np.asarray(coord_scalar, dtype=np.float64)
        s = s[np.isfinite(s) & (s != 0)]
        factor = float(np.median(s)) if s.size else 1.0
        if abs(factor) < 1e-9 or abs(factor) > 1e6:
            factor = 1.0
        mul = factor if factor > 0 else 1.0 / abs(factor)
        if abs(mul - 1.0) > 1e-12:
            for key in ("source_x", "source_y", "group_x", "group_y", "cdp_x", "cdp_y"):
                f = table.get(key)
                if f is not None:
                    f.values = np.asarray(f.values, dtype=np.float64) * mul
                    f.note = f.note + "（已乘坐标标定因子）"
    return table


def _scan_traces(
    f: Any,
    start: int,
    size: int,
    ns_bin: int,
    dt_bin: int,
    bps: int,
    endian: str,
    max_traces: int | None,
) -> dict:
    """扫描道头，返回每道起始偏移 / ns / dt 与道头原始字节。"""
    limit = int(max_traces) if max_traces else None
    tlen_bin = TRACE_HEADER_SIZE + ns_bin * bps
    fixed = False
    n_total = 0
    if _plausible_ns(ns_bin) and tlen_bin > 0 and size >= start + tlen_bin:
        rest = size - start
        if rest % tlen_bin == 0:
            fixed = True
            n_total = int(rest // tlen_bin)

    if fixed and n_total >= 1:
        n_read = min(n_total, limit) if limit else n_total
        starts = np.empty(n_read, dtype=np.int64)
        ns_arr = np.empty(n_read, dtype=np.int32)
        dt_arr = np.empty(n_read, dtype=np.int32)
        header_buf = bytearray(n_read * TRACE_HEADER_SIZE)
        mismatch = 0  # 道头 ns 与二进制头不一致的道数（用于识别“假固定道”）
        for k in range(n_read):
            pos = start + k * tlen_bin
            f.seek(pos)
            hb = f.read(TRACE_HEADER_SIZE)
            if len(hb) < TRACE_HEADER_SIZE:
                # 扫描途中发现文件被截断：丢掉这一道及之后
                n_read = k
                starts = starts[:n_read]
                ns_arr = ns_arr[:n_read]
                dt_arr = dt_arr[:n_read]
                header_buf = header_buf[: n_read * TRACE_HEADER_SIZE]
                break
            hb_ns = _i16(hb, 114, endian)
            hb_dt = _i16(hb, 116, endian)
            if _plausible_ns(hb_ns) and hb_ns != ns_bin:
                mismatch += 1
            starts[k] = pos
            ns_arr[k] = hb_ns if _plausible_ns(hb_ns) else ns_bin
            dt_arr[k] = hb_dt if _plausible_dt_us(hb_dt) else dt_bin
            header_buf[k * TRACE_HEADER_SIZE:(k + 1) * TRACE_HEADER_SIZE] = hb
        if mismatch == 0:
            # 文件长度整除、且每道道头 ns 都与二进制头一致：按固定道长处理
            truncated = limit is not None and n_read < n_total
            pos_end = start + n_read * tlen_bin
            return {
                "n": int(n_read),
                "n_total": int(n_total),
                "fixed": True,
                "starts": starts,
                "ns": ns_arr,
                "dt_us": dt_arr,
                "header_bytes": bytes(header_buf),
                "truncated": bool(truncated),
                "trailing": int(max(0, size - pos_end)),
                "start": int(start),
            }
        # 长度虽整除，但道头 ns 不一致：多半是变长道，继续走逐道扫描
        fixed = False

    # ---- 变长道：逐道扫描 ----
    pos = start
    starts_l: list[int] = []
    ns_l: list[int] = []
    dt_l: list[int] = []
    chunks: list[bytes] = []
    while pos + TRACE_HEADER_SIZE <= size:
        if limit is not None and len(starts_l) >= limit:
            break
        f.seek(pos)
        hb = f.read(TRACE_HEADER_SIZE)
        if len(hb) < TRACE_HEADER_SIZE:
            break
        ns_k = _i16(hb, 114, endian)
        dt_k = _i16(hb, 116, endian)
        if not _plausible_ns(ns_k):
            ns_k = ns_bin
        if not _plausible_ns(ns_k):
            break
        dt_k = dt_k if _plausible_dt_us(dt_k) else dt_bin
        starts_l.append(pos)
        ns_l.append(int(ns_k))
        dt_l.append(int(dt_k))
        chunks.append(hb)
        pos += TRACE_HEADER_SIZE + ns_k * bps

    n = len(starts_l)
    if n == 0:
        raise ValueError("逐道扫描没有找到合法道（ns 读取失败）")
    truncated = limit is not None and pos + TRACE_HEADER_SIZE <= size
    return {
        "n": int(n),
        "n_total": int(n),  # 变长道只有扫完才知道总数；截断时这是下界
        "fixed": False,
        "starts": np.asarray(starts_l, dtype=np.int64),
        "ns": np.asarray(ns_l, dtype=np.int32),
        "dt_us": np.asarray(dt_l, dtype=np.int32),
        "header_bytes": b"".join(chunks),
        "truncated": bool(truncated),
        "trailing": int(max(0, size - pos)),
        "start": int(start),
    }


# --------------------------------------------------------------------------
# 样本解码
# --------------------------------------------------------------------------
def _decode_samples(seg: np.ndarray, ns: int, fmt: int, endian: str) -> np.ndarray:
    """把 ``(count, ns * bps)`` 的 uint8 原始样本解码成 ``(count, ns)`` float32。"""
    count = int(seg.shape[0])
    if count == 0 or ns <= 0:
        return np.zeros((max(count, 0), max(ns, 0)), dtype=np.float32)
    code = _FMT_DTYPE[fmt]
    itemsize = _FMT_BPS[fmt]
    arr = np.ascontiguousarray(seg)
    if itemsize == 1:
        samples = arr.reshape(count, ns).view(np.dtype(code))
    else:
        dt = np.dtype(endian + code)
        samples = arr.reshape(count, ns * itemsize).view(dt).reshape(count, ns)
    if fmt == 1:
        return ibm_to_ieee(samples).astype(np.float32)
    return samples.astype(np.float32)


def _read_first_raw(path: Path, starts: np.ndarray, ns_data: np.ndarray, n: int, ns_rect: int) -> bytes:
    """读取前若干道样本原始字节，用于 IBM/IEEE 判别。"""
    take = min(int(n), 4)
    chunks: list[bytes] = []
    with path.open("rb") as f:
        for k in range(take):
            ns_k = int(min(ns_data[k], ns_rect))
            if ns_k <= 0:
                continue
            f.seek(int(starts[k]) + TRACE_HEADER_SIZE)
            chunks.append(f.read(ns_k * 4))
    return b"".join(chunks)


def _float_plausible(v: np.ndarray) -> bool:
    v = np.asarray(v, dtype=np.float32)
    if v.size == 0:
        return True
    finite = np.isfinite(v)
    if float(finite.mean()) < 0.98:
        return False
    vals = np.abs(v[finite])
    if vals.size == 0:
        return True
    mx = float(vals.max())
    return mx == 0.0 or (1e-30 <= mx <= 1e30)


def _decide_ibm_layout(
    path: Path, starts: np.ndarray, ns_data: np.ndarray, n: int, ns_rect: int, endian: str
) -> tuple[str, str | None]:
    """格式码=1 时判断样本按 IBM 还是按 IEEE 读，返回 (layout, warning)。"""
    raw = _read_first_raw(path, starts, ns_data, n, ns_rect)
    if not raw:
        return "ibm", None
    as_ibm = ibm_to_ieee(np.frombuffer(raw, dtype=endian + "u4"))
    as_ieee = np.frombuffer(raw, dtype=endian + "f4").astype(np.float32)
    ibm_ok = _float_plausible(as_ibm)
    ieee_ok = _float_plausible(as_ieee)
    if ieee_ok and not ibm_ok:
        return "ieee", "样本格式码为 1，但按 IEEE 浮点解读才合理，已按 IEEE 读取"
    return "ibm", None


def _read_data(
    path: Path,
    scan: dict,
    ns_rect: int,
    ns_data: np.ndarray,
    fmt: int,
    endian: str,
    ibm_layout: str,
    bps: int,
    n: int,
    chunk_bytes: int = 1024 * 1024,
) -> tuple[np.ndarray, list[str]]:
    """按道读取样本，返回 ``(ns_rect, n)`` float32 数组与警告。"""
    warnings: list[str] = []
    out = np.zeros((int(ns_rect), int(n)), dtype=np.float32)
    starts = np.asarray(scan["starts"])
    tlen = TRACE_HEADER_SIZE + ns_rect * bps
    equal_len = bool(np.all(ns_data == ns_rect))

    code = _FMT_DTYPE[fmt]
    data_dtype = np.dtype(endian + code) if bps > 1 else np.dtype(code)
    if fmt == 1 and ibm_layout == "ieee":
        data_dtype = np.dtype(endian + "f4")

    contiguous = False
    if equal_len and n > 1:
        expected = np.arange(int(n), dtype=np.int64) * tlen + int(scan["start"])
        contiguous = bool(np.array_equal(starts, expected))

    if contiguous:
        chunk = max(1, int(chunk_bytes) // max(1, tlen))
        with path.open("rb") as f:
            for i0 in range(0, int(n), chunk):
                k = min(chunk, int(n) - i0)
                f.seek(int(scan["start"]) + i0 * tlen)
                buf = f.read(k * tlen)
                if len(buf) < k * tlen:
                    warnings.append(f"文件在读取第 {i0 + k} 道附近被截断，缺失部分补 0")
                    buf = buf + bytes(k * tlen - len(buf))
                block = np.frombuffer(buf, dtype=np.uint8).reshape(k, tlen)
                seg = block[:, TRACE_HEADER_SIZE:TRACE_HEADER_SIZE + ns_rect * bps]
                if fmt == 1 and ibm_layout == "ibm":
                    u = np.ascontiguousarray(seg).reshape(k, ns_rect * 4).view(
                        np.dtype(endian + "u4")
                    )
                    out[:, i0:i0 + k] = ibm_to_ieee(u).reshape(k, ns_rect).T
                else:
                    if bps > 1:
                        vals = np.ascontiguousarray(seg).reshape(k, ns_rect * bps).view(data_dtype)
                    else:
                        vals = np.ascontiguousarray(seg).reshape(k, ns_rect).view(data_dtype)
                    out[:, i0:i0 + k] = vals.astype(np.float32).reshape(k, ns_rect).T
        return out, warnings

    # ---- 逐道读取（变长 / 不连续）----
    with path.open("rb") as f:
        for k in range(int(n)):
            ns_k = int(ns_data[k]) if k < ns_data.size else ns_rect
            take = min(ns_k, ns_rect)
            if take <= 0:
                continue
            f.seek(int(starts[k]) + TRACE_HEADER_SIZE)
            want = take * bps
            buf = f.read(want)
            if len(buf) < want:
                take = len(buf) // bps
                if take <= 0:
                    continue
                buf = buf[: take * bps]
                warnings.append(f"第 {k + 1} 道数据不完整，缺失部分补 0")
            seg = np.frombuffer(buf, dtype=np.uint8).reshape(1, take * bps)
            vals = _decode_samples(seg, take, fmt, endian)
            out[:take, k] = vals[0]

    if not equal_len:
        longer = int(np.sum(ns_data > ns_rect))
        shorter = int(np.sum(ns_data < ns_rect))
        if longer:
            warnings.append(f"有 {longer} 道比对齐采样点数长，超出部分被截断")
        if shorter:
            warnings.append(f"有 {shorter} 道比对齐采样点数短，尾部补 0")
    return out, warnings


# --------------------------------------------------------------------------
# 对外：头段定位 / 道头
# --------------------------------------------------------------------------
def _locate_data_start(path: Path, size: int, bin_info: dict) -> tuple[int, int]:
    """根据扩展文本头数计算第一道道头偏移；返回 (start, n_ext)。"""
    n_ext = int(bin_info.get("n_ext", 0))
    if n_ext == -1:
        n_ext = _count_ext_text(path, BASE_SIZE, size)
        return BASE_SIZE + n_ext * TEXT_SIZE, int(n_ext)
    if n_ext < 0 or n_ext > 200:
        return BASE_SIZE, 0
    start = BASE_SIZE + n_ext * TEXT_SIZE
    if start + TRACE_HEADER_SIZE > size:
        return BASE_SIZE, 0
    return start, n_ext


def _read_header_only(path: Path) -> tuple[Path, int, str, dict, dict]:
    """确定端序 / 二进制头 / 数据起点，供 read_segy 与 read_segy_headers 复用。"""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"文件不存在：{p}")
    size = int(p.stat().st_size)
    if size < BASE_SIZE:
        raise ValueError(f"文件太小（{size} 字节），不足 SEG-Y 最小头段 {BASE_SIZE} 字节")
    with p.open("rb") as f:
        head = f.read(BASE_SIZE)
    if len(head) < BASE_SIZE:
        raise ValueError(f"文件头段不完整：{len(head)} < {BASE_SIZE} 字节")

    candidates = []
    for endian in (">", "<"):
        info = _parse_binary_header(head, endian)
        info["score"] = _bin_score(info, size)
        candidates.append((info["score"], endian, info))
    candidates.sort(key=lambda x: -x[0])
    score, endian, bin_info = candidates[0]
    if score < 0.55:
        raise ValueError(
            f"SEG-Y 头段不可信：大小端最高匹配度只有 {score:.2f}；"
            f"文件大小 {size} 字节，可能不是 SEG-Y 或头段被破坏"
        )
    start, n_ext = _locate_data_start(p, size, bin_info)
    bin_info["n_ext_used"] = n_ext
    bin_info["endian"] = endian
    bin_info["score_best"] = score
    bin_info["score_alt"] = candidates[1][0]
    bin_info["ambig_endian"] = bool(candidates[1][0] >= 0.55)
    return p, size, endian, bin_info, {"start": start, "n_ext": n_ext}


def _effective_format(bin_info: dict, size: int, start: int, warnings: list[str]) -> int:
    """取实际使用的采样格式码；格式码异常时按道长整除关系反推。"""
    fmt = int(bin_info["fmt"])
    if fmt in _FMT_BPS:
        return fmt
    ns_guess = int(bin_info["ns"])
    guessed = None
    if _plausible_ns(ns_guess):
        for cand in (5, 1, 2, 3, 8):
            tlen = TRACE_HEADER_SIZE + ns_guess * _FMT_BPS[cand]
            if 0 < tlen and (size - start) >= tlen and (size - start) % tlen == 0:
                guessed = cand
                break
    if guessed is None:
        raise ValueError(
            f"不支持的 SEG-Y 采样格式码 {fmt}（支持 {sorted(_FMT_BPS)}）；"
            f"文件大小 {size} 字节"
        )
    warnings.append(f"二进制头采样格式码异常（{fmt}），按道长整除关系推断为格式码 {guessed}")
    return guessed


def read_segy_headers(path: str | Path) -> HeaderTable:
    """只读道头（不加载样本），返回 HeaderTable；字段字节为 1-based。"""
    p, size, endian, bin_info, loc = _read_header_only(Path(path))
    warnings: list[str] = []
    fmt = _effective_format(bin_info, size, loc["start"], warnings)
    bps = _FMT_BPS[fmt]
    ns_bin = int(bin_info["ns"])
    if not _plausible_ns(ns_bin):
        with p.open("rb") as f:
            f.seek(loc["start"])
            hb = f.read(TRACE_HEADER_SIZE)
        ns_bin = _i16(hb, 114, endian)
    if not _plausible_ns(ns_bin):
        raise ValueError("无法确定道采样点数 ns（二进制头与道头都为 0/非法）")
    with p.open("rb") as f:
        scan = _scan_traces(
            f, loc["start"], size, ns_bin, int(bin_info["dt_us"]), bps, endian, None
        )
    if scan["n"] <= 0:
        raise ValueError("没有扫描到任何道")
    table = _parse_trace_headers(scan["header_bytes"], int(scan["n"]), endian)
    # HeaderTable 是普通 dataclass，可挂附加信息（报告/调试时能读）
    try:
        table.meta = {
            "endian": "big" if endian == ">" else "little",
            "n_traces": int(scan["n"]),
            "ns_binary": int(bin_info["ns"]),
            "sample_format": int(fmt),
            "n_ext_text": int(loc["n_ext"]),
            "header_start": int(loc["start"]),
            "file_size": int(size),
        }
    except AttributeError:  # pragma: no cover - 防御未来改 slots
        pass
    return table


# --------------------------------------------------------------------------
# 对外：完整读取
# --------------------------------------------------------------------------
def read_segy(path: str | Path, max_traces: int | None = None) -> SeismicDataset:
    """读取 SEG-Y 文件为 SeismicDataset。"""
    p, size, endian, bin_info, loc = _read_header_only(Path(path))
    warnings: list[str] = []

    endian_name = "大端（big-endian）" if endian == ">" else "小端（little-endian）"
    if bin_info.get("ambig_endian"):
        warnings.append(
            f"大小端判别不确定：{endian_name} 得分 {bin_info['score_best']:.2f}，"
            f"另一种得分 {bin_info['score_alt']:.2f}，已按前者解析"
        )

    fmt = _effective_format(bin_info, size, loc["start"], warnings)
    bps = _FMT_BPS[fmt]

    ns_bin = int(bin_info["ns"])
    if not _plausible_ns(ns_bin):
        with p.open("rb") as f:
            f.seek(loc["start"])
            hb = f.read(TRACE_HEADER_SIZE)
        ns_bin = _i16(hb, 114, endian)
        if _plausible_ns(ns_bin):
            warnings.append("二进制头 ns 非法，改用第一道道头里的 ns")
    if not _plausible_ns(ns_bin):
        raise ValueError("无法确定采样点数 ns（二进制头与第一道道头都是 0/非法）")

    dt_bin = int(bin_info["dt_us"])
    if not _plausible_dt_us(dt_bin):
        dt_bin = 0

    with p.open("rb") as f:
        scan = _scan_traces(f, loc["start"], size, ns_bin, dt_bin, bps, endian, max_traces)
    n = int(scan["n"])
    if n <= 0:
        raise ValueError("没有扫描到任何道")

    if scan["truncated"]:
        if scan["fixed"]:
            warnings.append(
                f"按 max_traces={int(max_traces)} 截断：文件共 {scan['n_total']} 道，仅读取前 {n} 道"
            )
        else:
            warnings.append(f"按 max_traces={int(max_traces)} 截断：仅读取前 {n} 道（文件还有更多道）")
    if not scan["fixed"]:
        warnings.append("文件为变长道（各道 ns 不同），已按中位数对齐，短道补 0、长道截断")
    if scan["trailing"] > TRACE_HEADER_SIZE and not scan["truncated"]:
        warnings.append(f"文件末尾有 {scan['trailing']} 字节未对齐的残余数据（可能被截断）")

    ns_trace = np.asarray(scan["ns"], dtype=np.int64)
    valid_ns = (ns_trace >= 16) & (ns_trace <= 1_000_000)
    if valid_ns.any():
        ns_median = int(np.median(ns_trace[valid_ns]))
        ns_consistent = float(np.mean(ns_trace == ns_bin)) if scan["fixed"] else float(np.mean(valid_ns))
        ns_min = int(ns_trace[valid_ns].min())
        ns_max = int(ns_trace[valid_ns].max())
    else:
        ns_median, ns_consistent, ns_min, ns_max = ns_bin, 0.0, ns_bin, ns_bin
    variable_ns = ns_min != ns_max
    if scan["fixed"]:
        ns_rect = int(ns_bin)
        ns_data = np.full(n, ns_bin, dtype=np.int32)
    else:
        ns_rect = int(ns_median)
        ns_data = ns_trace.astype(np.int32)
    if ns_consistent < 0.95:
        warnings.append(
            f"道头 ns 与二进制头 ns={ns_bin} 的一致率只有 {ns_consistent:.0%}，已按文件对齐关系读取"
        )

    # ---- 读取样本 ----
    ibm_layout: str | None = None
    if fmt == 1:
        ibm_layout, ibm_warning = _decide_ibm_layout(p, scan["starts"], ns_data, n, ns_rect, endian)
        if ibm_warning:
            warnings.append(ibm_warning)
    data, read_warnings = _read_data(
        p, scan, ns_rect, ns_data, fmt, endian, ibm_layout or "ibm", bps, n
    )
    warnings.extend(read_warnings)
    finite_ratio = _finite_ratio_chunked(data)
    data, clean_warnings = clean_array(data)
    warnings.extend(clean_warnings)
    amplitude_max = _max_abs_chunked(data)
    if amplitude_max > 1e20:
        warnings.append(f"样本振幅异常大（max|值|={amplitude_max:.3g}），格式码或端序可能识别有误")

    # ---- dt ----
    dt_us_arr = np.asarray(scan["dt_us"], dtype=np.int64)
    valid_dt = (dt_us_arr >= 1) & (dt_us_arr <= 1_000_000)
    dt_us_median = 0.0
    if valid_dt.any():
        dt_us_median = float(np.median(dt_us_arr[valid_dt]))
        uniq = np.unique(dt_us_arr[valid_dt])
        if uniq.size > 1:
            warnings.append(
                f"道头 dt 不唯一（{int(uniq.min())}–{int(uniq.max())} μs），"
                f"取中位数 {dt_us_median:.0f} μs"
            )
        if dt_bin and abs(dt_us_median - dt_bin) > max(1.0, 0.01 * dt_bin):
            warnings.append(
                f"道头 dt（{dt_us_median:.0f} μs）与二进制头 dt（{dt_bin} μs）不一致，以道头为准"
            )
    elif dt_bin:
        dt_us_median = float(dt_bin)
        warnings.append("道头 dt 全为 0，改用二进制头采样间隔")
    else:
        warnings.append("道头与二进制头的 dt 都无效，采样间隔 dt 置 None（频域分析需显式提供）")

    dt: float | None = None
    if dt_us_median > 0:
        dt = dt_us_median / 1e6
        if not (1e-6 <= dt <= 0.5):
            warnings.append(f"采样间隔 dt={dt:.6g} s 不在常规范围，已置 None")
            dt = None

    # ---- 道头与 dz ----
    headers = _parse_trace_headers(scan["header_bytes"], n, endian)
    elev = headers.values("receiver_elev")
    elev_scalar = headers.values("elevation_scalar")
    dz, dz_info = estimate_dz_from_series(elev, elev_scalar, "接收点高程")
    if dz is None:
        depth = headers.values("receiver_depth")
        dz, dz_info = estimate_dz_from_series(depth, elev_scalar, "接收点深度")
        if dz is not None:
            warnings.append("接收点高程无法估计道间距，已改用接收点深度")
    if dz is None:
        warnings.append("无法从道头高程/深度估计道间距 dz，dz 置 None")
    else:
        dz_warning = dz_irregular_warning(dz, dz_info)
        if dz_warning:
            warnings.append(dz_warning)

    # ---- 文本头 ----
    with p.open("rb") as f:
        text_raw = f.read(TEXT_SIZE)
    text_header, text_encoding, text_info = _decode_text(text_raw)
    if text_info["printable"] < 0.8:
        warnings.append(f"文本头可读性差（可打印比例 {text_info['printable']:.0%}，编码 {text_encoding}）")

    # ---- 置信度 ----
    confidence = 0.0
    confidence += 0.30 if bin_info["score_best"] >= 0.75 else 0.20
    if text_encoding in ("cp500", "ascii") and text_info["marker"]:
        confidence += 0.15
    elif text_info["printable"] >= 0.8:
        confidence += 0.05
    if ns_consistent >= 0.95:
        confidence += 0.20
    elif ns_consistent >= 0.80:
        confidence += 0.10
    if scan["fixed"]:
        confidence += 0.15
    elif not scan["truncated"]:
        confidence += 0.10
    if n >= 2:
        confidence += 0.10
    if finite_ratio >= 0.999 and amplitude_max <= 1e20:
        confidence += 0.10
    confidence = clamp01(confidence)

    meta = {
        "endian": "big" if endian == ">" else "little",
        "endian_symbol": endian,
        "endian_name": endian_name,
        "sample_format": int(fmt),
        "sample_format_binary": int(bin_info["fmt"]),
        "sample_format_name": _FMT_NAME.get(fmt, f"格式码 {fmt}"),
        "bytes_per_sample": bps,
        "ns_binary": int(bin_info["ns"]),
        "dt_us_binary": int(bin_info["dt_us"]),
        "ns_trace_median": int(ns_rect),
        "dt_us_trace_median": float(dt_us_median),
        "n_ext_text": int(loc["n_ext"]),
        "header_start": int(loc["start"]),
        "trace_length_bytes": int(TRACE_HEADER_SIZE + ns_rect * bps),
        "n_traces_in_file": int(scan["n_total"]),
        "n_traces_read": int(n),
        "variable_length": bool(variable_ns),
        "trace_ns_min": int(ns_min),
        "trace_ns_max": int(ns_max),
        "text_encoding": text_encoding,
        "text_printable": text_info["printable"],
        "text_marker": bool(text_info["marker"]),
        "file_size": int(size),
        "file_size_text": human_size(size),
        "sorting_code": int(bin_info["sorting"]),
        "measurement_system": int(bin_info["measure"]),
        "ibm_layout": ibm_layout,
        "dz_method": (dz_info or {}).get("method"),
        "dz_info": dz_info,
        "max_traces": None if max_traces is None else int(max_traces),
        "trailing_bytes": int(scan["trailing"]),
    }

    return SeismicDataset(
        path=str(p),
        format="segy",
        data=data,
        dt=dt,
        dz=dz,
        headers=headers,
        text_header=text_header,
        meta=meta,
        confidence=confidence,
        warnings=warnings,
        source_kind="segy",
    )


# --------------------------------------------------------------------------
# 对外：头段试探（analyze_binary 用）
# --------------------------------------------------------------------------
def probe_segy(path: str | Path) -> dict:
    """只读头段做一次 SEG-Y 试探，返回解析摘要（不加载样本）。"""
    p, size, endian, bin_info, loc = _read_header_only(Path(path))
    fmt = int(bin_info["fmt"])
    ns = int(bin_info["ns"])
    dt_us = int(bin_info["dt_us"])
    bps = _FMT_BPS.get(fmt)
    with p.open("rb") as f:
        text_raw = f.read(TEXT_SIZE)
    text, encoding, text_info = _decode_text(text_raw)

    n_traces = None
    trace_len = None
    aligned = False
    if bps and _plausible_ns(ns):
        trace_len = TRACE_HEADER_SIZE + ns * bps
        rest = size - loc["start"]
        if rest >= trace_len:
            n_traces = int(rest // trace_len)
            aligned = rest % trace_len == 0

    preview_lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    return {
        "endian": "big" if endian == ">" else "little",
        "endian_name": "大端（big-endian）" if endian == ">" else "小端（little-endian）",
        "score": float(bin_info["score_best"]),
        "score_alt": float(bin_info["score_alt"]),
        "ns": ns,
        "dt_us": dt_us,
        "dt_ms": (dt_us / 1000.0) if dt_us > 0 else None,
        "sample_format": fmt,
        "sample_format_name": _FMT_NAME.get(fmt, f"格式码 {fmt}"),
        "bytes_per_sample": bps,
        "n_ext_text": int(loc["n_ext"]),
        "header_start": int(loc["start"]),
        "trace_length_bytes": trace_len,
        "n_traces_est": n_traces,
        "aligned": bool(aligned),
        "sorting_code": int(bin_info["sorting"]),
        "measurement_system": int(bin_info["measure"]),
        "text_encoding": encoding,
        "text_printable": text_info["printable"],
        "text_preview": preview_lines[:4],
        "file_size": int(size),
        "file_size_text": human_size(size),
    }


# --------------------------------------------------------------------------
# 对外：写 SEG-Y（合成测试 / pipeline 中间结果落盘用，非契约必需）
# --------------------------------------------------------------------------
def _make_text_header(text: str) -> bytes:
    """按 40 字符一行构造 3200 字节 ASCII 文本头，空位补空格。"""
    raw_lines = [ln.rstrip("\r") for ln in str(text).splitlines() if ln.strip()]
    if not raw_lines:
        raw_lines = ["C01 SYNTHETIC SEG-Y (dsh-seismic-denoise)", "C40 END TEXT"]
    out_lines: list[str] = []
    for i, ln in enumerate(raw_lines[:80]):
        m = re.match(r"^\s*C\s*\d{1,2}\s?(.*)$", ln, re.IGNORECASE)
        body = m.group(1) if m else ln
        out_lines.append(f"C{i + 1:02d} {body}"[:80].ljust(80))
    raw = "\n".join(out_lines)
    data = raw.encode("ascii", errors="replace")[:TEXT_SIZE]
    return data.ljust(TEXT_SIZE, b" ")


def write_segy(
    path: str | Path,
    data: Any,
    dt: float | None,
    *,
    fmt: int = 5,
    text: str = "",
    headers: HeaderTable | None = None,
    endian: str = ">",
) -> Path:
    """把 ``(ns, ntr)`` 数据写成 SEG-Y（默认大端 IEEE float，供测试/落盘用）。

    ``headers`` 若提供，会按字段自身 byte/size 写回第一道模板；
    无论如何都会重写道头的道序号（1/5 字节）、ns(115) 与 dt(117)。
    """
    p = Path(path)
    arr = np.asarray(data)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError(f"data 必须是二维非空 (ns, ntr)，实际 shape={arr.shape}")
    ns, ntr = int(arr.shape[0]), int(arr.shape[1])
    if ns > 65535:
        raise ValueError(f"SEG-Y 道头 ns 字段只有 2 字节，ns={ns} 过大")
    if fmt not in _FMT_DTYPE:
        raise ValueError(f"不支持的 SEG-Y 格式码 {fmt}")
    bps = _FMT_BPS[fmt]
    if endian not in (">", "<"):
        raise ValueError("endian 只能是 '>' 或 '<'")

    dt_us = 0
    if dt is not None and float(dt) > 0:
        dt_us = int(round(float(dt) * 1e6))
    dt_us = max(0, min(65535, dt_us))

    # ---- 二进制头（400 字节，块内 0-based 偏移）----
    bin_hdr = bytearray(BIN_SIZE)
    struct_data = {
        12: min(ntr, 65535),   # 3213 每炮道数
        16: dt_us,             # 3217 采样间隔 μs
        20: min(ns, 65535),    # 3221 采样点数
        24: fmt,               # 3225 采样格式码
        28: 1,                 # 3229 排序：1 = 无排序/未定义
        52: 1,                 # 3253 测量系统 1 = 米
        304: 0,                # 3505 扩展文本头数
    }
    for off, value in struct_data.items():
        bin_hdr[off:off + 2] = np.asarray([value], dtype=endian + "i2").tobytes()

    text_bytes = _make_text_header(text)

    # ---- 道头模板（来自 headers 的第 0 道）----
    header_tpl = bytearray(TRACE_HEADER_SIZE)
    if headers is not None:
        for field in headers.fields.values():
            byte = int(getattr(field, "byte", 0))
            size = int(getattr(field, "size", 0))
            if byte < 1 or size not in (2, 4):
                continue
            try:
                v = float(np.asarray(field.values)[0])
            except (IndexError, TypeError, ValueError):
                continue
            if not math.isfinite(v):
                continue
            if size == 2:
                code = endian + "i2"
                v = int(max(-32768, min(32767, round(v))))
            else:
                code = endian + "i4"
                v = int(max(-2147483648, min(2147483647, round(v))))
            header_tpl[byte - 1:byte - 1 + size] = np.asarray([v], dtype=code).tobytes()
    # 逐道变化的字段：values 长度 == ntr 时按道写入（高程/坐标/炮检距等）
    per_trace_fields: list[tuple[int, int, np.ndarray]] = []
    if headers is not None:
        for field in headers.fields.values():
            byte = int(getattr(field, "byte", 0))
            size = int(getattr(field, "size", 0))
            if byte < 1 or size not in (2, 4):
                continue
            try:
                vals = np.asarray(field.values, dtype=np.float64).ravel()
            except (TypeError, ValueError):
                continue
            if vals.size != ntr or not np.isfinite(vals).all():
                continue
            if size == 2:
                code = endian + "i2"
                vals = np.clip(np.rint(vals), -32768, 32767).astype(code)
            else:
                code = endian + "i4"
                vals = np.clip(np.rint(vals), -2147483648, 2147483647).astype(code)
            per_trace_fields.append((byte - 1, size, vals))
    # 道序号 / ns / dt 固定写在模板里
    header_tpl[0:4] = np.asarray([1], dtype=endian + "i4").tobytes()
    header_tpl[4:8] = np.asarray([1], dtype=endian + "i4").tobytes()
    header_tpl[114:116] = np.asarray([ns], dtype=endian + "i2").tobytes()
    header_tpl[116:118] = np.asarray([dt_us], dtype=endian + "i2").tobytes()

    with p.open("wb") as f:
        f.write(text_bytes)
        f.write(bytes(bin_hdr))
        for k in range(ntr):
            hdr = bytearray(header_tpl)
            # 先铺逐道变化的字段，再用固定的道序号/ns/dt 覆盖，保证一致
            for off, size, vals in per_trace_fields:
                # 注意用切片而不是标量：numpy 标量 .tobytes() 会退回本机字节序
                hdr[off:off + size] = vals[k:k + 1].tobytes()
            hdr[0:4] = np.asarray([k + 1], dtype=endian + "i4").tobytes()
            hdr[4:8] = np.asarray([k + 1], dtype=endian + "i4").tobytes()
            hdr[114:116] = np.asarray([ns], dtype=endian + "i2").tobytes()
            hdr[116:118] = np.asarray([dt_us], dtype=endian + "i2").tobytes()
            f.write(bytes(hdr))
            trace = np.ascontiguousarray(arr[:, k], dtype=np.float64)
            if fmt == 1:
                payload = ieee_to_ibm(trace).astype(np.dtype(endian + "u4")).tobytes()
            elif bps == 1:
                payload = trace.astype(np.dtype(_FMT_DTYPE[fmt])).tobytes()
            else:
                payload = trace.astype(np.dtype(endian + _FMT_DTYPE[fmt])).tobytes()
            f.write(payload)
    return p
