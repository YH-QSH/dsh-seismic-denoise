"""SEG-Y 读写。

支持
----
- 3200 字节 EBCDIC/ASCII 文本头 + 400 字节二进制头
- 数据格式：1=IBM float、2=int32、3=int16、5=IEEE float、8=int8
- 大端/小端自动判别
- 变长道（逐道读取时按道头的 ns 覆盖）

单位约定：对外统一用 ``dt`` 秒、``dz`` 米。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np


# ==========================================================================
# IBM float -> IEEE
# ==========================================================================
def ibm2ieee(ibm: np.ndarray) -> np.ndarray:
    """32 位 IBM 浮点转 IEEE754（SEG-Y 老格式 format=1）。"""
    u = np.asarray(ibm, dtype=np.uint32)
    sign = (u >> 31) & 0x01
    exp = ((u >> 24) & 0x7F).astype(np.int32) - 64
    mant = (u & 0x00FFFFFF).astype(np.float64) / float(0x01000000)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        val = mant * np.power(2.0, exp, dtype=np.float64)
    val = np.where(sign == 1, -val, val)
    val = np.where(u == 0, 0.0, val)
    return val.astype(np.float32)


def ieee2ibm(x: np.ndarray) -> np.ndarray:
    """IEEE754 -> IBM 浮点（写 SEG-Y format=1 时用）。"""
    x = np.asarray(x, dtype=np.float64).ravel()
    out = np.zeros(x.size, dtype=np.uint32)
    nz = x != 0
    if not nz.any():
        return out
    v = np.abs(x[nz])
    sign = (x[nz] < 0).astype(np.uint32)
    exp = np.floor(np.log2(v)) + 1.0
    mant = v / np.power(2.0, exp)
    # 规格化到 [1/16, 1)
    shift = np.where(mant < 0.0625, 4, 0).astype(np.int32)
    mant = mant * np.power(2.0, shift)
    exp = exp - shift
    mant_i = np.clip((mant * float(0x01000000)).astype(np.int64), 0, 0x00FFFFFF)
    e = np.clip((exp + 64).astype(np.int64), 0, 127).astype(np.uint32)
    out[nz] = (sign << 31) | (e << 24) | mant_i.astype(np.uint32)
    return out


# ==========================================================================
# 头段
# ==========================================================================
_DTYPE = {1: "ibm", 2: ">i4", 3: ">i2", 5: ">f4", 8: "i1"}
_BPS = {1: 4, 2: 4, 3: 2, 5: 4, 8: 1}


@dataclass
class SegyMeta:
    path: str = ""
    ns: int = 0
    dt_us: int = 0
    fmt: int = 1
    n_traces: int = 0
    text: str = ""
    endian: str = ">"
    extra: Dict = field(default_factory=dict)

    @property
    def dt(self) -> float:
        return self.dt_us / 1e6

    def summary(self) -> str:
        return (
            f"{Path(self.path).name}: ns={self.ns} dt={self.dt_us}us "
            f"fmt={self.fmt} ntraces={self.n_traces} endian={self.endian}"
        )


def _i16(b: bytes, off: int, endian: str) -> int:
    return int(np.frombuffer(b[off:off + 2], dtype=endian + "i2")[0])


def _i32(b: bytes, off: int, endian: str) -> int:
    return int(np.frombuffer(b[off:off + 4], dtype=endian + "i4")[0])


def _plausible(ns: int, dt_us: int, fmt: int) -> bool:
    return (
        16 <= ns <= 200_000
        and 10 <= dt_us <= 100_000
        and fmt in _BPS
    )


def read_segy(path: str | Path, max_traces: Optional[int] = None) -> Tuple[np.ndarray, SegyMeta]:
    """读取 SEG-Y，返回 ``(data, meta)``，data 形状 ``(ns, ntraces)`` float32。"""
    path = Path(path)
    raw = path.read_bytes()
    if len(raw) < 3600:
        raise ValueError(f"{path.name} 太小，不是有效 SEG-Y（{len(raw)} 字节）")

    # ---- 判断大小端 ----
    chosen = None
    for endian in (">", "<"):
        ns_bin = _i16(raw, 3220, endian)
        dt_bin = _i16(raw, 3216, endian)
        fmt = _i16(raw, 3224, endian)
        if _plausible(ns_bin, dt_bin, fmt):
            chosen = (endian, ns_bin, dt_bin, fmt)
            break
    if chosen is None:
        raise ValueError(
            f"{path.name} 的二进制头无法解析（ns/dt/format 不合理），"
            "可能不是标准 SEG-Y，或起始处带私有头。"
        )
    endian, ns, dt_us, fmt = chosen

    # ---- 扩展文本头个数（字节 3505-3506）----
    n_ext = _i16(raw, 3504, endian)
    if not (0 <= n_ext <= 50):
        n_ext = 0
    start = 3600 + 3200 * n_ext

    bps = _BPS[fmt]
    tlen = 240 + ns * bps
    n_total = (len(raw) - start) // tlen
    if max_traces:
        n_total = min(n_total, int(max_traces))
    if n_total <= 0:
        raise ValueError(f"{path.name} 没有可用道（tlen={tlen}, 剩余={len(raw) - start}）")

    if fmt == 1:
        raw_rec = np.frombuffer(raw, dtype=np.dtype((np.uint8, tlen)), count=n_total, offset=start)
        data_bytes = np.ascontiguousarray(raw_rec[:, 240:240 + ns * 4])
        data = ibm2ieee(data_bytes.view(">u4")).reshape(n_total, ns).T
        # 有些文件头里写 format=1，实际却存的是 IEEE float，这里做一次兜底
        if not np.all(np.isfinite(data)) or float(np.max(np.abs(data))) > 1e20:
            alt = data_bytes.view(">f4").astype(np.float32).reshape(n_total, ns).T
            if np.all(np.isfinite(alt)) and float(np.max(np.abs(alt))) < 1e20:
                data = alt
    else:
        dt_ = np.dtype(_DTYPE[fmt])
        raw_rec = np.frombuffer(raw, dtype=np.uint8, count=n_total * tlen, offset=start)
        rec = raw_rec.reshape(n_total, tlen)
        seg = np.ascontiguousarray(rec[:, 240:240 + ns * bps])
        data = seg.view(dt_).reshape(n_total, ns).T.astype(np.float32)
    data = np.ascontiguousarray(data, dtype=np.float32)

    text = raw[:3200].decode("latin-1", errors="replace")
    meta = SegyMeta(
        path=str(path), ns=int(ns), dt_us=int(dt_us), fmt=int(fmt),
        n_traces=int(n_total), text=text, endian=endian,
        extra={"file_size": len(raw), "n_ext_text": n_ext},
    )
    return data, meta


def read_trace_headers(
    path: str | Path,
    fields: Optional[Dict[str, Tuple[int, int]]] = None,
    max_traces: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """读取常见道头字段。

    Parameters
    ----------
    fields : {名字: (起始偏移(0基), 字节数)}，默认为几个常用量。
    """
    path = Path(path)
    raw = path.read_bytes()
    endian = ">"
    for e in (">", "<"):
        if _plausible(_i16(raw, 3220, e), _i16(raw, 3216, e), _i16(raw, 3224, e)):
            endian = e
            break
    ns = _i16(raw, 3220, endian)
    dt_us = _i16(raw, 3216, endian)
    fmt = _i16(raw, 3224, endian)
    bps = _BPS.get(fmt, 4)
    tlen = 240 + ns * bps
    n_ext = _i16(raw, 3504, endian)
    if not (0 <= n_ext <= 50):
        n_ext = 0
    start = 3600 + 3200 * n_ext
    n = (len(raw) - start) // tlen
    if max_traces:
        n = min(n, int(max_traces))

    if fields is None:
        fields = {
            "trace_seq": (0, 4),
            "receiver_elev": (40, 4),
            "source_x": (72, 4),
            "source_y": (76, 4),
            "ns": (114, 2),
            "dt": (116, 2),
            "cdp_x": (180, 4),
            "cdp_y": (184, 4),
        }

    out: Dict[str, np.ndarray] = {}
    for name, (off, width) in fields.items():
        if width == 4:
            out[name] = np.fromiter(
                (_i32(raw, start + k * tlen + off, endian) for k in range(n)),
                dtype=np.float64, count=n,
            )
        else:
            out[name] = np.fromiter(
                (_i16(raw, start + k * tlen + off, endian) for k in range(n)),
                dtype=np.float64, count=n,
            )
    out["_ns"] = np.full(n, ns, dtype=np.int32)
    out["_dt_us"] = np.full(n, dt_us, dtype=np.int32)
    return out


def estimate_dz(path: str | Path, max_traces: int = 300) -> Optional[float]:
    """从道头接收点高程差估计道间距（米）。

    不同软件写道头的单位不统一：SERCEL / 部分处理系统用**厘米**（差 205 表示 2.05 m），
    也有用米的。这里按量级判断：差大于 100 就当作厘米处理。
    高程全相同（道头没写）时返回 None。
    """
    try:
        h = read_trace_headers(
            path, max_traces=max_traces,
            fields={"receiver_elev": (40, 4), "elev_scalar": (68, 2)},
        )
        el = np.asarray(h["receiver_elev"], dtype=np.float64)
        if el.size < 3 or np.unique(el).size < 2:
            return None
        d = float(np.median(np.abs(np.diff(el))))
        if not np.isfinite(d) or d <= 0:
            return None
        if d > 100.0:          # 几乎可以肯定是厘米
            d /= 100.0
        return d if 0.1 <= d <= 100.0 else None
    except Exception:
        return None


# ==========================================================================
# 写
# ==========================================================================
def write_segy(
    path: str | Path,
    data: np.ndarray,
    dt: float,
    text: str = "C01 VS-DENOISE OUTPUT",
    fmt: int = 5,
    receiver_depths: Optional[np.ndarray] = None,
) -> SegyMeta:
    """把 ``(ns, ntraces)`` 数据写成 SEG-Y（默认 IEEE float）。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    ns, ntr = data.shape
    dt_us = int(round(dt * 1e6))

    hdr = 3600
    bps = _BPS[fmt]
    buf = bytearray(hdr + ntr * (240 + ns * bps))

    txt = (text or "").ljust(3200)[:3200]
    buf[0:3200] = txt.encode("latin-1", errors="replace")

    def put_i16(off, val):
        buf[off:off + 2] = int(val).to_bytes(2, "big", signed=True)

    def put_i32(off, val):
        buf[off:off + 4] = int(val).to_bytes(4, "big", signed=True)

    put_i16(3204, 1)          # job id 的高位不用管
    put_i16(3216, dt_us)      # 采样间隔
    put_i16(3220, ns)         # 每道采样点
    put_i16(3224, fmt)        # 数据格式
    put_i16(3226, 1)          # 道集类型
    put_i16(3228, 1)
    put_i16(3252, 0)          # 测量系统 = 米

    dt_trace = np.dtype(">f4") if fmt == 5 else None
    for k in range(ntr):
        base = hdr + k * (240 + ns * bps)
        put_i32(base + 0, k + 1)
        put_i16(base + 114, ns)
        put_i16(base + 116, dt_us)
        if receiver_depths is not None:
            put_i32(base + 40, int(round(float(receiver_depths[k]))))
        if fmt == 1:
            ib = ieee2ibm(data[:, k])
            buf[base + 240: base + 240 + ns * 4] = ib.astype(">u4").tobytes()
        else:
            buf[base + 240: base + 240 + ns * 4] = data[:, k].astype(dt_trace).tobytes()

    path.write_bytes(bytes(buf))
    return SegyMeta(path=str(path), ns=ns, dt_us=dt_us, fmt=fmt, n_traces=ntr)


# ==========================================================================
# 与 MAT 互转（方便和 MATLAB 后处理衔接）
# ==========================================================================
def save_mat(
    path: str | Path,
    data: np.ndarray,
    dt: float,
    var: str = "data",
    extra: Optional[Dict[str, np.ndarray]] = None,
) -> None:
    from scipy.io import savemat

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    d = {var: np.asarray(data, dtype=np.float64)}
    d["dt"] = float(dt)
    if extra:
        for k, v in extra.items():
            d[k] = np.asarray(v, dtype=np.float64)
    savemat(path, d)


def mat_candidates(path: str | Path) -> list[str]:
    """列出 MAT 里可用于去噪的条目名（供界面下拉/报错提示用）。

    - 嵌套结构里带 ``data`` 的，列出 ``变量名.data``；
    - 顶层二维数值数组，直接列出变量名。
    """
    from scipy.io import loadmat

    try:
        d = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    except Exception:
        return []
    out: list[str] = []
    for name, val in d.items():
        if name.startswith("__"):
            continue
        obj = _unwrap(val)
        if "data" in _struct_fields(obj):
            arr = np.asarray(_unwrap(getattr(obj, "data")), dtype=np.float32) if getattr(obj, "data", None) is not None else None
            if arr is not None and arr.ndim == 2:
                out.append(f"{name}.data")
        elif isinstance(val, np.ndarray) and val.dtype != object and val.ndim == 2 and np.issubdtype(val.dtype, np.number):
            out.append(name)
    return out


def load_mat_vars(path: str | Path) -> Dict[str, np.ndarray]:
    from scipy.io import loadmat

    d = loadmat(path)
    return {k: v for k, v in d.items() if not k.startswith("__")}


def _struct_fields(obj) -> list[str]:
    """mat_struct / ndarray-of-struct 的字段名清单。"""
    if hasattr(obj, "_fieldnames"):
        return list(obj._fieldnames)
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        inner = obj.ravel()[0]
        if hasattr(inner, "_fieldnames"):
            return list(inner._fieldnames)
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        return list(obj.dtype.names)
    return []


def _unwrap(obj):
    """把 (1,1) 的 object 数组剥成内部对象；普通数组原样返回。"""
    if isinstance(obj, np.ndarray) and obj.dtype == object and obj.size == 1:
        return obj.ravel()[0]
    if isinstance(obj, np.ndarray) and obj.dtype.names and obj.size == 1:
        return {n: np.asarray(obj[n].ravel()[0]) for n in obj.dtype.names}
    return obj


def load_seismic_mat(path: str | Path) -> Tuple[np.ndarray, Optional[float], Optional[float], str]:
    """读取地震处理软件导出的 MAT（SeismicLab / SeismicMAT 风格嵌套结构）。

    这些文件的典型形状是::

        seismic_shot            (1,1) struct
          ├── data              (ns, ntraces) float
          ├── header            (1,1) struct，含 dt / ns（dt 单位微秒）
          ├── trace_headers     (1,ntr) struct 数组，含 ReceiverGroupElevation
          └── shot_number

    返回 ``(data(ns,ntr) float32, dt 秒 或 None, dz 米 或 None, 说明)``。
    找不到嵌套结构时退回「挑最大的二维数组」。

    Raises
    ------
    ValueError
        文件里既没有可识别的地震结构，也找不到二维数组。
    """
    from scipy.io import loadmat

    path = Path(path)
    # struct_as_record=False：让 struct 成为带属性的对象，比 object 数组好拆
    d = loadmat(str(path), squeeze_me=False, struct_as_record=False)
    top = {k: v for k, v in d.items() if not k.startswith("__")}

    # ---- 1) 先找带 data 字段的嵌套 struct（不依赖变量名）----
    for name, val in top.items():
        obj = _unwrap(val)
        fields = _struct_fields(obj)
        if "data" not in fields:
            continue
        raw = getattr(obj, "data", None)
        if raw is None:
            continue
        arr = np.asarray(_unwrap(raw), dtype=np.float32)
        if arr.ndim != 2 or arr.size == 0:
            continue

        dt = None
        dz = None
        notes = [f"嵌套结构 {name}.data {arr.shape}"]

        # 头段里的 dt（单位微秒）与 ns
        if "header" in fields:
            hdr = _unwrap(getattr(obj, "header"))
            for key in ("dt", "dtOrig"):
                try:
                    v = float(np.asarray(getattr(hdr, key)).ravel()[0])
                    if 1.0 <= v <= 200_000.0:          # 微秒量级
                        dt = v / 1e6
                        notes.append(f"header.{key}={v:g}us")
                        break
                except Exception:
                    pass

        # 道头里的检波点高程差 → 道间距
        if dz is None and "trace_headers" in fields:
            try:
                th = getattr(obj, "trace_headers")
                elev = []
                for k in range(np.asarray(th).size):
                    one = _unwrap(np.asarray(th).ravel()[k])
                    try:
                        elev.append(float(np.asarray(getattr(one, "ReceiverGroupElevation")).ravel()[0]))
                    except Exception:
                        elev.append(np.nan)
                e = np.asarray(elev, dtype=np.float64)
                e = e[np.isfinite(e)]
                if e.size >= 3 and np.unique(e).size >= 2:
                    step = float(np.median(np.abs(np.diff(e))))
                    if step > 100.0:       # 与 SEG-Y 一致：差 >100 视为厘米
                        step /= 100.0
                    if 0.1 <= step <= 100.0:
                        dz = step
                        notes.append(f"dz≈{step:.3f}m（道头高程差）")
            except Exception:
                pass

        return np.ascontiguousarray(arr), dt, dz, "MAT " + "；".join(notes)

    # ---- 2) 退回：挑最大的二维数值数组 ----
    cand: Dict[str, np.ndarray] = {}
    for name, val in top.items():
        arr = np.asarray(val)
        if arr.dtype != object and arr.ndim == 2 and arr.size and np.issubdtype(arr.dtype, np.number):
            cand[name] = arr
    if not cand:
        raise ValueError(
            f"{path.name} 里没有可用的二维数据。"
            f"顶层变量：{sorted(top)}；若是嵌套结构，请确认含 data 字段。"
        )
    best = max(cand, key=lambda k: cand[k].size)
    arr = np.ascontiguousarray(cand[best], dtype=np.float32)

    # 顶层标量 dt（秒）兜底
    dt = None
    for k in ("dt", "DT", "sample_interval"):
        if k in top:
            try:
                v = float(np.asarray(top[k]).ravel()[0])
                dt = v / 1e6 if v > 1.0 else v       # 常见两种写法
                break
            except Exception:
                pass
    return arr, dt, None, f"MAT 变量 {best} {arr.shape}（顶层二维数组）"

