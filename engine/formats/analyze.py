"""任意二进制初判：给 CLI / 诊断报告一个“这文件可能是什么”的摘要。

不加载整份数据；只读文件头若干字节 + SEG-Y 头段 + RAW 候选扫描。
返回 dict 只含 JSON 友好值（最后过一遍 ``model.to_jsonable``）。
"""
from __future__ import annotations

from pathlib import Path

from ..model import to_jsonable
from .common import clamp01, human_size
from .detect import detect_format
from .raw import scan_candidates
from .segy import BASE_SIZE, probe_segy

_HEXDUMP_BYTES = 64


def _hexdump(data: bytes, limit: int = _HEXDUMP_BYTES) -> list[str]:
    """经典 16 字节一行的 hexdump（只取前 limit 字节）。"""
    lines: list[str] = []
    for off in range(0, min(len(data), limit), 16):
        chunk = data[off:off + 16]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        txt = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        lines.append(f"{off:08x}  {hex_part:<47}  |{txt}|")
    return lines


def analyze_binary(path: str | Path, max_bytes: int = 2_000_000) -> dict:
    """对任意二进制文件做初判，返回魔数、候选解释、置信度与建议。"""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"文件不存在或不是普通文件：{p}")
    size = int(p.stat().st_size)
    if size <= 0:
        raise ValueError(f"空文件：{p}")

    read_len = int(min(size, max(64, int(max_bytes))))
    with p.open("rb") as f:
        head = f.read(read_len)

    fmt = detect_format(p)
    warnings: list[str] = []
    suggestions: list[str] = []
    if read_len < size:
        warnings.append(
            f"只分析了前 {human_size(read_len)}（max_bytes={max_bytes}），"
            f"文件共 {human_size(size)}"
        )

    # ---- SEG-Y 头段试探 ----
    segy_info = None
    segy_strong = False
    if size >= BASE_SIZE:
        try:
            info = probe_segy(p)
            segy_strong = float(info.get("score", 0.0)) >= 0.55 and bool(info.get("bytes_per_sample"))
            if segy_strong or fmt == "segy":
                segy_info = info
        except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
            warnings.append(f"SEG-Y 头段试探失败：{exc}")
    elif fmt == "segy":
        warnings.append(f"文件只有 {size} 字节，不足 SEG-Y 最小头段（{BASE_SIZE} 字节）")

    # ---- RAW 候选扫描（只有 SEG-Y 不成立时才值得做）----
    raw_candidates: list[dict] = []
    if not segy_strong and fmt in ("raw", "unknown") and size >= 64:
        try:
            raw_candidates = scan_candidates(p, max_bytes=min(size, max(4096, int(max_bytes))), top=5)
        except (ValueError, OSError, MemoryError) as exc:
            warnings.append(f"RAW 候选扫描失败：{exc}")

    # ---- 置信度 ----
    confidence = 0.10
    if segy_strong and segy_info is not None:
        confidence = clamp01(0.35 + 0.55 * float(segy_info["score"]))
    elif fmt in ("npy", "npz", "mat", "hdf5"):
        confidence = 0.85
    elif raw_candidates:
        confidence = min(0.45, 0.10 + 0.45 * float(raw_candidates[0]["score"]))
    elif fmt == "segy":
        confidence = 0.30

    # ---- 建议 ----
    if segy_strong and segy_info is not None:
        dt_text = f"{segy_info['dt_ms']:.6g} ms" if segy_info.get("dt_ms") else "未知"
        suggestions.append(
            f"标准 SEG-Y 头段成立：{segy_info['endian_name']}、"
            f"格式码 {segy_info['sample_format']}（{segy_info['sample_format_name']}）、"
            f"ns={segy_info['ns']}、dt={dt_text}、估计 {segy_info['n_traces_est']} 道；"
            "可直接 read_any()，用 read_segy_headers() 可查看道头字段"
        )
        if not segy_info.get("aligned") and segy_info.get("n_traces_est"):
            suggestions.append("道数与文件长度不整除，可能存在变长道或尾部附加数据")
    elif fmt in ("npy", "npz", "mat", "hdf5"):
        suggestions.append(f"按魔数/扩展名识别为 {fmt}，可直接 read_any() 读取")
    elif raw_candidates:
        best = raw_candidates[0]
        suggestions.append(
            f"没有可识别的 SEG-Y/容器头段；若这是裸样本，可尝试 "
            f"{best['dtype_label']}、ns={best['ns']}（约 {best['ntr']} 道），"
            "但 dt/dz 必须另行提供"
        )
        suggestions.append("建议改用带 3200+400 字节头段的 SEG-Y，或提供 ns/dt 后显式读取")
    else:
        suggestions.append(
            "未能识别格式：既不是 SEG-Y/常见容器，也没有找到合理的裸二进制解释；"
            "请确认文件未加密/未压缩，或补充数据说明"
        )

    result = {
        "path": str(p),
        "file_size": size,
        "file_size_text": human_size(size),
        "max_bytes": int(max_bytes),
        "bytes_read": read_len,
        "magic_hex": head[:32].hex(" "),
        "magic_ascii": "".join(chr(b) if 32 <= b < 127 else "." for b in head[:16]),
        "hexdump": _hexdump(head),
        "format": fmt,
        "segy": segy_info,
        "segy_strong": segy_strong,
        "raw_candidates": raw_candidates,
        "confidence": round(float(confidence), 3),
        "suggestions": suggestions,
        "warnings": warnings,
    }
    return to_jsonable(result)
