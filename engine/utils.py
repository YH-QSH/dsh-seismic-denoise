"""通用工具：路径、日志、JSON、数组清洗、重采样。"""
from __future__ import annotations

import json
import math
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .model import to_jsonable


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_name(name: str, maxlen: int = 80) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", str(name)).strip(" .")
    return name[:maxlen] or "未命名"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, obj: Any) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def read_json(path: str | Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return default


class Logger:
    """同时写文件和 stderr；stderr 只放进度，最终 JSON 走 --json-out。"""

    def __init__(self, logfile: str | Path | None = None):
        self.lines: list[str] = []
        self.logfile = Path(logfile) if logfile else None
        if self.logfile:
            self.logfile.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        self.lines.append(line)
        try:
            print(line, file=sys.stderr, flush=True)
        except Exception:
            pass
        if self.logfile:
            try:
                with self.logfile.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except Exception:
                pass

    def section(self, title: str) -> None:
        self(f"—— {title} ——")


def clean_array(data: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """转 float32、非有限值置 0；返回 (数组, 警告)。不改幅度。"""
    warnings: list[str] = []
    arr = np.asarray(data, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"数据必须是二维 (道数, 采样点)，实际 shape={arr.shape}")
    bad = ~np.isfinite(arr)
    if bad.any():
        ratio = float(bad.mean())
        warnings.append(f"数据含 {bad.sum()} 个非有限值（{ratio:.3%}），已置 0")
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return arr, warnings


def resample_time(data: np.ndarray, dt: float, target_dt: float) -> np.ndarray:
    if dt is None or target_dt is None or abs(dt - target_dt) < 1e-9:
        return data
    ratio = dt / target_dt
    # 用有理数近似，避免超大 up/down
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(1000)
    return signal.resample_poly(data, frac.numerator, frac.denominator, axis=0).astype(np.float32)


def resample_space(data: np.ndarray, dz: float | None, target_dz: float | None) -> np.ndarray:
    if not dz or not target_dz or abs(dz - target_dz) < 1e-9:
        return data
    ratio = dz / target_dz
    from fractions import Fraction
    frac = Fraction(ratio).limit_denominator(1000)
    return signal.resample_poly(data, frac.numerator, frac.denominator, axis=1).astype(np.float32)


def resample_to(data: np.ndarray, dt: float | None, dz: float | None,
                target_dt: float | None, target_dz: float | None) -> np.ndarray:
    out = data
    if target_dt:
        out = resample_time(out, dt or target_dt, target_dt)
    if target_dz:
        out = resample_space(out, dz, target_dz)
    return out.astype(np.float32)


def safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.size < 3 or b.size != a.size:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    den = float(np.linalg.norm(a) * np.linalg.norm(b))
    if den < 1e-12:
        return 0.0
    return float(np.dot(a, b) / den)


def db(x: float, floor: float = 1e-12) -> float:
    return float(10.0 * math.log10(max(float(x), floor)))


def trace_offsets(ds) -> np.ndarray | None:
    """从道头推算偏移距（米），失败返回 None。"""
    h = ds.headers
    off = h.values("offset")
    if off is not None and np.isfinite(off).any() and np.nanmax(np.abs(off)) > 0:
        return np.abs(off)
    sx, sy = h.values("source_x"), h.values("source_y")
    gx, gy = h.values("cdp_x"), h.values("cdp_y")
    if sx is not None and gx is not None:
        if sy is not None and gy is not None:
            d = np.sqrt((sx - gx) ** 2 + (sy - gy) ** 2)
        else:
            d = np.abs(sx - gx)
        # 坐标常以 厘米/分米 存；量级异常时归一化为相对值
        if np.isfinite(d).any() and np.nanmedian(d) > 10000:
            d = d / 100.0
        return d
    return None


def amplitude_range(data: np.ndarray) -> dict:
    a = np.asarray(data)
    return {
        "min": float(np.nanmin(a)) if a.size else 0.0,
        "max": float(np.nanmax(a)) if a.size else 0.0,
        "p99": float(np.nanpercentile(np.abs(a), 99)) if a.size else 0.0,
    }


def elapsed(t0: float) -> float:
    return round(time.time() - t0, 3)
