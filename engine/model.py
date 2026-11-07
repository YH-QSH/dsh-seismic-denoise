"""地震数据智能去噪插件 —— 全模块共享的数据模型。

设计原则：
- 数据矩阵只在本进程内传递，不写进 JSON；跨阶段用 数据缓存.npz + 数据集信息.json。
- 所有模型都能 to_info() 成可 JSON 序列化的 dict。
- 道头字段保留完整 values（供后续来源匹配/报告），JSON 里可按需只放统计。
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np


# --------------------------------------------------------------------------
# JSON 序列化
# --------------------------------------------------------------------------
def to_jsonable(obj: Any) -> Any:
    """递归转成 json.dumps 能处理的类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        if not math.isfinite(value):
            return None
        # DSH 的“无损 JSON”检查不接受 -0.0（JSON.stringify 后会变成 0）。
        return 0.0 if value == 0.0 else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    if is_dataclass(obj):
        return to_jsonable(asdict(obj))
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj


def dumps(obj: Any, indent: int = 2) -> str:
    return json.dumps(to_jsonable(obj), ensure_ascii=False, indent=indent)


# --------------------------------------------------------------------------
# 道头
# --------------------------------------------------------------------------
def _array_stats(values: np.ndarray) -> dict:
    a = np.asarray(values, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"count": 0}
    uniq = np.unique(a)
    return {
        "count": int(a.size),
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
        "median": float(np.median(a)),
        "unique": int(uniq.size),
        "first": float(a[0]),
        "last": float(a[-1]),
    }


@dataclass
class HeaderField:
    """一个道头字段。byte 从 1 开始，size 单位字节。"""

    key: str
    name: str
    byte: int
    size: int
    kind: str  # int16 / int32 / float32 / uint16 ...
    values: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.float64))
    note: str = ""

    def stats(self) -> dict:
        return _array_stats(self.values)

    def to_info(self, max_values: int = 48) -> dict:
        vals = np.asarray(self.values).ravel()
        sample = vals[:max_values].tolist()
        return {
            "key": self.key,
            "name": self.name,
            "byte": self.byte,
            "size": self.size,
            "kind": self.kind,
            "note": self.note,
            "stats": self.stats(),
            "values_sample": sample,
            "values_count": int(vals.size),
        }


@dataclass
class HeaderTable:
    fields: dict[str, HeaderField] = field(default_factory=dict)

    def add(self, key: str, name: str, byte: int, size: int, kind: str,
            values, note: str = "") -> None:
        self.fields[key] = HeaderField(
            key=key, name=name, byte=byte, size=size, kind=kind,
            values=np.asarray(values, dtype=np.float64), note=note,
        )

    def get(self, key: str) -> HeaderField | None:
        return self.fields.get(key)

    def values(self, key: str) -> np.ndarray | None:
        f = self.fields.get(key)
        return None if f is None else f.values

    def to_info(self) -> dict:
        return {k: f.to_info() for k, f in self.fields.items()}

    def to_rows(self) -> list[dict]:
        """展开成 一行一个字段 的表格，写 CSV 用。"""
        rows = []
        for f in self.fields.values():
            s = f.stats()
            rows.append({
                "字段": f.name,
                "键": f.key,
                "起始字节": f.byte,
                "字节数": f.size,
                "类型": f.kind,
                "最小值": s.get("min"),
                "最大值": s.get("max"),
                "均值": s.get("mean"),
                "标准差": s.get("std"),
                "中位数": s.get("median"),
                "唯一值数": s.get("unique"),
                "备注": f.note,
            })
        return rows

    def to_dict(self) -> dict:
        return {k: asdict(f) for k, f in self.fields.items()}

    @classmethod
    def from_dict(cls, obj: dict) -> "HeaderTable":
        t = cls()
        for k, d in (obj or {}).items():
            vals = d.get("values", [])
            t.fields[k] = HeaderField(
                key=d.get("key", k),
                name=d.get("name", k),
                byte=int(d.get("byte", 1)),
                size=int(d.get("size", 4)),
                kind=d.get("kind", "int32"),
                values=np.asarray(vals, dtype=np.float64),
                note=d.get("note", ""),
            )
        return t


# --------------------------------------------------------------------------
# 地震数据集
# --------------------------------------------------------------------------
@dataclass
class SeismicDataset:
    path: str
    format: str
    data: np.ndarray  # (n_samples, n_traces) float32
    dt: float | None = None  # 秒
    dz: float | None = None  # 米
    headers: HeaderTable = field(default_factory=HeaderTable)
    text_header: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    source_kind: str = "unknown"

    @property
    def n_samples(self) -> int:
        return int(self.data.shape[0]) if self.data.ndim >= 2 else 0

    @property
    def n_traces(self) -> int:
        return int(self.data.shape[1]) if self.data.ndim >= 2 else 0

    def to_info(self, include_headers: bool = True, header_values: bool = False) -> dict:
        info = {
            "path": self.path,
            "format": self.format,
            "n_samples": self.n_samples,
            "n_traces": self.n_traces,
            "dt_ms": None if self.dt is None else round(self.dt * 1000.0, 6),
            "dz_m": None if self.dz is None else round(self.dz, 6),
            "duration_s": None if self.dt is None else round(self.dt * self.n_samples, 6),
            "data_bytes": int(self.data.nbytes),
            "amplitude": {
                "min": float(np.nanmin(self.data)) if self.data.size else 0.0,
                "max": float(np.nanmax(self.data)) if self.data.size else 0.0,
                "mean": float(np.nanmean(self.data)) if self.data.size else 0.0,
                "std": float(np.nanstd(self.data)) if self.data.size else 0.0,
                "finite_ratio": float(np.isfinite(self.data).mean()) if self.data.size else 1.0,
            },
            "confidence": round(float(self.confidence), 4),
            "warnings": list(self.warnings),
            "source_kind": self.source_kind,
            "meta": to_jsonable(self.meta),
            "text_header": self.text_header[:4000] if self.text_header else "",
        }
        if include_headers:
            info["headers"] = {
                k: (f.to_info() if not header_values else {
                    **f.to_info(), "values": f.values.tolist()
                })
                for k, f in self.headers.fields.items()
            }
        return info

    # ---- 缓存：跨 CLI 阶段传递 ----
    def save_cache(self, folder: str | Path) -> Path:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            folder / "数据缓存.npz",
            data=np.asarray(self.data, dtype=np.float32),
            dt=np.asarray([self.dt if self.dt is not None else np.nan], dtype=np.float64),
            dz=np.asarray([self.dz if self.dz is not None else np.nan], dtype=np.float64),
        )
        info = self.to_info(include_headers=True)
        info["_headers_raw"] = self.headers.to_dict()
        (folder / "数据集信息.json").write_text(dumps(info), encoding="utf-8")
        return folder / "数据缓存.npz"

    @classmethod
    def load_cache(cls, folder: str | Path, path: str = "") -> "SeismicDataset":
        folder = Path(folder)
        with np.load(folder / "数据缓存.npz", allow_pickle=False) as z:
            data = np.asarray(z["data"], dtype=np.float32)
            dt = float(z["dt"][0])
            dz = float(z["dz"][0])
        info = json.loads((folder / "数据集信息.json").read_text(encoding="utf-8"))
        headers = HeaderTable.from_dict(info.pop("_headers_raw", {}))
        info.pop("headers", None)
        ds = cls(
            path=path or info.get("path", ""),
            format=info.get("format", "cache"),
            data=data,
            dt=None if not np.isfinite(dt) else dt,
            dz=None if not np.isfinite(dz) else dz,
            headers=headers,
            text_header=info.get("text_header", ""),
            meta=info.get("meta", {}),
            confidence=float(info.get("confidence", 0.0)),
            warnings=list(info.get("warnings", [])),
            source_kind=info.get("source_kind", "unknown"),
        )
        return ds


# --------------------------------------------------------------------------
# 来源匹配 / 数据诊断
# --------------------------------------------------------------------------
@dataclass
class SourceAssessment:
    domain: str = "未知"
    acquisition: str = "未知"
    geometry_type: str = "未知"
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    geometry: dict[str, Any] = field(default_factory=dict)
    cleanliness: dict[str, Any] = field(default_factory=dict)
    noise_types: dict[str, float] = field(default_factory=dict)
    needs_denoise: bool = True
    recommended: list[str] = field(default_factory=list)
    clean_reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    notes: str = ""

    def to_info(self) -> dict:
        return to_jsonable(asdict(self))


# --------------------------------------------------------------------------
# 质量评价 / 门禁
# --------------------------------------------------------------------------
@dataclass
class QualityMetrics:
    snr_in_db: float | None = None
    snr_out_db: float | None = None
    snr_gain_db: float | None = None
    reference_snr_in_db: float | None = None
    reference_snr_out_db: float | None = None
    reference_gain_db: float | None = None
    noise_before_db: float | None = None
    noise_after_db: float | None = None
    noise_reduction_db: float | None = None
    energy_ratio: float | None = None
    signal_leakage: float | None = None
    spectrum_corr: float | None = None
    coherence_residual: float | None = None
    clip_ratio: float | None = None
    local_snr_gain_db: float | None = None
    estimator: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def to_info(self) -> dict:
        return to_jsonable(asdict(self))


@dataclass
class GateCheck:
    name: str
    value: float | None
    threshold: str
    passed: bool
    note: str = ""


@dataclass
class GateResult:
    passed: bool = False
    score: float = 0.0
    checks: list[GateCheck] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    confidence: float = 0.0
    advice: list[str] = field(default_factory=list)

    def to_info(self) -> dict:
        return to_jsonable(asdict(self))


# --------------------------------------------------------------------------
# 去噪方法与尝试记录
# --------------------------------------------------------------------------
@dataclass
class MethodSpec:
    key: str
    name: str
    params: dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    targets: list[str] = field(default_factory=list)
    strength: float = 1.0
    priority: int = 0
    source: str = "rule"

    def to_info(self) -> dict:
        return to_jsonable(asdict(self))


@dataclass
class MethodResult:
    spec: MethodSpec
    metrics: QualityMetrics = field(default_factory=QualityMetrics)
    gate: GateResult = field(default_factory=GateResult)
    runtime_s: float = 0.0
    warnings: list[str] = field(default_factory=list)
    notes: str = ""
    output_path: str = ""
    data: np.ndarray | None = field(default=None, repr=False)  # 仅内存，to_info 不输出

    def to_info(self) -> dict:
        return {
            "method": self.spec.to_info(),
            "metrics": self.metrics.to_info(),
            "gate": self.gate.to_info(),
            "runtime_s": round(self.runtime_s, 3),
            "warnings": list(self.warnings),
            "notes": self.notes,
            "output_path": self.output_path,
        }
