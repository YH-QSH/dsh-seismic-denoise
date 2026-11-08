"""CaseCard：把一次任务压缩成可比较、可入库的特征卡。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from engine.analysis.diagnostics import diagnostic_summary
from engine.analysis.source import match_source
from engine.model import SeismicDataset


NOISE_KEYS = ["面波", "随机噪声", "工频干扰", "散射绕射", "折射波"]


@dataclass
class CaseCard:
    domain: str
    acquisition: str
    geometry_type: str
    ns: int
    ntr: int
    dt_ms: float
    dz_m: float | None
    bad_trace_ratio: float
    noise_scores: dict[str, float]
    dominant_noise: str
    snr_db: float | None
    snr_reliable: bool
    hf_ratio: float
    spike_ratio: float
    coherence: float
    has_reference: bool
    confidence: float
    needs_denoise: bool = True
    recommended: list[str] = field(default_factory=list)
    ood_score: float | None = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_vector(self) -> list[float]:
        return [
            float(self.ns),
            float(self.ntr),
            float(self.dt_ms),
            float(self.dz_m or 0.0),
            float(self.bad_trace_ratio),
            float(self.snr_db if self.snr_db is not None else 0.0),
            float(self.hf_ratio),
            float(self.spike_ratio),
            float(self.coherence),
            float(self.confidence),
            *[float(self.noise_scores.get(k, 0.0)) for k in NOISE_KEYS],
        ]

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "ns",
            "ntr",
            "dt_ms",
            "dz_m",
            "bad_trace_ratio",
            "snr_db",
            "hf_ratio",
            "spike_ratio",
            "coherence",
            "source_confidence",
            *[f"noise_{k}" for k in NOISE_KEYS],
        ]


def _bad_trace_ratio(data: np.ndarray, ratio: float = 50.0) -> float:
    if data.ndim != 2 or data.shape[1] == 0:
        return 0.0
    rms = np.sqrt(np.mean(np.asarray(data, dtype=np.float64) ** 2, axis=0))
    med = float(np.median(rms))
    if not np.isfinite(med) or med <= 0:
        return 0.0
    return float(np.mean(rms > ratio * med))


def build_case_card(
    ds: SeismicDataset,
    assessment=None,
    diagnostics: dict | None = None,
    *,
    has_reference: bool = False,
) -> CaseCard:
    """从阶段三的数据集/来源匹配/诊断结果构建 CaseCard。"""
    if assessment is None:
        assessment = match_source(ds)
    if diagnostics is None:
        diagnostics = diagnostic_summary(ds.data, ds.dt, ds.dz)

    snr = diagnostics.get("snr", {}) or {}
    spectral = diagnostics.get("spectral", {}) or {}
    spike = diagnostics.get("spike", {}) or {}
    coherence = diagnostics.get("coherence", {}) or {}
    noise = diagnostics.get("noise", {}) or assessment.noise_types or {}

    return CaseCard(
        domain=str(getattr(assessment, "domain", "未知")),
        acquisition=str(getattr(assessment, "acquisition", "未知")),
        geometry_type=str(getattr(assessment, "geometry_type", "未知")),
        ns=int(ds.data.shape[0]),
        ntr=int(ds.data.shape[1]),
        dt_ms=float(ds.dt * 1000.0),
        dz_m=float(ds.dz) if ds.dz is not None else None,
        bad_trace_ratio=_bad_trace_ratio(ds.data),
        noise_scores={k: float(noise.get(k, 0.0)) for k in NOISE_KEYS},
        dominant_noise=str(diagnostics.get("dominant_noise", "未知")),
        snr_db=float(snr["snr_db"]) if snr.get("snr_db") is not None else None,
        snr_reliable=bool(snr.get("reliable", False)),
        hf_ratio=float(spectral.get("hf_ratio", 0.0) or 0.0),
        spike_ratio=float(spike.get("spike_ratio", 0.0) or 0.0),
        coherence=float(coherence.get("adjacent_coherence", 0.0) or 0.0),
        has_reference=bool(has_reference),
        confidence=float(getattr(assessment, "confidence", 0.0) or 0.0),
        needs_denoise=bool(getattr(assessment, "needs_denoise", True)),
        recommended=list(getattr(assessment, "recommended", []) or []),
        warnings=list(getattr(assessment, "warnings", []) or []),
    )
