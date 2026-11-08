"""P1 编排：读数据 → CaseCard → 阶段三候选 → 集成 UQ → 证据卡。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from engine.analysis.diagnostics import diagnostic_summary
from engine.analysis.source import match_source
from engine.formats import read_any
from engine.methods.planner import plan
from engine.methods.runner import pick_best, run_attempts

from .case_card import CaseCard, build_case_card
from .config import DEFAULT_CONFIG
from .evidence import Evidence, build_evidence
from .uq import UQReport, ensemble_uq


@dataclass
class CaseResult:
    input_path: str
    reference_path: str | None
    card: CaseCard
    candidates: list[dict[str, Any]]
    attempts: list[dict[str, Any]]
    best_method: str | None
    uq: UQReport
    evidence: Evidence
    mode: str = "denoise"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_path": self.input_path,
            "reference_path": self.reference_path,
            "mode": self.mode,
            "card": self.card.to_dict(),
            "candidates": self.candidates,
            "attempts": self.attempts,
            "best_method": self.best_method,
            "uq": self.uq.to_dict(),
            "evidence": self.evidence.to_dict(),
            "notes": self.notes,
        }


def _load_reference(path: str | None) -> tuple[np.ndarray | None, str | None]:
    if not path:
        return None, None
    ref = read_any(path)
    return np.asarray(ref.data, dtype=np.float32), str(path)


def _result_summary(result) -> dict[str, Any]:
    metrics = getattr(result, "metrics", None)
    gate = getattr(result, "gate", None)
    spec = getattr(result, "spec", None)
    return {
        "method": getattr(spec, "name", None) or getattr(spec, "key", "未知"),
        "key": getattr(spec, "key", None),
        "reference_gain_db": getattr(metrics, "reference_gain_db", None) if metrics else None,
        "snr_gain_db": getattr(metrics, "snr_gain_db", None) if metrics else None,
        "energy_ratio": getattr(metrics, "energy_ratio", None) if metrics else None,
        "signal_leakage": getattr(metrics, "signal_leakage", None) if metrics else None,
        "spectrum_corr": getattr(metrics, "spectrum_corr", None) if metrics else None,
        "gate_passed": bool(getattr(gate, "passed", False)) if gate else False,
        "gate_score": float(getattr(gate, "score", 0.0) or 0.0) if gate else 0.0,
        "warnings": list(getattr(result, "warnings", []) or []),
    }


def _reference_close(data: np.ndarray, reference: np.ndarray | None) -> bool:
    """输入与参考几乎一致：corr≥0.995 且相对差异≤0.02。"""
    if reference is None:
        return False
    a = np.asarray(data, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape:
        return False
    a = a.ravel()
    b = b.ravel()
    if a.size < 16 or b.size < 16:
        return False
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return False
    corr = float(np.dot(a, b) / denom)
    rel = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-12))
    return corr >= 0.995 and rel <= 0.02


def run_case(
    input_path: str | Path,
    reference_path: str | Path | None = None,
    *,
    cfg: dict | None = None,
    max_methods: int = 4,
    allow_cnn: bool = False,
) -> CaseResult:
    """P1 单案例：只读调用阶段三，不写阶段三目录。"""
    input_path = str(input_path)
    ref_path = str(reference_path) if reference_path else None
    config = cfg or DEFAULT_CONFIG
    stage3_cfg = {
        "allow_cnn": bool(allow_cnn),
        "planner": {"max_methods": int(max_methods), "allow_cnn": bool(allow_cnn)},
    }

    ds = read_any(input_path)
    assessment = match_source(ds)
    diagnostics = diagnostic_summary(ds.data, ds.dt, ds.dz)
    ref, ref_loaded = _load_reference(ref_path)
    card = build_case_card(ds, assessment, diagnostics, has_reference=ref is not None)

    if not card.needs_denoise or _reference_close(ds.data, ref):
        uq = ensemble_uq([])
        clean_reason = ("来源匹配判定数据接近干净，不进入去噪"
                        if not card.needs_denoise else
                        "输入与提供的纯净参考几乎一致，不进入去噪")
        evidence = Evidence(
            decision="accept",
            confidence=round(min(0.90, 0.55 + 0.35 * float(card.confidence)), 4),
            best_method="无需去噪",
            gate_passed=True,
            gate_score=0.0,
            reference_gain_db=None,
            energy_ratio=None,
            signal_leakage=None,
            weak_signal_ratio=None,
            spectrum_corr=None,
            uq=uq.to_dict(),
            flags=[],
            reasons=[clean_reason],
            notes=list(diagnostics.get("evidence", []) or []),
        )
        return CaseResult(
            input_path=input_path,
            reference_path=ref_loaded,
            card=card,
            candidates=[],
            attempts=[],
            best_method="无需去噪",
            uq=uq,
            evidence=evidence,
            mode="clean",
            notes=list(diagnostics.get("evidence", []) or []),
        )

    specs = plan(assessment, cfg=stage3_cfg)
    results = run_attempts(ds, specs, cfg=stage3_cfg, reference=ref, log=lambda *a, **k: None)
    best = pick_best(results)
    uq = ensemble_uq(results, confidence=float(config["uq"]["confidence"]), best=best)
    evidence = build_evidence(best, card, uq, cfg=config)

    attempts = [_result_summary(r) for r in results]
    return CaseResult(
        input_path=input_path,
        reference_path=ref_loaded,
        card=card,
        candidates=[
            {"key": s.key, "name": s.name, "source": s.source, "why": s.rationale}
            for s in specs
        ],
        attempts=attempts,
        best_method=evidence.best_method,
        uq=uq,
        evidence=evidence,
        notes=list(diagnostics.get("evidence", []) or []),
    )
