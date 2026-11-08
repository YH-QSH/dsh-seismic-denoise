"""阶段四 P1 配置。只放判断层参数，不改阶段三门禁。"""
from __future__ import annotations

DEFAULT_CONFIG: dict = {
    "decision": {
        "min_reference_gain_db": 0.5,
        "min_energy_ratio": 0.60,
        "max_energy_ratio": 1.05,
        "max_signal_leakage": 0.45,
        "min_weak_signal_ratio": 0.85,
        "min_spectrum_corr": 0.80,
        "max_resid_ratio": 250.0,
        "no_reference_max_decision": "conditional",
        "min_independent_evidence": 3,
    },
    "uq": {
        "confidence": 0.90,
        "min_samples": 2,
        "min_reliable_samples": 3,
        "max_std_db": 2.0,
    },
    "ood": {
        "enabled": False,
        "abstain_score": 0.70,
    },
}
