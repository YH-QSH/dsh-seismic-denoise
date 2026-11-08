"""证据评分卡：把门禁、保真、UQ、伪影和参考增益放在一张卡上。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .config import DEFAULT_CONFIG


DECISIONS = ("accept", "conditional", "reject", "abstain")


@dataclass
class Evidence:
    decision: str
    confidence: float
    best_method: str
    gate_passed: bool
    gate_score: float
    reference_gain_db: float | None
    energy_ratio: float | None
    signal_leakage: float | None
    weak_signal_ratio: float | None
    spectrum_corr: float | None
    uq: dict[str, Any]
    flags: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _finite(value) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and abs(out) != float("inf") else None


def build_evidence(
    best,
    card,
    uq,
    *,
    cfg: dict | None = None,
    ood_score: float | None = None,
) -> Evidence:
    """由最佳结果 + CaseCard + UQ 生成四档判断。"""
    config = (cfg or DEFAULT_CONFIG)["decision"]
    if best is None:
        return Evidence(
            decision="reject",
            confidence=0.0,
            best_method="无",
            gate_passed=False,
            gate_score=0.0,
            reference_gain_db=None,
            energy_ratio=None,
            signal_leakage=None,
            weak_signal_ratio=None,
            spectrum_corr=None,
            uq=uq.to_dict() if hasattr(uq, "to_dict") else dict(uq or {}),
            flags=["no_candidate"],
            reasons=["没有任何候选方法产生结果"],
        )

    metrics = getattr(best, "metrics", None)
    gate = getattr(best, "gate", None)
    extra = getattr(metrics, "extra", None) or {}
    visual = extra.get("visual") or {}
    spec = getattr(best, "spec", None)
    method = getattr(spec, "name", None) or getattr(spec, "key", "未知")

    ref_gain = _finite(getattr(metrics, "reference_gain_db", None))
    energy = _finite(getattr(metrics, "energy_ratio", None))
    leak = _finite(getattr(metrics, "signal_leakage", None))
    weak = _finite(visual.get("weak_signal_ratio"))
    spectrum = _finite(getattr(metrics, "spectrum_corr", None))
    resid = _finite(extra.get("trace_resid_ratio_out"))
    clip = _finite(getattr(metrics, "clip_ratio", None))

    flags: list[str] = []
    reasons: list[str] = []

    if gate is not None and not bool(getattr(gate, "passed", False)):
        flags.append("gate_failed")
        reasons.extend(list(getattr(gate, "failures", []) or [])[:3])
    if ref_gain is not None and ref_gain < float(config["min_reference_gain_db"]):
        flags.append("reference_gain_low")
        reasons.append(f"参考增益 {ref_gain:.2f} dB 低于 {config['min_reference_gain_db']:.1f} dB")
    if energy is not None and not (float(config["min_energy_ratio"]) <= energy <= float(config["max_energy_ratio"])):
        flags.append("energy_out_of_range")
        reasons.append(f"能量比 {energy:.3f} 不在 [{config['min_energy_ratio']}, {config['max_energy_ratio']}]")
    if leak is not None and leak > float(config["max_signal_leakage"]):
        flags.append("leakage_high")
        reasons.append(f"信号泄漏 {leak:.1%} 超过 {config['max_signal_leakage']:.0%}")
    if weak is not None and weak < float(config["min_weak_signal_ratio"]):
        flags.append("weak_signal_lost")
        reasons.append(f"弱区能量保留 {weak:.1%} 低于 {config['min_weak_signal_ratio']:.0%}")
    if spectrum is not None and spectrum < float(config["min_spectrum_corr"]):
        flags.append("spectrum_mismatch")
        reasons.append(f"谱相关 {spectrum:.3f} 低于 {config['min_spectrum_corr']:.2f}")
    if resid is not None and resid > float(config["max_resid_ratio"]):
        flags.append("vertical_stripe")
        reasons.append(f"垂直条带/单道尖峰 {resid:.1f}× 超过 {config['max_resid_ratio']:.0f}×")
    if clip is not None and clip > 1e-3:
        flags.append("clipping")
        reasons.append(f"削波比例 {clip:.2%} 偏高")
    if ood_score is not None and ood_score > float(DEFAULT_CONFIG["ood"]["abstain_score"]):
        flags.append("ood_high")
        reasons.append(f"OOD 分数 {ood_score:.2f} 偏高")

    # 一票否决
    hard_flags = {"gate_failed", "reference_gain_low", "energy_out_of_range",
                  "leakage_high", "weak_signal_lost", "spectrum_mismatch",
                  "vertical_stripe", "clipping", "ood_high"}
    if hard_flags & set(flags):
        decision = "abstain" if "ood_high" in flags else "reject"
        confidence = min(0.45, float(getattr(gate, "confidence", 0.0) or 0.0) * 0.5)
        if not reasons:
            reasons.append("证据卡存在硬失败项")
        return Evidence(
            decision=decision,
            confidence=round(confidence, 4),
            best_method=str(method),
            gate_passed=bool(getattr(gate, "passed", False)),
            gate_score=float(getattr(gate, "score", 0.0) or 0.0),
            reference_gain_db=ref_gain,
            energy_ratio=energy,
            signal_leakage=leak,
            weak_signal_ratio=weak,
            spectrum_corr=spectrum,
            uq=uq.to_dict() if hasattr(uq, "to_dict") else dict(uq or {}),
            flags=sorted(set(flags)),
            reasons=reasons,
        )

    # 没有参考：只能条件通过，不能自动 accept
    if card is not None and not bool(getattr(card, "has_reference", False)):
        decision = "conditional"
        reasons.append("没有纯净参考，无法验证真实增益；按条件通过，需人工复核")
        confidence = min(0.70, float(getattr(gate, "confidence", 0.0) or 0.0) + 0.15)
    else:
        uq_low = None
        uq_available = bool(getattr(uq, "available", False))
        uq_reliable = bool(getattr(uq, "reliable", False))
        if uq_available:
            uq_low = _finite(uq.gain_ci_low)
        ref_ok = ref_gain is not None and ref_gain >= float(config["min_reference_gain_db"])
        unstable = bool(uq_available and uq_low is not None
                        and uq_low < float(config["min_reference_gain_db"]))
        if ref_ok and not unstable:
            decision = "accept"
            reasons.append("参考增益达标，门禁/能量/泄漏/弱区/谱相关均通过")
            confidence = min(0.95, 0.60 + 0.20 * min(1.0, ref_gain / 5.0) + (0.10 if uq_reliable else 0.0))
        else:
            decision = "conditional"
            if unstable:
                reasons.append("参考增益点估计达标，但 UQ 下界不足，需人工复核")
            else:
                reasons.append("有参考但证据不足以无条件接受")
            confidence = min(0.85, 0.45 + 0.20 * min(1.0, max(0.0, ref_gain or 0.0) / 5.0))

    if hasattr(uq, "notes"):
        notes = list(getattr(uq, "notes") or [])
    else:
        notes = []
    return Evidence(
        decision=decision,
        confidence=round(float(confidence), 4),
        best_method=str(method),
        gate_passed=bool(getattr(gate, "passed", False)),
        gate_score=float(getattr(gate, "score", 0.0) or 0.0),
        reference_gain_db=ref_gain,
        energy_ratio=energy,
        signal_leakage=leak,
        weak_signal_ratio=weak,
        spectrum_corr=spectrum,
        uq=uq.to_dict() if hasattr(uq, "to_dict") else dict(uq or {}),
        flags=sorted(set(flags)),
        reasons=reasons,
        notes=notes,
    )
