"""阶段四判断层（engine/judgment）单元测试：UQ、证据卡、四档决策。"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from engine.judgment.case_card import CaseCard  # noqa: E402
from engine.judgment.evidence import build_evidence  # noqa: E402
from engine.judgment.orchestrator import _reference_close  # noqa: E402
from engine.judgment.uq import ensemble_uq  # noqa: E402


def fake_result(ref=None, blind=None, energy=0.9, leak=0.2, spec=0.99, weak=0.95,
                passed=True, unreliable=False):
    extra = {
        "reliable": not unreliable,
        "visual": {"weak_signal_ratio": weak},
        "trace_resid_ratio_out": 30.0,
    }
    metrics = SimpleNamespace(
        reference_gain_db=ref,
        snr_gain_db=blind,
        energy_ratio=energy,
        signal_leakage=leak,
        spectrum_corr=spec,
        clip_ratio=0.0,
        extra=extra,
    )
    gate = SimpleNamespace(passed=passed, score=60.0, confidence=0.4, failures=[])
    spec_obj = SimpleNamespace(key="bandpass", name="带通滤波")
    return SimpleNamespace(metrics=metrics, gate=gate, spec=spec_obj)


def card(has_ref=True):
    return CaseCard(
        domain="地面地震",
        acquisition="2D/3D 地面反射",
        geometry_type="地面排列",
        ns=1000,
        ntr=100,
        dt_ms=2.0,
        dz_m=5.0,
        bad_trace_ratio=0.0,
        noise_scores={"面波": 0.8, "随机噪声": 0.2, "工频干扰": 0.0,
                      "散射绕射": 0.1, "折射波": 0.0},
        dominant_noise="面波",
        snr_db=10.0,
        snr_reliable=True,
        hf_ratio=0.01,
        spike_ratio=0.01,
        coherence=0.5,
        has_reference=has_ref,
        confidence=0.9,
    )


def test_ensemble_uq_uses_reference_gains():
    results = [fake_result(ref=v) for v in (1.0, 1.5, 2.0)]
    uq = ensemble_uq(results)
    assert uq.available is True
    assert uq.n == 3
    assert abs(uq.gain_mean_db - 1.5) < 1e-9
    assert uq.reliable is True
    assert uq.gain_ci_low < uq.gain_mean_db < uq.gain_ci_high


def test_no_reference_is_conditional():
    best = fake_result(ref=None, blind=None)
    uq = ensemble_uq([best])
    ev = build_evidence(best, card(has_ref=False), uq)
    assert ev.decision == "conditional"
    assert "没有纯净参考" in " ".join(ev.reasons)


def test_reject_on_leakage():
    best = fake_result(ref=1.2, leak=0.60)
    uq = ensemble_uq([best, fake_result(ref=1.1, leak=0.58), fake_result(ref=1.3, leak=0.62)])
    ev = build_evidence(best, card(has_ref=True), uq)
    assert ev.decision == "reject"
    assert "leakage_high" in ev.flags


def test_accept_with_reference_and_uq():
    vals = (1.8, 2.1, 2.2)
    results = [fake_result(ref=v) for v in vals]
    best = results[1]
    uq = ensemble_uq(results)
    ev = build_evidence(best, card(has_ref=True), uq)
    assert ev.decision == "accept"
    assert ev.reference_gain_db == 2.1
    assert ev.confidence >= 0.7


def test_abstain_on_ood():
    best = fake_result(ref=1.2)
    uq = ensemble_uq([best])
    ev = build_evidence(best, card(has_ref=True), uq, ood_score=0.9)
    assert ev.decision == "abstain"
    assert "ood_high" in ev.flags


def test_trace_bootstrap_uq_prefers_interval():
    best = fake_result(ref=2.0)
    best.metrics.extra["reference_gain_bootstrap"] = {
        "low_db": 1.5, "high_db": 2.5, "std_db": 0.3,
        "n_boot": 400, "n_traces": 100, "method": "trace_bootstrap",
    }
    uq = ensemble_uq([best], best=best)
    assert uq.available is True
    assert uq.method == "trace_bootstrap"
    assert uq.gain_ci_low == 1.5 and uq.gain_ci_high == 2.5
    assert uq.reliable is True

    weak = fake_result(ref=2.0)
    weak.metrics.extra["reference_gain_bootstrap"] = {
        "low_db": 0.2, "high_db": 2.5, "std_db": 0.9,
        "n_boot": 400, "n_traces": 100, "method": "trace_bootstrap",
    }
    ev = build_evidence(weak, card(has_ref=True), ensemble_uq([weak], best=weak))
    assert ev.decision == "conditional"
    assert "UQ 下界不足" in " ".join(ev.reasons)


def test_reference_close_detects_identity():
    rng = np.random.default_rng(0)
    a = rng.normal(size=(100, 20)).astype(np.float32)
    assert _reference_close(a, a) is True
    assert _reference_close(a, a + 0.05) is False


if __name__ == "__main__":
    total = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            total += 1
            print(f"[PASS] {name}")
    print(f"\n{total}/{total} 个测试通过")
