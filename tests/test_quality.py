"""质量评价指标 / 行业门禁测试。

pytest 未安装时可直接运行::

    python tests/test_quality.py
"""
from __future__ import annotations

import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.methods.classical import run_method  # noqa: E402
from engine.model import SourceAssessment  # noqa: E402
from engine.quality.fidelity import enhance_fidelity  # noqa: E402
from engine.quality.gate import DEFAULT_THRESHOLDS, gate_confidence, industry_gate  # noqa: E402
from engine.quality.metrics import blind_snr, evaluate, signal_band  # noqa: E402


# --------------------------------------------------------------------------
# 辅助数据
# --------------------------------------------------------------------------
def ricker(t: np.ndarray, f: float = 22.0) -> np.ndarray:
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def make_events(ns: int = 1024, ntr: int = 60, dt: float = 0.002,
                dz: float = 5.0, vel: float = 2500.0) -> np.ndarray:
    """只有 4 条线性同相轴的干净剖面。"""
    t = np.arange(ns, dtype=np.float64) * dt
    x = np.arange(ntr, dtype=np.float64) * dz
    clean = np.zeros((ns, ntr))
    for t0, amp in ((0.22, 1.0), (0.42, 0.8), (0.62, 0.6), (0.82, 0.45)):
        for j in range(ntr):
            clean[:, j] += amp * ricker(t - (t0 + x[j] / vel))
    return clean.astype(np.float32)


def make_noisy(seed: int = 1, noise: float = 0.8, line: float = 0.5,
               sw_amp: float = 2.0, ns: int = 1024, ntr: int = 60,
               dt: float = 0.002, dz: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """含面波、工频、随机噪声的强噪声剖面，返回 (noisy, clean)。"""
    rng = np.random.default_rng(seed)
    clean = make_events(ns=ns, ntr=ntr, dt=dt, dz=dz)
    t = np.arange(ns, dtype=np.float64) * dt
    x = np.arange(ntr, dtype=np.float64) * dz
    sw = np.zeros_like(clean, dtype=np.float64)
    for j in range(ntr):
        tt = t - x[j] / 300.0
        env = np.exp(-((tt - 1.0) / 0.35) ** 2)
        sw[:, j] = sw_amp * env * np.sin(2.0 * np.pi * 14.0 * tt) * (tt > 0.0)
    line_wave = (line * np.sin(2.0 * np.pi * 50.0 * t)[:, None]
                 * (1.0 + 0.1 * rng.standard_normal((1, ntr))))
    noisy = clean + sw + line_wave + noise * rng.standard_normal(clean.shape)
    return noisy.astype(np.float32), clean


# --------------------------------------------------------------------------
# blind_snr
# --------------------------------------------------------------------------
def test_blind_snr_white_noise():
    rng = np.random.default_rng(2)
    data = (0.5 * rng.standard_normal((2048, 32))).astype(np.float32)
    res = blind_snr(data, 0.002)
    for f in ("snr_db", "signal_db", "noise_db", "reliable", "method"):
        assert f in res
    assert math.isfinite(res["snr_db"])
    # 纯白噪声没有可检测信号：实现把 p_sig 卡在极小值，给出 -90 dB 地板；
    # 关键是不能把纯噪声误报成高信噪比。
    assert -120.0 <= res["snr_db"] <= 1.0, f"白噪声盲 SNR 异常: {res['snr_db']:.2f}"
    assert isinstance(res["reliable"], bool)
    assert res["method"]


def test_blind_snr_recovers_positive_snr():
    clean = make_events(ns=1024, ntr=40)
    rng = np.random.default_rng(5)
    noise = rng.standard_normal(clean.shape)
    target_db = 8.0
    scale = math.sqrt(float(np.mean(clean ** 2)) / (10.0 ** (target_db / 10.0))
                      / float(np.mean(noise ** 2)))
    noisy = (clean + scale * noise).astype(np.float32)
    res = blind_snr(noisy, 0.002)
    assert math.isfinite(res["snr_db"])
    assert res["snr_db"] > 0.0, f"含有效波数据盲 SNR 应大于 0: {res['snr_db']:.2f}"
    assert res["snr_db"] < 25.0
    assert res["signal_db"] > res["noise_db"]


def test_blind_snr_handles_zero_data():
    res = blind_snr(np.zeros((512, 8), dtype=np.float32), 0.002)
    assert res["reliable"] is False
    assert res["signal_db"] <= -100.0
    assert res["noise_db"] <= -100.0
    assert math.isfinite(res["snr_db"])


def test_signal_band():
    lo, hi = signal_band(0.001)
    assert lo < hi and lo >= 4.0 and hi <= 90.0
    lo2, hi2 = signal_band(0.004)
    assert lo2 < hi2
    assert hi2 <= 0.5 * (1.0 / 0.004) + 1e-9


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------
def test_evaluate_identity():
    noisy, _ = make_noisy(seed=3)
    metrics = evaluate(noisy, noisy.copy(), 0.002, 5.0)
    assert abs(metrics.snr_gain_db) <= 1e-6
    assert abs(metrics.energy_ratio - 1.0) <= 1e-4
    assert metrics.spectrum_corr >= 0.999
    assert metrics.clip_ratio <= 1e-6
    assert metrics.signal_leakage is None or abs(metrics.signal_leakage) < 0.05
    assert metrics.local_snr_gain_db is not None
    assert abs(metrics.local_snr_gain_db) <= 1.0
    assert metrics.estimator


def test_evaluate_bandpass_and_reference():
    noisy, clean = make_noisy(seed=3)
    out, _ = run_method("bandpass", noisy, 0.002, 5.0)
    metrics = evaluate(noisy, out, 0.002, 5.0, reference=clean)
    assert 0.5 <= metrics.energy_ratio <= 1.05
    assert metrics.spectrum_corr >= 0.9
    assert math.isfinite(metrics.snr_gain_db)
    assert metrics.signal_leakage is not None and 0.0 <= metrics.signal_leakage <= 1.0
    assert metrics.local_snr_gain_db is not None
    assert metrics.reference_snr_in_db is not None
    assert metrics.reference_snr_out_db is not None
    assert abs(metrics.reference_gain_db
               - (metrics.reference_snr_out_db - metrics.reference_snr_in_db)) < 1e-6
    # 无参考时不应产生参考字段
    m2 = evaluate(noisy, out, 0.002, 5.0)
    assert m2.reference_snr_in_db is None and m2.reference_gain_db is None


def test_evaluate_bad_input():
    try:
        evaluate(np.zeros((100, 4), np.float32), np.zeros((100, 3), np.float32), 0.002)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("形状不一致应抛 ValueError")
    try:
        evaluate(np.zeros((100, 4), np.float32), np.zeros((100, 4), np.float32), 0.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("dt=0 应抛 ValueError")


# --------------------------------------------------------------------------
# 行业门禁
# --------------------------------------------------------------------------
def test_gate_identity_fails():
    noisy, _ = make_noisy(seed=4)
    assessment = SourceAssessment(noise_types={"随机噪声": 0.6}, needs_denoise=True, confidence=0.6)
    metrics = evaluate(noisy, noisy.copy(), 0.002, 5.0)
    gate = industry_gate(metrics, assessment, {})
    assert gate.passed is False, "原样返回必须判失败"
    assert gate.failures
    text = " ".join(gate.failures)
    assert ("原样" in text) or ("信噪比增益" in text), f"失败原因不明确: {text}"
    assert 0.0 <= gate.score < 60.0
    assert 0.0 <= gate.confidence <= 1.0
    assert gate.advice


def test_gate_passes_denoised_strong_noise():
    noisy, _ = make_noisy(seed=6)
    assessment = SourceAssessment(noise_types={"随机噪声": 0.6, "工频干扰": 0.3},
                                  needs_denoise=True, confidence=0.6)
    passed = []
    for key in ("bandpass", "median", "svd"):
        out, _ = run_method(key, noisy, 0.002, 5.0)
        out, _fid = enhance_fidelity(noisy, out, 0.002)
        metrics = evaluate(noisy, out, 0.002, 5.0)
        gate = industry_gate(metrics, assessment, {})
        # runner 会用 gate_confidence 回填 confidence
        gate.confidence = gate_confidence(metrics, gate, assessment)
        assert 0.0 <= gate.score <= 100.0
        assert 0.0 <= gate.confidence <= 1.0
        if gate.passed:
            passed.append((key, gate, metrics))
    assert passed, "强噪声下至少应有一个方法通过门禁（bandpass/median/svd）"
    key, gate, metrics = passed[0]
    assert gate.passed is True and not gate.failures
    if metrics.extra.get("reliable", True):
        assert metrics.snr_gain_db >= DEFAULT_THRESHOLDS["min_snr_gain_db"]
    else:
        # 盲 SNR 不可靠时增益是软约束，改由能量/频谱等保真指标把关。
        assert float(metrics.energy_ratio) >= 0.70
        assert float(metrics.spectrum_corr) >= 0.80
    print(f"  通过门禁的方法: {[p[0] for p in passed]}，"
          f"最佳 score={max(p[1].score for p in passed):.0f}")


def test_gate_clean_data_skipped():
    noisy, _ = make_noisy(seed=7, noise=0.01, line=0.0, sw_amp=0.0)
    assessment = SourceAssessment(needs_denoise=False, confidence=0.9,
                                  clean_reasons=["信噪比高"])
    metrics = evaluate(noisy, noisy.copy(), 0.002, 5.0)
    gate = industry_gate(metrics, assessment, {})
    assert gate.passed is True
    assert len(gate.checks) == 1
    assert gate.score >= 80.0
    assert 0.0 <= gate.confidence <= 1.0


def test_gate_threshold_aliases_and_override():
    noisy, _ = make_noisy(seed=8)
    out, _ = run_method("bandpass", noisy, 0.002, 5.0)
    metrics = evaluate(noisy, out, 0.002, 5.0)
    assessment = SourceAssessment(needs_denoise=True, confidence=0.5)

    # 模拟“盲 SNR 可靠”的场景，验证 min_gain_db 别名仍是硬阈值。
    metrics.extra["reliable"] = True
    strict = industry_gate(metrics, assessment, {"gate": {"min_gain_db": 50.0}})
    assert strict.passed is False, "pipeline 别名 min_gain_db 应覆盖阈值"
    assert any("信噪比增益" in f for f in strict.failures)

    strict2 = industry_gate(metrics, assessment, {"gate": {"energy_ratio_max": 0.05}})
    assert strict2.passed is False, "energy_ratio_max 别名应生效"


def test_gate_soft_snr_out():
    """盲估计不可靠时输出 SNR 只做软检查（不应因此直接判失败）。"""
    noisy, _ = make_noisy(seed=9)
    out, _ = run_method("bandpass", noisy, 0.002, 5.0)
    metrics = evaluate(noisy, out, 0.002, 5.0)
    metrics.extra["reliable"] = False
    metrics.snr_out_db = 1.0  # 明显低于 10 dB
    gate = industry_gate(metrics, SourceAssessment(needs_denoise=True), {})
    snr_checks = [c for c in gate.checks if "输出信噪比" in c.name]
    assert snr_checks, "应包含输出信噪比检查"
    assert snr_checks[0].passed is True
    assert DEFAULT_THRESHOLDS["min_snr_out_db"] == 10.0


def test_gate_rejects_single_trace_spike():
    """单道尖峰/垂直条带必须在门禁里被拦下。"""
    noisy, _ = make_noisy(seed=11)
    out = noisy.copy()
    out[500:520, 30] += 2.0e4  # 单道短时强尖峰
    metrics = evaluate(noisy, out, 0.002, 5.0)
    assert metrics.extra.get("trace_resid_ratio_out") is not None
    gate = industry_gate(metrics, SourceAssessment(needs_denoise=True), {})
    checks = [c for c in gate.checks if "单道尖峰" in c.name]
    assert checks, "应包含单道尖峰/垂直条带检查"
    assert checks[0].passed is False
    assert any("单道尖峰" in f for f in gate.failures)


# --------------------------------------------------------------------------
# 无 pytest 时的运行器
# --------------------------------------------------------------------------
def main() -> int:
    tests = [(name, obj) for name, obj in sorted(globals().items())
             if name.startswith("test_") and callable(obj)]
    failed = 0
    for name, fn in tests:
        t0 = time.perf_counter()
        try:
            fn()
            print(f"[PASS] {name}  ({time.perf_counter() - t0:.2f}s)")
        except Exception:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {name}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} 个测试通过")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
