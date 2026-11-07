"""去噪方法 / 规划器 / 执行器测试。

pytest 未安装时可直接运行::

    python tests/test_methods.py

真实数据冒烟测试默认跳过，需要时设置环境变量::

    $env:SEISMIC_REAL_DATA = "1"; python tests/test_methods.py
"""
from __future__ import annotations

import math
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from engine.methods import classical, cnn, planner, runner  # noqa: E402
from engine.model import MethodResult, MethodSpec, QualityMetrics, SourceAssessment  # noqa: E402
from engine.quality.gate import industry_gate  # noqa: E402
from engine.quality.metrics import evaluate  # noqa: E402

_REAL_ENV = os.environ.get("SEISMIC_REAL_DATA_FILE")
REAL_DATA: Path | None = Path(_REAL_ENV) if _REAL_ENV else None

ALL_METHODS = {
    "bandpass", "notch", "despike", "median", "fk",
    "fx_decon", "svd", "spectral_sub", "wavelet",
}


# --------------------------------------------------------------------------
# 合成数据：线性同相轴 + 高斯噪声 + 50 Hz 工频 + 低速面波
# --------------------------------------------------------------------------
def ricker(t: np.ndarray, f: float = 22.0) -> np.ndarray:
    """Ricker 子波。"""
    a = (np.pi * f * t) ** 2
    return (1.0 - 2.0 * a) * np.exp(-a)


def synthetic_section(ns: int = 768, ntr: int = 48, dt: float = 0.002,
                      dz: float = 5.0, seed: int = 7, noise: float = 0.8,
                      line: float = 0.5, sw_amp: float = 2.0,
                      vel: float = 2500.0) -> tuple[np.ndarray, np.ndarray]:
    """生成含有效波/面波/工频/随机噪声的合成剖面。

    返回 (noisy, clean)；clean 只含 4 条线性同相轴，可作为参考。
    同相轴视速度 vel（米/秒），按道间距 dz 计算每道延迟。
    """
    rng = np.random.default_rng(seed)
    t = np.arange(ns, dtype=np.float64) * dt
    x = np.arange(ntr, dtype=np.float64) * dz

    clean = np.zeros((ns, ntr), dtype=np.float64)
    for t0, amp in ((0.22, 1.0), (0.42, 0.8), (0.62, 0.6), (0.82, 0.45)):
        for j in range(ntr):
            clean[:, j] += amp * ricker(t - (t0 + x[j] / vel))

    # 低速面波（300 m/s，14 Hz）
    sw = np.zeros_like(clean)
    for j in range(ntr):
        tt = t - x[j] / 300.0
        env = np.exp(-((tt - 1.0) / 0.35) ** 2)
        sw[:, j] = sw_amp * env * np.sin(2.0 * np.pi * 14.0 * tt) * (tt > 0.0)

    # 50 Hz 工频（道间幅度略有差异）
    line_wave = (line * np.sin(2.0 * np.pi * 50.0 * t)[:, None]
                 * (1.0 + 0.1 * rng.standard_normal((1, ntr))))

    noisy = clean + sw + line_wave + noise * rng.standard_normal((ns, ntr))
    return noisy.astype(np.float32), clean.astype(np.float32)


def _tail_energy(a: np.ndarray) -> float:
    return float(np.mean(np.asarray(a, dtype=np.float64) ** 2))


# --------------------------------------------------------------------------
# classical 方法
# --------------------------------------------------------------------------
def test_available_methods_structure():
    info = classical.available_methods()
    assert set(info) == ALL_METHODS, f"方法清单不符: {sorted(info)}"
    for key, meta in info.items():
        assert meta["key"] == key
        for f in ("name", "description", "targets", "params", "params_schema"):
            assert f in meta, f"{key} 缺少字段 {f}"
        assert isinstance(meta["params"], dict) and meta["params"], f"{key} 默认参数为空"
        assert isinstance(meta["params_schema"], dict) and meta["params_schema"]
        assert isinstance(meta["targets"], list) and meta["targets"]
        for pkey, schema in meta["params_schema"].items():
            assert pkey in meta["params"], f"{key}.{pkey} 不在默认参数中"
            assert "default" in schema and "type" in schema


def test_run_all_methods_shape_finite():
    data, _clean = synthetic_section(ns=640, ntr=40)
    dt, dz = 0.002, 5.0
    changed = 0
    for key in sorted(ALL_METHODS):
        out, info = classical.run_method(key, data, dt, dz)
        assert isinstance(out, np.ndarray)
        assert out.shape == data.shape, f"{key} 形状改变: {out.shape} != {data.shape}"
        assert out.dtype == np.float32, f"{key} dtype={out.dtype}"
        assert np.isfinite(out).all(), f"{key} 输出含 NaN/Inf"
        for f in ("notes", "params_used", "warnings"):
            assert f in info, f"{key} info 缺少 {f}"
        assert isinstance(info["params_used"], dict)
        assert np.array_equal(out, data) is False or key == "despike"
        if not np.array_equal(out, data):
            changed += 1
    assert changed >= 6, f"只有 {changed} 个方法改变了数据，疑似实现退化"


def test_run_method_invalid_inputs():
    data, _ = synthetic_section(ns=256, ntr=12)
    for bad_dt in (0.0, -0.001, float("nan"), None):
        try:
            classical.run_method("bandpass", data, bad_dt, 5.0)
        except ValueError:
            pass
        else:  # pragma: no cover
            raise AssertionError(f"dt={bad_dt!r} 应抛 ValueError")
    try:
        classical.run_method("不存在的方法", data, 0.002, 5.0)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("未知方法应抛 ValueError")
    # 非二维输入
    try:
        classical.run_method("bandpass", np.zeros(100), 0.002, 5.0)
    except (ValueError, TypeError):
        pass
    else:  # pragma: no cover
        raise AssertionError("一维输入应报错")


def test_fk_without_dz_warns():
    data, _ = synthetic_section(ns=512, ntr=32)
    out, info = classical.run_method("fk", data, 0.002, None)
    assert out.shape == data.shape
    assert info["warnings"], "缺少 dz 时 fk 应给出警告"


def test_notch_suppresses_50hz():
    ns, ntr, dt = 2048, 24, 0.002
    t = np.arange(ns) * dt
    rng = np.random.default_rng(3)
    data = (2.0 * np.sin(2.0 * np.pi * 50.0 * t)[:, None]
            + 0.02 * rng.standard_normal((ns, ntr))).astype(np.float32)

    def amp50(a: np.ndarray) -> float:
        spec = np.abs(np.fft.rfft(a, axis=0)).mean(axis=1)
        freqs = np.fft.rfftfreq(ns, dt)
        return float(spec[np.argmin(np.abs(freqs - 50.0))])

    out, _ = classical.run_method("notch", data, dt, 5.0, {"harmonics": 1})
    assert amp50(out) < 0.2 * amp50(data), "陷波后 50 Hz 幅度没有明显下降"


def test_fk_keeps_fast_event_rejects_slow():
    """FK 应保留高速同相轴、压制低速面波。"""
    ns, ntr, dt, dz = 768, 64, 0.002, 5.0
    t = np.arange(ns) * dt
    x = np.arange(ntr) * dz

    def event(v: float) -> np.ndarray:
        a = np.zeros((ns, ntr))
        for j in range(ntr):
            a[:, j] = ricker(t - 0.3 - x[j] / v)
        return a.astype(np.float32)

    fast, slow = event(2500.0), event(300.0)
    out_fast, _ = classical.run_method("fk", fast, dt, dz, {"vmin": 1200, "taper": 300})
    out_slow, _ = classical.run_method("fk", slow, dt, dz, {"vmin": 1200, "taper": 300})
    keep_fast = _tail_energy(out_fast) / max(_tail_energy(fast), 1e-12)
    keep_slow = _tail_energy(out_slow) / max(_tail_energy(slow), 1e-12)
    assert keep_fast > 0.6, f"高速同相轴保留率过低: {keep_fast:.3f}"
    assert keep_slow < 0.3, f"低速同相轴压制不足: {keep_slow:.3f}"
    assert keep_fast > keep_slow + 0.2


def test_despike_suppresses_spikes():
    data, _ = synthetic_section(ns=512, ntr=24, noise=0.05, line=0.0, sw_amp=0.0)
    data = data.copy()
    data[100, 5] = 50.0
    data[300, 12] = -40.0
    out, info = classical.run_method("despike", data, 0.002, 5.0, {"threshold": 6.0})
    assert abs(float(out[100, 5])) < 5.0
    assert abs(float(out[300, 12])) < 5.0
    assert np.isfinite(out).all()
    assert info["notes"]


# --------------------------------------------------------------------------
# CNN
# --------------------------------------------------------------------------
def test_cnn_smoke_cpu():
    if not cnn.cnn_available():
        print("  [skip] CNN 权重/环境不可用")
        return
    data, _ = synthetic_section(ns=512, ntr=32, seed=11)
    t0 = time.perf_counter()
    out, info = cnn.run_cnn(data, 0.002, 5.0, {
        "device": "cpu", "max_traces": 32, "batch_sep": 4, "batch_cls": 8,
    })
    seconds = time.perf_counter() - t0
    assert out.shape == data.shape
    assert out.dtype == np.float32
    assert np.isfinite(out).all()
    for f in ("class_probs", "gating_weights", "alpha", "device", "energy_ratio"):
        assert f in info, f"cnn info 缺少 {f}"
    assert isinstance(info["class_probs"], dict) and len(info["class_probs"]) == 5
    assert isinstance(info["gating_weights"], dict) and set(info["gating_weights"]) == set(info["class_probs"])
    assert 0.0 <= float(info["alpha"]) <= 1.0
    assert info["device"] in ("cpu", "cuda")
    assert math.isfinite(float(info["energy_ratio"]))
    print(f"  CNN CPU 512x32 耗时 {seconds:.2f}s，alpha={info['alpha']}，device={info['device']}")


# --------------------------------------------------------------------------
# 规划器 / 执行器
# --------------------------------------------------------------------------
def test_plan_by_noise_type():
    assessment = SourceAssessment(
        noise_types={"面波": 0.8, "随机噪声": 0.5}, needs_denoise=True, confidence=0.7,
    )
    specs = planner.plan(assessment, {})
    assert 1 <= len(specs) <= 4
    keys = [s.key for s in specs]
    assert keys[0] in {"fk", "bandpass", "median"}, f"面波主噪声首选异常: {keys}"
    assert set(keys) <= ALL_METHODS | {"cnn"}
    for s in specs:
        assert isinstance(s.params, dict) and s.params
        assert s.rationale
        assert isinstance(s.targets, list)
        assert s.source == "rule"

    specs = planner.plan(SourceAssessment(noise_types={"工频干扰": 0.9}), {})
    assert specs and specs[0].key == "notch", f"工频主噪声应首选 notch: {[s.key for s in specs]}"

    specs = planner.plan(SourceAssessment(noise_types={"面波": 0.9}, needs_denoise=True),
                         {"allow_cnn": False})
    assert "cnn" not in [s.key for s in specs]

    assert planner.plan(SourceAssessment(needs_denoise=False), {}) == []


def test_plan_unknown_noise_falls_back():
    specs = planner.plan(SourceAssessment(noise_types={}), {})
    keys = [s.key for s in specs]
    assert keys[:2] == ["bandpass", "cnn"], f"未知噪声兜底方案异常: {keys}"


def test_refine_changes_params_after_failure():
    assessment = SourceAssessment(noise_types={"面波": 0.9}, needs_denoise=True)
    fk_params = dict(classical.available_methods()["fk"]["params"])
    spec = MethodSpec(key="fk", name="FK", params=fk_params)
    metrics = QualityMetrics(
        snr_gain_db=5.0, energy_ratio=0.9, signal_leakage=0.7,
        spectrum_corr=0.98, clip_ratio=0.0, estimator="test",
    )
    gate = industry_gate(metrics, None, {})
    assert gate.passed is False
    history = [MethodResult(spec=spec, metrics=metrics, gate=gate)]

    specs = planner.refine(assessment, history, {})
    assert 1 <= len(specs) <= 4
    sources = {s.source for s in specs}
    assert "refine" in sources, f"失败方法应给出调参重试方案: {sources}"
    assert sources <= {"rule", "refine"}
    assert all(isinstance(s.params, dict) and s.params for s in specs)
    # 不应按原参数重试同一方法
    assert not any(s.key == "fk" and s.params == fk_params for s in specs)

    # 尝试次数达到上限后不再给方案
    many = history * 6
    assert planner.refine(assessment, many, {"max_attempts": 6}) == []


def test_refine_skips_passed_methods():
    from engine.model import GateResult

    assessment = SourceAssessment(noise_types={"面波": 0.9}, needs_denoise=True)
    spec = MethodSpec(key="fk", name="FK", params=dict(classical.available_methods()["fk"]["params"]))
    passed = MethodResult(spec=spec, metrics=QualityMetrics(snr_gain_db=4.0),
                          gate=GateResult(passed=True, score=80.0, confidence=0.8))
    specs = planner.refine(assessment, [passed], {"max_attempts": 6})
    assert all(s.key != "fk" for s in specs), "已通过的方法不应再被重试"


def test_runner_attempts_and_pick_best():
    data, clean = synthetic_section(ns=640, ntr=40)
    ds = SimpleNamespace(data=data, dt=0.002, dz=5.0)
    specs = [
        MethodSpec(key="bandpass", name="带通", params={}),
        MethodSpec(key="not_a_method", name="坏方法", params={}),
        MethodSpec(key="notch", name="陷波", params={"freq": 50.0}),
    ]
    results = runner.run_attempts(ds, specs, cfg={}, reference=clean, log=None)
    assert len(results) == 3
    ok = results[0]
    assert isinstance(ok.data, np.ndarray) and ok.data.shape == data.shape
    assert ok.metrics is not None and ok.gate is not None
    assert 0.0 <= ok.gate.confidence <= 1.0
    assert ok.metrics.reference_snr_in_db is not None, "传了 reference 应产生参考指标"

    bad = results[1]
    assert bad.gate.passed is False
    assert bad.gate.failures
    assert bad.data is None

    best = runner.pick_best(results)
    assert best is not None and best in results
    passed = [r for r in results if r.gate.passed]
    if passed:
        assert best.gate.passed is True

    assert runner.run_attempts(ds, [], cfg={}, log=None) == []

    # dt 非法直接抛错
    try:
        runner.run_attempts(SimpleNamespace(data=data, dt=0.0, dz=5.0), specs, log=None)
    except ValueError:
        pass
    else:  # pragma: no cover
        raise AssertionError("dt=0 应抛 ValueError")


# --------------------------------------------------------------------------
# 真实数据冒烟（可选）
# --------------------------------------------------------------------------
def test_real_data_smoke():
    if os.environ.get("SEISMIC_REAL_DATA") != "1":
        print("  [skip] 设 SEISMIC_REAL_DATA=1 后运行真实数据测试")
        return
    if REAL_DATA is None or not REAL_DATA.exists():
        print(f"  [skip] 找不到真实数据 {REAL_DATA or '未设置 SEISMIC_REAL_DATA_FILE'}")
        return
    from engine.assets.cnnv2.vsdn2.sgy import read_segy

    data, meta = read_segy(str(REAL_DATA))
    dt = float(meta.dt_us) / 1e6
    dz = 2.05
    sub = np.asarray(data[:, ::4], dtype=np.float32)  # 抽道，控制测试耗时
    print(f"  真实数据 {data.shape} -> 测试子集 {sub.shape}，dt={dt}s")
    for key in ("bandpass", "fk"):
        t0 = time.perf_counter()
        out, _info = classical.run_method(key, sub, dt, dz)
        seconds = time.perf_counter() - t0
        metrics = evaluate(sub, out, dt, dz)
        gate = industry_gate(metrics, None, {})
        print(f"  {key:10s} {seconds:6.2f}s gain={metrics.snr_gain_db:+.2f}dB "
              f"er={metrics.energy_ratio:.3f} leak={metrics.signal_leakage:.3f} "
              f"sc={metrics.spectrum_corr:.3f} pass={gate.passed} score={gate.score:.0f}")
        assert out.shape == sub.shape and np.isfinite(out).all()

    if cnn.cnn_available():
        t0 = time.perf_counter()
        out, info = cnn.run_cnn(sub, dt, dz, {"device": "auto"})
        seconds = time.perf_counter() - t0
        metrics = evaluate(sub, out, dt, dz)
        gate = industry_gate(metrics, None, {})
        probs = info["class_probs"]
        print(f"  cnn        {seconds:6.2f}s gain={metrics.snr_gain_db:+.2f}dB "
              f"er={metrics.energy_ratio:.3f} pass={gate.passed}")
        print(f"  类别概率: " + ", ".join(f"{k}={v:.2f}" for k, v in probs.items()))
        assert out.shape == sub.shape and np.isfinite(out).all()


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
