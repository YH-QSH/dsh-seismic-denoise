"""P1 的异构集成不确定性：零训练，先用多方法/多组合的离散度。"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass
class UQReport:
    available: bool
    method: str
    n: int
    gain_mean_db: float | None = None
    gain_std_db: float | None = None
    gain_ci_low: float | None = None
    gain_ci_high: float | None = None
    weak_mean: float | None = None
    weak_std: float | None = None
    reliable: bool = False
    source: str = "none"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _get_gain(metrics) -> tuple[float | None, str]:
    if metrics is None:
        return None, "none"
    ref = getattr(metrics, "reference_gain_db", None)
    if ref is not None and np.isfinite(float(ref)):
        return float(ref), "reference"
    extra = getattr(metrics, "extra", None) or {}
    if extra.get("reliable", False):
        blind = getattr(metrics, "snr_gain_db", None)
        if blind is not None and np.isfinite(float(blind)):
            return float(blind), "blind_reliable"
    return None, "none"


def _get_weak(metrics) -> float | None:
    if metrics is None:
        return None
    extra = getattr(metrics, "extra", None) or {}
    visual = extra.get("visual") or {}
    weak = visual.get("weak_signal_ratio")
    if weak is None:
        return None
    try:
        value = float(weak)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _bootstrap_from_best(best) -> UQReport | None:
    """优先使用最佳候选的“按道 bootstrap”参考增益区间。

    这比“多方法离散度”更贴近真实验收：它回答的是“换一批道，这个
    去噪增益还稳不稳”。只有最佳候选有参考增益且门禁通过时才使用。
    """
    if best is None:
        return None
    gate = getattr(best, "gate", None)
    if gate is None or not bool(getattr(gate, "passed", False)):
        return None
    metrics = getattr(best, "metrics", None)
    extra = getattr(metrics, "extra", None) or {}
    boot = extra.get("reference_gain_bootstrap") or {}
    gain, source = _get_gain(metrics)
    if gain is None or source != "reference":
        return None
    try:
        low = float(boot["low_db"])
        high = float(boot["high_db"])
        std = float(boot["std_db"])
        n_traces = int(boot["n_traces"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (np.isfinite(low) and np.isfinite(high) and np.isfinite(std)):
        return None
    width = max(0.0, high - low)
    reliable = bool(n_traces >= 20 and width <= 2.0)
    notes = [f"按道 bootstrap {int(boot.get('n_boot', 0))} 次，5%–95% 区间宽 {width:.2f} dB"]
    if not reliable:
        notes.append("区间宽度大于 2dB，增益稳定性不足")
    return UQReport(
        available=True,
        method="trace_bootstrap",
        n=n_traces,
        gain_mean_db=float(gain),
        gain_std_db=std,
        gain_ci_low=low,
        gain_ci_high=high,
        reliable=reliable,
        source="reference_trace_bootstrap",
        notes=notes,
    )


def ensemble_uq(results, confidence: float = 0.90, only_passed: bool = True,
                best=None) -> UQReport:
    """对候选结果做异构集成；有按道 bootstrap 时优先使用。"""
    boot = _bootstrap_from_best(best)
    if boot is not None:
        return boot

    all_results = list(results or [])
    selected = all_results
    filter_note: list[str] = []
    if only_passed:
        passed = [r for r in all_results if bool(getattr(getattr(r, "gate", None), "passed", False))]
        if passed:
            selected = passed
            filter_note.append(f"仅统计门禁通过的 {len(passed)} 个候选")
        elif all_results:
            filter_note.append("没有门禁通过的候选，退化为全部候选")

    gains: list[float] = []
    weaks: list[float] = []
    sources: set[str] = set()
    notes: list[str] = list(filter_note)
    for result in selected:
        metrics = getattr(result, "metrics", None)
        gain, source = _get_gain(metrics)
        if gain is not None:
            gains.append(gain)
            sources.add(source)
        weak = _get_weak(metrics)
        if weak is not None:
            weaks.append(weak)

    if len(gains) < 2:
        return UQReport(
            available=False,
            method="heterogeneous_ensemble",
            n=len(gains),
            weak_mean=float(np.mean(weaks)) if weaks else None,
            weak_std=float(np.std(weaks, ddof=1)) if len(weaks) > 1 else None,
            reliable=False,
            source="+".join(sorted(sources)) if sources else "none",
            notes=notes + ["可信增益少于 2 个，无法估计离散度"],
        )

    arr = np.asarray(gains, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    z = 1.645 if abs(confidence - 0.90) < 1e-9 else 1.96
    half = z * std / np.sqrt(len(arr))
    reliable = bool(len(arr) >= 3 and std <= 2.0)
    if len(arr) < 3:
        notes.append("样本少于 3 个，区间仅作参考")
    if std > 2.0:
        notes.append("候选离散度大于 2dB，结果不稳定")
    if "reference" not in sources:
        notes.append("没有参考增益，集成只能依赖盲指标")

    return UQReport(
        available=True,
        method="heterogeneous_ensemble",
        n=len(arr),
        gain_mean_db=mean,
        gain_std_db=std,
        gain_ci_low=mean - half,
        gain_ci_high=mean + half,
        weak_mean=float(np.mean(weaks)) if weaks else None,
        weak_std=float(np.std(weaks, ddof=1)) if len(weaks) > 1 else None,
        reliable=reliable,
        source="+".join(sorted(sources)),
        notes=notes,
    )
