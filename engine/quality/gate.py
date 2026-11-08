"""行业门禁：判断一次去噪尝试是否"能用"。

门禁分两类检查：
- **硬检查**：不过就判失败（信噪比增益、能量保持、信号泄漏、频谱一致性、削波）。
- **软检查**：只在盲信噪比不可靠时降级（输出信噪比绝对值），不过只扣分不判失败。

"原样返回"（输出≈输入）必须判失败——去噪的底线是至少不劣化，而原样返回没有产生价值。
数据本身被评估为干净（``needs_denoise=False``）时走单独分支：此时原样保留是正确的，
直接通过，但分数与置信度都标明"无需去噪"。
"""
from __future__ import annotations

import math

from ..model import GateCheck, GateResult, QualityMetrics

# 默认阈值（可被 cfg["gate"] 覆盖）
DEFAULT_THRESHOLDS: dict[str, float] = {
    "min_snr_gain_db": 3.0,     # 硬：输出相对输入的盲 SNR 增益
    "energy_min": 0.60,         # 硬：信号带内能量保持下限（高保真，防削光）
    "energy_max": 1.05,         # 硬：信号带内能量保持上限（防放大）
    "max_signal_leakage": 0.40,  # 硬：被去除分量与输出局部相关的上限
    "max_signal_leakage_ref": 0.45,  # 有纯净参考时的泄漏上限（短/高频数据更宽容）
    "min_reference_gain_db": 0.5,    # 有纯净参考时的真实 SNR 增益下限
    "min_spectrum_corr": 0.80,   # 硬：信号带振幅谱相关下限
    "max_clip_ratio": 1e-3,     # 硬：额外削波样点比例上限
    "min_snr_out_db": 10.0,     # 软（盲估计不可靠时）：输出绝对 SNR
    "min_weak_signal_ratio": 0.85,  # 硬：输入弱能量区的输出能量保留（高保真）
    "max_trace_resid_ratio": 250.0,  # 硬：横向中值残差 / 中位残差（单道尖峰/垂直条带）
    "identity_gain_db": 0.2,    # 判"原样返回"的增益阈值
    "identity_energy": 0.03,    # 判"原样返回"的能量比偏离阈值
}

# pipeline.DEFAULT_CFG / 命令行可能使用另一套键名，统一映射到内部标准键
_ALIASES: dict[str, str] = {
    "min_gain_db": "min_snr_gain_db",
    "energy_ratio_min": "energy_min",
    "energy_ratio_max": "energy_max",
    "signal_leakage_max": "max_signal_leakage",
    "leakage_max_ref": "max_signal_leakage_ref",
    "min_reference_gain_db": "min_reference_gain_db",
    "spectrum_corr_min": "min_spectrum_corr",
    "clip_ratio_max": "max_clip_ratio",
    "snr_out_min_db": "min_snr_out_db",
    "weak_signal_ratio_min": "min_weak_signal_ratio",
}

# 评分权重（缺项自动归一化）
_SCORE_WEIGHTS = {
    "gain": 0.30,
    "energy": 0.16,
    "leakage": 0.18,
    "spectrum": 0.12,
    "clip": 0.05,
    "snr_out": 0.09,
    "visual": 0.10,
}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def _thresholds(cfg) -> dict[str, float]:
    th = dict(DEFAULT_THRESHOLDS)
    if not cfg:
        return th
    gate_cfg = None
    if isinstance(cfg, dict):
        gate_cfg = cfg.get("gate")
        if gate_cfg is None:
            # 兼容直接平铺在 cfg 里的写法
            gate_cfg = {k: v for k, v in cfg.items()
                        if k in DEFAULT_THRESHOLDS or k in _ALIASES}
    if isinstance(gate_cfg, dict):
        for k, v in gate_cfg.items():
            key = _ALIASES.get(k, k)
            if key not in th:
                continue
            try:
                th[key] = float(v)
            except (TypeError, ValueError):
                continue
    return th


def _is_clean(assessment) -> bool:
    return bool(assessment is not None and getattr(assessment, "needs_denoise", True) is False)


def _fmt(x, digits: int = 3, unit: str = "") -> str:
    if x is None:
        return "不可算"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "不可算"
    if not math.isfinite(v):
        return "不可算"
    if abs(v) >= 1e5 or (0 < abs(v) < 1e-3):
        return f"{v:.2e}{unit}"
    return f"{v:.{digits}f}{unit}"


# --------------------------------------------------------------------------
# 主入口
# --------------------------------------------------------------------------
def industry_gate(metrics: QualityMetrics | None,
                  assessment=None,
                  cfg=None) -> GateResult:
    """按行业可接受标准做门禁判定。

    返回 :class:`GateResult`：``passed``/``score``(0–100)/``checks``/``failures``/``advice``。
    ``confidence`` 由 :func:`gate_confidence` 另行填写（runner 会覆盖）。
    """
    th = _thresholds(cfg)
    checks: list[GateCheck] = []
    failures: list[str] = []
    advice: list[str] = []

    # ---- 干净数据：不走去噪门禁 ----
    if _is_clean(assessment):
        score = 90.0
        checks.append(GateCheck(
            name="是否需要去噪", value=0.0,
            threshold="assessment.needs_denoise=False",
            passed=True,
            note="评估认为数据较干净，原样保留即可；无需按去噪标准要求增益",
        ))
        return GateResult(passed=True, score=score, checks=checks, failures=[],
                          confidence=0.0,
                          advice=["数据较干净：如只想做轻度增强，可手动指定 bandpass 或 cnn 小腿力处理"])

    if metrics is None:
        return GateResult(
            passed=False, score=0.0,
            checks=[GateCheck("质量指标", None, "需要 QualityMetrics", False, "没有指标，无法判定")],
            failures=["缺少质量指标，无法判定去噪是否有效"],
            confidence=0.0,
            advice=["先运行 evaluate() 生成指标，再做门禁判定"],
        )

    ex = metrics.extra or {}
    snr_gain = metrics.snr_gain_db
    energy = metrics.energy_ratio
    leakage = metrics.signal_leakage
    spec_corr = metrics.spectrum_corr
    clip = metrics.clip_ratio
    snr_out = metrics.snr_out_db
    reliable = bool(ex.get("reliable", True))
    reliable_in = bool(ex.get("snr_in_reliable", reliable))
    reliable_out = bool(ex.get("snr_out_reliable", reliable))

    # ---- 1) 原样返回检测 ----
    identity = False
    identity_reason = ""
    if snr_gain is not None and energy is not None:
        if abs(snr_gain) < th["identity_gain_db"] and abs(energy - 1.0) < th["identity_energy"]:
            identity = True
            identity_reason = (f"能量比 {_fmt(energy, 4)}、SNR 增益 {_fmt(snr_gain, 2, ' dB')}，"
                               f"输出与输入几乎一致")
    elif snr_gain is not None and abs(snr_gain) < th["identity_gain_db"]:
        identity = True
        identity_reason = f"SNR 增益仅 {_fmt(snr_gain, 2, ' dB')}，输出与输入几乎一致"
    checks.append(GateCheck(
        name="是否真的去噪",
        value=None if snr_gain is None else round(float(snr_gain), 4),
        threshold="增益≈0 且 能量比≈1 视为原样返回",
        passed=not identity,
        note=identity_reason if identity else "输出与输入存在可测差异",
    ))
    if identity:
        failures.append(f"结果等于原样返回（{identity_reason}），没有产生去噪作用")

    # ---- 2) 信噪比增益（可靠时硬，不可靠时软） ----
    if snr_gain is None:
        checks.append(GateCheck("信噪比增益", None, f">={th['min_snr_gain_db']:.1f} dB", True,
                                "盲信噪比不可算，本项跳过"))
        advice.append("盲信噪比不可算：建议补充带参考的合成数据校验")
    elif not reliable:
        checks.append(GateCheck(
            "信噪比增益", round(float(snr_gain), 3),
            f">={th['min_snr_gain_db']:.1f} dB（盲估计不可靠时仅参考）",
            True,
            "盲信噪比不可靠，增益受噪声类型/记录长度影响；本项不硬性判失败，改由能量/泄漏/频谱门禁把关",
        ))
        advice.append("盲信噪比不可靠：建议人工复核剖面，或补充纯净参考数据计算有参考 SNR")
    else:
        ok = float(snr_gain) >= th["min_snr_gain_db"]
        note = ("增益达标" if ok else
                f"增益不足（差 {th['min_snr_gain_db'] - float(snr_gain):.1f} dB），去噪强度可能不够")
        checks.append(GateCheck("信噪比增益", round(float(snr_gain), 3),
                                f">={th['min_snr_gain_db']:.1f} dB", ok, note))
        if not ok:
            failures.append(f"信噪比增益 {float(snr_gain):.2f} dB 低于最低要求 {th['min_snr_gain_db']:.1f} dB")
            advice.append("提高去噪强度：加大滤波/预测强度、提高 cnn alpha，或叠加第二种方法")

    # ---- 2b) 纯净参考的真实 SNR 增益（有参考时硬） ----
    ref_gain = getattr(metrics, "reference_gain_db", None)
    if ref_gain is not None:
        try:
            rg = float(ref_gain)
        except (TypeError, ValueError):
            rg = None
        if rg is not None and math.isfinite(rg):
            ok = rg >= th["min_reference_gain_db"]
            note = ("参考 SNR 增益达标" if ok else
                    f"相对纯净参考反而下降 {abs(rg):.2f} dB，去噪伤及有效信号")
            checks.append(GateCheck("参考 SNR 增益", round(rg, 3),
                                    f">={th['min_reference_gain_db']:.1f} dB", ok, note))
            if not ok:
                failures.append(
                    f"参考 SNR 增益 {rg:.2f} dB 低于最低要求 {th['min_reference_gain_db']:.1f} dB")
                advice.append("换更温和的方法/参数：降低过滤强度，保住有效波后再比较参考增益")

    # ---- 3) 能量保持（硬） ----
    if energy is None:
        checks.append(GateCheck("能量保持", None,
                                f"[{th['energy_min']:.2f}, {th['energy_max']:.2f}]", True,
                                "信号带内输入能量为 0，本项跳过"))
    else:
        e = float(energy)
        ok = th["energy_min"] <= e <= th["energy_max"]
        if ok:
            note = "能量保持在合理范围"
        elif e < th["energy_min"]:
            note = f"能量削得太狠（保留 {e:.1%}），有效波可能被过度压制"
        else:
            note = f"输出能量超过输入（{e:.1%}），可能在放大噪声"
        checks.append(GateCheck("能量保持", round(e, 4),
                                f"[{th['energy_min']:.2f}, {th['energy_max']:.2f}]", ok, note))
        if not ok:
            failures.append(f"信号带能量比 {e:.3f} 不在 [{th['energy_min']:.2f}, {th['energy_max']:.2f}]")
            if e < th["energy_min"]:
                advice.append("降低去噪强度或加保护带：减小 alpha/滤波阶数，保住有效波能量")
            else:
                advice.append("输出被放大：检查是否有增益补偿或叠加重复施加，降低强度后重试")

    # ---- 3b) 弱信号保真（高保真硬检查） ----
    visual = ex.get("visual") if isinstance(ex.get("visual"), dict) else {}
    weak_ratio = visual.get("weak_signal_ratio") if visual.get("available") else None
    if weak_ratio is not None:
        wr = float(weak_ratio)
        ok = wr >= th["min_weak_signal_ratio"]
        if ok:
            note = "输入弱能量区的有效波保留良好"
        else:
            note = ("弱区能量保留不足，弱有效波可能被当成噪声削掉；"
                    "需要降低去噪强度或启用高保真回注")
        checks.append(GateCheck("弱信号保真", round(wr, 4),
                                f">={th['min_weak_signal_ratio']:.2f}", ok, note))
        if not ok:
            failures.append(f"弱区能量保留 {wr:.1%} 低于 {th['min_weak_signal_ratio']:.0%}")
            advice.append("对弱有效波做保护：降低滤波/预测强度，或开启 fidelity 弱信号回注")

    # ---- 4) 信号泄漏（硬） ----
    leak_limit = (th["max_signal_leakage_ref"]
                  if getattr(metrics, "reference_gain_db", None) is not None
                  else th["max_signal_leakage"])
    if leakage is None:
        checks.append(GateCheck("信号泄漏", None, f"<={leak_limit:.2f}", True,
                                "局部相关样本不足，本项跳过"))
        advice.append("信号泄漏不可算：建议用更长的数据或带参考的合成数据复核")
    else:
        lk = float(leakage)
        ok = lk <= leak_limit
        note = ("被去除分量与有效波相关性低，信号保护良好" if ok else
                "被去除分量里含有大量有效波成分，去噪伤及信号")
        checks.append(GateCheck("信号泄漏", round(lk, 4),
                                f"<={leak_limit:.2f}", ok, note))
        if not ok:
            failures.append(f"信号泄漏 {lk:.1%} 超过上限 {leak_limit:.0%}")
            advice.append("缩窄去噪频带/慢度范围，优先保住同相轴；必要时降低 cnn alpha")

    # ---- 5) 频谱一致性（硬） ----
    if spec_corr is None:
        checks.append(GateCheck("频谱一致性", None, f">={th['min_spectrum_corr']:.2f}", True,
                                "信号带振幅谱不可算，本项跳过"))
    else:
        sc = float(spec_corr)
        ok = sc >= th["min_spectrum_corr"]
        note = ("频谱形状保持良好" if ok else "频谱形状被明显改变，可能滤掉了整段频带")
        checks.append(GateCheck("频谱一致性", round(sc, 4),
                                f">={th['min_spectrum_corr']:.2f}", ok, note))
        if not ok:
            failures.append(f"频谱相关 {sc:.3f} 低于下限 {th['min_spectrum_corr']:.2f}")
            advice.append("检查滤波频带/小波阈值是否过激，减小对振幅谱的改变")

    # ---- 6) 削波/过冲（硬） ----
    if clip is None:
        checks.append(GateCheck("削波控制", None, f"<={th['max_clip_ratio']:.1e}", True,
                                "输入全零，本项跳过"))
    else:
        c = float(clip)
        ok = c <= th["max_clip_ratio"]
        checks.append(GateCheck("削波控制", round(c, 8),
                                f"<={th['max_clip_ratio']:.1e}", ok,
                                "无新增削波/过冲" if ok else "输出出现削波或异常过冲"))
        if not ok:
            failures.append(f"额外削波比例 {c:.2e} 超过 {th['max_clip_ratio']:.1e}")
            advice.append("检查输出是否越界或被放大：必要时做幅度归一化/限幅")

    # ---- 7) 道级异常（坏道/垂直条带） ----
    ex = getattr(metrics, "extra", None) or {}
    ratio_in = ex.get("trace_rms_ratio_in")
    ratio_out = ex.get("trace_rms_ratio_out")
    if ratio_out is not None:
        limit = 50.0
        introduced = bool(ratio_in and float(ratio_out) > float(ratio_in) * 2.0)
        bad = bool(float(ratio_out) > limit and introduced)
        checks.append(GateCheck(
            "道级异常", round(float(ratio_out), 2),
            f"<={limit:g}×中位 RMS（且不超过输入 2 倍）", not bad,
            (f"输出出现异常能量道（{float(ratio_out):.1f}×中位，输入 {float(ratio_in or 0):.1f}×），疑似坏道/垂直条带"
             if bad else "输出道能量分布正常"),
        ))
        if bad:
            failures.append(f"输出出现异常能量道（{float(ratio_out):.1f}×中位 RMS），疑似坏道或垂直条带")
            advice.append("检查坏道/边界伪影；对输出做道级异常修复，或换不会放大边界的去噪方法")

    # ---- 7b) 单道尖峰/垂直条带（横向中值残差） ----
    resid_in = ex.get("trace_resid_ratio_in")
    resid_out = ex.get("trace_resid_ratio_out")
    if resid_out is not None:
        limit = float(th.get("max_trace_resid_ratio", 150.0))
        base = float(resid_in) if resid_in is not None else 1.0
        bad = bool(float(resid_out) > limit)
        checks.append(GateCheck(
            "单道尖峰/垂直条带", round(float(resid_out), 2),
            f"<={limit:g}×中位残差（输入 {base:.1f}×）", not bad,
            (f"输出出现单道尖峰/垂直条带（{float(resid_out):.1f}×中位残差，输入 {base:.1f}×），"
             "疑似边界伪影或坏道被放大" if bad else "输出无明显单道尖峰/垂直条带"),
        ))
        if bad:
            failures.append(f"输出出现单道尖峰/垂直条带（{float(resid_out):.1f}×中位残差）")
            advice.append("检查边界/坏道处理；换更平滑的方法或先修掉异常道再重跑")

    # ---- 8) 输出绝对信噪比（软：盲估计不可靠时只扣分） ----
    if snr_out is None:
        checks.append(GateCheck("输出信噪比", None, f">={th['min_snr_out_db']:.1f} dB", True,
                                "盲信噪比不可算，本项跳过"))
    else:
        so = float(snr_out)
        ok = so >= th["min_snr_out_db"]
        if reliable:
            passed = ok
            note = "输出信噪比达标" if ok else "输出信噪比仍偏低"
        else:
            passed = True
            note = (f"盲估计不可靠（输入{'可靠' if reliable_in else '不可靠'}/"
                    f"输出{'可靠' if reliable_out else '不可靠'}），仅作减分项："
                    f"{'达标' if ok else '偏低'}")
        checks.append(GateCheck("输出信噪比", round(so, 3),
                                f">={th['min_snr_out_db']:.1f} dB" + ("" if reliable else "（软）"),
                                passed, note))
        if not ok:
            if reliable:
                failures.append(f"输出信噪比 {so:.1f} dB 低于 {th['min_snr_out_db']:.1f} dB")
                advice.append("继续迭代：换更强方法或叠加多方法，直到输出 SNR 达标")
            else:
                advice.append("盲信噪比不可靠，输出 SNR 仅参考；建议用带参考数据复核绝对值")

    # ---- 评分 ----
    score = _score(metrics, th, identity=identity)
    if failures:
        score = min(score, 55.0) * 0.85 if identity else score * 0.55
    score = round(_clip01(score / 100.0) * 100.0, 1)

    if not failures and score < 60:
        advice.append("指标擦线通过：建议换更强的参数变体再比较一次")
    return GateResult(passed=not failures, score=score, checks=checks,
                      failures=failures, confidence=0.0, advice=advice)


def _score(metrics: QualityMetrics, th: dict[str, float], identity: bool = False) -> float:
    """0–100：能量/增益/泄漏/频谱/削波/输出 SNR 的加权得分（缺项归一化）。"""
    parts: dict[str, float] = {}
    gain = metrics.snr_gain_db
    if gain is not None:
        parts["gain"] = _clip01(float(gain) / 12.0)
    energy = metrics.energy_ratio
    if energy is not None:
        e = float(energy)
        lo, hi = th["energy_min"], th["energy_max"]
        if lo <= e <= hi:
            # 带内越靠中间越好
            center = 0.5 * (lo + min(hi, 1.0))
            span = max(0.05, 0.5 * (min(hi, 1.0) - lo))
            parts["energy"] = 1.0 - 0.35 * min(1.0, abs(e - center) / max(span, 1e-6))
        elif e < lo:
            parts["energy"] = _clip01((e - lo + 0.15) / 0.15)
        else:
            parts["energy"] = _clip01(1.0 - (e - hi) / 0.15)
    lk = metrics.signal_leakage
    if lk is not None:
        parts["leakage"] = _clip01(1.0 - max(float(lk), 0.0) / 0.6)
    sc = metrics.spectrum_corr
    if sc is not None:
        parts["spectrum"] = _clip01((float(sc) - 0.4) / 0.6)
    clip = metrics.clip_ratio
    if clip is not None:
        parts["clip"] = _clip01(1.0 - float(clip) / (2.0 * th["max_clip_ratio"]))
    so = metrics.snr_out_db
    if so is not None:
        parts["snr_out"] = _clip01((float(so) - 3.0) / 17.0)
    visual = (metrics.extra or {}).get("visual") if isinstance(metrics.extra, dict) else None
    if isinstance(visual, dict) and visual.get("available"):
        vs = visual.get("visual_score")
        if vs is not None:
            parts["visual"] = _clip01(float(vs))

    if not parts:
        return 0.0
    wsum = sum(_SCORE_WEIGHTS[k] for k in parts)
    score = sum(parts[k] * _SCORE_WEIGHTS[k] for k in parts) / wsum * 100.0
    if identity:
        score = min(score, 15.0)
    return float(score)


# --------------------------------------------------------------------------
# 置信度
# --------------------------------------------------------------------------
def gate_confidence(metrics: QualityMetrics | None,
                    gate: GateResult | None,
                    assessment=None) -> float:
    """门禁结论的置信度 0–1。

    三个因子相乘：
    1. **指标可信度**：盲 SNR 两路是否都可靠、关键指标是否有缺项；
    2. **门禁余量**：离阈值越远越可信（压线通过时置信度低）；
    3. **来源匹配**：评估置信度越高、且与当前数据一致，结论越可信。
    """
    if gate is None:
        return 0.0
    if _is_clean(assessment) and gate.passed:
        src = float(getattr(assessment, "confidence", 0.6) or 0.6)
        return round(_clip01(0.72 + 0.22 * _clip01(src)), 4)

    ex = (metrics.extra if metrics is not None else None) or {}
    if metrics is None:
        return 0.0
    key_missing = sum(1 for k in ("snr_gain_db", "energy_ratio", "signal_leakage",
                                  "spectrum_corr", "clip_ratio")
                      if getattr(metrics, k, None) is None)
    reli_in = bool(ex.get("snr_in_reliable", False))
    reli_out = bool(ex.get("snr_out_reliable", False))
    if reli_in and reli_out:
        metric_rel = 1.0
    elif reli_in or reli_out:
        metric_rel = 0.78
    else:
        metric_rel = 0.55
    metric_rel *= max(0.55, 1.0 - 0.12 * key_missing)
    if ex.get("spectrum_corr_bins", 1) and int(ex.get("spectrum_corr_bins", 1)) < 10:
        metric_rel *= 0.9

    th = _thresholds(None)
    margins: list[tuple[float, float]] = []
    gain = metrics.snr_gain_db
    if gain is not None:
        margins.append((_clip01((float(gain) - th["min_snr_gain_db"]) / 6.0), 0.34))
    energy = metrics.energy_ratio
    if energy is not None:
        e = float(energy)
        lo, hi = th["energy_min"], th["energy_max"]
        if lo <= e <= hi:
            m = min(e - lo, hi - e) / 0.2
        else:
            m = 0.0
        margins.append((_clip01(m), 0.16))
    lk = metrics.signal_leakage
    if lk is not None:
        margins.append((_clip01((th["max_signal_leakage"] - float(lk)) / 0.25), 0.20))
    sc = metrics.spectrum_corr
    if sc is not None:
        margins.append((_clip01((float(sc) - th["min_spectrum_corr"]) / 0.15), 0.14))
    clip = metrics.clip_ratio
    if clip is not None:
        margins.append((_clip01((th["max_clip_ratio"] - float(clip)) / th["max_clip_ratio"]), 0.06))
    so = metrics.snr_out_db
    if so is not None:
        w = 0.10 if (ex.get("snr_out_reliable", False)) else 0.05
        margins.append((_clip01((float(so) - th["min_snr_out_db"]) / 15.0), w))

    if margins:
        wsum = sum(w for _, w in margins)
        margin = sum(m * w for m, w in margins) / max(wsum, 1e-9)
    else:
        margin = 0.0

    src = float(getattr(assessment, "confidence", 0.6) or 0.6) if assessment is not None else 0.6
    src = _clip01(src)
    src_factor = 0.65 + 0.35 * src

    if gate.passed:
        conf = metric_rel * (0.25 + 0.75 * margin) * src_factor
    else:
        conf = metric_rel * (0.15 + 0.30 * (1.0 - margin)) * src_factor
    return round(_clip01(conf), 4)
