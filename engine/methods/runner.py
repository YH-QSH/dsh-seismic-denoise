"""执行器：依次跑规划出的方法，逐个评价 + 门禁，最后挑最优。

只做内存计算，不写任何文件（成果保存由 pipeline 负责）。
单个方法抛异常不会中断整轮：该次尝试记为失败（``GateResult.passed=False``、
``failures`` 写人话），继续跑后面的方法。
"""
from __future__ import annotations

import json
import math
import time

import numpy as np

from ..model import MethodResult, QualityMetrics, GateCheck, GateResult
from ..quality.fidelity import enhance_fidelity
from ..quality.gate import gate_confidence, industry_gate
from ..quality.metrics import evaluate


def _log(log, msg: str) -> None:
    if log is None:
        return
    try:
        log(msg)
    except Exception:
        pass


def _fmt_params(params: dict) -> str:
    try:
        s = json.dumps(params or {}, ensure_ascii=False, sort_keys=True, default=str)
    except Exception:
        s = str(params)
    return s if len(s) <= 160 else s[:157] + "..."


def _method_summary(metrics: QualityMetrics | None) -> str:
    """短指标摘要，方便日志/报告一眼看懂。"""
    if metrics is None:
        return "无指标"
    parts: list[str] = []
    gain = getattr(metrics, "snr_gain_db", None)
    if isinstance(gain, (int, float)) and math.isfinite(float(gain)):
        parts.append(f"SNR增益 {float(gain):+.2f} dB")
    er = getattr(metrics, "energy_ratio", None)
    if isinstance(er, (int, float)) and math.isfinite(float(er)):
        parts.append(f"能量比 {float(er):.2f}")
    leak = getattr(metrics, "signal_leakage", None)
    if isinstance(leak, (int, float)) and math.isfinite(float(leak)):
        parts.append(f"泄漏 {float(leak):.2f}")
    sc = getattr(metrics, "spectrum_corr", None)
    if isinstance(sc, (int, float)) and math.isfinite(float(sc)):
        parts.append(f"谱相关 {float(sc):.2f}")
    return "，".join(parts) if parts else "无指标"


def _execute(spec, data: np.ndarray, dt: float, dz: float | None):
    """调用对应方法，返回 (out, info)。"""
    params = dict(getattr(spec, "params", None) or {})
    key = getattr(spec, "key", "")
    if key == "combo":
        from .combo import run_chain
        return run_chain(data, dt, dz, params.get("chain") or [])
    if key == "cnn":
        from .cnn import run_cnn
        return run_cnn(data, dt, dz, params)
    from .classical import run_method
    return run_method(key, data, dt, dz, params)


def _failed_result(spec, seconds: float, exc: Exception) -> MethodResult:
    msg = f"{type(exc).__name__}: {exc}"
    gate = GateResult(
        passed=False,
        score=0.0,
        checks=[GateCheck(
            name="方法执行",
            value=None,
            threshold="需要正常完成",
            passed=False,
            note=f"{getattr(spec, 'name', spec.key)} 执行异常：{msg}",
        )],
        failures=[f"{getattr(spec, 'name', spec.key)} 执行失败：{msg}"],
        confidence=0.0,
        advice=["检查该方法参数是否合法、依赖是否齐全；可换一种方法或降低数据规模"],
    )
    return MethodResult(
        spec=spec,
        metrics=QualityMetrics(estimator=""),
        gate=gate,
        runtime_s=round(float(seconds), 3),
        warnings=[f"执行失败：{msg}"],
        notes="方法抛出异常，未产出可用结果",
    )


def run_attempts(ds, specs: list, cfg: dict | None = None,
                 reference: np.ndarray | None = None, log=print) -> list[MethodResult]:
    """依次执行 ``specs``，返回每次尝试的 :class:`MethodResult`。

    参数
    ----
    ds : SeismicDataset。使用其 ``data/dt/dz``；``dt`` 缺失或非法直接抛 ValueError。
    cfg : 流水线配置（``cfg["gate"]`` 覆盖门禁阈值）。
    reference : 可选参考（同 shape）；形状不符时忽略参考指标并记警告。
    log : 日志回调（默认 print；传 None 静默）。
    """
    if ds is None:
        raise ValueError("ds 不能为空")
    data = np.asarray(getattr(ds, "data", None))
    if data.ndim != 2:
        raise ValueError(f"数据集 data 必须是二维 (采样点, 道数)，实际 shape={data.shape}")
    if data.size == 0:
        raise ValueError("数据集 data 为空")
    dt = getattr(ds, "dt", None)
    try:
        dt = float(dt)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"数据集 dt 无效（{dt!r}），无法去噪") from exc
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"数据集 dt 无效（{dt!r}），必须为正的有限秒数")
    dz = getattr(ds, "dz", None)
    try:
        dz = float(dz) if dz is not None else None
    except (TypeError, ValueError):
        dz = None
    if dz is not None and (not math.isfinite(dz) or dz <= 0):
        dz = None

    cfg = cfg if isinstance(cfg, dict) else {}
    specs = list(specs or [])
    if not specs:
        return []
    # 评估结果（若调用方通过 cfg 或 ds 属性传入）：只影响"干净数据"分支
    assessment = cfg.get("assessment") if isinstance(cfg, dict) else None
    if assessment is None:
        assessment = getattr(ds, "assessment", None)

    ref = None
    if reference is not None:
        ref = np.asarray(reference, dtype=np.float32)
        if ref.shape != data.shape:
            _log(log, f"参考数据形状 {ref.shape} 与输入 {data.shape} 不一致，忽略参考指标")
            ref = None

    results: list[MethodResult] = []
    total = len(specs)
    for i, spec in enumerate(specs, 1):
        name = getattr(spec, "name", None) or getattr(spec, "key", "?")
        key = getattr(spec, "key", "?")
        _log(log, f"[{i}/{total}] {name}（{key}）参数 {_fmt_params(getattr(spec, 'params', {}) or {})}")
        # 每次尝试都用干净副本，避免方法就地修改污染输入
        work = np.array(data, dtype=np.float32, copy=True)
        t0 = time.perf_counter()
        try:
            out, info = _execute(spec, work, dt, dz)
            out = np.asarray(out)
            if out.shape != data.shape:
                raise ValueError(f"方法输出形状 {out.shape} 与输入 {data.shape} 不一致")
            out = np.asarray(out, dtype=np.float32)
            seconds = time.perf_counter() - t0
            if not np.isfinite(out).all():
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                warn_finite = "方法输出含 NaN/Inf，已置 0"
            else:
                warn_finite = ""
        except Exception as exc:  # 单方法失败不影响其他方法
            seconds = time.perf_counter() - t0
            result = _failed_result(spec, seconds, exc)
            results.append(result)
            _log(log, f"    执行失败（{seconds:.2f}s）：{type(exc).__name__}: {exc}")
            continue

        info = info if isinstance(info, dict) else {}
        warnings = [str(w) for w in (info.get("warnings") or [])]
        if warn_finite:
            warnings.append(warn_finite)
        notes = str(info.get("notes") or "")

        # 高保真后处理：只补“弱能量 + 横向相干”的残差，强面波区不动。
        fid_info = {"enabled": False}
        try:
            fid_cfg = cfg.get("fidelity") if isinstance(cfg, dict) else None
            fid_cfg = fid_cfg if isinstance(fid_cfg, dict) else {}
            out, fid_info = enhance_fidelity(data, out, dt, fid_cfg)
            out = np.asarray(out, dtype=np.float32)
            if not np.isfinite(out).all():
                out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
                warnings.append("高保真后处理输出含 NaN/Inf，已置 0")
            if fid_info.get("enabled") and abs(float(fid_info.get("restore_db") or 0.0)) > 0.2:
                notes = (notes + "；" if notes else "") + (
                    f"弱信号保护回注 {fid_info['restore_db']:+.2f} dB"
                )
        except Exception as exc:
            warnings.append(f"高保真后处理失败：{type(exc).__name__}: {exc}")

        t_eval = time.perf_counter()
        try:
            metrics = evaluate(data, out, dt, dz, reference=ref)
            metrics.extra["fidelity"] = fid_info
            gate = industry_gate(metrics, assessment=assessment, cfg=cfg)
            gate.confidence = gate_confidence(metrics, gate, assessment)
        except Exception as exc:  # 评价异常也要留下记录
            metrics = QualityMetrics(estimator="")
            gate = GateResult(
                passed=False, score=0.0,
                checks=[GateCheck("质量评价", None, "需要正常完成", False,
                                  f"评价失败：{type(exc).__name__}: {exc}")],
                failures=[f"质量评价失败：{type(exc).__name__}: {exc}"],
                confidence=0.0,
                advice=["检查输入数据与采样参数后重试"],
            )
            warnings.append(f"质量评价异常：{type(exc).__name__}: {exc}")
        eval_s = time.perf_counter() - t_eval

        result = MethodResult(
            spec=spec,
            metrics=metrics,
            gate=gate,
            runtime_s=round(float(seconds), 3),
            warnings=warnings,
            notes=notes,
            data=out,
        )
        results.append(result)

        status = "通过" if gate.passed else "未通过"
        _log(log, f"    {seconds:.2f}s（评价 {eval_s:.2f}s），门禁{status}，"
                  f"score={gate.score:.1f}，conf={gate.confidence:.2f}；{_method_summary(metrics)}")
        if not gate.passed and gate.failures:
            _log(log, f"    原因：{'；'.join(gate.failures[:3])}")
    return results


def pick_best(results: list[MethodResult] | None) -> MethodResult | None:
    """挑最优：优先门禁通过者，按 score 降序；否则全按 score 降序。"""
    valid = [r for r in (results or [])
             if r is not None and getattr(r, "gate", None) is not None]
    if not valid:
        return None
    passed = [r for r in valid if bool(getattr(r.gate, "passed", False))]

    def _key(r: MethodResult):
        g = r.gate
        score = float(getattr(g, "score", 0.0) or 0.0)
        conf = float(getattr(g, "confidence", 0.0) or 0.0)
        rt = float(getattr(r, "runtime_s", 0.0) or 0.0)
        # 有纯净参考时，真实参考 SNR 增益比盲估计分数更可信；优先用它排序。
        ref = getattr(r.metrics, "reference_gain_db", None)
        ref_key = float(ref) if ref is not None and math.isfinite(float(ref)) else None
        return (1 if ref_key is not None else 0,
                ref_key if ref_key is not None else -1e9,
                score, conf, -rt)

    pool = passed if passed else valid
    return max(pool, key=_key)
