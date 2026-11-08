"""去噪流水线：读取 → 来源匹配 → 规划 → 尝试 → 门禁 → 整理成果。

所有阶段都可以单独调用；`auto_flow` 串起全流程。
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .analysis.inspect import READ_DIR, assess_stage, load_dataset, read_stage
from .model import MethodResult, MethodSpec, SeismicDataset, SourceAssessment, to_jsonable
from .utils import Logger, ensure_dir, now_iso, read_json, safe_name, stamp, write_json

DEFAULT_CFG: dict[str, Any] = {
    "clean_snr_db": 28.0,
    "clean_noise_score": 0.35,
    "max_attempts": 6,
    "gate": {
        "min_gain_db": 3.0,
        "energy_ratio_min": 0.60,
        "energy_ratio_max": 1.05,
        "signal_leakage_max": 0.40,
        "leakage_max_ref": 0.45,
        "min_reference_gain_db": 0.5,
        "spectrum_corr_min": 0.80,
        "clip_ratio_max": 1e-3,
        "snr_out_min_db": 10.0,
        "weak_signal_ratio_min": 0.85,
    },
    "fidelity": {
        "enabled": True,
        "restore": 1.0,
        "weak_quantile": 0.40,
        "weak_boost": 0.05,
        "min_residual_coherence": 0.30,
        "trace_window": 5,
    },
    "cnn": {"alpha": 0.65, "gate": True},
}

STATE_NAME = "任务状态.json"


def merge_cfg(base: dict, override: dict | None) -> dict:
    out = json.loads(json.dumps(base, ensure_ascii=False))
    if not override:
        return out
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k].update(v)
        else:
            out[k] = v
    return out


def create_job_folder(input_path: str | Path, root: str | Path | None = None) -> Path:
    input_path = Path(input_path)
    base = Path(root) if root else input_path.parent
    ensure_dir(base)
    name = f"{safe_name(input_path.stem)}_地震去噪_{stamp()}"
    job = base / name
    n = 2
    while job.exists():
        job = base / f"{name}_{n}"
        n += 1
    for sub in ("01_原始数据", "02_读取与诊断", "03_去噪结果", "04_质量评价", "05_报告"):
        ensure_dir(job / sub)
    return job


def state_path(job_dir: str | Path) -> Path:
    return Path(job_dir) / STATE_NAME


def load_state(job_dir: str | Path) -> dict | None:
    return read_json(state_path(job_dir))


def save_state(job_dir: str | Path, state: dict) -> Path:
    state["updated_at"] = now_iso()
    return write_json(state_path(job_dir), state)


def _history(state: dict, stage: str, message: str) -> None:
    state.setdefault("history", []).append(
        {"time": now_iso(), "stage": stage, "message": message}
    )


def _new_state(job_dir: Path, input_path: Path, dataset_info: dict) -> dict:
    return {
        "version": "1.0",
        "job_id": job_dir.name,
        "input": str(input_path),
        "workdir": str(job_dir),
        "stage": "read",
        "created_at": now_iso(),
        "updated_at": now_iso(),
        "dataset": dataset_info,
        "source": None,
        "plan": [],
        "attempts": [],
        "best": None,
        "gate": None,
        "confidence": 0.0,
        "passed": None,
        "outputs": {"denoised": "", "report": "", "folder": str(job_dir)},
        "history": [],
        "warnings": [],
    }


def _public_dataset(info: dict) -> dict:
    """给工具返回值用的精简数据集信息（不重复塞道头全量值）。"""
    out = {k: v for k, v in info.items() if k not in ("headers", "_headers_raw")}
    headers = info.get("headers") or {}
    out["header_fields"] = len(headers)
    out["header_names"] = [v.get("name", k) for k, v in list(headers.items())[:16]]
    return out


def _ensure_dirs(job_dir: Path) -> None:
    for sub in ("01_原始数据", "02_读取与诊断", "03_去噪结果", "04_质量评价", "05_报告"):
        ensure_dir(job_dir / sub)


def init_job(input_path: str | Path, workdir: str | Path | None = None,
             root: str | Path | None = None, max_traces: int | None = None,
             log: Callable[[str], None] | None = None) -> tuple[Path, dict]:
    input_path = Path(input_path)
    log = log or Logger()
    if workdir:
        job_dir = Path(workdir)
        _ensure_dirs(job_dir)
        existing = load_state(job_dir)
        if existing:
            log(f"复用已有任务：{job_dir}")
            return job_dir, existing
    else:
        job_dir = create_job_folder(input_path, root)
    log(f"成果文件夹：{job_dir}")
    result = read_stage(input_path, job_dir, max_traces=max_traces, log=log)
    # 状态/报告保留道头统计；全量道头原值只在缓存 JSON 里，避免状态文件重复膨胀
    dataset_state = {k: v for k, v in result["dataset"].items() if k != "_headers_raw"}
    state = _new_state(job_dir, input_path, dataset_state)
    state["artifacts"] = result.get("artifacts", {})
    _history(state, "read", result.get("message", ""))
    save_state(job_dir, state)
    return job_dir, state


def inspect_flow(input_path: str | Path, workdir: str | Path | None = None,
                 root: str | Path | None = None, max_traces: int | None = None,
                 log: Callable[[str], None] | None = None) -> dict:
    t0 = time.time()
    job_dir, state = init_job(input_path, workdir, root, max_traces, log)
    return {
        "ok": True,
        "stage": "read",
        "workdir": str(job_dir),
        "dataset": _public_dataset(state["dataset"]),
        "repairs": state["dataset"].get("preprocess", {}).get("bad_traces_repaired", []),
        "artifacts": state.get("artifacts", {}),
        "message": (
            f"读取完成：{state['dataset'].get('n_traces')} 道 × "
            f"{state['dataset'].get('n_samples')} 点，格式 {state['dataset'].get('format')}。"
            "下一步可调用 seismic_assess 做来源匹配。"
        ),
        "elapsed_s": round(time.time() - t0, 2),
    }


def _assessment_from_state(state: dict) -> SourceAssessment:
    src = state.get("source") or {}
    names = {f.name for f in fields(SourceAssessment)}
    return SourceAssessment(**{k: v for k, v in src.items() if k in names})


def assess_flow(job_dir: str | Path, cfg: dict | None = None,
                log: Callable[[str], None] | None = None) -> dict:
    t0 = time.time()
    job_dir = Path(job_dir)
    log = log or Logger(job_dir / "运行日志.txt")
    state = load_state(job_dir)
    if not state:
        raise FileNotFoundError(f"{job_dir} 下没有任务状态，请先调用 seismic_read")
    cfg = merge_cfg(DEFAULT_CFG, cfg)
    result = assess_stage(job_dir, cfg, log=log)
    state["source"] = result["source"]
    state["stage"] = "assess"
    state["artifacts"] = {**state.get("artifacts", {}), **result.get("artifacts", {})}
    _history(state, "assess", result["message"])
    save_state(job_dir, state)
    return {
        "ok": True,
        "stage": "assess",
        "workdir": str(job_dir),
        "source": result["source"],
        "needs_denoise": result["needs_denoise"],
        "artifacts": result.get("artifacts", {}),
        "message": result["message"],
        "elapsed_s": round(time.time() - t0, 2),
    }


def _load_reference(path: str | Path, ds: SeismicDataset, log) -> np.ndarray:
    from .formats import read_any
    ref_ds = read_any(path)
    ref = np.asarray(ref_ds.data, dtype=np.float32)
    if ref.shape != ds.data.shape:
        raise ValueError(f"参考数据形状 {ref.shape} 与输入 {ds.data.shape} 不一致")
    log(f"已加载参考数据：{path}")
    return ref


def _reference_close(ds: SeismicDataset, reference: str | Path, log) -> bool:
    """输入与纯净参考几乎相同时，直接判为无需去噪（参考数据专用短路）。"""
    try:
        ref = _load_reference(reference, ds, log)
    except Exception as exc:
        log(f"参考数据无法用于干净判断：{exc}")
        return False
    x = np.asarray(ds.data, dtype=np.float32)
    a = x.ravel().astype(np.float64)
    b = np.asarray(ref, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return False
    corr = float(np.dot(a, b) / denom)
    rel = float(np.sqrt(np.mean((x - ref) ** 2)) / (np.sqrt(np.mean(x ** 2)) + 1e-12))
    log(f"参考一致性：corr={corr:.6f}，相对差异={rel:.4f}")
    return corr >= 0.995 and rel <= 0.02


def _user_specs(methods: str | list[str], cfg: dict) -> list[MethodSpec]:
    keys = methods if isinstance(methods, list) else [m.strip() for m in str(methods).split(",") if m.strip()]
    from .methods.classical import available_methods
    known = available_methods()
    specs: list[MethodSpec] = []
    for k in keys:
        params = {}
        if k == "cnn":
            params = dict(cfg.get("cnn", {}))
        name = str(known.get(k, {}).get("name") or k)
        specs.append(MethodSpec(key=k, name=name, params=params, rationale="用户指定", source="user"))
    return specs


def _result_data(result: MethodResult, ds: SeismicDataset) -> np.ndarray | None:
    if result.data is not None and np.asarray(result.data).shape == ds.data.shape:
        return np.asarray(result.data, dtype=np.float32)
    # 兜底：方法确定性，重跑一次拿数组
    try:
        if result.spec.key == "cnn":
            from .methods.cnn import run_cnn
            out, _ = run_cnn(ds.data, ds.dt, ds.dz, result.spec.params)
        else:
            from .methods.classical import run_method
            out, _ = run_method(result.spec.key, ds.data, ds.dt, ds.dz, result.spec.params)
        return np.asarray(out, dtype=np.float32) if np.asarray(out).shape == ds.data.shape else None
    except Exception:
        return None


def _write_segy(path: Path, data: np.ndarray, ds: SeismicDataset, text: str) -> bool:
    try:
        assets = Path(__file__).resolve().parent / "assets" / "cnnv2"
        if str(assets) not in sys.path:
            sys.path.insert(0, str(assets))
        from vsdn2 import sgy  # type: ignore
        sgy.write_segy(path, data, dt=ds.dt or 0.001, text=text, fmt=5)
        return True
    except Exception:
        return False


def _save_attempts(job_dir: Path, ds: SeismicDataset, results: list[MethodResult], log) -> None:
    outdir = job_dir / "03_去噪结果"
    outdir.mkdir(parents=True, exist_ok=True)
    # 同一次任务重复跑 denoise 时，旧的尝试 SEG-Y 会变成没有记录的孤儿文件；先清掉。
    for old in outdir.glob("去噪结果_*.segy"):
        try:
            old.unlink()
        except OSError:
            pass
    for i, r in enumerate(results, 1):
        data = _result_data(r, ds)
        if data is None:
            log(f"跳过保存 {r.spec.key}：拿不到去噪数组")
            continue
        name = f"去噪结果_{i:02d}_{safe_name(r.spec.name, 20)}.segy"
        path = outdir / name
        ok = _write_segy(path, data, ds, text=f"C01 SEISMIC DENOISE {r.spec.key}")
        if ok:
            r.output_path = str(path)
            log(f"已保存尝试 {i}：{name}")
        else:
            log(f"SEG-Y 写入失败：{r.spec.key}")


def _save_quality(job_dir: Path, ds: SeismicDataset, best: MethodResult,
                  results: list[MethodResult], log) -> dict:
    qdir = job_dir / "04_质量评价"
    qdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "best": best.to_info(),
        "attempts": [r.to_info() for r in results],
        "source": "见 02_读取与诊断/来源匹配.json",
    }
    write_json(qdir / "质量评价.json", payload)
    arts: dict[str, str] = {}
    try:
        from .reporting.figures import (
            fig_compare, fig_local_zoom, fig_metrics, fig_residual, fig_visual_qc,
        )
        data = _result_data(best, ds)
        if data is not None:
            p1 = fig_compare(ds.data, data, ds.dt, ds.dz, qdir / "指标对比.png",
                             title=f"去噪前后（{best.spec.name}）", residual=ds.data - data)
            p2 = fig_metrics(best.metrics, qdir / "指标图.png", title="质量指标")
            p3 = fig_local_zoom(ds.data, data, ds.dt, ds.dz, qdir / "局部放大对比.png")
            p5 = fig_residual(ds.data, data, ds.dt, ds.dz,
                              job_dir / "03_去噪结果" / "去噪差分.png")
            p6 = fig_visual_qc(ds.data, data, ds.dt, ds.dz, qdir / "视觉对比.png",
                               visual=(best.metrics.extra or {}).get("visual")
                               if isinstance(best.metrics.extra, dict) else None)
            for p in (p1, p2, p3, p5, p6):
                if p:
                    arts[Path(p).stem] = str(p)
            # 03 里也放一张前后对比
            p4 = fig_compare(ds.data, data, ds.dt, ds.dz,
                             job_dir / "03_去噪结果" / "去噪后剖面.png",
                             title=f"去噪前后（{best.spec.name}）")
            if p4:
                arts["去噪后剖面"] = str(p4)
    except Exception as e:
        log(f"质量评价图生成失败：{e}")
    return arts


def _reconstruct_specs(state: dict) -> list[MethodSpec]:
    out = []
    for d in state.get("plan", []) or []:
        out.append(MethodSpec(
            key=d.get("key", ""), name=d.get("name", d.get("key", "")),
            params=d.get("params", {}), rationale=d.get("rationale", ""),
            targets=d.get("targets", []), strength=float(d.get("strength", 1.0)),
            priority=int(d.get("priority", 0)), source=d.get("source", "rule"),
        ))
    return out


def denoise_flow(job_dir: str | Path, cfg: dict | None = None,
                 reference: str | Path | None = None,
                 methods: str | list[str] | None = None,
                 max_attempts: int | None = None,
                 allow_cnn: bool = True,
                 log: Callable[[str], None] | None = None) -> dict:
    t0 = time.time()
    job_dir = Path(job_dir)
    log = log or Logger(job_dir / "运行日志.txt")
    state = load_state(job_dir)
    if not state:
        raise FileNotFoundError(f"{job_dir} 下没有任务状态，请先调用 seismic_read")
    cfg = merge_cfg(DEFAULT_CFG, cfg)
    cfg["allow_cnn"] = bool(allow_cnn)
    if max_attempts:
        cfg["max_attempts"] = int(max_attempts)
    if not state.get("source"):
        assess_flow(job_dir, cfg, log=log)
        state = load_state(job_dir)
    assessment = _assessment_from_state(state)

    if not assessment.needs_denoise:
        state["stage"] = "assess"
        state["passed"] = True
        _history(state, "denoise", "数据质量已较高，跳过去噪")
        save_state(job_dir, state)
        return {
            "ok": True,
            "stage": "assess",
            "workdir": str(job_dir),
            "passed": True,
            "skipped": True,
            "source": state["source"],
            "message": "数据质量已较高，无需去噪；可直接 seismic_finalize 整理成果。",
            "elapsed_s": round(time.time() - t0, 2),
        }

    ds = load_dataset(job_dir)
    ref = _load_reference(reference, ds, log) if reference else None

    from .methods.planner import plan, refine
    from .methods.runner import pick_best, run_attempts

    def _drop_risky_fx(items: list[MethodSpec]) -> list[MethodSpec]:
        """有坏道修复时，自动规划避开 F-X：修复边界容易被预测成垂直条带。"""
        if methods:
            return items
        repaired = state.get("dataset", {}).get("preprocess", {}).get("bad_traces_repaired") or []
        if not repaired:
            return items
        def _has_fx(spec) -> bool:
            if getattr(spec, "key", "") == "fx_decon":
                return True
            if getattr(spec, "key", "") == "combo":
                chain = (getattr(spec, "params", None) or {}).get("chain") or []
                return any(str(s.get("key")) == "fx_decon" for s in chain)
            return False

        kept = [s for s in items if not _has_fx(s)]
        if len(kept) != len(items):
            log("检测到坏道修复：自动规划跳过 F-X，避免修复边界产生垂直条带")
            state["plan_note"] = "检测到坏道修复，已跳过 F-X（避免垂直条带）"
        return kept

    if methods:
        specs = _user_specs(methods, cfg)
    else:
        specs = _drop_risky_fx(plan(assessment, cfg))
    if not allow_cnn:
        specs = [s for s in specs if s.key != "cnn"]

    seen: set[tuple] = set()
    results: list[MethodResult] = []
    rounds = 0
    max_total = int(cfg.get("max_attempts", 6))

    while specs and len(results) < max_total and rounds < 5:
        batch: list[MethodSpec] = []
        for s in specs:
            key = (s.key, json.dumps(to_jsonable(s.params), sort_keys=True, ensure_ascii=False))
            if key in seen:
                continue
            seen.add(key)
            batch.append(s)
            if len(results) + len(batch) >= max_total:
                break
        if not batch:
            break
        log(f"第 {rounds + 1} 轮：尝试 {', '.join(s.name for s in batch)}")
        got = run_attempts(ds, batch, cfg, reference=ref, log=log)
        for r in got:
            log(f"  {r.spec.name}: score={r.gate.score:.1f}, passed={r.gate.passed}, "
                f"{'；'.join(r.gate.failures[:2]) if r.gate.failures else '通过'}")
        results.extend(got)
        state["attempts"] = [r.to_info() for r in results]
        state["plan"] = [s.to_info() for s in (specs or [])]
        save_state(job_dir, state)

        best_now = pick_best(results)
        if best_now and best_now.gate.passed:
            break
        rounds += 1
        if methods:
            specs = []
        else:
            specs = _drop_risky_fx(refine(assessment, results, cfg))
            if not allow_cnn:
                specs = [s for s in specs if s.key != "cnn"]

    if not results:
        return {
            "ok": False,
            "stage": "denoise",
            "workdir": str(job_dir),
            "passed": False,
            "message": "没有可执行的方法（可能方法名无效或依赖缺失）。",
            "elapsed_s": round(time.time() - t0, 2),
        }

    best = pick_best(results)
    assert best is not None
    _save_attempts(job_dir, ds, results, log)
    # 保存尝试后 output_path 才回写，这里同步进状态，避免清单里缺 SEG-Y 路径
    state["attempts"] = [r.to_info() for r in results]
    arts = _save_quality(job_dir, ds, best, results, log)
    state["best"] = best.to_info()
    state["gate"] = best.gate.to_info()
    state["confidence"] = round(float(best.gate.confidence), 4)
    state["passed"] = bool(best.gate.passed)
    state["stage"] = "denoise"
    state["outputs"]["denoised"] = best.output_path
    state["artifacts"] = {**state.get("artifacts", {}), **arts}
    _history(state, "denoise", f"最佳方案：{best.spec.name}，通过={best.gate.passed}")
    save_state(job_dir, state)

    if best.gate.passed:
        msg = (
            f"已通过行业门禁：{best.spec.name}，信噪比增益 {best.metrics.snr_gain_db}dB，"
            f"置信度 {best.gate.confidence:.0%}。可调用 seismic_finalize 生成 Word 报告。"
        )
    else:
        msg = (
            f"未通过行业门禁：最佳方案 {best.spec.name}，"
            f"失败项：{'；'.join(best.gate.failures[:3])}。"
            "按需求不能进入整理阶段；可换方法/参数重试，或 force 强制整理（会明确标注未达标）。"
        )
    log(msg)
    return {
        "ok": True,
        "stage": "denoise",
        "workdir": str(job_dir),
        "passed": bool(best.gate.passed),
        "best": best.to_info(),
        "attempts": [r.to_info() for r in results],
        "gate": best.gate.to_info(),
        "confidence": state["confidence"],
        "artifacts": arts,
        "output_segy": best.output_path,
        "message": msg,
        "elapsed_s": round(time.time() - t0, 2),
    }


def finalize_flow(job_dir: str | Path, force: bool = False,
                  log: Callable[[str], None] | None = None) -> dict:
    job_dir = Path(job_dir)
    log = log or Logger(job_dir / "运行日志.txt")
    state = load_state(job_dir)
    if not state:
        raise FileNotFoundError(f"{job_dir} 下没有任务状态")
    clean = bool(state.get("source") and not state["source"].get("needs_denoise", True))
    passed = state.get("passed") is True or clean
    if not passed and not force:
        return {
            "ok": False,
            "stage": "finalize",
            "workdir": str(job_dir),
            "passed": False,
            "message": "去噪结果未通过行业门禁，按需求不进入整理阶段；如确认接受可 force=true。",
        }
    if not passed and force:
        log("警告：强制整理未达标结果，报告中会醒目标注")

    from .reporting.docx_report import build_report
    from .reporting.package import write_manifest, write_readme

    state["stage"] = "done"
    manifest = write_manifest(job_dir, state)
    readme = write_readme(job_dir, state)
    report = build_report(job_dir, state)
    state["outputs"]["folder"] = str(job_dir)
    state["outputs"]["report"] = str(report)
    state["outputs"]["manifest"] = str(manifest)
    state["outputs"]["readme"] = str(readme)
    if not passed:
        state["warnings"].append("强制整理：未通过行业门禁")
    _history(state, "finalize", f"报告：{report}")
    save_state(job_dir, state)
    log(f"整理完成：{report}")
    return {
        "ok": True,
        "stage": "done",
        "workdir": str(job_dir),
        "passed": passed,
        "report": str(report),
        "folder": str(job_dir),
        "manifest": str(manifest),
        "readme": str(readme),
        "confidence": state.get("confidence", 0.0),
        "message": ("成果已整理完成，Word 报告：" if passed else "未达标但已强制整理，Word 报告：") + str(report),
    }


def auto_flow(input_path: str | Path, workdir: str | Path | None = None,
              root: str | Path | None = None, cfg: dict | None = None,
              reference: str | Path | None = None,
              methods: str | list[str] | None = None,
              max_attempts: int | None = None,
              allow_cnn: bool = True,
              force: bool = False,
              log: Callable[[str], None] | None = None) -> dict:
    t0 = time.time()
    input_path = Path(input_path)
    job_dir = Path(workdir) if workdir else create_job_folder(input_path, root)
    log = log or Logger(job_dir / "运行日志.txt")
    log(f"全流程开始：{input_path.name}")
    read = inspect_flow(input_path, job_dir, max_traces=None, log=log)
    assess = assess_flow(job_dir, cfg, log=log)
    state = load_state(job_dir) or {}
    # 有纯净参考且输入与参考几乎一致时，说明这份数据本身就是干净参考；
    # 不应因为盲 SNR 估计器把相干强能量当“高 SNR”而误判后继续去噪。
    if reference and assess.get("needs_denoise"):
        ds = load_dataset(job_dir)
        if ds is not None and _reference_close(ds, reference, log):
            src = state.get("source") or assess.get("source") or {}
            src["needs_denoise"] = False
            src["clean_reasons"] = ["输入与提供的纯净参考几乎一致（参考一致性检查）"]
            src["cleanliness"] = dict(src.get("cleanliness") or {})
            src["cleanliness"]["reference_match"] = True
            state["source"] = src
            state["needs_denoise"] = False
            save_state(job_dir, state)
            assess["needs_denoise"] = False
            assess["source"] = src
            assess["message"] = "输入与纯净参考几乎一致，判定为无需去噪。"
    if not assess["needs_denoise"]:
        fin = finalize_flow(job_dir, force=False, log=log)
        return {
            "ok": bool(fin.get("ok")),
            "mode": "clean",
            "workdir": str(job_dir),
            "read": read,
            "assess": assess,
            "finalize": fin,
            "passed": fin.get("passed", True),
            "report": fin.get("report"),
            "folder": fin.get("folder", str(job_dir)),
            "message": "数据质量已较高，未做去噪；已整理读取/诊断/来源匹配成果与报告。",
            "elapsed_s": round(time.time() - t0, 2),
        }
    den = denoise_flow(job_dir, cfg=cfg, reference=reference,
                       methods=methods, max_attempts=max_attempts,
                       allow_cnn=allow_cnn, log=log)
    if not den.get("passed"):
        return {
            "ok": True,
            "mode": "denoise",
            "workdir": str(job_dir),
            "read": read,
            "assess": assess,
            "denoise": den,
            "finalize": None,
            "passed": False,
            "message": "未通过行业门禁，未生成验收报告。可调整方法后重试，或 force=true 强制整理。",
            "elapsed_s": round(time.time() - t0, 2),
        }
    fin = finalize_flow(job_dir, force=force, log=log)
    return {
        "ok": bool(fin.get("ok")),
        "mode": "denoise",
        "workdir": str(job_dir),
        "read": read,
        "assess": assess,
        "denoise": den,
        "finalize": fin,
        "passed": bool(fin.get("passed")),
        "report": fin.get("report"),
        "folder": str(job_dir),
        "message": fin.get("message", "完成"),
        "elapsed_s": round(time.time() - t0, 2),
    }
