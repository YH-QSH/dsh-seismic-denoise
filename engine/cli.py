"""命令行入口：被 Node 插件以 `python -m engine.cli <stage> ...` 调用。

约定：stdout 始终输出一行 JSON；进度走 stderr；--json-out 再落一份文件。
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .model import to_jsonable
from .pipeline import (
    DEFAULT_CFG, assess_flow, auto_flow, denoise_flow, finalize_flow,
    inspect_flow, merge_cfg, preview_flow,
)
from .utils import Logger, read_json, write_json


def _common(p: argparse.ArgumentParser, input_required: bool = False) -> None:
    p.add_argument("--input", default=None, required=input_required, help="输入地震文件")
    p.add_argument("--root", default=None, help="成果文件夹的父目录（默认输入文件同目录）")
    p.add_argument("--json-out", default=None, help="结果 JSON 输出路径")
    p.add_argument("--config", default=None, help="JSON 配置文件，覆盖默认阈值/参数")
    p.add_argument("--max-traces", type=int, default=None, help="最多读取道数（大文件调试用）")


def _cfg(args) -> dict:
    cfg = merge_cfg(DEFAULT_CFG, read_json(args.config, {}) if args.config else None)
    if getattr(args, "target_snr", None):
        cfg.setdefault("gate", {})["snr_out_min_db"] = float(args.target_snr)
    if getattr(args, "clean_snr", None):
        cfg["clean_snr_db"] = float(args.clean_snr)
    if getattr(args, "alpha", None) is not None:
        cfg.setdefault("cnn", {})["alpha"] = float(args.alpha)
    if getattr(args, "device", None):
        cfg.setdefault("cnn", {})["device"] = args.device
    return cfg


def _emit(result: dict, json_out: str | None) -> int:
    clean = to_jsonable(result)
    text = json.dumps(clean, ensure_ascii=False, default=str)
    if json_out:
        write_json(json_out, clean)
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    print(text)
    return 0 if result.get("ok", True) else 1


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="seismic-denoise", description="地震数据智能去噪引擎")
    sub = ap.add_subparsers(dest="stage", required=True)

    p = sub.add_parser("inspect", help="读取文件并生成诊断图/道头表")
    _common(p, input_required=True)
    p.add_argument("--workdir", default=None, help="复用已有成果文件夹")

    p = sub.add_parser("assess", help="来源匹配与数据诊断")
    _common(p)
    p.add_argument("--workdir", required=True)

    p = sub.add_parser("preview", help="上传后先读取+来源匹配，返回来源/是否需去噪并等待确认")
    _common(p, input_required=True)
    p.add_argument("--workdir", default=None, help="复用已有成果文件夹")

    p = sub.add_parser("denoise", help="规划并执行去噪，做质量门禁")
    _common(p)
    p.add_argument("--workdir", required=True)
    p.add_argument("--reference", default=None, help="可选纯净参考数据")
    p.add_argument("--methods", default=None, help="逗号分隔，如 fk,cnn；不填则自动规划")
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--no-cnn", action="store_true", help="禁用 CNN 方案")
    p.add_argument("--target-snr", type=float, default=None, help="输出 SNR 门槛 dB")
    p.add_argument("--alpha", type=float, default=None, help="CNN 混合强度 0–1")
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])

    p = sub.add_parser("finalize", help="整理成果文件夹并生成 Word 报告")
    _common(p)
    p.add_argument("--workdir", required=True)
    p.add_argument("--force", action="store_true", help="未达标也强制整理（报告会标注）")

    p = sub.add_parser("auto", help="全流程：读取→匹配→去噪→门禁→报告")
    _common(p, input_required=True)
    p.add_argument("--workdir", default=None, help="复用已有成果文件夹")
    p.add_argument("--reference", default=None)
    p.add_argument("--methods", default=None)
    p.add_argument("--max-attempts", type=int, default=None)
    p.add_argument("--no-cnn", action="store_true")
    p.add_argument("--force", action="store_true")
    p.add_argument("--target-snr", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])

    p = sub.add_parser("methods", help="列出可用去噪方法与 CNN 状态")
    p.add_argument("--json-out", default=None)

    p = sub.add_parser("judge", help="阶段四判断层：CaseCard + UQ + 证据卡（只读调用去噪引擎）")
    _common(p, input_required=True)
    p.add_argument("--reference", default=None, help="可选纯净参考数据")
    p.add_argument("--max-methods", type=int, default=4, help="最多尝试方法数")
    p.add_argument("--no-cnn", action="store_true", help="禁用 CNN 候选")

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    stage = args.stage
    try:
        if stage == "judge":
            from .judgment.orchestrator import run_case
            case = run_case(
                args.input,
                args.reference,
                max_methods=args.max_methods,
                allow_cnn=not args.no_cnn,
            )
            result = {"ok": True, "stage": "judge", **case.to_dict()}
            return _emit(result, args.json_out)

        if stage == "methods":
            from .methods.classical import available_methods
            from .methods.cnn import cnn_available
            methods = available_methods()
            result = {
                "ok": True,
                "methods": methods,
                "cnn_available": bool(cnn_available()),
                "gate_defaults": DEFAULT_CFG["gate"],
                "note": "方法 key 可直接传给 seismic_denoise 的 methods 参数",
            }
            return _emit(result, args.json_out)

        if stage == "inspect":
            result = inspect_flow(args.input, args.workdir, args.root,
                                  args.max_traces, log=None)
        elif stage == "preview":
            result = preview_flow(args.input, args.workdir, args.root,
                                  args.max_traces, cfg=_cfg(args), log=None)
        elif stage == "assess":
            result = assess_flow(args.workdir, _cfg(args), log=None)
        elif stage == "denoise":
            result = denoise_flow(
                args.workdir, cfg=_cfg(args), reference=args.reference,
                methods=args.methods, max_attempts=args.max_attempts,
                allow_cnn=not args.no_cnn, log=None,
            )
        elif stage == "finalize":
            result = finalize_flow(args.workdir, force=args.force, log=None)
        elif stage == "auto":
            result = auto_flow(
                args.input, workdir=args.workdir, root=args.root, cfg=_cfg(args),
                reference=args.reference, methods=args.methods,
                max_attempts=args.max_attempts,
                allow_cnn=not args.no_cnn, force=args.force, log=None,
            )
        else:
            raise ValueError(f"未知阶段：{stage}")
        return _emit(result, args.json_out)
    except Exception as e:
        err = {
            "ok": False,
            "stage": stage,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
        }
        try:
            Logger(Path(args.workdir) / "运行日志.txt" if getattr(args, "workdir", None) else None)(err["error"])
        except Exception:
            pass
        return _emit(err, getattr(args, "json_out", None))


if __name__ == "__main__":
    raise SystemExit(main())
