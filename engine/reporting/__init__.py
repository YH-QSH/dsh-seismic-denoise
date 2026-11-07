"""报告与成果打包层。

按 docs/接口约定.md §6 对外提供：
- figures：8 张报告图 + 统一样式；
- package：任务目录、原始数据副本、文件清单、中文说明；
- docx_report：完整 Word 报告。

只依赖 engine.model / engine.utils，不 import 分析层与流水线层。
"""
from __future__ import annotations

from .docx_report import build_report
from .figures import (
    cjk_available,
    fig_compare,
    fig_data_overview,
    fig_fk,
    fig_header_analysis,
    fig_local_zoom,
    fig_metrics,
    fig_residual,
    fig_source_assessment,
    fig_spectrum,
    generate_report_figures,
    setup_style,
)
from .package import (
    JOB_SUBDIRS,
    MANIFEST_NAME,
    README_NAME,
    REPORT_NAME,
    copy_original,
    create_job_folder,
    describe_file,
    write_manifest,
    write_readme,
)

__all__ = [
    # figures
    "setup_style",
    "fig_data_overview",
    "fig_header_analysis",
    "fig_spectrum",
    "fig_fk",
    "fig_source_assessment",
    "fig_compare",
    "fig_metrics",
    "fig_local_zoom",
    # figures 额外提供
    "fig_residual",
    "generate_report_figures",
    "cjk_available",
    # package
    "create_job_folder",
    "copy_original",
    "write_manifest",
    "write_readme",
    "describe_file",
    "JOB_SUBDIRS",
    "MANIFEST_NAME",
    "README_NAME",
    "REPORT_NAME",
    # docx
    "build_report",
]
