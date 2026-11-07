# 地震数据智能去噪（dsh-seismic-denoise）

GitHub: <https://github.com/YH-QSH/dsh-seismic-denoise>

给 DSH 加一套「地震勘探数据去噪工作流」：读取任意常见地震数据 → 判断来源与噪声 → 自主选方法 → 去噪 → 质量门禁 → 整理成果 + Word 报告。

## 安装

```powershell
# 在 DSH 里执行（plugin_manager 工具）：
# action=install_bundle, target=<你本机克隆目录的绝对路径>\dsh-seismic-denoise
```

安装后工具会出现在 DSH 工具列表里；不需要改 profile 的 `package.json` / `cordis.patch.yml`。

## 工具

| 工具 | 作用 |
| --- | --- |
| `seismic_auto` | 一键全流程（推荐） |
| `seismic_read` | 读取文件、解析道头、生成总览/频谱/F-K/道头图 |
| `seismic_assess` | 来源匹配、噪声诊断、判断是否需要去噪 |
| `seismic_denoise` | 规划并执行去噪，做 SNR/能量/泄漏/谱相关门禁，不达标自动重试 |
| `seismic_finalize` | 整理文件、写清单/说明/Word 报告 |
| `seismic_methods` | 列可用方法、门禁阈值、CNN/GPU 状态 |

典型对话：

> 帮我把 `C:\data\line1.sgy` 去噪，出个报告。

AI 会调用 `seismic_auto`，最后给出成果文件夹和 Word 报告路径。

如果数据本身已经足够干净（盲信噪比高、噪声打分低），流程会在来源匹配后直接结束，不强行去噪，并同样整理成果和报告；报告会写“无需去噪”。

## 支持格式

- SEG-Y / SGY / SU（自动判别 IBM float、int32、int16、IEEE float、int8，大小端自动）
- MAT（嵌套 `data` / `header` / `trace_headers` 结构也支持）
- NPY / NPZ
- HDF5 / HDF5（自动找二维数据）
- RAW / DAT（需要能推断采样点数和道数，失败会给出提示）
- 目录：自动选其中最大的支持文件

道头会按 SEG-Y 标准 1-based 字节位置解析，并额外给出统计、唯一值、首尾值；识别不出来时仍会保留原始道头字节表。

## 成果文件夹

每次任务生成：

```
<输入名>_地震去噪_<时间>/
  01_原始数据/
  02_读取与诊断/     数据集信息.json、道头表.csv、数据总览/频谱/F-K/道头图
  03_去噪结果/       每次尝试的 SEG-Y + 前后对比图/去噪差分
  04_质量评价/       质量评价.json、指标对比/局部放大/视觉对比（弱信号保留）
  05_报告/           Word 报告、文件清单、说明
  任务状态.json       全流程状态
  运行日志.txt
```

## 高保真机制

去噪不是“越狠越好”。插件在方法输出后做一层**弱信号保护**：

- 用局部包络找输入里的弱能量区，只对“弱且横向相干”的样点回注原始残差；
- 对回注的弱信号再轻推 5%，强面波/强直达波区域不动；
- 同时生成 `04_质量评价/视觉对比.png`（输入/输出/残差 + SSIM、边缘相关、弱区能量保留、视觉综合分）；
- 门禁要求弱区能量保留 ≥ 85%，能量保留 60%–105%；不够就换方法/降强度重跑。

报告会内嵌这张视觉图；工具提示也要求 AI 交付前必须看图，不能只看 SNR。

## 质量门禁

默认阈值（可在 `engine/pipeline.py` 的 `DEFAULT_CFG` 或调用参数里改）：

- SNR 增益 ≥ 3 dB（盲 SNR 不可靠时改为软约束，报告会注明）
- 输出 SNR ≥ 10 dB（仅可靠时硬判）
- 能量保留 60%–105%
- 信号泄漏 ≤ 0.40
- 谱相关 ≥ 0.80
- 弱区能量保留 ≥ 85%（视觉保真）
- 单道尖峰/垂直条带 ≤ 250×中位残差（横向中值残差；明显伪影拦下）
- 有坏道修复时，自动规划跳过 F-X（避免修复边界产生垂直条带；手动指定 `methods=fx_decon` 仍可执行）
- 削波比例 ≤ 1e-3

未通过不会进入整理阶段（符合「达标才下一步」的要求）；`force=true` 可强制整理，报告会醒目标注。

## 依赖

- Python 3.12（默认用 DSH 运行时 Python）
- numpy / scipy / matplotlib / python-docx
- 可选：obspy、segyio、h5py、PyWavelets（装了会启用更多格式/方法）
- CNN 权重已随插件放在 `engine/assets/cnnv2/权重/`；torch 可用时启用 GPU

插件默认 Python 路径（按当前机器的 DSH 主目录自动拼接，不写死用户名）：

```
%DSH_HOME%\dsh-runtimes\dsh-primary-runtime\dependencies\python\python.exe
```

换机器时在插件配置里改 `pythonPath`，或设环境变量 `SEISMIC_PYTHON`。

## 安全与隐私

- 仓库不包含真实地震数据；示例/测试数据由脚本现场合成，产物写到 `tests/_out/`（已 gitignore）。
- 插件在本机运行，不主动联网上传数据；所有结果写在用户指定的成果文件夹。
- 不读取、不写入 API key / token / 账号凭据；唯一外部路径配置是 Python 解释器（`pythonPath` 或 `SEISMIC_PYTHON`）。
- CNN 权重随仓库分发（约 7.7 MB），不含个人路径字符串；如需更小仓库可改为 Release 附件。

## 说明

- 去噪不是「越狠越好」：默认限制能量保留，避免把有效信号一起抹掉。
- CNN 只作为候选方案之一，是否采用由门禁分数决定。
- 报告中的数字全部来自实际计算；没有参考纯净数据时，SNR 是盲估计，会在报告里注明。

## 测试

pytest 未安装时可直接用 Python 跑：

```powershell
python tests/test_formats.py
python tests/test_methods.py
python tests/test_quality.py
python tests/test_reporting.py
```

真实数据快速验证（把路径换成你本机的数据；也可用 `SEISMIC_REAL_DATA_DIR` / `SEISMIC_REAL_DATA_FILE`）：

```powershell
python -m engine.cli auto --input "真实地震数据\whataroa_0_hDVS_2239.segy" --root "_验收"
```
