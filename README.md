# 地震数据智能去噪（dsh-seismic-denoise）

GitHub: <https://github.com/YH-QSH/dsh-seismic-denoise>

版本：`0.1.1`

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

有纯净参考数据时，把参考路径一起传给 `seismic_auto`/`seismic_denoise`（工具参数 `reference`，或 CLI 的 `--reference`）：主流程会用它计算真实 SNR 增益、选择最佳方法，并拒绝“看起来通过、实际变差”的结果；输入与参考几乎一致时会直接判为无需去噪。

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

## 质量门禁（第一阶段基线）

默认阈值（可在 `engine/pipeline.py` 的 `DEFAULT_CFG` 或调用参数里改）：

- SNR 增益 ≥ 3 dB
- 输出 SNR ≥ 10 dB
- 能量保留 30%–105%
- 信号泄漏 ≤ 0.40
- 谱相关 ≥ 0.80
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
python tests/test_judgment.py
```

真实数据快速验证（把路径换成你本机的数据；也可用 `SEISMIC_REAL_DATA_DIR` / `SEISMIC_REAL_DATA_FILE`）：

```powershell
python -m engine.cli auto --input "真实地震数据\whataroa_0_hDVS_2239.segy" --root "_验收"
```

## 改动

### 0.1.1（2026-11-08，框架升级阶段）

- **新增 `seismic_judge` 判断层**：`engine/judgment/` 提供 `CaseCard`、集成 UQ、证据卡与四档决策 `accept / conditional / reject / abstain`；只读调用去噪引擎，不生成报告、不改已有成果。无参考时只能 `conditional`，有参考且门禁通过、参考增益达标、证据一致时才 `accept`。
- **高保真去噪**：去噪后对输入弱能量区中横向相干较高的样点回注原始残差，并轻推 5%；生成 `04_质量评价/视觉对比.png`（SSIM、边缘相关、弱区能量保留、视觉综合分）。门禁增加弱区能量保留 ≥ 85%，能量下限由 30% 提到 60%。
- **多方法组合链**：`engine/methods/combo.py` 顺序执行 `FK → 带通`、`FK → CNN`、`带通 → 中值` 等链；自动规划在单方法后追加最多 2 条组合，整条链只做一次弱信号保护/视觉评价/门禁。12 例模拟数据中，01 `FK→带通` +2.586 dB、11 组合 +2.860 dB、12 `带通→CNN` +3.304 dB；04 散射 / 05 折射因噪声与有效波速度带重叠、泄漏过大，保留为失败案例，不放宽阈值硬放行。
- **参考数据门禁**：有纯净参考时按 `reference_gain_db` 选最佳方法（不再只看盲 score）；参考增益 ≥ 0.5 dB、泄漏上限 0.45；输入与参考 corr≥0.995、相对差异≤0.02 直接判净。
- **坏道与伪影**：多轮修复整道/局部异常（全局 RMS、局部中位、横向残差），有坏道修复时自动规划跳过 F-X，避免修复边界产生垂直条带；门禁增加单道尖峰/垂直条带 ≤ 250× 中位残差检查。
- **自适应带通**：高频/超声数据按 Nyquist 自动调整带通范围，避免固定 5–90 Hz 把有效频带切掉。
- **判断层工具**：新增 `seismic_judge`，参数 `path`、`reference`、`maxMethods`、`allowCnn`；返回 `evidence.decision`、`best_method`、`uq`、`card` 与四档理由。
- **测试**：`tests/test_formats.py` 9/9、`tests/test_methods.py` 15/15、`tests/test_quality.py` 13/13、`tests/test_reporting.py` 通过、`tests/test_judgment.py` 6/6。

### 0.1.0（第一阶段基线）

- 六工具主流程：读取、来源匹配、单方法去噪、门禁、整理、方法列表。
- 支持 SEG-Y/MAT/NPY/NPZ/HDF5/RAW；中文成果文件夹；Word 报告；CNNv2 分类型噪声模型可选。

creative by Deepseek Harness
