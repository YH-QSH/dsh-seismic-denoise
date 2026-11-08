/**
 * 地震波勘探数据智能去噪 —— DSH Host 插件入口。
 *
 * 所有重活交给 Python 引擎（engine/cli.py）；这里只做：
 * 工具注册、参数校验、子进程调用、结果摘要渲染。
 */
import { spawn } from 'node:child_process'
import { existsSync, readFileSync, readdirSync, statSync } from 'node:fs'
import { homedir, tmpdir } from 'node:os'
import { extname, join, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const ENGINE_ROOT = fileURLToPath(new URL('..', import.meta.url))
const DSH_HOME = process.env.DSH_HOME || join(homedir(), '.dsh')
const DEFAULT_PYTHON = join(
  DSH_HOME,
  'dsh-runtimes',
  'dsh-primary-runtime',
  'dependencies',
  'python',
  process.platform === 'win32' ? 'python.exe' : 'python3',
)
const SUPPORTED_EXT = new Set([
  '.segy', '.sgy', '.seg-y', '.su', '.mat', '.npy', '.npz', '.h5', '.hdf5', '.dat', '.bin',
])

export const inject = ['tools', 'systemPrompt']

export const name = 'seismic-denoise'

function resolveInput(input) {
  const p = resolve(String(input))
  if (!existsSync(p)) throw new Error(`路径不存在：${p}`)
  const st = statSync(p)
  if (!st.isDirectory()) return { path: p, directory: null, candidates: [] }
  const candidates = readdirSync(p, { withFileTypes: true })
    .filter((d) => d.isFile() && SUPPORTED_EXT.has(extname(d.name).toLowerCase()))
    .map((d) => {
      const full = join(p, d.name)
      return { path: full, name: d.name, size: statSync(full).size }
    })
    .sort((a, b) => b.size - a.size)
  if (!candidates.length) throw new Error(`目录里没有可识别的地震数据文件：${p}`)
  return { path: candidates[0].path, directory: p, candidates }
}

function runPython(python, args, timeoutMs) {
  const jsonOut = join(tmpdir(), `seismic-${Date.now()}-${Math.random().toString(16).slice(2)}.json`)
  const fullArgs = ['-m', 'engine.cli', ...args, '--json-out', jsonOut]
  return new Promise((resolvePromise, reject) => {
    let stderr = ''
    let settled = false
    const child = spawn(python, fullArgs, {
      cwd: ENGINE_ROOT,
      windowsHide: true,
      env: { ...process.env, PYTHONIOENCODING: 'utf-8', PYTHONUTF8: '1' },
    })
    const timer = setTimeout(() => {
      if (!settled) {
        settled = true
        try { child.kill() } catch {}
        reject(new Error(`处理超时（${Math.round(timeoutMs / 1000)}s）。可先用 maxTraces 限制数据量再试。`))
      }
    }, timeoutMs)
    child.stdout.on('data', () => {})
    child.stderr.on('data', (d) => {
      stderr += String(d)
      if (stderr.length > 40000) stderr = stderr.slice(-40000)
    })
    child.on('error', (e) => {
      if (!settled) {
        settled = true
        clearTimeout(timer)
        reject(new Error(`无法启动 Python：${e.message}`))
      }
    })
    child.on('close', (code) => {
      if (settled) return
      settled = true
      clearTimeout(timer)
      let result = null
      if (existsSync(jsonOut)) {
        try { result = JSON.parse(readFileSync(jsonOut, 'utf8')) } catch {}
      }
      if (!result) {
        reject(new Error(`Python 引擎没有返回结果（退出码 ${code}）。\n${stderr.slice(-2500)}`))
        return
      }
      if (!result.ok) {
        reject(new Error(result.error || result.message || '引擎执行失败'))
        return
      }
      resolvePromise(result)
    })
  })
}

function text(s) {
  return [{ type: 'text', text: String(s) }]
}

function fmtNum(v, digits = 1, suffix = '') {
  return v === null || v === undefined || Number.isNaN(v) ? '未知' : `${Number(v).toFixed(digits)}${suffix}`
}

function renderRead(v) {
  const d = v.dataset || {}
  const lines = [
    `读取完成：${d.n_traces} 道 × ${d.n_samples} 点，格式 ${d.format}`,
    `采样 dt=${fmtNum(d.dt_ms, 4, 'ms')}，道间距 ${fmtNum(d.dz_m, 2, 'm')}，读取置信度 ${fmtNum((d.confidence || 0) * 100, 0, '%')}`,
    `成果文件夹：${v.workdir}`,
  ]
  if (d.warnings?.length) lines.push(`警告：${d.warnings.join('；')}`)
  const arts = Object.entries(v.artifacts || {})
  if (arts.length) lines.push(`已生成：${arts.map(([k, p]) => `${k}（${p}）`).join('；')}`)
  if (v.picked_from_directory) lines.push(`提示：输入是目录，已自动选最大的文件；候选 ${v.candidates?.length || 0} 个。`)
  lines.push('下一步：调用 seismic_assess（workdir 同上）做来源匹配与诊断。')
  return text(lines.join('\n'))
}

function renderAssess(v) {
  const s = v.source || {}
  const noise = Object.entries(s.noise_types || {}).sort((a, b) => b[1] - a[1])
  const lines = [
    `来源判断：${s.domain} / ${s.acquisition}（${s.geometry_type}）`,
    `置信度：${fmtNum((s.confidence || 0) * 100, 0, '%')}`,
    `是否需要去噪：${v.needs_denoise ? '是' : '否'}`,
    `噪声排序：${noise.slice(0, 4).map(([k, x]) => `${k} ${fmtNum(x * 100, 0, '%')}`).join('，')}`,
  ]
  if (s.clean_reasons?.length) lines.push(`干净依据：${s.clean_reasons.join('；')}`)
  if (s.recommended?.length) lines.push(`建议方法：${s.recommended.join('、')}`)
  if (s.evidence?.length) lines.push(`证据：${s.evidence.slice(0, 5).join('；')}`)
  if (s.warnings?.length) lines.push(`警告：${s.warnings.join('；')}`)
  lines.push(v.needs_denoise
    ? '下一步：调用 seismic_denoise（不填 methods 则自动规划）。'
    : '数据已较干净：直接告诉用户前期结论并结束，无需去噪。若仍要整理成果，调用 seismic_finalize。')
  return text(lines.join('\n'))
}

function renderPreview(v) {
  const s = v.source || {}
  const noise = Object.entries(s.noise_types || {}).sort((a, b) => b[1] - a[1])
  const lines = [
    '预检完成（尚未去噪）。',
    `来源：${s.domain || '未知'} / ${s.acquisition || '未知'}（${s.geometry_type || '未知'}）`,
    `来源置信度：${fmtNum((s.confidence || 0) * 100, 0, '%')}`,
    `是否需要去噪：${v.needs_denoise ? '是' : '否'}`,
    `主噪声：${noise.slice(0, 3).map(([k, x]) => `${k} ${fmtNum(x * 100, 0, '%')}`).join('，') || '未知'}`,
    `建议方法：${(s.recommended || []).join('、') || '自动规划'}`,
    `成果文件夹：${v.workdir}`,
  ]
  if (s.clean_reasons?.length) lines.push(`干净依据：${s.clean_reasons.join('；')}`)
  if (s.evidence?.length) lines.push(`证据：${s.evidence.slice(0, 4).join('；')}`)
  lines.push('请先询问用户：是否继续去噪/整理？确认前不要调用 seismic_auto 或 seismic_denoise。')
  return text(lines.join('\n'))
}

function renderDenoise(v) {
  if (v.skipped) return text(v.message || '数据无需去噪')
  const best = v.best || {}
  const m = best.metrics || {}
  const g = best.gate || {}
  const lines = [
    `${v.passed ? '✅ 通过行业门禁' : '❌ 未通过行业门禁'}：${best.method?.name || '无'}（${best.method?.key || ''}）`,
    `SNR：输入 ${fmtNum(m.snr_in_db, 2, 'dB')} → 输出 ${fmtNum(m.snr_out_db, 2, 'dB')}，增益 ${fmtNum(m.snr_gain_db, 2, 'dB')}`,
    `能量保留 ${fmtNum((m.energy_ratio || 0) * 100, 1, '%')}，信号泄漏 ${fmtNum(m.signal_leakage, 3)}，谱相关 ${fmtNum(m.spectrum_corr, 3)}`,
    `门禁分数 ${fmtNum(g.score, 1)}，置信度 ${fmtNum((g.confidence || 0) * 100, 0, '%')}`,
  ]
  if (g.failures?.length) lines.push(`失败项：${g.failures.join('；')}`)
  if (g.advice?.length) lines.push(`建议：${g.advice.join('；')}`)
  const attempts = v.attempts || []
  if (attempts.length) {
    lines.push('尝试记录：')
    for (const a of attempts) {
      lines.push(`- ${a.method?.name || a.method?.key}：score ${fmtNum(a.gate?.score, 1)}，${a.gate?.passed ? '通过' : '未通过'}，${fmtNum(a.runtime_s, 1, 's')}`)
    }
  }
  if (v.output_segy) lines.push(`去噪 SEG-Y：${v.output_segy}`)
  lines.push(`成果文件夹：${v.workdir}`)
  lines.push(v.passed
    ? '下一步：调用 seismic_finalize 整理文件并生成 Word 报告。'
    : '按需求不进入整理阶段。可换 methods/参数重试，或 force=true 强制整理（报告会标注未达标）。')
  return text(lines.join('\n'))
}

function renderFinalize(v) {
  return text([
    v.passed ? '成果已整理完成。' : '⚠️ 未达标结果被强制整理，报告已醒目标注。',
    `报告：${v.report}`,
    `成果文件夹：${v.folder}`,
    `文件清单：${v.manifest}`,
    `说明：${v.readme}`,
    `综合置信度：${fmtNum((v.confidence || 0) * 100, 0, '%')}`,
  ].join('\n'))
}

function renderAuto(v) {
  const d = v.denoise || {}
  const best = d.best || {}
  const lines = [
    `全流程完成：${v.mode === 'clean' ? '数据较干净，未去噪' : `去噪方案 ${best.method?.name || ''}`}`,
    `是否通过门禁：${v.passed ? '是' : '否'}`,
    `成果文件夹：${v.workdir}`,
  ]
  if (v.report) lines.push(`Word 报告：${v.report}`)
  if (v.passed && best.gate) {
    lines.push(`门禁分数 ${fmtNum(best.gate.score, 1)}，置信度 ${fmtNum((best.gate.confidence || 0) * 100, 0, '%')}`)
    lines.push(`SNR 增益 ${fmtNum(best.metrics?.snr_gain_db, 2, 'dB')}，能量保留 ${fmtNum((best.metrics?.energy_ratio || 0) * 100, 1, '%')}`)
  }
  if (!v.passed) lines.push(v.message || '未达标，未生成验收报告。')
  return text(lines.join('\n'))
}

function renderMethods(v) {
  const lines = ['可用去噪方法：']
  for (const [k, m] of Object.entries(v.methods || {})) {
    lines.push(`- ${k}：${m.name}；适用 ${ (m.targets || []).join('/') || '通用' }`)
  }
  lines.push(`CNN 可用：${v.cnn_available ? '是' : '否（权重或 torch 不可用）'}`)
  lines.push(`默认门禁：${JSON.stringify(v.gate_defaults)}`)
  return text(lines.join('\n'))
}

function renderJudge(v) {
  const e = v.evidence || {}
  const c = v.card || {}
  const uq = v.uq || {}
  const lines = [
    `AI 判断：${e.decision || '未知'}（综合置信度 ${fmtNum((e.confidence || 0) * 100, 0, '%')}）`,
    `模式：${v.mode === 'clean' ? '无需去噪' : '去噪'}；最佳方法：${e.best_method || v.best_method || '无'}`,
    `数据：${c.domain || '未知'} / ${c.acquisition || '未知'}；${c.ntr || '?'} 道 × ${c.ns || '?'} 点；主噪声 ${c.dominant_noise || '未知'}`,
  ]
  if (v.mode === 'clean') {
    lines.push(`理由：${(e.reasons || []).join('；') || '数据接近干净'}`)
  } else {
    lines.push(`门禁：${e.gate_passed ? '通过' : '未通过'}，分数 ${fmtNum(e.gate_score, 1)}`)
    if (e.reference_gain_db !== null && e.reference_gain_db !== undefined) {
      lines.push(`参考增益：${fmtNum(e.reference_gain_db, 2, 'dB')}`)
    } else {
      lines.push('参考增益：无（无纯净参考，只能有条件接受）')
    }
    if (e.energy_ratio !== null && e.energy_ratio !== undefined) {
      lines.push(`能量保留 ${fmtNum(e.energy_ratio * 100, 1, '%')}，泄漏 ${fmtNum(e.signal_leakage, 3)}，弱区 ${fmtNum(e.weak_signal_ratio, 3)}`)
    }
    if (uq.available) {
      lines.push(`UQ：n=${uq.n}，均值 ${fmtNum(uq.gain_mean_db, 2, 'dB')}，区间 [${fmtNum(uq.gain_ci_low, 2)}, ${fmtNum(uq.gain_ci_high, 2)}]，${uq.reliable ? '可靠' : '样本不足/离散度大'}`)
    } else {
      lines.push(`UQ：不可用（${(uq.notes || []).slice(0, 2).join('；') || '候选不足'}）`)
    }
  }
  if (e.flags?.length) lines.push(`证据标记：${e.flags.join('、')}`)
  if (e.reasons?.length) lines.push(`理由：${e.reasons.join('；')}`)
  lines.push('说明：judge 只做只读判断，不生成 Word 报告；报告仍由 seismic_finalize/seismic_auto 负责。')
  return text(lines.join('\n'))
}

function register(ctx, spec) {
  const output = {
    schema: { type: 'object', additionalProperties: true },
    // DSH 的 render 签名是 (args, value)；业务渲染函数只需要结果值。
    render: (_args, value) => spec.render(value),
  }
  ctx.tools.register({
    name: spec.name,
    description: spec.description,
    parameters: spec.parameters,
    output,
    execute: spec.execute,
    timeoutMs: spec.timeoutMs || 30 * 60 * 1000,
  })
}

export function apply(ctx, config) {
  const python = config?.pythonPath || process.env.SEISMIC_PYTHON || DEFAULT_PYTHON
  if (!existsSync(python)) {
    console.warn(`[seismic-denoise] Python 不存在：${python}；请改插件配置 pythonPath`)
  }

  const run = (args, timeoutMs) => runPython(python, args, timeoutMs)

  const ensureJob = async (args) => {
    if (args.workdir) return args.workdir
    if (!args.path) throw new Error('需要 workdir（seismic_read 的返回值），或提供 path 让插件先读取')
    const input = resolveInput(args.path)
    const res = await run(['inspect', '--input', input.path, ...(args.root ? ['--root', args.root] : []),
      ...(args.maxTraces ? ['--max-traces', String(args.maxTraces)] : [])], 10 * 60 * 1000)
    return res.workdir
  }

  register(ctx, {
    name: 'seismic_read',
    description: '读取地震勘探二进制/数据文件（SEG-Y/SGY/SU/MAT/NPY/NPZ/HDF5/RAW），自动解析道头与采样信息，生成数据总览、频谱、F-K、道头分析图，建立中文成果文件夹。返回 workdir，后续工具都用它。',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '输入文件路径，或包含数据文件的目录（自动选最大的支持文件）' },
        workdir: { type: 'string', description: '可选：指定成果文件夹；不填则自动创建' },
        root: { type: 'string', description: '可选：成果文件夹的父目录' },
        maxTraces: { type: 'number', description: '可选：最多读取道数，大文件调试用' },
      },
      required: ['path'],
      additionalProperties: false,
    },
    async execute(args) {
      const input = resolveInput(args.path)
      const out = await run(['inspect', '--input', input.path,
        ...(args.workdir ? ['--workdir', args.workdir] : []),
        ...(args.root ? ['--root', args.root] : []),
        ...(args.maxTraces ? ['--max-traces', String(args.maxTraces)] : [])], 10 * 60 * 1000)
      return { ...out, picked_from_directory: input.directory, candidates: input.candidates.map((c) => c.name) }
    },
    render: renderRead,
  })

  register(ctx, {
    name: 'seismic_assess',
    description: '对已读取的数据做来源匹配与诊断：判断 VSP/DAS-VSP/地面/海洋/被动源/实验室等领域，分析噪声类型（面波、随机、工频、散射、折射），估计信噪比，并判断数据是否已足够干净、是否需要去噪。不填 workdir 但给 path 时会先自动读取。',
    parameters: {
      type: 'object',
      properties: {
        workdir: { type: 'string', description: 'seismic_read 返回的成果文件夹' },
        path: { type: 'string', description: '可选：输入文件；没有 workdir 时用' },
        root: { type: 'string', description: '可选：成果文件夹父目录' },
      },
      required: [],
      additionalProperties: false,
    },
    async execute(args) {
      const workdir = await ensureJob(args)
      return run(['assess', '--workdir', workdir], 10 * 60 * 1000)
    },
    render: renderAssess,
  })

  register(ctx, {
    name: 'seismic_preview',
    description: '用户上传文件后的第一步：只读取数据并做来源匹配/诊断，返回数据来源、置信度、噪声排序、是否需要去噪和建议方法；不进入去噪、不生成报告，等待用户确认后再调用 seismic_auto/seismic_denoise。用于避免未经确认就消耗大量 token。',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '输入文件路径，或包含数据文件的目录' },
        workdir: { type: 'string', description: '可选：复用已有成果文件夹' },
        root: { type: 'string', description: '可选：成果文件夹父目录' },
        maxTraces: { type: 'number', description: '可选：最多读取道数（大文件快速预检）' },
      },
      required: ['path'],
      additionalProperties: false,
    },
    async execute(args) {
      const input = resolveInput(args.path)
      return run(['preview', '--input', input.path,
        ...(args.workdir ? ['--workdir', args.workdir] : []),
        ...(args.root ? ['--root', args.root] : []),
        ...(args.maxTraces ? ['--max-traces', String(args.maxTraces)] : [])], 10 * 60 * 1000)
    },
    render: renderPreview,
  })

  register(ctx, {
    name: 'seismic_denoise',
    description: '根据来源匹配结果自主规划去噪方案（单方法或“先压主噪声、再处理残余”的组合链，如 FK→CNN、带通→中值），逐个执行并用 SNR、能量保留、信号泄漏、谱相关等指标做行业门禁；不达标自动换方法重试。返回每次尝试、最佳方案、门禁明细与置信度。',
    parameters: {
      type: 'object',
      properties: {
        workdir: { type: 'string', description: '成果文件夹' },
        path: { type: 'string', description: '可选：输入文件；没有 workdir 时用' },
        reference: { type: 'string', description: '可选：纯净参考数据路径' },
        methods: { type: 'string', description: '可选：逗号分隔的方法 key（fk,cnn,bandpass...）；不填则自动规划' },
        maxAttempts: { type: 'number', description: '最多尝试次数，默认 6' },
        allowCnn: { type: 'boolean', description: '是否允许 CNN 方案，默认 true' },
        targetSnr: { type: 'number', description: '可选：输出 SNR 门槛 dB' },
        alpha: { type: 'number', description: '可选：CNN 混合强度 0–1' },
        device: { type: 'string', enum: ['auto', 'cpu', 'cuda'], description: 'CNN 设备' },
        force: { type: 'boolean', description: '未达标也整理（仅 seismic_finalize 用）' },
      },
      required: [],
      additionalProperties: false,
    },
    async execute(args) {
      const workdir = await ensureJob(args)
      return run(['denoise', '--workdir', workdir,
        ...(args.reference ? ['--reference', args.reference] : []),
        ...(args.methods ? ['--methods', args.methods] : []),
        ...(args.maxAttempts ? ['--max-attempts', String(args.maxAttempts)] : []),
        ...(args.allowCnn === false ? ['--no-cnn'] : []),
        ...(args.targetSnr ? ['--target-snr', String(args.targetSnr)] : []),
        ...(args.alpha !== undefined ? ['--alpha', String(args.alpha)] : []),
        ...(args.device ? ['--device', args.device] : [])], 30 * 60 * 1000)
    },
    render: renderDenoise,
  })

  register(ctx, {
    name: 'seismic_finalize',
    description: '把读取、诊断、去噪、质量评价的所有文件整理进一个中文成果文件夹，生成文件清单、说明和 Word 报告（含过程、方法选择理由、指标、置信度）。默认只在门禁通过后执行；force=true 可强制整理未达标结果并在报告中醒目标注。',
    parameters: {
      type: 'object',
      properties: {
        workdir: { type: 'string', description: '成果文件夹' },
        force: { type: 'boolean', description: '未达标也强制整理，默认 false' },
      },
      required: ['workdir'],
      additionalProperties: false,
    },
    async execute(args) {
      return run(['finalize', '--workdir', args.workdir,
        ...(args.force ? ['--force'] : [])], 10 * 60 * 1000)
    },
    render: renderFinalize,
  })

  register(ctx, {
    name: 'seismic_auto',
    description: '一键完成地震数据智能去噪全流程：读取任意格式 → 来源匹配 → 干净数据直接结束/需去噪则自主规划 → 去噪 → 质量门禁（不达标自动重试）→ 整理文件并写 Word 报告。仅在用户已确认继续后调用；首次上传请先用 seismic_preview 展示来源与是否需要去噪并询问。',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '输入文件或目录' },
        workdir: { type: 'string', description: '可选：复用已有成果文件夹' },
        root: { type: 'string', description: '可选：成果文件夹父目录' },
        reference: { type: 'string', description: '可选：纯净参考数据' },
        methods: { type: 'string', description: '可选：逗号分隔的方法 key' },
        maxAttempts: { type: 'number', description: '最多尝试次数，默认 6' },
        allowCnn: { type: 'boolean', description: '是否允许 CNN，默认 true' },
        targetSnr: { type: 'number', description: '可选：输出 SNR 门槛 dB' },
        alpha: { type: 'number', description: '可选：CNN 混合强度 0–1' },
        device: { type: 'string', enum: ['auto', 'cpu', 'cuda'], description: 'CNN 设备' },
        force: { type: 'boolean', description: '未达标也整理，默认 false' },
      },
      required: ['path'],
      additionalProperties: false,
    },
    async execute(args) {
      const input = resolveInput(args.path)
      const out = await run(['auto', '--input', input.path,
        ...(args.workdir ? ['--workdir', args.workdir] : []),
        ...(args.root ? ['--root', args.root] : []),
        ...(args.reference ? ['--reference', args.reference] : []),
        ...(args.methods ? ['--methods', args.methods] : []),
        ...(args.maxAttempts ? ['--max-attempts', String(args.maxAttempts)] : []),
        ...(args.allowCnn === false ? ['--no-cnn'] : []),
        ...(args.targetSnr ? ['--target-snr', String(args.targetSnr)] : []),
        ...(args.alpha !== undefined ? ['--alpha', String(args.alpha)] : []),
        ...(args.device ? ['--device', args.device] : []),
        ...(args.force ? ['--force'] : [])], 40 * 60 * 1000)
      return { ...out, picked_from_directory: input.directory, candidates: input.candidates.map((c) => c.name) }
    },
    render: renderAuto,
  })

  register(ctx, {
    name: 'seismic_methods',
    description: '列出去噪引擎当前可用的方法、适用噪声、默认门禁阈值，以及 CNN 权重/GPU 是否可用。在向用户解释“有哪些去噪手段”或手动指定 methods 前调用。',
    parameters: { type: 'object', properties: {}, additionalProperties: false },
    async execute() {
      return run(['methods'], 60 * 1000)
    },
    render: renderMethods,
  })

  register(ctx, {
    name: 'seismic_judge',
    description: '阶段四判断层：对一份地震数据（可带纯净参考）构建 CaseCard，只读调用去噪引擎生成候选，计算集成 UQ 与证据卡，返回 accept/conditional/reject/abstain 四档判断。用于验收前二次判断；不生成 Word 报告，不修改已有成果。',
    parameters: {
      type: 'object',
      properties: {
        path: { type: 'string', description: '输入文件或目录' },
        reference: { type: 'string', description: '可选：纯净参考数据路径' },
        maxMethods: { type: 'number', description: '最多尝试方法数，默认 4' },
        allowCnn: { type: 'boolean', description: '是否允许 CNN 候选，默认 false（更快）' },
      },
      required: ['path'],
      additionalProperties: false,
    },
    async execute(args) {
      const input = resolveInput(args.path)
      return run(['judge', '--input', input.path,
        ...(args.reference ? ['--reference', args.reference] : []),
        ...(args.maxMethods ? ['--max-methods', String(args.maxMethods)] : []),
        ...(args.allowCnn ? [] : ['--no-cnn'])], 30 * 60 * 1000)
    },
    render: renderJudge,
  })

  ctx.systemPrompt.section({
    name: 'seismic-denoise:guide',
    order: 150,
    text: [
      '地震数据去噪插件已安装。用户上传/给出地震文件后，第一步必须先调用 seismic_preview：只读取并做来源匹配，返回数据来源、置信度、是否需要去噪和建议方法；',
      '把预检结果告诉用户并询问“是否继续去噪/整理”，得到明确确认后才调用 seismic_auto 或 seismic_denoise。未经确认不要直接进入去噪或生成报告。',
      '需要逐步展示或调试时，按 seismic_read → seismic_assess → seismic_denoise → seismic_finalize 调用，workdir 用上一步返回值。',
      '规则：',
      '1. 不要凭空编造 SNR、道头、来源结论；所有数字必须来自工具返回。',
      '2. seismic_assess 返回 needs_denoise=false 时，直接总结读取结果与“无需去噪”的理由并结束，不要强行去噪。',
      '3. seismic_denoise 返回 passed=false 时，明确告诉用户未通过行业门禁、失败项和建议；不要调用 finalize 冒充成功（除非用户明确要求强制整理，用 force=true）。',
      '4. 交付时先给证据（passed、SNR 增益、能量保留、置信度、报告路径），再给结论；成果文件夹路径要原样给出。',
      '5. 工具返回的 artifacts 是图片路径；需要查看图像内容时用 read_image 读取这些路径。',
      '6. 高保真优先：去噪后必须看 04_质量评价/视觉对比.png 与 03_去噪结果/去噪后剖面.png，重点检查弱同相轴是否被削；若弱区能量保留低于 85% 或视觉上弱波消失，应降强度/换方法重跑，不要只看 SNR 通过就交付。',
      '7. 视觉对比图随去噪结果自动生成；报告里也会嵌入，交付时把该图路径一并给用户。',
      '8. 自动规划会尝试“单方法 + 多方法组合”（如 FK→CNN、带通→中值）；组合通过时报告里要写清组合链和总效果，不要把组合拆成单方法汇报。组合全部失败时按未通过门禁处理，不要为了通过而放宽阈值。',
      '9. 需要验收判断或用户问“结果可不可信”时，调用 seismic_judge（可带 reference）获取 accept/conditional/reject/abstain；judge 只读判断，不生成报告。',
      '10. judge 返回 reject/abstain 时不要宣称成功；conditional 时明确说明“无参考，需人工复核或补参考”；不要把 judge 的 decision 与阶段三门禁混为一谈。',
    ].join('\n'),
  })
}
