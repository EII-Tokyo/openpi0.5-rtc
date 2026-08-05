import { useState } from 'react'

import { runPreflightSession } from './api'
import type { PreflightSession } from './api'

type Workspace = 'Preview' | 'Dataset' | 'Solve' | 'Validate' | 'Export'
type CameraRole = 'cam_high' | 'cam_low' | 'wrist_left' | 'wrist_right'
export type AppMode = 'preview' | 'live'

type PreflightUiState =
  | { kind: 'idle' }
  | { kind: 'running' }
  | { kind: 'complete'; session: PreflightSession }
  | { kind: 'error'; message: string }

const workspaces: Workspace[] = ['Preview', 'Dataset', 'Solve', 'Validate', 'Export']

const stages = [
  { id: 0, title: '预检与相机身份', detail: '序列号 · 独占权 · 生产 Profile', state: 'ready' },
  { id: 1, title: '内参与正式 Profile', detail: '工厂 K/D · ChArUco 验证', state: 'locked' },
  { id: 2, title: '世界原点板与固定相机', detail: 'cam_high · cam_low 分别注册', state: 'locked' },
  { id: 3, title: '左腕 / 右腕手眼', detail: '两条运动链 · 两套数据集', state: 'locked' },
  { id: 4, title: '独立三维检查点', detail: '留出数据 · 至少两个高度', state: 'locked' },
  { id: 5, title: 'Isaac 叠加与带标夹具', detail: 'Session Layer · 三瓶传递', state: 'locked' },
] as const

const cameras: Array<{
  role: CameraRole
  label: string
  kind: string
  focus: string
  tint: string
}> = [
  { role: 'cam_high', label: 'cam_high', kind: '固定 · 俯视', focus: 'ACTIVE', tint: '#24a1ff' },
  { role: 'cam_low', label: 'cam_low', kind: '固定 · 低位', focus: 'STANDBY', tint: '#8aa0b6' },
  { role: 'wrist_left', label: 'wrist_left', kind: '左臂 · eye-in-hand', focus: 'STANDBY', tint: '#8aa0b6' },
  { role: 'wrist_right', label: 'wrist_right', kind: '右臂 · eye-in-hand', focus: 'STANDBY', tint: '#8aa0b6' },
]

const outcomes = [
  { label: 'INTRINSICS_VALIDATED', state: 'candidate', note: '示例状态' },
  { label: 'WORLD_REGISTRATION_VALIDATED', state: 'locked', note: '尚未求解' },
  { label: 'HAND_EYE_NONCONTACT_VALIDATED', state: 'locked', note: '尚未采集' },
  { label: 'ISAAC_IMPORT_PASS', state: 'locked', note: '尚未连接' },
  { label: 'TAGGED_FIXTURE_TRANSFER_PASS', state: 'locked', note: 'V1 最终演示' },
] as const

function TargetBoard({ compact = false }: { compact?: boolean }) {
  const cells = Array.from({ length: 35 }, (_, index) => index)
  return (
    <div className={`target-board ${compact ? 'compact' : ''}`} aria-label="模拟刚性多标记原点板">
      <div className="target-grid" aria-hidden="true">
        {cells.map((cell) => (
          <span className={(cell + Math.floor(cell / 5)) % 2 === 0 ? 'dark' : 'light'} key={cell}>
            {cell % 6 === 0 ? <i /> : null}
          </span>
        ))}
      </div>
      <span className="origin-dot">O</span>
      <span className="axis axis-x"><b />+X</span>
      <span className="axis axis-y"><b />+Y</span>
    </div>
  )
}

function CameraTile({ camera, index, preflight }: { camera: (typeof cameras)[number]; index: number; preflight: PreflightUiState }) {
  const liveCamera = preflight.kind === 'complete'
    ? preflight.session.latest_preflight.cameras.find((item) => item.role === camera.role)
    : undefined
  const cameraState = liveCamera
    ? liveCamera.connected && liveCamera.identity_match ? 'IDENTITY OK' : 'CHECK FAILED'
    : camera.focus
  return (
    <article
      className={`camera-tile ${camera.focus === 'ACTIVE' ? 'active' : ''}`}
      data-testid={`camera-${camera.role}`}
      style={{ '--camera-tint': camera.tint } as React.CSSProperties}
    >
      <header className="camera-titlebar">
        <div>
          <strong>{camera.label}</strong>
          <span>{camera.kind}</span>
        </div>
        <div className="camera-state"><i />{cameraState}</div>
      </header>
      <div className={`synthetic-feed feed-${index}`}>
        <div className="feed-grid" />
        <div className="arm arm-left"><span /><span /><span /></div>
        <div className="arm arm-right"><span /><span /><span /></div>
        <div className="table-plane" />
        <TargetBoard compact={index !== 0} />
        {index === 0 ? (
          <>
            <div className="coverage-heatmap" aria-label="模拟覆盖热图"><i /><i /><i /><i /></div>
            <div className="corner-points" aria-hidden="true">
              {Array.from({ length: 18 }, (_, point) => <i key={point} />)}
            </div>
          </>
        ) : null}
        <div className="feed-meta">
          <span>SYNTHETIC</span>
          <span>640 × 480 · 60</span>
        </div>
      </div>
    </article>
  )
}

function StageRail() {
  return (
    <aside className="stage-rail" aria-label="校准阶段">
      <div className="rail-heading">
        <span>CALIBRATION FLOW</span>
        <strong>六阶段</strong>
      </div>
      <ol>
        {stages.map((stage) => (
          <li className={stage.state} key={stage.id}>
            <div className="stage-index">{String(stage.id).padStart(2, '0')}</div>
            <div className="stage-copy">
              <strong>{stage.title}</strong>
              <span>{stage.detail}</span>
              <em>{stage.state === 'ready' ? 'READY · 等待确认' : 'LOCKED'}</em>
            </div>
          </li>
        ))}
      </ol>
      <div className="rail-note">
        <span className="pulse-dot" />
        Preview 使用模拟数据，不读取设备
      </div>
    </aside>
  )
}

function InstructionPanel({ mode, preflight, onRun }: { mode: AppMode; preflight: PreflightUiState; onRun: () => void }) {
  const passingCameras = preflight.kind === 'complete'
    ? preflight.session.latest_preflight.cameras.filter((camera) => (
      camera.connected && camera.identity_match && camera.production_profile_supported && camera.ownership === 'FREE'
    )).length
    : 0
  const actionLabel = mode === 'preview'
    ? '开始检测（预览模式）'
    : preflight.kind === 'running' ? '正在运行只读预检…' : '运行只读预检'
  return (
    <aside className="instruction-panel">
      <section className="instruction-card primary">
        <div className="instruction-number">01</div>
        <div>
          <h2>现在放什么 / 怎么摆</h2>
          <p>将刚性多标记原点板放在桌面中心定位止挡内。</p>
          <ul>
            <li>板上的 O 对准桌面中心</li>
            <li>+X / +Y 对准实体刻线</li>
            <li>保持平整、无翘曲、无遮挡</li>
          </ul>
        </div>
        <TargetBoard compact />
      </section>
      <section className="instruction-card">
        <div className="instruction-number">02</div>
        <div>
          <h2>系统现在检查什么</h2>
          <p>实时质量只决定当前帧是否可采，不代表标定已经成功。</p>
          <div className="check-list">
            <span><i className="good" />角点完整可见</span>
            <span><i className="good" />画面清晰、曝光稳定</span>
            <span><i />设备身份与 Profile</span>
            <span><i />采集服务独占权</span>
          </div>
        </div>
      </section>
      <section className="instruction-card">
        <div className="instruction-number">03</div>
        <div>
          <h2>通过后会得到什么</h2>
          <p>这里只完成设备身份预检，不产生“系统校准成功”。</p>
          <div className="artifact-preview">
            <span>camera_registry.json</span>
            <span>profile_snapshot.json</span>
            <span>preflight_report.html</span>
          </div>
        </div>
      </section>
      {preflight.kind === 'complete' ? (
        <section className={`preflight-result ${preflight.session.latest_preflight.status.toLowerCase()}`} aria-live="polite">
          <strong>{preflight.session.state}</strong>
          <span>{passingCameras} / 4 相机身份通过</span>
          {preflight.session.latest_preflight.issues.length > 0 ? (
            <div className="preflight-issues">
              {preflight.session.latest_preflight.issues.map((issue) => (
                <span className={issue.severity.toLowerCase()} key={`${issue.code}-${issue.camera_role ?? 'system'}`}>
                  {issue.code} · {issue.camera_role ?? 'system'}
                </span>
              ))}
            </div>
          ) : null}
          <em>session · {preflight.session.id}</em>
        </section>
      ) : null}
      {preflight.kind === 'error' ? <p className="preflight-error" role="alert">{preflight.message}</p> : null}
      <button
        className="primary-action"
        disabled={mode === 'preview' || preflight.kind === 'running' || preflight.kind === 'complete'}
        onClick={onRun}
      >{actionLabel}</button>
      <p className="action-hint">
        {mode === 'preview' ? '连接真实设备前，必须先确认此页面的操作顺序。' : '只枚举设备、Profile 与占用状态；不会启动图像 pipeline。'}
      </p>
    </aside>
  )
}

function SampleStrip() {
  const samples = [
    { id: 'S-001', state: 'accepted', reason: '覆盖新增' },
    { id: 'S-002', state: 'accepted', reason: '尺度变化' },
    { id: 'S-003', state: 'accepted', reason: '边缘视角' },
    { id: 'S-004', state: 'accepted', reason: '倾斜新增' },
    { id: 'S-005', state: 'rejected', reason: '运动模糊' },
    { id: 'S-006', state: 'rejected', reason: '角点遮挡' },
  ]
  return (
    <section className="sample-strip">
      <div className="sample-summary">
        <span>当前相机样本</span>
        <strong>4 accepted <em>/ 2 rejected</em></strong>
      </div>
      <div className="sample-list">
        {samples.map((sample, index) => (
          <article className={`sample ${sample.state}`} key={sample.id}>
            <div className={`mini-board angle-${index}`}><i /><i /><i /></div>
            <div><strong>{sample.id}</strong><span>{sample.reason}</span></div>
          </article>
        ))}
      </div>
      <div className="coverage-metrics">
        {[
          ['X 覆盖', '78%', 78],
          ['Y 覆盖', '71%', 71],
          ['Size', '64%', 64],
          ['Skew', '42%', 42],
          ['Rotation', '18%', 18],
        ].map(([label, value, width]) => (
          <div className="metric" key={label as string}>
            <span>{label}</span><strong>{value}</strong>
            <i><b style={{ width: `${width}%` }} /></i>
          </div>
        ))}
      </div>
    </section>
  )
}

function PreviewWorkspace({ mode, preflight, onRun }: { mode: AppMode; preflight: PreflightUiState; onRun: () => void }) {
  return (
    <div className="preview-layout">
      <StageRail />
      <section className="camera-workspace">
        <div className="workspace-heading">
          <div>
            <span>SIMULATED CAMERA WALL</span>
            <h2>模拟画面 <i>·</i> 当前活动：<strong>cam_high</strong></h2>
          </div>
          <div className="quality-legend"><i className="active" />活动 <i />待机</div>
        </div>
        <div className="camera-grid">
          {cameras.map((camera, index) => <CameraTile camera={camera} index={index} key={camera.role} preflight={preflight} />)}
        </div>
        <SampleStrip />
      </section>
      <InstructionPanel mode={mode} preflight={preflight} onRun={onRun} />
    </div>
  )
}

function DatasetWorkspace() {
  return (
    <WorkspaceFrame eyebrow="DATASET · IMMUTABLE RAW TAKE" title="数据集审查" description="逐帧查看接受与拒绝原因；求解数据和 held-out 数据在采集前分区。">
      <div className="dataset-toolbar">
        <div><span>数据集</span><strong>intrinsics_cam_high_preview_001</strong></div>
        <div><span>Profile</span><strong>640 × 480 @ 60 · RGB</strong></div>
        <div><span>来源</span><strong>MOCK · 无设备连接</strong></div>
        <button disabled>移除上一帧</button>
      </div>
      <div className="dataset-grid">
        {Array.from({ length: 12 }, (_, index) => (
          <article className={`dataset-sample ${index === 7 || index === 10 ? 'bad' : ''}`} key={index}>
            <div className="dataset-image"><div className={`mini-board angle-${index % 6}`}><i /><i /><i /></div></div>
            <div className="dataset-sample-title"><strong>S-{String(index + 1).padStart(3, '0')}</strong><span>{index < 8 ? 'SOLVE' : 'HELD-OUT'}</span></div>
            <p>{index === 7 ? '拒绝 · 角点遮挡' : index === 10 ? '拒绝 · 运动模糊' : '接受 · 新增覆盖'}</p>
          </article>
        ))}
      </div>
    </WorkspaceFrame>
  )
}

function SolveWorkspace() {
  return (
    <WorkspaceFrame eyebrow="SOLVE · REVIEW BEFORE VALIDATE" title="求解结果" description="候选解只能比较，不能通过人工拖动变换制造 PASS。">
      <div className="solve-layout">
        <section className="large-card">
          <div className="card-heading"><span>变换语义</span><em>MOCK RESULT</em></div>
          <div className="transform-diagram">
            <div><span>W</span><small>桌面世界坐标系</small></div>
            <i><b>T<sub>W</sub><sup>C-high</sup></b><small>parent: W · child: cam_high</small></i>
            <div><span>C</span><small>cam_high 光学坐标系</small></div>
          </div>
          <pre>{`units: meters\nquaternion: wxyz\nsolver: preview-placeholder\ninput_hash: unavailable`}</pre>
        </section>
        <section className="large-card candidate-table">
          <div className="card-heading"><span>候选解比较</span><em>不作为验收</em></div>
          <div className="table-row header"><span>候选</span><span>Median px</span><span>Held-out</span><span>状态</span></div>
          <div className="table-row selected"><span>Factory K/D</span><span>0.42*</span><span>未运行</span><span>待验证</span></div>
          <div className="table-row"><span>Software K/D</span><span>0.39*</span><span>未运行</span><span>仅比较</span></div>
          <p>* 示例数值，不是 NVIDIA 或项目验收门限。</p>
        </section>
      </div>
    </WorkspaceFrame>
  )
}

function ValidateWorkspace() {
  return (
    <WorkspaceFrame eyebrow="VALIDATE · NO REFIT ALLOWED" title="独立验证" description="冻结求解参数后，使用未参与拟合的图像、三维工装和重复会话检验。">
      <div className="validation-grid">
        {[
          ['Held-out 图像', '检查未参与求解的视角', 'WAITING'],
          ['三维检查工装', '覆盖桌面与离面高度', 'LOCKED'],
          ['跨相机一致性', '只作诊断，不作独立真值', 'LOCKED'],
          ['冷启动重复性', '至少三次独立会话', 'LOCKED'],
          ['反向接近重复', '暴露腕部背隙与柔顺误差', 'LOCKED'],
          ['TCP 指针触碰', '需要用户另行授权', 'AUTH REQUIRED'],
        ].map(([title, note, state], index) => (
          <article className={index === 0 ? 'waiting' : ''} key={title}>
            <div className="validation-icon">{String(index + 1).padStart(2, '0')}</div>
            <div><h3>{title}</h3><p>{note}</p></div>
            <span>{state}</span>
          </article>
        ))}
      </div>
      <div className="validation-warning"><strong>禁止循环证明</strong><span>同一标定板、同一标签尺寸或同一拟合数据不能同时充当独立真值。</span></div>
    </WorkspaceFrame>
  )
}

function ExportWorkspace() {
  return (
    <WorkspaceFrame eyebrow="EXPORT · EXPLICIT COMMIT" title="导出与提交" description="只有独立验收通过并绑定 AcceptancePolicy 后，才能生成正式校准包。">
      <div className="export-layout">
        <section className="large-card artifact-list">
          {[
            ['camera_registry.json', '设备身份与正式 Profile', 'DRAFT'],
            ['dataset_manifest.json', '原始数据、分区与哈希', 'MISSING'],
            ['transforms.yaml', '父子坐标系、单位与协方差', 'MISSING'],
            ['validation_report.html', '独立验证与重复性报告', 'MISSING'],
            ['camera_calibration.usda', 'Isaac 独立校准层', 'LOCKED'],
          ].map(([name, note, state]) => (
            <div className="artifact-row" key={name}><i /><div><strong>{name}</strong><span>{note}</span></div><em>{state}</em></div>
          ))}
        </section>
        <section className="commit-card">
          <span>COMMIT GATE</span>
          <h3>当前不能提交</h3>
          <p>Preview 没有原始数据、冻结变换、独立验证或批准的门限策略。</p>
          <div><i />SOLVED <i />VALIDATED <i />POLICY APPROVED</div>
          <button disabled>导出正式校准包</button>
        </section>
      </div>
    </WorkspaceFrame>
  )
}

function WorkspaceFrame({ eyebrow, title, description, children }: { eyebrow: string; title: string; description: string; children: React.ReactNode }) {
  return (
    <section className="secondary-workspace">
      <header><span>{eyebrow}</span><h2>{title}</h2><p>{description}</p></header>
      {children}
    </section>
  )
}

function SafetyBanner({ mode, preflight }: { mode: AppMode; preflight: PreflightUiState }) {
  let captureStatus = mode === 'preview' ? 'OFF / EXCLUSIVE' : 'NOT CHECKED'
  if (preflight.kind === 'running') captureStatus = 'CHECKING'
  if (preflight.kind === 'complete') captureStatus = preflight.session.latest_preflight.status === 'READY' ? 'FREE / EXCLUSIVE' : preflight.session.latest_preflight.status
  if (preflight.kind === 'error') captureStatus = 'ERROR'
  return (
    <section className="safety-banner" aria-label="系统所有权状态">
      <div><i className="host" /><span>101 本机浏览器</span></div>
      <div><i className="off" /><span>103 Capture: {captureStatus}</span></div>
      <div><i className="blocked" /><span>Robot command APIs: NONE</span></div>
      <div><i className="cloud" /><span>Isaac: DISCONNECTED</span></div>
      <div><i className="clock" /><span>Browser time: <strong>NOT USED</strong></span></div>
    </section>
  )
}

function OutcomeBar() {
  return (
    <section className="outcome-bar" aria-label="独立验收状态">
      {outcomes.map((outcome) => (
        <article className={outcome.state} key={outcome.label}>
          <i>{outcome.state === 'candidate' ? '◇' : '⌁'}</i>
          <div><strong>{outcome.label}</strong><span>{outcome.note}</span></div>
        </article>
      ))}
    </section>
  )
}

export default function App({ mode }: { mode?: AppMode }) {
  const [workspace, setWorkspace] = useState<Workspace>('Preview')
  const [preflight, setPreflight] = useState<PreflightUiState>({ kind: 'idle' })
  const activeMode: AppMode = mode ?? (import.meta.env.VITE_CALIBRATION_API_MODE === 'live' ? 'live' : 'preview')

  async function handleRunPreflight() {
    if (activeMode !== 'live' || preflight.kind !== 'idle') return
    setPreflight({ kind: 'running' })
    try {
      const session = await runPreflightSession()
      setPreflight({ kind: 'complete', session })
    } catch (error) {
      setPreflight({ kind: 'error', message: error instanceof Error ? error.message : 'Unknown preflight error' })
    }
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="brand-mark">A</div>
        <div className="title-block">
          <span>ROBOTICS METROLOGY</span>
          <h1>ALOHA 四相机标定工作台</h1>
        </div>
        <div className={`preview-badge ${activeMode === 'live' ? 'live' : ''}`}>
          <i />{activeMode === 'live' ? 'LIVE PREFLIGHT · READ ONLY' : 'PREVIEW · 不执行设备操作'}
        </div>
        <nav role="tablist" aria-label="工作区">
          {workspaces.map((item) => (
            <button
              aria-selected={workspace === item}
              className={workspace === item ? 'active' : ''}
              key={item}
              onClick={() => setWorkspace(item)}
              role="tab"
            >{item}</button>
          ))}
        </nav>
        <div className="session-meta"><span>SESSION</span><strong>preview_001</strong></div>
      </header>

      <div className="workspace-container">
        {workspace === 'Preview' ? <PreviewWorkspace mode={activeMode} preflight={preflight} onRun={handleRunPreflight} /> : null}
        {workspace === 'Dataset' ? <DatasetWorkspace /> : null}
        {workspace === 'Solve' ? <SolveWorkspace /> : null}
        {workspace === 'Validate' ? <ValidateWorkspace /> : null}
        {workspace === 'Export' ? <ExportWorkspace /> : null}
      </div>

      <SafetyBanner mode={activeMode} preflight={preflight} />
      <OutcomeBar />
    </main>
  )
}
