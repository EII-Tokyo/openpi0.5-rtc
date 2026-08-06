import { useMemo, useState } from 'react'

import {
  captureAndSolveWorldOrigin,
  captureBottleTrial,
  captureTableSnapshot,
  exportCalibrationBundle,
  freezeBottleContract,
  freezeFactoryIntrinsics,
  freezeTableContract,
  runPreflightSession,
  solveTableRegistration,
  validateBottleTrials,
} from './api'
import type {
  BottleCaptureResult,
  BottleValidationResult,
  ExportResult,
  FactorySnapshotBundle,
  FrozenBottleContract,
  FrozenTableContract,
  PreflightSession,
  TableResult,
  TableSnapshot,
  TransformRecord,
  WorldOriginResult,
} from './api'

export type AppMode = 'preview' | 'live'
type Workspace = 'Guide' | 'Table Dots' | 'Bottle' | 'Export'
type TrialId = 'B-A' | 'B-B' | 'B-C'

const cameraRoles = [
  ['cam_high', '固定俯视 · ACTIVE'],
  ['cam_low', '固定低位 · FACTORY ONLY'],
  ['wrist_left', '左腕 · FACTORY ONLY'],
  ['wrist_right', '右腕 · FACTORY ONLY'],
] as const

const identity = (source: string, target: string): TransformRecord => ({
  source_frame: source,
  target_frame: target,
  matrix: [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
  length_unit: 'meter',
  matrix_order: 'row-major',
  vector_convention: 'column-vector',
  quaternion_order: 'wxyz',
})

const taskFromAsset: TransformRecord = {
  ...identity('bottle_asset', 'bottle_task'),
  matrix: [[0, 0, 1, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
}

interface DotForm {
  id: string
  color: 'blue' | 'magenta' | 'lime'
  partition: 'SOLVE' | 'HELD_OUT'
  x1: number
  y1: number
  x2: number
  y2: number
  u: string
  v: string
  confirmed: boolean
}

const heldOut = new Set(['P11', 'P23', 'P32'])
const initialDots: DotForm[] = [0.18, 0, -0.18].flatMap((y, row) =>
  [-0.35, 0, 0.35].map((x, column) => {
    const id = `P${row + 1}${column + 1}`
    return {
      id,
      color: (['blue', 'magenta', 'lime'] as const)[row],
      partition: heldOut.has(id) ? 'HELD_OUT' : 'SOLVE',
      x1: x,
      y1: y,
      x2: x,
      y2: y,
      u: '',
      v: '',
      confirmed: false,
    }
  }),
)

function eulerTransform(
  tx: number,
  ty: number,
  tz: number,
  rollDeg: number,
  pitchDeg: number,
  yawDeg: number,
): TransformRecord {
  const [roll, pitch, yaw] = [rollDeg, pitchDeg, yawDeg].map((value) => value * Math.PI / 180)
  const [cr, sr, cp, sp, cy, sy] = [Math.cos(roll), Math.sin(roll), Math.cos(pitch), Math.sin(pitch), Math.cos(yaw), Math.sin(yaw)]
  return {
    ...identity('bottle_task', 'tag'),
    matrix: [
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr, tx],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr, ty],
      [-sp, cp * sr, cp * cr, tz],
      [0, 0, 0, 1],
    ],
  }
}

function StageRail({ state }: { state: string }) {
  const stages = [
    ['预检与身份', 'PREFLIGHT_READY'],
    ['冻结出厂 K/D', 'FACTORY_INTRINSICS_FROZEN'],
    ['实验一 · 世界锚点', 'WORLD_ORIGIN_SOLVED'],
    ['实验二 · 9 点桌面', 'WORLD_REGISTRATION_VALIDATED'],
    ['实验三 · Bottle500', 'TAGGED_FIXTURE_TRANSFER_PASS'],
    ['独立 USD 导出', 'EXPORT_READY'],
  ]
  const current = Math.max(0, stages.findIndex((item) => item[1] === state))
  return <aside className="stage-rail"><div className="rail-heading"><span>CALIBRATION FLOW</span><strong>三实验</strong></div><ol>
    {stages.map(([title, gate], index) => <li className={index < current ? 'done' : index === current ? 'active' : 'locked'} key={gate}>
      <div className="stage-index">{String(index).padStart(2, '0')}</div>
      <div className="stage-copy"><strong>{title}</strong><span>{gate}</span><em>{index < current ? 'COMPLETE' : index === current ? 'CURRENT' : 'LOCKED'}</em></div>
    </li>)}
  </ol></aside>
}

function CameraWall({ factory }: { factory?: FactorySnapshotBundle }) {
  return <section className="camera-workspace"><div className="workspace-heading"><div><span>RGB ONLY · FACTORY K/D</span><h2>仅 camera_high 参与三个实验</h2></div></div>
    <div className="camera-grid">{cameraRoles.map(([role, label]) => {
      const snapshot = factory?.cameras.find((camera) => camera.role === role)
      return <article className={`camera-tile ${role === 'cam_high' ? 'active' : ''}`} data-testid={`camera-${role}`} key={role}>
        <header className="camera-titlebar"><div><strong>{role}</strong><span>{label}</span></div><div className="camera-state"><i />{snapshot ? `FROZEN · ${snapshot.serial}` : 'NOT FROZEN'}</div></header>
        <div className="synthetic-feed"><div className="feed-grid" /><div className="table-plane" /><div className="feed-meta"><span>{role === 'cam_high' ? 'APRILTAG / DOTS / BOTTLE' : 'NO EXPERIMENT STREAM'}</span><span>640×480@60 RGB8</span></div></div>
      </article>
    })}</div>
  </section>
}

function Guide({
  mode, session, factory, world, busy, error, onPreflight, onFactory, onWorld,
}: {
  mode: AppMode
  session?: PreflightSession
  factory?: FactorySnapshotBundle
  world?: WorldOriginResult
  busy: string | null
  error: string | null
  onPreflight: () => void
  onFactory: () => void
  onWorld: (tagSize: number, tagHeight: number) => void
}) {
  const [tagSizeMm, setTagSizeMm] = useState(80)
  const [tagHeightMm, setTagHeightMm] = useState(3)
  return <aside className="instruction-panel">
    <section className="instruction-card primary"><div className="instruction-number">01</div><div><h2>实验一 · 世界锚点</h2><p>将 AprilTag ID0 中心对准桌面原点，印刷 +X/+Y 对准真实桌面 +X/+Y。</p><ul><li>标签贴在刚性平板上</li><li>输入黑边实测尺寸和印刷面高度</li><li>一次采 200 个不可覆盖 RGB 帧</li></ul></div></section>
    <section className="form-card"><label>Tag 黑边 / mm<input type="number" value={tagSizeMm} onChange={(event) => setTagSizeMm(Number(event.target.value))} /></label><label>印刷面高出桌面 / mm<input type="number" value={tagHeightMm} onChange={(event) => setTagHeightMm(Number(event.target.value))} /></label></section>
    {!session ? <button className="primary-action" disabled={mode === 'preview' || busy !== null} onClick={onPreflight}>{mode === 'preview' ? '运行只读预检（预览禁用）' : '运行只读预检'}</button> : null}
    {session && !factory ? <><strong className="gate-status">{session.state}</strong><button className="primary-action" disabled={busy !== null} onClick={onFactory}>冻结四台出厂 K/D</button></> : null}
    {factory && !world ? <><strong className="gate-status">{factory.status}</strong><button className="primary-action" disabled={busy !== null} onClick={() => onWorld(tagSizeMm / 1000, tagHeightMm / 1000)}>采集 200 帧并求解世界锚点</button></> : null}
    {world ? <section className="preflight-result ready"><strong>{world.status}</strong><span>{world.accepted_frames}/{world.total_frames} accepted</span><span>median {world.median_reprojection_rms_px.toFixed(3)} px</span><span>jitter {(world.translation_jitter_m * 1000).toFixed(2)} mm · {world.rotation_jitter_deg.toFixed(3)}°</span></section> : null}
    {busy ? <p className="action-hint">正在执行：{busy}</p> : null}{error ? <p className="preflight-error">{error}</p> : null}
  </aside>
}

function TableDots({
  enabled, contract, result, snapshot, busy, onSnapshot, onFreeze, onSolve,
}: {
  enabled: boolean
  contract?: FrozenTableContract
  result?: TableResult
  snapshot?: TableSnapshot & { url: string }
  busy: string | null
  onSnapshot: () => void
  onFreeze: (dots: DotForm[]) => void
  onSolve: (dots: DotForm[]) => void
}) {
  const [dots, setDots] = useState(initialDots)
  const [selectedDot, setSelectedDot] = useState('P11')
  const update = (id: string, patch: Partial<DotForm>) => setDots((current) => current.map((dot) => dot.id === id ? { ...dot, ...patch } : dot))
  const clickSnapshot = (event: React.MouseEvent<HTMLImageElement>) => {
    const image = event.currentTarget
    const rect = image.getBoundingClientRect()
    const u = (event.clientX - rect.left) * image.naturalWidth / rect.width
    const v = (event.clientY - rect.top) * image.naturalHeight / rect.height
    update(selectedDot, { u: u.toFixed(2), v: v.toFixed(2) })
  }
  return <section className="secondary-workspace"><header><span>EXPERIMENT 2 · TABLETOP XY</span><h2>9 个彩色圆点</h2><p>先冻结钢尺/直角尺测量，再输入图像中心；求解请求不能携带或修改物理真值。</p></header>
    <div className="validation-warning"><strong>范围</strong><span>范围：桌面平面 XY 交叉验证，不声明离面 Z 已验证</span></div>
    <section className="table-snapshot-card"><div><strong>同一张不可变证据图</strong><span>{snapshot ? `${snapshot.attemptId} · frame ${snapshot.frameNumber} · ${snapshot.imageSha256.slice(0, 12)}…` : '先让 9 个圆点全部可见，再拍摄'}</span><button className="primary-action" disabled={!enabled || busy !== null} onClick={onSnapshot}>{snapshot ? '拍摄新的不可覆盖 attempt' : '拍摄 camera_high 桌面快照'}</button></div>{snapshot ? <div className="clickable-snapshot"><img alt="camera_high 桌面圆点证据" src={snapshot.url} onClick={clickSnapshot} />{dots.filter((dot) => dot.u !== '' && dot.v !== '').map((dot) => <i key={dot.id} style={{ left: `${Number(dot.u) / 640 * 100}%`, top: `${Number(dot.v) / 480 * 100}%` }}><b>{dot.id}</b></i>)}</div> : null}</section>
    <div className="dot-table">{dots.map((dot) => <article className={`dot-row ${dot.partition.toLowerCase()}`} key={dot.id}>
      <button className={selectedDot === dot.id ? 'point-selector selected' : 'point-selector'} onClick={() => setSelectedDot(dot.id)}>{dot.id} · {dot.partition}</button><i className={`color-${dot.color}`} />
      {(['x1', 'y1', 'x2', 'y2'] as const).map((field) => <label key={field}>{field}<input type="number" step="0.001" value={dot[field]} onChange={(event) => update(dot.id, { [field]: Number(event.target.value) })} /></label>)}
      <label>u px<input value={dot.u} onChange={(event) => update(dot.id, { u: event.target.value })} /></label><label>v px<input value={dot.v} onChange={(event) => update(dot.id, { v: event.target.value })} /></label>
      <label className="confirm"><input type="checkbox" checked={dot.confirmed} onChange={(event) => update(dot.id, { confirmed: event.target.checked })} />中心已确认</label>
    </article>)}</div>
    {!contract ? <button className="primary-action" disabled={!enabled || busy !== null} onClick={() => onFreeze(dots)}>冻结 9 点测量 contract</button> : null}
    {contract && !result ? <><strong className="gate-status">{contract.status}</strong><button className="primary-action" disabled={busy !== null} onClick={() => onSolve(dots)}>运行或载入 6 点求解 + 3 点盲测</button></> : null}
    {result ? <section className="preflight-result ready"><strong>{result.status}</strong><span>{result.validation_scope}</span><span>held-out RMS {(result.held_out_rms_m * 1000).toFixed(2)} mm · max {(result.held_out_max_m * 1000).toFixed(2)} mm</span></section> : null}
  </section>
}

function Bottle({
  enabled, contract, captures, result, busy, onFreeze, onCapture, onValidate,
}: {
  enabled: boolean
  contract?: FrozenBottleContract
  captures: Partial<Record<TrialId, BottleCaptureResult>>
  result?: BottleValidationResult
  busy: string | null
  onFreeze: (input: unknown) => void
  onCapture: (trial: TrialId) => void
  onValidate: () => void
}) {
  const [lengthMm, setLengthMm] = useState(206)
  const [diameterMm, setDiameterMm] = useState(68)
  const [blockMm, setBlockMm] = useState(50)
  const [tagPose, setTagPose] = useState({ tx: 0, ty: 0, tz: 0, roll: 0, pitch: 0, yaw: 0 })
  const [repeatMm, setRepeatMm] = useState(0)
  const trials: Array<[TrialId, string]> = [
    ['B-A', 'P22 · 长轴 +X · 桌面支撑'],
    ['B-B', 'P23 · 长轴 +Y · 桌面支撑'],
    ['B-C', 'P11 · 长轴 -X · 已知垫块'],
  ]
  const freeze = () => onFreeze({
    fixture_id: 'bottle500-v-block-001', revision: 1,
    measured_length_m: lengthMm / 1000, measured_diameter_m: diameterMm / 1000,
    tag_from_bottle: eulerTransform(tagPose.tx / 1000, tagPose.ty / 1000, tagPose.tz / 1000, tagPose.roll, tagPose.pitch, tagPose.yaw),
    task_from_asset: taskFromAsset, block_height_m: blockMm / 1000,
    measurement_method: 'steel-ruler-square-and-rigid-stops', repeated_installation_delta_m: repeatMm / 1000,
  })
  return <section className="secondary-workspace"><header><span>EXPERIMENT 3 · TAGGED RIGID FIXTURE</span><h2>Bottle500 三位置传递</h2><p>expected pose 由冻结圆点与夹具 contract 在服务端产生；浏览器不能提交 expected pose。</p></header>
    <div className="validation-warning"><strong>能力边界</strong><span>不代表无标签透明瓶识别、碰撞或动力学通过</span></div>
    <div className="form-card bottle-form"><label>瓶长 / mm<input type="number" value={lengthMm} onChange={(e) => setLengthMm(Number(e.target.value))} /></label><label>最大直径 / mm<input type="number" value={diameterMm} onChange={(e) => setDiameterMm(Number(e.target.value))} /></label><label>垫块 / mm<input type="number" value={blockMm} onChange={(e) => setBlockMm(Number(e.target.value))} /></label><label>重复安装差 / mm<input type="number" value={repeatMm} onChange={(e) => setRepeatMm(Number(e.target.value))} /></label>
      {(['tx', 'ty', 'tz', 'roll', 'pitch', 'yaw'] as const).map((field) => <label key={field}>{field}{field.length === 2 ? ' / mm' : ' / deg'}<input type="number" value={tagPose[field]} onChange={(event) => setTagPose((current) => ({ ...current, [field]: Number(event.target.value) }))} /></label>)}
    </div>
    {!contract ? <button className="primary-action" disabled={!enabled || busy !== null} onClick={freeze}>冻结瓶子夹具 contract</button> : null}
    {contract ? <div className="validation-grid">{trials.map(([trial, note]) => <article key={trial}><div className="validation-icon">{trial}</div><div><h3>{note}</h3><p>{captures[trial] ? `${captures[trial]?.stability.accepted_frames} frames · captured` : '按物理挡块摆放后采集'}</p></div><button disabled={busy !== null} onClick={() => onCapture(trial)}>{captures[trial] ? '重新采集新 attempt' : '采集 150 帧'}</button></article>)}</div> : null}
    {contract && !result ? <button className="primary-action" disabled={trials.some(([trial]) => !captures[trial]) || busy !== null} onClick={onValidate}>运行三位置独立验收</button> : null}
    {result ? <section className="preflight-result ready"><strong>{result.status}</strong><span>{result.claim_scope}</span><span>center RMS {(result.center_rms_m * 1000).toFixed(2)} mm · axis RMS {result.long_axis_rms_deg.toFixed(2)}°</span></section> : null}
    <strong className="gate-status">TAGGED_FIXTURE_TRANSFER_PASS</strong>
  </section>
}

function ExportPanel({ enabled, result, busy, onExport }: { enabled: boolean; result?: ExportResult; busy: string | null; onExport: () => void }) {
  return <section className="secondary-workspace"><header><span>USD · EXPLICIT COMPOSITION</span><h2>独立校准层与组合 Stage</h2><p>源 Stage 与 Bottle500 都先核验哈希；校准层强于冻结源层，源资产不被保存或改写。</p></header>
    <button className="primary-action" disabled={!enabled || busy !== null} onClick={onExport}>生成 calibration.usda 与 calibrated_review.usda</button>
    {result ? <section className="artifact-list large-card"><div className="artifact-row"><i /><div><strong>{result.calibration_json}</strong><span>列向量 JSON 与冻结 manifest</span></div><em>READY</em></div><div className="artifact-row"><i /><div><strong>{result.calibration_layer}</strong><span>CameraHigh + 三个 Bottle ghost prim</span></div><em>AUTHORED</em></div><div className="artifact-row"><i /><div><strong>{result.review_stage}</strong><span>等待 Isaac runtime readback</span></div><em>PENDING ISAAC</em></div></section> : null}
  </section>
}

export default function App({ mode }: { mode?: AppMode }) {
  const activeMode = mode ?? (import.meta.env.VITE_CALIBRATION_API_MODE === 'live' ? 'live' : 'preview')
  const [workspace, setWorkspace] = useState<Workspace>('Guide')
  const [session, setSession] = useState<PreflightSession>()
  const [factory, setFactory] = useState<FactorySnapshotBundle>()
  const [world, setWorld] = useState<WorldOriginResult>()
  const [tableContract, setTableContract] = useState<FrozenTableContract>()
  const [tableResult, setTableResult] = useState<TableResult>()
  const [tableSnapshot, setTableSnapshot] = useState<(TableSnapshot & { url: string })>()
  const [bottleContract, setBottleContract] = useState<FrozenBottleContract>()
  const [captures, setCaptures] = useState<Partial<Record<TrialId, BottleCaptureResult>>>({})
  const [bottleResult, setBottleResult] = useState<BottleValidationResult>()
  const [exportResult, setExportResult] = useState<ExportResult>()
  const [busy, setBusy] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const sessionId = session?.id
  const workflowState = exportResult ? 'EXPORT_READY' : bottleResult ? bottleResult.status : tableResult ? tableResult.status : world ? world.status : factory ? factory.status : session ? session.state : 'PREFLIGHT_READY'

  async function run<T>(label: string, action: () => Promise<T>, commit: (value: T) => void) {
    setBusy(label); setError(null)
    try { commit(await action()) } catch (caught) { setError(caught instanceof Error ? caught.message : String(caught)) } finally { setBusy(null) }
  }

  const dotsContractInput = (dots: DotForm[]) => ({ contract_id: 'table-dots-20260805', revision: 1, measurement_method: 'steel-ruler-and-square', points: dots.map((dot) => ({ id: dot.id, color: dot.color, measurement_1_xy_m: [dot.x1, dot.y1], measurement_2_xy_m: [dot.x2, dot.y2] })) })
  const dotObservationInput = (dots: DotForm[]) => ({ observations: dots.map((dot) => ({ id: dot.id, image_uv_px: [Number(dot.u), Number(dot.v)], operator_confirmed: dot.confirmed })) })
  const tabs: Workspace[] = ['Guide', 'Table Dots', 'Bottle', 'Export']
  const content = useMemo(() => {
    if (workspace === 'Table Dots') return <TableDots enabled={Boolean(world)} contract={tableContract} result={tableResult} snapshot={tableSnapshot} busy={busy} onSnapshot={() => sessionId && void run('拍摄桌面圆点快照', () => captureTableSnapshot(sessionId), (value) => setTableSnapshot((current) => { if (current) URL.revokeObjectURL(current.url); return { ...value, url: URL.createObjectURL(value.blob) } }))} onFreeze={(dots) => sessionId && void run('冻结桌面点', () => freezeTableContract(sessionId, dotsContractInput(dots)), setTableContract)} onSolve={(dots) => sessionId && void run('求解桌面外参', () => solveTableRegistration(sessionId, dotObservationInput(dots)), setTableResult)} />
    if (workspace === 'Bottle') return <Bottle enabled={Boolean(tableResult)} contract={bottleContract} captures={captures} result={bottleResult} busy={busy} onFreeze={(input) => sessionId && void run('冻结瓶子夹具', () => freezeBottleContract(sessionId, input), setBottleContract)} onCapture={(trial) => sessionId && void run(`采集 ${trial}`, () => captureBottleTrial(sessionId, trial, { tag_size_m: 0.080, frame_count: 150 }), (value) => setCaptures((current) => ({ ...current, [trial]: value })))} onValidate={() => sessionId && void run('验收瓶子传递', () => validateBottleTrials(sessionId), setBottleResult)} />
    if (workspace === 'Export') return <ExportPanel enabled={Boolean(bottleResult)} result={exportResult} busy={busy} onExport={() => sessionId && void run('导出 USD 标定包', () => exportCalibrationBundle(sessionId, {
      stage: { path: '/home/eii/project/openpi0.5-rtc-reward-learning/assets/Trossen/ALOHA1/1.0/diagnostics/table_support_alignment/1.0/aloha1_table_support_aligned_workcell.usda', sha256: '2b3f76365ed67532f478d995ae859a88b5639975ac07cb7ac8a53ac679e8205c' },
      bottle_asset_path: '/home/eii/project/openpi0.5-rtc-reward-learning/assets/bottle_500ml/isaac/bottle_500ml_sim.usd', bottle_asset_sha256: '16427135f152ec951de2321fd689366d745a2dd389cbe260976631783952533e', bottle_asset_prim: '/Bottle500',
    }), setExportResult)} />
    return <div className="preview-layout"><StageRail state={workflowState} /><CameraWall factory={factory} /><Guide mode={activeMode} session={session} factory={factory} world={world} busy={busy} error={error} onPreflight={() => void run('相机预检', runPreflightSession, setSession)} onFactory={() => sessionId && void run('冻结出厂内参', () => freezeFactoryIntrinsics(sessionId), setFactory)} onWorld={(tagSize, tagHeight) => sessionId && void run('世界锚点 200 帧', () => captureAndSolveWorldOrigin(sessionId, { tag_size_m: tagSize, tag_plane_height_m: tagHeight, frame_count: 200 }), setWorld)} /></div>
  }, [workspace, session, factory, world, tableContract, tableResult, tableSnapshot, bottleContract, captures, bottleResult, exportResult, busy, error, workflowState, activeMode, sessionId])

  return <main className="app-shell"><header className="app-header"><div className="brand-mark">A</div><div className="title-block"><span>ROBOTICS METROLOGY</span><h1>ALOHA 桌面与瓶子标定工作台</h1></div><div className={`preview-badge ${activeMode === 'live' ? 'live' : ''}`}><i />{activeMode === 'live' ? 'LIVE · GATED CAMERA ACCESS' : 'PREVIEW · 不执行设备操作'}</div><nav role="tablist" aria-label="工作区">{tabs.map((tab) => <button aria-selected={workspace === tab} className={workspace === tab ? 'active' : ''} key={tab} onClick={() => setWorkspace(tab)} role="tab">{tab}</button>)}</nav><div className="session-meta"><span>SESSION</span><strong>{sessionId ?? 'preview_001'}</strong></div></header><div className="workspace-container">{content}</div><section className="safety-banner"><div><span>DEPTH OFF · ROBOT API NONE</span></div><div><span>camera_high RGB ONLY</span></div><div><span>Isaac timeline: PAUSED</span></div><div><span>{workflowState}</span></div></section></main>
}
