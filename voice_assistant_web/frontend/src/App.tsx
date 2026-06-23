import { useEffect, useMemo, useState } from 'react'
import { CameraGrid } from './components/CameraGrid'
import { KeyRegionsPage } from './components/KeyRegionsPage'
import { RLTConfigPage } from './components/RLTConfigPage'
import { RLTConfigPanel, RLTControlPanel, RLTStatsPanel } from './components/RLTControlPanel'
import { RolloutBrowser } from './components/RolloutBrowser'
import { SystemPage } from './components/SystemPage'
import { AppLanguage, translations } from './i18n'
import { RLTControlState, sendRobotTask, wsBase } from './services/api'
import { truncateLabel } from './utils/text'

type RealtimeState = {
  robot: {
    timestamp: number | null
    mode: string
    current_task: string | null
    qpos: number[]
    latest_action: number[]
  }
  camera_status: Record<string, boolean>
  camera_timestamps: Record<string, number | null>
  camera_jpeg_b64: Record<string, string>
  rlt: RLTControlState
}

const initialRLT: RLTControlState = {
  phase: 'idle',
  training_phase: 'warmup',
  warmup_target: 100,
  warmup_count: 0,
  warmup_success: 0,
  warmup_failure: 0,
  warmup_attempts: 0,
  warmup_invalid: 0,
  auto_rollout_count: 0,
  auto_rollout_success: 0,
  auto_rollout_failure: 0,
  auto_rollout_attempts: 0,
  auto_rollout_invalid: 0,
  trainer_enabled: false,
  trainer_running: false,
  actor_enabled: false,
  actor_effective: false,
  actor_ready: false,
  actor_locked_reason: 'warmup',
  beta: 10,
  auto_beta_enabled: true,
  auto_beta_target_delta_norm: 0.06,
  auto_beta_min: 1,
  auto_beta_max: 30,
  auto_beta_lr: 0.03,
  auto_beta_ema_decay: 0.8,
  auto_beta_update_interval: 100,
  auto_beta_q_margin: 0.01,
  auto_beta_delta_norm_ema: null,
  auto_beta_q_advantage_ema: null,
  auto_beta_critic_loss_ema: null,
  auto_beta_reason: null,
  intervention_scale: 0.25,
  max_delta: 0.1,
  critic_gate_enabled: true,
  critic_gate_margin: 0,
  critic_gate_temperature: 0.05,
  critic_ready: false,
  inference_actor_active: false,
  inference_delta_norm: null,
  inference_gate_reason: null,
  key_region_probability: null,
  loaded_actor_step: null,
  inference_reference_q_value: null,
  inference_actor_q_value: null,
  inference_q_advantage: null,
  active_key_region_id: null,
  score_deadline: null,
  last_reward: null,
  last_event: null,
  wandb_url: null,
  critic_loss: null,
  critic_q1_loss: null,
  critic_q2_loss: null,
  actor_loss: null,
  actor_q_value: null,
  reference_q_value: null,
  q_advantage: null,
  actor_delta_norm: null,
  q1_mean: null,
  q2_mean: null,
  target_q_mean: null,
  q_gap: null,
  actor_updated: null,
  publish_actor: null,
  trainer_step: null,
  critic_burn_in_steps: 1000,
  target_sync_step: null,
  steps_per_sec: null,
  success_episodes: null,
  failure_episodes: null,
  replay_action_horizon: null,
  train_action_horizon: null,
  rlt_metrics_timestamp: null,
  replay_size: null,
  replay_shards: null,
  bad_shards: null,
  trainable_replay_count: 0,
  trainable_replay_success: 0,
  trainable_replay_failure: 0,
  trainable_replay_samples: 0,
  trainable_replay_shards: 0,
  invalid_replay_shards: 0,
  actor_checkpoint_path: null,
  actor_checkpoint_step: null,
  rl_token_checkpoint_path: null,
  events: [],
}

const initialState: RealtimeState = {
  robot: {
    timestamp: null,
    mode: 'waiting',
    current_task: null,
    qpos: [],
    latest_action: [],
  },
  camera_status: {},
  camera_timestamps: {},
  camera_jpeg_b64: {},
  rlt: initialRLT,
}

export default function App() {
  const [state, setState] = useState<RealtimeState>(initialState)
  const [language, setLanguage] = useState<AppLanguage>('en')
  const [cameraView, setCameraView] = useState<'focus' | 'quad'>('quad')
  const [page, setPage] = useState<'live' | 'config' | 'system' | 'rollouts' | 'key_regions'>('live')
  const [wsConnected, setWsConnected] = useState(false)
  const t = translations[language]
  const currentTaskLabel = state.robot.current_task ? truncateLabel(state.robot.current_task) : t.noActiveTask

  useEffect(() => {
    let isActive = true
    let socket: WebSocket | null = null
    let reconnectTimer: number | null = null

    const connect = () => {
      const ws = new WebSocket(`${wsBase}/ws/realtime`)
      socket = ws

      ws.onmessage = (event) => {
        if (!isActive) return
        setWsConnected(true)
        const payload = JSON.parse(event.data)
        setState({ ...payload, rlt: payload.rlt || initialRLT })
      }

      ws.onclose = () => {
        if (!isActive) return
        setWsConnected(false)
        reconnectTimer = window.setTimeout(connect, 1000)
      }

      ws.onerror = () => {
        setWsConnected(false)
        ws.close()
      }
    }

    connect()

    return () => {
      isActive = false
      if (reconnectTimer !== null) {
        window.clearTimeout(reconnectTimer)
      }
      socket?.close()
    }
  }, [])

  const freshness = useMemo(() => {
    if (!state.robot.timestamp) return t.waitingForRobot
    const age = Date.now() / 1000 - state.robot.timestamp
    return age < 1 ? t.live : t.stale(age.toFixed(1))
  }, [state.robot.timestamp, t])

  const setRLTState = (rlt: RLTControlState) => {
    setState((current) => ({ ...current, rlt }))
  }

  const runRobotTask = async (taskNum: '1' | '4' | '5') => {
    await sendRobotTask(taskNum)
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="header-brand">
          <h1>{t.title}</h1>
        </div>
        <div className="header-actions">
          <div className="robot-command-bar">
            <button className="robot-command task" type="button" onClick={() => void runRobotTask('1')}>
              twist bottle
            </button>
            <button className="robot-command home" type="button" onClick={() => void runRobotTask('4')}>
              home
            </button>
            <button className="robot-command sleep" type="button" onClick={() => void runRobotTask('5')}>
              sleep
            </button>
          </div>
          <nav className="page-tabs" aria-label="Primary">
            <button className={page === 'live' ? 'active' : ''} type="button" onClick={() => setPage('live')}>
              RLT Control
            </button>
            <button className={page === 'config' ? 'active' : ''} type="button" onClick={() => setPage('config')}>
              Config
            </button>
            <button className={page === 'system' ? 'active' : ''} type="button" onClick={() => setPage('system')}>
              System
            </button>
            <button className={page === 'rollouts' ? 'active' : ''} type="button" onClick={() => setPage('rollouts')}>
              Rollouts
            </button>
            <button
              className={page === 'key_regions' ? 'active' : ''}
              type="button"
              onClick={() => setPage('key_regions')}
            >
              Key Regions
            </button>
          </nav>
          <span className={`status-pill ${state.robot.timestamp ? 'live' : 'offline'}`}>{freshness}</span>
          <span className="status-pill mode">{state.robot.mode}</span>
          <span className="robot-task-badge" title={state.robot.current_task || t.noActiveTask}>
            {currentTaskLabel}
          </span>
          <label className="language-switch">
            <span>{t.language}</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value as AppLanguage)}>
              <option value="en">{t.english}</option>
              <option value="ja">{t.japanese}</option>
            </select>
          </label>
        </div>
      </header>

      {page === 'live' ? (
        <>
          <section className="rlt-status-strip">
            <StatusItem label="Control" value={state.rlt.phase} />
            <StatusItem label="Phase" value={state.rlt.training_phase} />
            <StatusItem label="Trainable" value={`${state.rlt.trainable_replay_count} / ${state.rlt.warmup_target}`} />
            <StatusItem label="Success" value={state.rlt.trainable_replay_success} />
            <StatusItem label="Failure" value={state.rlt.trainable_replay_failure} />
            <StatusItem label="RL Token" value={state.rlt.rl_token_checkpoint_path ? 'query/12000' : '-'} tone={state.rlt.rl_token_checkpoint_path ? 'ok' : 'watch'} />
            <StatusItem label="Actor" value={state.rlt.actor_effective ? 'active' : 'locked'} tone={state.rlt.actor_effective ? 'ok' : 'watch'} />
            <StatusItem label="Step" value={state.rlt.trainer_step ?? '-'} />
            <StatusItem label="Q Adv" value={formatStatusMetric(state.rlt.q_advantage)} tone={state.rlt.q_advantage !== null && state.rlt.q_advantage < 0 ? 'watch' : undefined} />
            <StatusItem label="Task" value={currentTaskLabel} />
            <StatusItem label="Alert" value={state.rlt.actor_locked_reason || 'OK'} tone={state.rlt.actor_locked_reason ? 'watch' : 'ok'} />
          </section>
          <section className="layout rlt-layout">
            <CameraGrid
              cameraStatus={state.camera_status}
              cameraTimestamps={state.camera_timestamps}
              cameraFrames={state.camera_jpeg_b64}
              language={language}
              currentTask={state.robot.current_task}
              cameraView={cameraView}
              onCameraViewChange={setCameraView}
            />
            <aside className="control-rail rlt-rail">
              <RLTControlPanel rlt={state.rlt} onState={setRLTState} />
              <RLTConfigPanel rlt={state.rlt} onState={setRLTState} />
              <RLTStatsPanel rlt={state.rlt} />
            </aside>
          </section>
        </>
      ) : page === 'config' ? (
        <RLTConfigPage rlt={state.rlt} onState={setRLTState} />
      ) : page === 'system' ? (
        <SystemPage
          rlt={state.rlt}
          onState={setRLTState}
          wsConnected={wsConnected}
          cameraStatus={state.camera_status}
          cameraTimestamps={state.camera_timestamps}
        />
      ) : page === 'key_regions' ? (
        <KeyRegionsPage title="Key Regions" />
      ) : (
        <RolloutBrowser title="Collected Files" excludeRootPaths={["key_regions"]} />
      )}
    </main>
  )
}

function StatusItem({ label, value, tone }: { label: string; value: number | string; tone?: 'ok' | 'watch' }) {
  return (
    <div className="rlt-status-item">
      <span>{label}</span>
      <strong className={tone ? `tone-${tone}` : ''}>{value}</strong>
    </div>
  )
}

function formatStatusMetric(value: number | null | undefined) {
  return value === null || value === undefined || !Number.isFinite(value) ? '-' : value.toFixed(4)
}
