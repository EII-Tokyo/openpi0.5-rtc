import { useEffect, useMemo, useState } from 'react'
import { CameraGrid } from './components/CameraGrid'
import { RLTConfigPanel, RLTControlPanel, RLTStatsPanel } from './components/RLTControlPanel'
import { RolloutBrowser } from './components/RolloutBrowser'
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
  auto_rollout_count: 0,
  auto_rollout_success: 0,
  auto_rollout_failure: 0,
  actor_enabled: false,
  actor_effective: false,
  actor_locked_reason: 'warmup',
  beta: 10,
  intervention_scale: 0.25,
  max_delta: 0.1,
  active_key_region_id: null,
  score_deadline: null,
  last_reward: null,
  last_event: null,
  wandb_url: null,
  critic_loss: null,
  actor_loss: null,
  replay_size: null,
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
  const [page, setPage] = useState<'live' | 'rollouts' | 'key_regions'>('live')
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
        const payload = JSON.parse(event.data)
        setState({ ...payload, rlt: payload.rlt || initialRLT })
      }

      ws.onclose = () => {
        if (!isActive) return
        reconnectTimer = window.setTimeout(connect, 1000)
      }

      ws.onerror = () => {
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
            <StatusItem label="Phase" value={state.rlt.training_phase} />
            <StatusItem label="Warmup" value={`${state.rlt.warmup_count} / ${state.rlt.warmup_target}`} />
            <StatusItem label="Replay" value={state.rlt.replay_size ?? '-'} />
            <StatusItem label="RL Token" value={state.rlt.rl_token_checkpoint_path ? 'query/12000' : '-'} tone={state.rlt.rl_token_checkpoint_path ? 'ok' : 'watch'} />
            <StatusItem label="Actor" value={state.rlt.actor_effective ? 'active' : 'locked'} tone={state.rlt.actor_effective ? 'ok' : 'watch'} />
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
              <RLTStatsPanel rlt={state.rlt} />
              <RLTConfigPanel rlt={state.rlt} onState={setRLTState} />
            </aside>
          </section>
        </>
      ) : page === 'key_regions' ? (
        <RolloutBrowser
          title="Key Regions"
          rootPath="key_regions"
          defaultCamera="cam_right_wrist.mp4"
          showManifest
        />
      ) : (
        <RolloutBrowser title="Collected Files" />
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
