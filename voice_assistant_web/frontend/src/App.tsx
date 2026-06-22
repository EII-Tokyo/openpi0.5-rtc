import { useEffect, useMemo, useState } from 'react'
import { ActionDrawer } from './components/ActionDrawer'
import { CameraGrid } from './components/CameraGrid'
import { MotorDrawer } from './components/MotorDrawer'
import { RobotViewer } from './components/RobotViewer'
import { RLTTrajectoryReview } from './components/RLTTrajectoryReview'
import { VoicePanel } from './components/VoicePanel'
import { AppLanguage, translations } from './i18n'
import { apiBase, wsBase } from './services/api'
import { truncateLabel } from './utils/text'

type RealtimeState = {
  robot: {
    timestamp: number | null
    mode: string
    current_task: string | null
    qpos: number[]
    effort: number[]
    joint_effort: Record<string, { names?: string[]; values?: number[] }>
    joint_temperature: Record<string, { names?: string[]; values?: number[] }>
    latest_action: number[]
    rlt_actor_enabled: boolean
    rlt_chunk_q_min: number | null
    rlt_vla_chunk_q_min: number | null
    rlt_actor_chunk_q_min: number | null
  }
  camera_status: Record<string, boolean>
  camera_timestamps: Record<string, number | null>
  camera_jpeg_b64: Record<string, string>
}

type RLTReplayStatus = {
  replay_dir: string
  latest_episode: string | null
  terminal_label: string | null
  terminal_success: number | null
  num_steps: number | null
  num_chunks: number | null
}

const initialState: RealtimeState = {
  robot: {
    timestamp: null,
    mode: 'waiting',
    current_task: null,
    qpos: [],
    effort: [],
    joint_effort: {},
    joint_temperature: {},
    latest_action: [],
    rlt_actor_enabled: false,
    rlt_chunk_q_min: null,
    rlt_vla_chunk_q_min: null,
    rlt_actor_chunk_q_min: null,
  },
  camera_status: {},
  camera_timestamps: {},
  camera_jpeg_b64: {},
}

const TASK_NUMBERS = ['1', '2', '3', '4', '5', '6'] as const

export default function App() {
  const [state, setState] = useState<RealtimeState>(initialState)
  const [language, setLanguage] = useState<AppLanguage>('en')
  const [dispatchError, setDispatchError] = useState('')
  const [cameraView, setCameraView] = useState<'focus' | 'quad'>('quad')
  const [motorDrawerOpen, setMotorDrawerOpen] = useState(false)
  const [actionDrawerOpen, setActionDrawerOpen] = useState(false)
  const [rltBusy, setRltBusy] = useState(false)
  const [rltSavingReplay, setRltSavingReplay] = useState(false)
  const [rltError, setRltError] = useState('')
  const [rltStatus, setRltStatus] = useState<RLTReplayStatus | null>(null)
  const [labelModalOpen, setLabelModalOpen] = useState(false)
  const [rltStarted, setRltStarted] = useState(false)
  const [rltRecordingStarted, setRltRecordingStarted] = useState(false)
  const [trajectoryReviewOpen, setTrajectoryReviewOpen] = useState(false)
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
        setState(JSON.parse(event.data))
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

  const dispatchTask = async (taskNumber: string) => {
    setDispatchError('')
    try {
      const response = await fetch(`${apiBase}/api/tasks/${taskNumber}`, { method: 'POST' })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
    } catch {
      setDispatchError(t.dispatchFailed)
    }
  }

  const fetchRltStatus = async () => {
    const response = await fetch(`${apiBase}/api/rlt/status`)
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    return (await response.json()) as RLTReplayStatus
  }

  const refreshRltStatus = async () => {
    try {
      setRltStatus(await fetchRltStatus())
    } catch {
      // Status is optional; keep the main control path usable.
    }
  }

  const startRltCollection = async () => {
    setRltBusy(true)
    setRltError('')
    try {
      const response = await fetch(`${apiBase}/api/rlt/start`, { method: 'POST' })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setRltStarted(true)
      setRltRecordingStarted(false)
      await refreshRltStatus()
    } catch {
      setRltError(t.rltRequestFailed)
    } finally {
      setRltBusy(false)
    }
  }

  const beginRltRecording = async () => {
    setRltBusy(true)
    setRltError('')
    try {
      const response = await fetch(`${apiBase}/api/rlt/record`, { method: 'POST' })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setRltRecordingStarted(true)
    } catch {
      setRltError(t.rltRequestFailed)
    } finally {
      setRltBusy(false)
    }
  }

  const setRltActorSampling = async (enabled: boolean) => {
    setRltBusy(true)
    setRltError('')
    const previousEnabled = state.robot.rlt_actor_enabled
    setState((current) => ({
      ...current,
      robot: {
        ...current.robot,
        rlt_actor_enabled: enabled,
      },
    }))
    try {
      const response = await fetch(`${apiBase}/api/rlt/actor/${enabled ? 'enable' : 'disable'}`, { method: 'POST' })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
    } catch {
      setState((current) => ({
        ...current,
        robot: {
          ...current.robot,
          rlt_actor_enabled: previousEnabled,
        },
      }))
      setRltError(t.rltRequestFailed)
    } finally {
      setRltBusy(false)
    }
  }

  const endRltCollection = async () => {
    setRltBusy(true)
    setRltSavingReplay(true)
    setRltError('')
    const shouldLabelReplay = rltRecordingStarted
    try {
      const response = await fetch(`${apiBase}/api/rlt/end`, { method: 'POST' })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      if (shouldLabelReplay) {
        setLabelModalOpen(true)
      } else {
        setLabelModalOpen(false)
        await refreshRltStatus()
      }
      setRltStarted(false)
      setRltRecordingStarted(false)
    } catch {
      setRltError(t.rltRequestFailed)
    } finally {
      setRltSavingReplay(false)
      setRltBusy(false)
    }
  }

  const submitRltLabel = async (label: 'success' | 'failure') => {
    setRltBusy(true)
    setRltSavingReplay(true)
    setRltError('')
    try {
      const response = await fetch(`${apiBase}/api/rlt/label`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ label }),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      setRltStatus(await response.json())
      setLabelModalOpen(false)
    } catch {
      setRltError(t.rltLabelFailed)
    } finally {
      setRltSavingReplay(false)
      setRltBusy(false)
    }
  }

  useEffect(() => {
    void refreshRltStatus()
  }, [])

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.shiftKey || event.repeat) return
      const active = document.activeElement
      if (
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement ||
        active instanceof HTMLSelectElement ||
        active?.getAttribute('contenteditable') === 'true'
      ) {
        return
      }
      if (TASK_NUMBERS.includes(event.key as (typeof TASK_NUMBERS)[number])) {
        void dispatchTask(event.key)
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [t.dispatchFailed])

  return (
    <main className="app-shell">
      <header className="app-header">
        <div className="header-brand">
          <h1>{t.title}</h1>
        </div>
        <div className="header-actions">
          <span className={`status-pill ${state.robot.timestamp ? 'live' : 'offline'}`}>{freshness}</span>
          <span className="status-pill mode">{state.robot.mode}</span>
          <span className="robot-task-badge" title={state.robot.current_task || t.noActiveTask}>
            {currentTaskLabel}
          </span>
          <button className="ghost-button" type="button" onClick={() => setMotorDrawerOpen(true)}>
            Motors
          </button>
          <button className="ghost-button" type="button" onClick={() => setActionDrawerOpen(true)}>
            Actions
          </button>
          <button className="ghost-button" type="button" onClick={() => setTrajectoryReviewOpen(true)}>
            RLT Data
          </button>
          <label className="language-switch">
            <span>{t.language}</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value as AppLanguage)}>
              <option value="en">{t.english}</option>
              <option value="ja">{t.japanese}</option>
            </select>
          </label>
        </div>
      </header>

      {trajectoryReviewOpen ? (
        <RLTTrajectoryReview onClose={() => setTrajectoryReviewOpen(false)} />
      ) : (
      <section className="layout">
        <CameraGrid
          cameraStatus={state.camera_status}
          cameraTimestamps={state.camera_timestamps}
          cameraFrames={state.camera_jpeg_b64}
          language={language}
          currentTask={state.robot.current_task}
          cameraView={cameraView}
          onCameraViewChange={setCameraView}
        />
        <aside className="control-rail">
          <RobotViewer
            latestAction={state.robot.latest_action.length ? state.robot.latest_action : null}
            qpos={state.robot.qpos.length ? state.robot.qpos : null}
            mode={state.robot.mode}
            currentTask={state.robot.current_task}
            language={language}
          />
          <section className="panel rlt-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.rltEyebrow}</p>
                <h2>{t.rltTitle}</h2>
              </div>
              <div className="header-actions">
                <span className={`status-pill ${state.robot.rlt_actor_enabled ? 'live' : 'mode'}`}>
                  {state.robot.rlt_actor_enabled ? t.rltActorOn : t.rltActorOff}
                </span>
                <span className="status-pill q-value">
                  {t.rltVlaChunkQ(state.robot.rlt_vla_chunk_q_min)}
                </span>
                <span className="status-pill q-value">
                  {t.rltActorChunkQ(state.robot.rlt_actor_chunk_q_min)}
                </span>
                <span className="status-pill mode">{rltStatus?.terminal_label || t.rltUnlabeled}</span>
              </div>
            </div>
            <div className="rlt-actions">
              <button className="primary-command start" type="button" disabled={rltBusy} onClick={() => void startRltCollection()}>
                {t.rltStart}
              </button>
              <button
                className={`primary-command ${state.robot.rlt_actor_enabled ? 'stop' : 'start'}`}
                type="button"
                disabled={rltBusy}
                onClick={() => void setRltActorSampling(!state.robot.rlt_actor_enabled)}
              >
                {state.robot.rlt_actor_enabled ? t.rltActorDisable : t.rltActorEnable}
              </button>
              <button
                className="primary-command start"
                type="button"
                disabled={rltBusy || !rltStarted || rltRecordingStarted}
                onClick={() => void beginRltRecording()}
              >
                {t.rltRecord}
              </button>
              <button className="primary-command stop" type="button" disabled={rltBusy} onClick={() => void endRltCollection()}>
                {rltSavingReplay ? t.rltSavingReplay : t.rltEnd}
              </button>
            </div>
            <div className="rlt-meta">
              <span>{t.rltLatest}</span>
              <strong title={rltStatus?.latest_episode || ''}>
                {rltStatus?.latest_episode ? rltStatus.latest_episode.split('/').pop() : t.rltNoEpisode}
              </strong>
              <span>{t.rltCounts(rltStatus?.num_steps ?? 0, rltStatus?.num_chunks ?? 0)}</span>
            </div>
            {rltError ? <p className="voice-error inline-error">{rltError}</p> : null}
          </section>
          <VoicePanel
            mode={state.robot.mode}
            language={language}
            dispatchTask={dispatchTask}
            dispatchError={dispatchError}
          />
        </aside>
      </section>
      )}
      <MotorDrawer
        open={motorDrawerOpen}
        onClose={() => setMotorDrawerOpen(false)}
        jointEffort={state.robot.joint_effort}
        jointTemperature={state.robot.joint_temperature}
      />
      <ActionDrawer
        open={actionDrawerOpen}
        onClose={() => setActionDrawerOpen(false)}
        latestAction={state.robot.latest_action}
      />
      {labelModalOpen ? (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label={t.rltLabelTitle}>
          <div className="label-modal">
            <h2>{t.rltLabelTitle}</h2>
            <p>{t.rltLabelPrompt}</p>
            {rltSavingReplay ? <p>{t.rltSavingReplay}</p> : null}
            {rltError ? <p className="voice-error inline-error">{rltError}</p> : null}
            <div className="label-modal-actions">
              <button className="primary-command start" type="button" disabled={rltBusy} onClick={() => void submitRltLabel('success')}>
                {t.rltSuccess}
              </button>
              <button className="primary-command stop" type="button" disabled={rltBusy} onClick={() => void submitRltLabel('failure')}>
                {t.rltFailure}
              </button>
            </div>
          </div>
        </div>
      ) : null}
    </main>
  )
}
