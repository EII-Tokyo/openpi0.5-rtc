import { useEffect, useMemo, useState } from 'react'
import { CameraGrid } from './components/CameraGrid'
import { RobotViewer } from './components/RobotViewer'
import { VoicePanel } from './components/VoicePanel'
import { AppLanguage, translations } from './i18n'
import { apiBase, wsBase } from './services/api'

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
}

const TASK_NUMBERS = ['1', '2', '3', '4', '5'] as const

export default function App() {
  const [state, setState] = useState<RealtimeState>(initialState)
  const [language, setLanguage] = useState<AppLanguage>('en')
  const [dispatchError, setDispatchError] = useState('')
  const [cameraView, setCameraView] = useState<'focus' | 'quad'>('quad')
  const t = translations[language]

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
            {state.robot.current_task || t.noActiveTask}
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
          <VoicePanel
            mode={state.robot.mode}
            language={language}
            dispatchTask={dispatchTask}
            dispatchError={dispatchError}
          />
        </aside>
      </section>
    </main>
  )
}
