import { useEffect, useMemo, useState } from 'react'
import { CameraGrid } from './components/CameraGrid'
import { RobotViewer } from './components/RobotViewer'
import { VoicePanel } from './components/VoicePanel'
import { AppLanguage, translations } from './i18n'
import { wsBase } from './services/api'

type RealtimeState = {
  robot: {
    timestamp: number | null
    mode: string
    current_task: string | null
    qpos: number[]
    latest_action: number[]
  }
  camera_status: Record<string, boolean>
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
}

export default function App() {
  const [state, setState] = useState<RealtimeState>(initialState)
  const [language, setLanguage] = useState<AppLanguage>('en')
  const [highOnly, setHighOnly] = useState(false)
  const t = translations[language]

  useEffect(() => {
    const ws = new WebSocket(`${wsBase}/ws/realtime`)
    ws.onmessage = (event) => setState(JSON.parse(event.data))
    return () => ws.close()
  }, [])

  const freshness = useMemo(() => {
    if (!state.robot.timestamp) return t.waitingForRobot
    const age = Date.now() / 1000 - state.robot.timestamp
    return age < 1 ? t.live : t.stale(age.toFixed(1))
  }, [state.robot.timestamp, t])

  return (
    <main className="app-shell">
      <header className="hero">
        <div>
          <p className="eyebrow">{t.eyebrow}</p>
          <h1>{t.title}</h1>
        </div>
        <div className="hero-badges">
          <label className="language-switch">
            <span>{t.language}</span>
            <select value={language} onChange={(event) => setLanguage(event.target.value as AppLanguage)}>
              <option value="en">{t.english}</option>
              <option value="ja">{t.japanese}</option>
            </select>
          </label>
          <span className="status ok">{freshness}</span>
          <span className="robot-task" title={state.robot.current_task || t.noActiveTask}>
            {state.robot.current_task || t.noActiveTask}
          </span>
        </div>
      </header>

      <section className="layout">
        <CameraGrid
          cameraStatus={state.camera_status}
          language={language}
          highOnly={highOnly}
          onToggleHighOnly={() => setHighOnly((current) => !current)}
        />
        <div className="sidebar">
          <RobotViewer
            latestAction={state.robot.latest_action.length ? state.robot.latest_action : null}
            qpos={state.robot.qpos.length ? state.robot.qpos : null}
            mode={state.robot.mode}
            currentTask={state.robot.current_task}
            language={language}
          />
          <VoicePanel mode={state.robot.mode} language={language} />
        </div>
      </section>
    </main>
  )
}
