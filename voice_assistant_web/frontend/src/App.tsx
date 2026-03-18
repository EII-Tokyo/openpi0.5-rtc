import { FormEvent, useEffect, useMemo, useState } from 'react'
import { AppLanguage, translations } from './i18n'

const apiBase = import.meta.env.VITE_API_BASE || 'http://localhost:8011'
const wsBase = import.meta.env.VITE_WS_BASE || 'ws://localhost:8011'
const cameraNames = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'] as const

type HierarchicalState = {
  task_prompt?: string
  low_level_prompt?: string
  high_level_text?: string
  bottle_description?: string | null
  bottle_position?: Record<string, unknown> | null
  bottle_state?: string | null
  subtask?: string | null
  high_level_server_timing?: Record<string, unknown>
  low_level_server_timing?: Record<string, unknown>
}

type RealtimeState = {
  robot: {
    timestamp: number | null
    mode: string
    current_task: string | null
    qpos: number[]
    latest_action: number[]
    hierarchical: HierarchicalState
  }
  camera_status: Record<string, boolean>
}

type VoiceResponse = {
  transcript: string
  reply_text: string
  task_number: string | null
  task_name: string | null
}

const initialState: RealtimeState = {
  robot: {
    timestamp: null,
    mode: 'waiting',
    current_task: null,
    qpos: [],
    latest_action: [],
    hierarchical: {},
  },
  camera_status: {},
}

function formatArray(values: number[], maxItems = 6) {
  if (!values.length) return '[]'
  const head = values.slice(0, maxItems).map((value) => value.toFixed(3)).join(', ')
  return values.length > maxItems ? `[${head}, ...]` : `[${head}]`
}

function formatTiming(value: Record<string, unknown> | undefined) {
  if (!value || !Object.keys(value).length) return 'N/A'
  return Object.entries(value)
    .map(([key, entry]) => `${key}=${typeof entry === 'number' ? entry.toFixed(1) : String(entry)}`)
    .join('  ')
}

export default function App() {
  const [state, setState] = useState<RealtimeState>(initialState)
  const [language, setLanguage] = useState<AppLanguage>('en')
  const [highOnly, setHighOnly] = useState(false)
  const [command, setCommand] = useState('')
  const [voiceResponse, setVoiceResponse] = useState<VoiceResponse | null>(null)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const t = translations[language]

  useEffect(() => {
    const ws = new WebSocket(`${wsBase}/ws/realtime`)
    ws.onmessage = (event) => {
      try {
        setState(JSON.parse(event.data))
      } catch (err) {
        console.error('Failed to parse realtime payload', err)
      }
    }
    ws.onerror = () => setError('Realtime websocket disconnected')
    return () => ws.close()
  }, [])

  const freshness = useMemo(() => {
    if (!state.robot.timestamp) return t.waitingForRobot
    const age = Date.now() / 1000 - state.robot.timestamp
    return age < 1 ? t.live : t.stale(age.toFixed(1))
  }, [state.robot.timestamp, t])

  const visibleCameras: ReadonlyArray<(typeof cameraNames)[number]> = highOnly ? ['cam_high'] : cameraNames
  const hierarchical = state.robot.hierarchical || {}

  const submitCommand = async (event: FormEvent) => {
    event.preventDefault()
    const text = command.trim()
    if (!text) return
    setSending(true)
    setError(null)
    try {
      const response = await fetch(`${apiBase}/api/voice/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, language }),
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const payload = (await response.json()) as VoiceResponse
      setVoiceResponse(payload)
      setCommand('')
    } catch (err) {
      setError(err instanceof Error ? err.message : t.requestFailed)
    } finally {
      setSending(false)
    }
  }

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
              <option value="zh">{t.chinese}</option>
            </select>
          </label>
          <span className="status ok">{freshness}</span>
          <span className="robot-task" title={state.robot.current_task || t.noActiveTask}>
            {state.robot.current_task || t.noActiveTask}
          </span>
        </div>
      </header>

      <section className="layout">
        <section className="panel camera-panel">
          <div className="panel-header">
            <div>
              <p className="eyebrow">{t.camerasEyebrow}</p>
              <h2>{t.camerasTitle}</h2>
            </div>
            <button className="view-toggle" onClick={() => setHighOnly((current) => !current)}>
              {highOnly ? t.showAllCameras : t.showHighCameraOnly}
            </button>
          </div>
          <div className={`camera-grid ${highOnly ? 'single' : ''}`}>
            {visibleCameras.map((cameraName) => (
              <article key={cameraName} className="camera-card">
                <div className="camera-frame">
                  <img src={`${apiBase}/api/cameras/${cameraName}/stream.mjpg`} alt={cameraName} />
                  <div className="camera-overlay">
                    <span className="camera-chip">{t.cameraLabels[cameraName] || cameraName}</span>
                    <span className={`status ${state.camera_status[cameraName] ? 'ok' : 'offline'}`}>
                      {state.camera_status[cameraName] ? t.live : t.offline}
                    </span>
                  </div>
                </div>
              </article>
            ))}
          </div>
        </section>

        <div className="sidebar">
          <section className="panel info-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.runtimeEyebrow}</p>
                <h2>{t.runtimeTitle}</h2>
              </div>
              <span className="status ok">{state.robot.mode || t.waiting}</span>
            </div>
            <div className="info-grid">
              <div className="info-block">
                <span className="info-label">{t.taskPrompt}</span>
                <pre>{hierarchical.task_prompt || state.robot.current_task || t.noActiveTask}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.highLevelText}</span>
                <pre>{hierarchical.high_level_text || 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.bottleDescription}</span>
                <pre>{hierarchical.bottle_description || 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.bottlePosition}</span>
                <pre>{hierarchical.bottle_position ? JSON.stringify(hierarchical.bottle_position, null, 2) : 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.bottleState}</span>
                <pre>{hierarchical.bottle_state || 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.subtask}</span>
                <pre>{hierarchical.subtask || 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.lowLevelPrompt}</span>
                <pre>{hierarchical.low_level_prompt || 'N/A'}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.highLevelTiming}</span>
                <pre>{formatTiming(hierarchical.high_level_server_timing)}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.lowLevelTiming}</span>
                <pre>{formatTiming(hierarchical.low_level_server_timing)}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.qpos}</span>
                <pre>{formatArray(state.robot.qpos)}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.latestAction}</span>
                <pre>{formatArray(state.robot.latest_action)}</pre>
              </div>
            </div>
          </section>

          <section className="panel voice-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.voiceEyebrow}</p>
                <h2>{t.voiceTitle}</h2>
              </div>
            </div>
            <form className="voice-form" onSubmit={submitCommand}>
              <textarea
                value={command}
                onChange={(event) => setCommand(event.target.value)}
                placeholder={t.typeFallback}
              />
              <div className="voice-actions">
                <button type="submit" disabled={sending || !command.trim()}>
                  {sending ? t.thinking : t.send}
                </button>
              </div>
            </form>
            <div className="conversation">
              <div className="info-block compact">
                <span className="info-label">{t.you}</span>
                <pre>{voiceResponse?.transcript || t.noConversation}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.aloha}</span>
                <pre>{voiceResponse?.reply_text || t.noReply}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.voiceTask}</span>
                <pre>{voiceResponse?.task_name || 'N/A'}</pre>
              </div>
              {error ? (
                <div className="error-banner">{error}</div>
              ) : null}
            </div>
          </section>
        </div>
      </section>
    </main>
  )
}
