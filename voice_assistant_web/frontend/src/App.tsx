import { FormEvent, useEffect, useMemo, useState } from 'react'
import { AppLanguage, translations } from './i18n'

const defaultHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const apiBase = import.meta.env.VITE_API_BASE || `http://${defaultHost}:8011`
const wsBase = import.meta.env.VITE_WS_BASE || `ws://${defaultHost}:8011`
const cameraNames = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'] as const
const hiddenConfigClicks = 5
const configStorageKey = 'aloha-ui-config-v1'

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

type UiConfig = {
  manualDatasetSubdir: string
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

const defaultConfig: UiConfig = {
  manualDatasetSubdir: '12345',
}

function loadUiConfig(): UiConfig {
  if (typeof window === 'undefined') return defaultConfig
  try {
    const raw = window.localStorage.getItem(configStorageKey)
    if (!raw) return defaultConfig
    const parsed = JSON.parse(raw)
    return {
      manualDatasetSubdir:
        typeof parsed?.manualDatasetSubdir === 'string' && parsed.manualDatasetSubdir.trim()
          ? parsed.manualDatasetSubdir.trim()
          : defaultConfig.manualDatasetSubdir,
    }
  } catch {
    return defaultConfig
  }
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
  const [command, setCommand] = useState('')
  const [voiceResponse, setVoiceResponse] = useState<VoiceResponse | null>(null)
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [secretClicks, setSecretClicks] = useState(0)
  const [uiConfig, setUiConfig] = useState<UiConfig>(() => loadUiConfig())
  const [cameraView, setCameraView] = useState<'focus' | 'quad'>('focus')
  const t = translations[language]

  useEffect(() => {
    const ws = new WebSocket(`${wsBase}/ws/realtime`)
    ws.onopen = () => setError(null)
    ws.onmessage = (event) => {
      try {
        setState(JSON.parse(event.data))
        setError(null)
      } catch (err) {
        console.error('Failed to parse realtime payload', err)
      }
    }
    ws.onerror = () => setError('Realtime websocket disconnected')
    return () => ws.close()
  }, [])

  useEffect(() => {
    window.localStorage.setItem(configStorageKey, JSON.stringify(uiConfig))
  }, [uiConfig])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === ',') {
        event.preventDefault()
        setConfigOpen((current) => !current)
      }
      if (event.key === 'Escape') {
        setConfigOpen(false)
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [])

  const freshness = useMemo(() => {
    if (!state.robot.timestamp) return t.waitingForRobot
    const age = Date.now() / 1000 - state.robot.timestamp
    return age < 1 ? t.live : t.stale(age.toFixed(1))
  }, [state.robot.timestamp, t])

  const hierarchical = state.robot.hierarchical || {}
  const primaryCamera = 'cam_high'
  const secondaryCameras = cameraNames.filter((name) => name !== primaryCamera)

  const revealConfig = () => {
    const next = secretClicks + 1
    if (next >= hiddenConfigClicks) {
      setConfigOpen(true)
      setSecretClicks(0)
      return
    }
    setSecretClicks(next)
  }

  const sendCommand = async (rawText: string) => {
    const text = rawText.trim()
    if (!text) return
    setSending(true)
    setError(null)
    try {
      const response = await fetch(`${apiBase}/api/voice/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          language,
          manual_dataset_subdir: uiConfig.manualDatasetSubdir.trim() || undefined,
        }),
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

  const submitCommand = async (event: FormEvent) => {
    event.preventDefault()
    await sendCommand(command)
  }

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow" onClick={revealConfig}>
            {t.eyebrow}
          </p>
          <h1 onClick={revealConfig}>{t.title}</h1>
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
          <span className="status-pill live">{freshness}</span>
          <span className="status-pill mode">{state.robot.mode || t.waiting}</span>
        </div>
      </header>

      <section className="layout">
        <section className="stage-panel">
          <article className="hero-camera">
            <div className="camera-panel-header">
              <div>
                <p className="eyebrow">{t.camerasEyebrow}</p>
                <h2>{cameraView === 'focus' ? t.cameraLabels[primaryCamera] : 'Quad View'}</h2>
              </div>
              <div className="camera-controls">
                <button
                  type="button"
                  className={`ghost-button ${cameraView === 'focus' ? 'active' : ''}`}
                  onClick={() => setCameraView('focus')}
                >
                  High
                </button>
                <button
                  type="button"
                  className={`ghost-button ${cameraView === 'quad' ? 'active' : ''}`}
                  onClick={() => setCameraView('quad')}
                >
                  Quad
                </button>
                <span className={`status-pill ${state.camera_status[primaryCamera] ? 'live' : 'offline'}`}>
                  {state.camera_status[primaryCamera] ? t.live : t.offline}
                </span>
              </div>
            </div>
            {cameraView === 'focus' ? (
              <div className="camera-stage-frame">
                <img src={`${apiBase}/api/cameras/${primaryCamera}/stream.mjpg`} alt={primaryCamera} />
                <div className="camera-stage-overlay">
                  <div className="stage-task">
                    <span>{t.taskPrompt}</span>
                    <strong>{hierarchical.task_prompt || state.robot.current_task || t.noActiveTask}</strong>
                  </div>
                </div>
              </div>
            ) : (
              <div className="camera-grid">
                {cameraNames.map((cameraName) => (
                  <article key={cameraName} className="mini-camera-card">
                    <div className="mini-camera-header">
                      <span>{t.cameraLabels[cameraName] || cameraName}</span>
                      <span className={`dot ${state.camera_status[cameraName] ? 'live' : 'offline'}`} />
                    </div>
                    <img src={`${apiBase}/api/cameras/${cameraName}/stream.mjpg`} alt={cameraName} />
                  </article>
                ))}
              </div>
            )}
          </article>

          {cameraView === 'focus' ? (
            <div className="camera-strip">
              {secondaryCameras.map((cameraName) => (
                <article key={cameraName} className="mini-camera-card">
                  <div className="mini-camera-header">
                    <span>{t.cameraLabels[cameraName] || cameraName}</span>
                    <span className={`dot ${state.camera_status[cameraName] ? 'live' : 'offline'}`} />
                  </div>
                  <img src={`${apiBase}/api/cameras/${cameraName}/stream.mjpg`} alt={cameraName} />
                </article>
              ))}
            </div>
          ) : null}
        </section>

        <aside className="control-rail">
          <section className="panel command-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.voiceEyebrow}</p>
                <h2>{t.voiceTitle}</h2>
              </div>
              <span className="command-hint">{t.directTaskHint}</span>
            </div>
            <form className="command-form" onSubmit={submitCommand}>
              <input
                value={command}
                onChange={(event) => setCommand(event.target.value)}
                placeholder={t.commandPlaceholder}
              />
              <button type="submit" disabled={sending || !command.trim()}>
                {sending ? t.thinking : t.send}
              </button>
            </form>
            <div className="quick-tasks">
              {['1', '2', '3', '4', '5'].map((taskNum) => (
                <button key={taskNum} type="button" className="quick-task" onClick={() => void sendCommand(taskNum)}>
                  {taskNum}
                </button>
              ))}
            </div>
            <div className="response-grid">
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
            </div>
            {error ? <div className="error-banner">{error}</div> : null}
          </section>

          <section className="panel runtime-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.runtimeEyebrow}</p>
                <h2>{t.runtimeTitle}</h2>
              </div>
              <span className="robot-task-badge" title={state.robot.current_task || t.noActiveTask}>
                {state.robot.current_task || t.noActiveTask}
              </span>
            </div>
            <div className="runtime-grid">
              <div className="info-block">
                <span className="info-label">{t.bottleDescription}</span>
                <pre>{hierarchical.bottle_description || 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.bottleState}</span>
                <pre>{hierarchical.bottle_state || 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.bottlePosition}</span>
                <pre>{hierarchical.bottle_position ? JSON.stringify(hierarchical.bottle_position, null, 2) : 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.subtask}</span>
                <pre>{hierarchical.subtask || 'N/A'}</pre>
              </div>
              <div className="info-block wide">
                <span className="info-label">{t.highLevelText}</span>
                <pre>{hierarchical.high_level_text || 'N/A'}</pre>
              </div>
              <div className="info-block wide">
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
        </aside>
      </section>

      {configOpen ? (
        <div className="config-overlay" onClick={() => setConfigOpen(false)}>
          <section className="config-panel" onClick={(event) => event.stopPropagation()}>
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.configEyebrow}</p>
                <h2>{t.configTitle}</h2>
              </div>
              <button type="button" className="ghost-button" onClick={() => setConfigOpen(false)}>
                {t.close}
              </button>
            </div>
            <label className="config-field">
              <span>{t.saveFolder}</span>
              <input
                value={uiConfig.manualDatasetSubdir}
                onChange={(event) =>
                  setUiConfig((current) => ({
                    ...current,
                    manualDatasetSubdir: event.target.value.replace(/[^A-Za-z0-9_-]/g, ''),
                  }))
                }
                placeholder="12345"
              />
            </label>
            <p className="config-help">{t.configHelp}</p>
          </section>
        </div>
      ) : null}
    </main>
  )
}
