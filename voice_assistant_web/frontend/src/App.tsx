import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { AppLanguage, translations } from './i18n'

const defaultHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const cameraNames = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'] as const
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
  audio_base64?: string | null
  audio_mime_type?: string | null
}

type UiConfig = {
  apiBase: string
  wsBase: string
  cameraRefreshMs: number
  datasetDir: string
  manualDatasetDir: string
}

type VoiceStatus = 'idle' | 'recording' | 'thinking' | 'speaking'

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
  apiBase: import.meta.env.VITE_API_BASE || `http://${defaultHost}:8011`,
  wsBase: import.meta.env.VITE_WS_BASE || `ws://${defaultHost}:8011`,
  cameraRefreshMs: 1000,
  datasetDir: '',
  manualDatasetDir: '',
}

function loadUiConfig(): UiConfig {
  if (typeof window === 'undefined') return defaultConfig
  try {
    const raw = window.localStorage.getItem(configStorageKey)
    if (!raw) return defaultConfig
    const parsed = JSON.parse(raw)
    return {
      apiBase: typeof parsed?.apiBase === 'string' && parsed.apiBase.trim() ? parsed.apiBase.trim() : defaultConfig.apiBase,
      wsBase: typeof parsed?.wsBase === 'string' && parsed.wsBase.trim() ? parsed.wsBase.trim() : defaultConfig.wsBase,
      cameraRefreshMs:
        typeof parsed?.cameraRefreshMs === 'number' && parsed.cameraRefreshMs > 0
          ? parsed.cameraRefreshMs
          : defaultConfig.cameraRefreshMs,
      datasetDir: typeof parsed?.datasetDir === 'string' ? parsed.datasetDir.trim() : defaultConfig.datasetDir,
      manualDatasetDir:
        typeof parsed?.manualDatasetDir === 'string'
          ? parsed.manualDatasetDir.trim()
          : defaultConfig.manualDatasetDir,
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
  const [uiConfig, setUiConfig] = useState<UiConfig>(() => loadUiConfig())
  const [state, setState] = useState<RealtimeState>(initialState)
  const [language, setLanguage] = useState<AppLanguage>('en')
  const [command, setCommand] = useState('')
  const [voiceResponse, setVoiceResponse] = useState<VoiceResponse | null>(null)
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>('idle')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [cameraView, setCameraView] = useState<'focus' | 'quad'>('focus')
  const [cameraRefreshToken, setCameraRefreshToken] = useState<number>(Date.now())
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaChunksRef = useRef<Blob[]>([])
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const t = translations[language]

  useEffect(() => {
    const ws = new WebSocket(`${uiConfig.wsBase}/ws/realtime`)
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
  }, [uiConfig.wsBase])

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

  useEffect(() => {
    return () => {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current.src = ''
      }
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        mediaRecorderRef.current.stop()
      }
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach((track) => track.stop())
      }
    }
  }, [])

  useEffect(() => {
    const intervalMs = Math.max(100, Number(uiConfig.cameraRefreshMs) || defaultConfig.cameraRefreshMs)
    const timer = window.setInterval(() => {
      setCameraRefreshToken(Date.now())
    }, intervalMs)
    return () => window.clearInterval(timer)
  }, [uiConfig.cameraRefreshMs])

  const freshness = useMemo(() => {
    if (!state.robot.timestamp) return t.waitingForRobot
    const age = Date.now() / 1000 - state.robot.timestamp
    return age < 1 ? t.live : t.stale(age.toFixed(1))
  }, [state.robot.timestamp, t])

  const hierarchical = state.robot.hierarchical || {}
  const primaryCamera = 'cam_high'
  const secondaryCameras = cameraNames.filter((name) => name !== primaryCamera)
  const cameraSrc = (cameraName: (typeof cameraNames)[number]) =>
    `${uiConfig.apiBase}/api/cameras/${cameraName}/latest.jpg?t=${cameraRefreshToken}`

  const sendCommand = async (rawText: string) => {
    const text = rawText.trim()
    if (!text) return
    setSending(true)
    setError(null)
    try {
      const response = await fetch(`${uiConfig.apiBase}/api/voice/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text,
          language,
          dataset_dir: uiConfig.datasetDir.trim() || undefined,
          manual_dataset_dir: uiConfig.manualDatasetDir.trim() || undefined,
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

  const uploadAudio = async (blob: Blob) => {
    setSending(true)
    setVoiceStatus('thinking')
    setError(null)
    try {
      const fileExtension = blob.type.includes('webm') ? 'webm' : 'wav'
      const formData = new FormData()
      formData.append('file', blob, `voice.${fileExtension}`)
      formData.append('language', language)
      if (uiConfig.datasetDir.trim()) {
        formData.append('dataset_dir', uiConfig.datasetDir.trim())
      }
      if (uiConfig.manualDatasetDir.trim()) {
        formData.append('manual_dataset_dir', uiConfig.manualDatasetDir.trim())
      }
      const response = await fetch(`${uiConfig.apiBase}/api/voice/audio`, {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      const payload = (await response.json()) as VoiceResponse
      setVoiceResponse(payload)
      if (payload.audio_base64 && payload.audio_mime_type) {
        const audio = new Audio(`data:${payload.audio_mime_type};base64,${payload.audio_base64}`)
        audioRef.current?.pause()
        audioRef.current = audio
        setVoiceStatus('speaking')
        audio.onended = () => setVoiceStatus('idle')
        audio.onerror = () => setVoiceStatus('idle')
        void audio.play().catch(() => setVoiceStatus('idle'))
      } else {
        setVoiceStatus('idle')
      }
    } catch (err) {
      setVoiceStatus('idle')
      setError(err instanceof Error ? err.message : t.requestFailed)
    } finally {
      setSending(false)
    }
  }

  const stopRecording = async () => {
    const recorder = mediaRecorderRef.current
    if (!recorder || recorder.state === 'inactive') return
    recorder.stop()
  }

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(t.micUnavailable)
      return
    }
    setError(null)
    try {
      if (audioRef.current) {
        audioRef.current.pause()
        audioRef.current = null
      }
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream
      mediaChunksRef.current = []
      const mimeType = MediaRecorder.isTypeSupported('audio/webm')
        ? 'audio/webm'
        : MediaRecorder.isTypeSupported('audio/mp4')
          ? 'audio/mp4'
          : ''
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      mediaRecorderRef.current = recorder
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          mediaChunksRef.current.push(event.data)
        }
      }
      recorder.onstop = () => {
        const blobType = recorder.mimeType || 'audio/webm'
        const audioBlob = new Blob(mediaChunksRef.current, { type: blobType })
        mediaChunksRef.current = []
        mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
        mediaStreamRef.current = null
        void uploadAudio(audioBlob)
      }
      recorder.start()
      setVoiceStatus('recording')
    } catch {
      setVoiceStatus('idle')
      setError(t.micFailed)
    }
  }

  const toggleRecording = async () => {
    if (voiceStatus === 'recording') {
      await stopRecording()
      return
    }
    if (sending || voiceStatus === 'thinking') return
    await startRecording()
  }

  const voiceStatusLabel =
    voiceStatus === 'recording'
      ? t.recording
      : voiceStatus === 'thinking'
        ? t.thinking
        : voiceStatus === 'speaking'
          ? t.speaking
          : t.idle

  return (
    <main className="app-shell">
      <header className="hero">
        <div className="hero-copy">
          <p className="eyebrow">{t.eyebrow}</p>
          <h1>{t.title}</h1>
        </div>
        <div className="hero-badges">
          <span className="status-pill live">{freshness}</span>
          <span className="status-pill mode">{state.robot.mode || t.waiting}</span>
          <button type="button" className="ghost-button" onClick={() => setConfigOpen(true)}>
            {t.openConfig}
          </button>
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
                <img src={cameraSrc(primaryCamera)} alt={primaryCamera} />
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
                    <img src={cameraSrc(cameraName)} alt={cameraName} />
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
                  <img src={cameraSrc(cameraName)} alt={cameraName} />
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
            <div className="voice-orb-wrap">
              <button
                type="button"
                className={`voice-orb halo ripple ${voiceStatus}`}
                onClick={() => void toggleRecording()}
                disabled={sending}
                aria-label={voiceStatus === 'recording' ? t.stop : t.talk}
              >
                <span>{voiceStatus === 'recording' ? t.stop : t.talk}</span>
              </button>
              <p className="voice-status">{voiceStatusLabel}</p>
              <p className="voice-hint">
                {voiceStatus === 'recording' ? t.autoSendHint : t.listeningHint}
              </p>
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
        <>
          <div className="drawer-backdrop" onClick={() => setConfigOpen(false)} />
          <aside className="config-drawer" onClick={(event) => event.stopPropagation()}>
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
              <span>{t.language}</span>
              <select value={language} onChange={(event) => setLanguage(event.target.value as AppLanguage)}>
                <option value="en">{t.english}</option>
                <option value="ja">{t.japanese}</option>
                <option value="zh">{t.chinese}</option>
              </select>
            </label>
            <label className="config-field">
              <span>{t.apiBaseLabel}</span>
              <input
                value={uiConfig.apiBase}
                onChange={(event) =>
                  setUiConfig((current) => ({
                    ...current,
                    apiBase: event.target.value,
                  }))
                }
                placeholder={`http://${defaultHost}:8011`}
              />
            </label>
            <label className="config-field">
              <span>{t.wsBaseLabel}</span>
              <input
                value={uiConfig.wsBase}
                onChange={(event) =>
                  setUiConfig((current) => ({
                    ...current,
                    wsBase: event.target.value,
                  }))
                }
                placeholder={`ws://${defaultHost}:8011`}
              />
            </label>
            <label className="config-field">
              <span>{t.cameraRefreshLabel}</span>
              <input
                type="number"
                min={100}
                step={100}
                value={uiConfig.cameraRefreshMs}
                onChange={(event) =>
                  setUiConfig((current) => ({
                    ...current,
                    cameraRefreshMs: Math.max(100, Number(event.target.value) || defaultConfig.cameraRefreshMs),
                  }))
                }
                placeholder="1000"
              />
            </label>
            <label className="config-field">
              <span>{t.inferenceSavePath}</span>
              <input
                value={uiConfig.datasetDir}
                onChange={(event) =>
                  setUiConfig((current) => ({
                    ...current,
                    datasetDir: event.target.value,
                  }))
                }
                placeholder="/app/examples/aloha_real/inference_hdf5"
              />
            </label>
            <label className="config-field">
              <span>{t.manualSavePath}</span>
              <input
                value={uiConfig.manualDatasetDir}
                onChange={(event) =>
                  setUiConfig((current) => ({
                    ...current,
                    manualDatasetDir: event.target.value,
                  }))
                }
                placeholder="/app/examples/aloha_real/manual_override"
              />
            </label>
            <p className="config-help">{t.configHelp}</p>
          </aside>
        </>
      ) : null}
    </main>
  )
}
