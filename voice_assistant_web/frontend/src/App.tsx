import { FormEvent, useEffect, useMemo, useRef, useState } from 'react'
import { AppLanguage, translations } from './i18n'

const defaultHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const cameraNames = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'] as const
const configStorageKey = 'aloha-ui-config-v2'
const lowLevelSubtaskOptions = [
  'Rotate so opening faces right',
  'Pick up with left hand',
  'Unscrew cap',
  'Bottle to left trash bin, cap to right trash bin',
  'Bottle to left trash bin',
  'Use right hand to remove and place into left trash bin',
  'Pick up cap and place into right trash bin',
  'Return to initial pose',
] as const

type HierarchicalState = {
  task_prompt?: string
  low_level_prompt?: string
  high_level_text?: string
  bottle_description?: string | null
  bottle_position?: Record<string, unknown> | null
  bottle_state?: string | null
  subtask?: string | null
  locked_bottle_description?: string | null
  raw_subtask?: string | null
  effective_subtask?: string | null
  pending_subtask?: string | null
  pending_subtask_count?: number
  high_level_server_timing?: Record<string, unknown>
  low_level_server_timing?: Record<string, unknown>
  history?: Array<{
    id: string
    timestamp: number
    obs_version: number
    task_prompt?: string
    high_level_text?: string
    bottle_description?: string | null
    bottle_position?: Record<string, unknown> | null
    bottle_state?: string | null
    subtask?: string | null
    images?: Record<string, string>
  }>
}

type RealtimeState = {
  robot: {
    timestamp: number | null
    mode: string
    current_task: string | null
    qpos: number[]
    runtime_qpos: number[]
    ros_qpos: number[]
    latest_action: number[]
    hierarchical: HierarchicalState
  }
  camera_status: Record<string, boolean>
  camera_timestamps: Record<string, number | null>
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
  includeBottleDescription: boolean
  includeBottlePosition: boolean
  includeBottleState: boolean
  includeSubtask: boolean
  forcedLowLevelSubtask: string | null
  announcementLanguage: 'zh' | 'ja'
}

type VoiceStatus = 'idle' | 'recording' | 'thinking' | 'speaking'

const initialState: RealtimeState = {
  robot: {
    timestamp: null,
    mode: 'waiting',
    current_task: null,
    qpos: [],
    runtime_qpos: [],
    ros_qpos: [],
    latest_action: [],
    hierarchical: {},
  },
  camera_status: {},
  camera_timestamps: {},
}

const defaultConfig: UiConfig = {
  apiBase: import.meta.env.VITE_API_BASE || `http://${defaultHost}:8011`,
  wsBase: import.meta.env.VITE_WS_BASE || `ws://${defaultHost}:8011`,
  cameraRefreshMs: 100,
  datasetDir: '',
  manualDatasetDir: '',
  includeBottleDescription: true,
  includeBottlePosition: false,
  includeBottleState: true,
  includeSubtask: true,
  forcedLowLevelSubtask: null,
  announcementLanguage: 'zh',
}

function getAnnouncementLocale(language: UiConfig['announcementLanguage']) {
  return language === 'ja' ? 'ja-JP' : 'zh-CN'
}

function getAnnouncementTranslation(language: UiConfig['announcementLanguage']) {
  return translations[language === 'ja' ? 'ja' : 'zh']
}

function pickSpeechVoice(locale: string) {
  if (typeof window === 'undefined' || !('speechSynthesis' in window)) return null
  const voices = window.speechSynthesis.getVoices()
  if (!voices.length) return null
  return (
    voices.find((voice) => voice.lang === locale) ||
    voices.find((voice) => voice.lang.toLowerCase().startsWith(locale.slice(0, 2).toLowerCase())) ||
    null
  )
}

function loadUiConfig(): UiConfig {
  if (typeof window === 'undefined') return defaultConfig
  try {
    const raw = window.localStorage.getItem(configStorageKey)
    if (!raw) return defaultConfig
    const parsed = JSON.parse(raw)
    const includeBottlePosition =
      typeof parsed?.includeBottlePosition === 'boolean'
        ? parsed.includeBottlePosition
        : typeof parsed?.includeBottlePosition === 'string'
          ? parsed.includeBottlePosition === 'true'
          : defaultConfig.includeBottlePosition
    const includeBottleDescription =
      typeof parsed?.includeBottleDescription === 'boolean'
        ? parsed.includeBottleDescription
        : typeof parsed?.includeBottleDescription === 'string'
          ? parsed.includeBottleDescription === 'true'
          : defaultConfig.includeBottleDescription
    const includeBottleState =
      typeof parsed?.includeBottleState === 'boolean'
        ? parsed.includeBottleState
        : typeof parsed?.includeBottleState === 'string'
          ? parsed.includeBottleState === 'true'
          : defaultConfig.includeBottleState
    const includeSubtask =
      typeof parsed?.includeSubtask === 'boolean'
        ? parsed.includeSubtask
        : typeof parsed?.includeSubtask === 'string'
          ? parsed.includeSubtask === 'true'
          : defaultConfig.includeSubtask
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
      includeBottleDescription,
      includeBottlePosition,
      includeBottleState,
      includeSubtask,
      forcedLowLevelSubtask:
        typeof parsed?.forcedLowLevelSubtask === 'string' && parsed.forcedLowLevelSubtask.trim()
          ? parsed.forcedLowLevelSubtask.trim()
          : defaultConfig.forcedLowLevelSubtask,
      announcementLanguage:
        parsed?.announcementLanguage === 'ja' || parsed?.announcementLanguage === 'zh'
          ? parsed.announcementLanguage
          : defaultConfig.announcementLanguage,
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

function formatCameraAge(timestamp: number | null | undefined) {
  if (!timestamp) return 'N/A'
  const age = Math.max(0, Date.now() / 1000 - timestamp)
  return `${age.toFixed(age < 1 ? 2 : 1)}s`
}

function parseBottleBox(value: Record<string, unknown> | null | undefined) {
  if (!value) return null
  const xMin = Number(value['x min'])
  const xMax = Number(value['x max'])
  const yMin = Number(value['y min'])
  const yMax = Number(value['y max'])
  if (![xMin, xMax, yMin, yMax].every(Number.isFinite)) return null
  if (xMax <= xMin || yMax <= yMin) return null
  return {
    left: `${Math.max(0, Math.min(100, xMin))}%`,
    top: `${Math.max(0, Math.min(100, yMin))}%`,
    width: `${Math.max(0, Math.min(100, xMax - xMin))}%`,
    height: `${Math.max(0, Math.min(100, yMax - yMin))}%`,
  }
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
  const [runtimeOpen, setRuntimeOpen] = useState(false)
  const [overrideOpen, setOverrideOpen] = useState(false)
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null)
  const [cameraView, setCameraView] = useState<'focus' | 'quad'>('focus')
  const [cameraRefreshToken, setCameraRefreshToken] = useState<number>(Date.now())
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaChunksRef = useRef<Blob[]>([])
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const vadAnimationRef = useRef<number | null>(null)
  const vadSilenceStartedAtRef = useRef<number | null>(null)
  const lastAnnouncedBottleDescriptionRef = useRef<string | null>(null)
  const translatedAnnouncementCacheRef = useRef<Record<string, string>>({})
  const t = translations[language]

  const clearHighLevelDisplay = () => {
    setSelectedHistoryId(null)
    setState((current) => ({
      ...current,
      robot: {
        ...current.robot,
        hierarchical: {},
      },
    }))
  }

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
      if (vadAnimationRef.current !== null) {
        window.cancelAnimationFrame(vadAnimationRef.current)
      }
      if (audioContextRef.current) {
        void audioContextRef.current.close()
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
  const history = Array.isArray(hierarchical.history) ? hierarchical.history : []
  const selectedHistory =
    history.find((entry) => entry.id === selectedHistoryId) ||
    history[history.length - 1] ||
    null
  const primaryCamera = 'cam_high'
  const secondaryCameras = cameraNames.filter((name) => name !== primaryCamera)
  const cameraSrc = (cameraName: (typeof cameraNames)[number]) =>
    `${uiConfig.apiBase}/api/cameras/${cameraName}/latest.jpg?t=${cameraRefreshToken}`
  const displayFps = useMemo(() => {
    const intervalMs = Math.max(100, Number(uiConfig.cameraRefreshMs) || defaultConfig.cameraRefreshMs)
    return 1000 / intervalMs
  }, [uiConfig.cameraRefreshMs])
  useEffect(() => {
    if (state.robot.current_task !== 'process all bottles') {
      lastAnnouncedBottleDescriptionRef.current = null
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
      return
    }
    const description =
      typeof hierarchical.locked_bottle_description === 'string'
        ? hierarchical.locked_bottle_description.trim()
        : ''
    if (!description) return
    if (lastAnnouncedBottleDescriptionRef.current === description) return
    lastAnnouncedBottleDescriptionRef.current = description
    if (!('speechSynthesis' in window)) return
    const targetLanguage = uiConfig.announcementLanguage
    const cacheKey = `${targetLanguage}:${description}`
    const announcementTranslation = getAnnouncementTranslation(targetLanguage)
    const announcementLocale = getAnnouncementLocale(targetLanguage)
    const speak = (translatedDescription: string) => {
      const utterance = new SpeechSynthesisUtterance(announcementTranslation.announceBottle(translatedDescription))
      utterance.lang = announcementLocale
      const voice = pickSpeechVoice(announcementLocale)
      if (voice) {
        utterance.voice = voice
      }
      window.speechSynthesis.cancel()
      window.speechSynthesis.speak(utterance)
    }
    const cached = translatedAnnouncementCacheRef.current[cacheKey]
    if (cached) {
      speak(cached)
      return
    }
    let cancelled = false
    void fetch(`${uiConfig.apiBase}/api/runtime/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: description,
        target_language: targetLanguage,
      }),
    })
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        return response.json() as Promise<{ translated_text: string }>
      })
      .then((payload) => {
        if (cancelled) return
        const translatedDescription = (payload.translated_text || description).trim() || description
        translatedAnnouncementCacheRef.current[cacheKey] = translatedDescription
        speak(translatedDescription)
      })
      .catch(() => {
        if (cancelled) return
        speak(description)
      })
    return () => {
      cancelled = true
    }
  }, [hierarchical.locked_bottle_description, state.robot.current_task, uiConfig.announcementLanguage, uiConfig.apiBase])
  const primaryCameraAge = formatCameraAge(state.camera_timestamps[primaryCamera])
  const primaryCameraStats = `${displayFps.toFixed(1)} FPS · ${primaryCameraAge}`
  const voiceTaskWithHighLevel = [voiceResponse?.task_name, hierarchical.high_level_text]
    .filter((value) => typeof value === 'string' && value.trim())
    .join('\n')
  const primaryCameraBbox = primaryCamera === 'cam_high' ? parseBottleBox(hierarchical.bottle_position || null) : null

  const renderCameraOverlay = (cameraName: (typeof cameraNames)[number]) => {
    const isLive = Boolean(state.camera_status[cameraName])
    const cameraAge = formatCameraAge(state.camera_timestamps[cameraName])
    return (
      <div className="camera-overlay-chip">
        <div className="camera-overlay-head">
          <span>{t.cameraLabels[cameraName] || cameraName}</span>
          <span className={`dot ${isLive ? 'live' : 'offline'}`} />
        </div>
        <div className="camera-overlay-meta">
          <span>{displayFps.toFixed(1)} FPS</span>
          <span>{cameraAge}</span>
        </div>
      </div>
    )
  }

  const sendCommand = async (rawText: string) => {
    const text = rawText.trim()
    if (!text) return
    if (text === '1') {
      clearHighLevelDisplay()
    }
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
          include_bottle_description: uiConfig.includeBottleDescription,
          include_bottle_position: uiConfig.includeBottlePosition,
          include_bottle_state: uiConfig.includeBottleState,
          include_subtask: uiConfig.includeSubtask,
          forced_low_level_subtask: uiConfig.forcedLowLevelSubtask || undefined,
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

  const pushRuntimeConfig = async (nextConfig: UiConfig) => {
    try {
      const response = await fetch(`${nextConfig.apiBase}/api/runtime/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          dataset_dir: nextConfig.datasetDir.trim(),
          manual_dataset_dir: nextConfig.manualDatasetDir.trim(),
          include_bottle_description: nextConfig.includeBottleDescription,
          include_bottle_position: nextConfig.includeBottlePosition,
          include_bottle_state: nextConfig.includeBottleState,
          include_subtask: nextConfig.includeSubtask,
          forced_low_level_subtask: nextConfig.forcedLowLevelSubtask,
        }),
      })
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`)
      }
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : t.requestFailed)
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
      formData.append('include_bottle_description', String(uiConfig.includeBottleDescription))
      formData.append('include_bottle_position', String(uiConfig.includeBottlePosition))
      formData.append('include_bottle_state', String(uiConfig.includeBottleState))
      formData.append('include_subtask', String(uiConfig.includeSubtask))
      if (uiConfig.forcedLowLevelSubtask) {
        formData.append('forced_low_level_subtask', uiConfig.forcedLowLevelSubtask)
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
    if (vadAnimationRef.current !== null) {
      window.cancelAnimationFrame(vadAnimationRef.current)
      vadAnimationRef.current = null
    }
    vadSilenceStartedAtRef.current = null
    if (audioContextRef.current) {
      void audioContextRef.current.close()
      audioContextRef.current = null
    }
    recorder.stop()
  }

  const startRecording = async () => {
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(t.micUnavailable)
      return
    }
    if (!window.isSecureContext && !['localhost', '127.0.0.1'].includes(window.location.hostname)) {
      setError(t.micRequiresSecureContext)
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
      const audioContext = new AudioContext()
      audioContextRef.current = audioContext
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      analyser.smoothingTimeConstant = 0.85
      source.connect(analyser)
      const sampleBuffer = new Uint8Array(analyser.fftSize)
      const silenceThreshold = 0.018
      const silenceDurationMs = 900
      const minRecordMs = 600
      const recordingStartedAt = performance.now()
      const monitorSilence = () => {
        if (mediaRecorderRef.current?.state !== 'recording') {
          vadAnimationRef.current = null
          return
        }
        analyser.getByteTimeDomainData(sampleBuffer)
        let sum = 0
        for (const value of sampleBuffer) {
          const normalized = value / 128 - 1
          sum += normalized * normalized
        }
        const rms = Math.sqrt(sum / sampleBuffer.length)
        const now = performance.now()
        if (rms < silenceThreshold) {
          if (vadSilenceStartedAtRef.current === null) {
            vadSilenceStartedAtRef.current = now
          } else if (
            now - recordingStartedAt >= minRecordMs &&
            now - vadSilenceStartedAtRef.current >= silenceDurationMs
          ) {
            void stopRecording()
            return
          }
        } else {
          vadSilenceStartedAtRef.current = null
        }
        vadAnimationRef.current = window.requestAnimationFrame(monitorSilence)
      }
      recorder.start()
      setVoiceStatus('recording')
      vadSilenceStartedAtRef.current = null
      vadAnimationRef.current = window.requestAnimationFrame(monitorSilence)
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
          <button type="button" className="ghost-button" onClick={() => setRuntimeOpen(true)}>
            {t.openRuntime}
          </button>
          <button type="button" className="ghost-button" onClick={() => setOverrideOpen(true)}>
            {t.forcedSubtask}
          </button>
          <button type="button" className="ghost-button" onClick={() => setConfigOpen(true)}>
            {t.openConfig}
          </button>
        </div>
      </header>

      <section className="layout">
        <section className={`stage-panel ${cameraView === 'quad' ? 'quad-mode' : 'focus-mode'}`}>
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
                  {state.camera_status[primaryCamera] ? `${t.live} · ${primaryCameraStats}` : `${t.offline} · ${primaryCameraStats}`}
                </span>
              </div>
            </div>
            {cameraView === 'focus' ? (
              <div className="camera-stage-frame">
                <img src={cameraSrc(primaryCamera)} alt={primaryCamera} />
                <div className="camera-frame-top">{renderCameraOverlay(primaryCamera)}</div>
                {primaryCameraBbox ? (
                  <div
                    className="camera-bbox"
                    style={{
                      left: primaryCameraBbox.left,
                      top: primaryCameraBbox.top,
                      width: primaryCameraBbox.width,
                      height: primaryCameraBbox.height,
                    }}
                  />
                ) : null}
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
                    <img src={cameraSrc(cameraName)} alt={cameraName} />
                    <div className="camera-frame-top">{renderCameraOverlay(cameraName)}</div>
                  </article>
                ))}
              </div>
            )}
          </article>
          {cameraView === 'focus' ? (
            <div className="camera-strip">
              {secondaryCameras.map((cameraName) => (
                <article key={cameraName} className="mini-camera-card">
                  <img src={cameraSrc(cameraName)} alt={cameraName} />
                  <div className="camera-frame-top">{renderCameraOverlay(cameraName)}</div>
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
              {['1', '2', '3', '4'].map((taskNum) => (
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
                <pre>{voiceTaskWithHighLevel || 'N/A'}</pre>
              </div>
            </div>
            {error ? <div className="error-banner">{error}</div> : null}
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
              <span>{t.announcementLanguage}</span>
              <select
                value={uiConfig.announcementLanguage}
                onChange={(event) =>
                  setUiConfig((current) => ({
                    ...current,
                    announcementLanguage: event.target.value as 'zh' | 'ja',
                  }))
                }
              >
                <option value="zh">{t.announcementChinese}</option>
                <option value="ja">{t.announcementJapanese}</option>
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
              <span>{t.includeBottleDescription}</span>
              <select
                value={uiConfig.includeBottleDescription ? 'true' : 'false'}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      includeBottleDescription: event.target.value === 'true',
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                <option value="false">{t.no}</option>
                <option value="true">{t.yes}</option>
              </select>
            </label>
            <label className="config-field">
              <span>{t.includeBottlePosition}</span>
              <select
                value={uiConfig.includeBottlePosition ? 'true' : 'false'}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      includeBottlePosition: event.target.value === 'true',
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                <option value="false">{t.no}</option>
                <option value="true">{t.yes}</option>
              </select>
            </label>
            <label className="config-field">
              <span>{t.includeBottleState}</span>
              <select
                value={uiConfig.includeBottleState ? 'true' : 'false'}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      includeBottleState: event.target.value === 'true',
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                <option value="false">{t.no}</option>
                <option value="true">{t.yes}</option>
              </select>
            </label>
            <label className="config-field">
              <span>{t.includeSubtask}</span>
              <select
                value={uiConfig.includeSubtask ? 'true' : 'false'}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      includeSubtask: event.target.value === 'true',
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                <option value="false">{t.no}</option>
                <option value="true">{t.yes}</option>
              </select>
            </label>
            <label className="config-field">
              <span>{t.inferenceSavePath}</span>
              <input
                value={uiConfig.datasetDir}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      datasetDir: event.target.value,
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
                placeholder="/app/examples/aloha_real/inference_hdf5"
              />
            </label>
            <label className="config-field">
              <span>{t.manualSavePath}</span>
              <input
                value={uiConfig.manualDatasetDir}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      manualDatasetDir: event.target.value,
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
                placeholder="/app/examples/aloha_real/manual_override"
              />
            </label>
            <p className="config-help">{t.configHelp}</p>
          </aside>
        </>
      ) : null}

      {overrideOpen ? (
        <>
          <div className="drawer-backdrop runtime-drawer-backdrop" onClick={() => setOverrideOpen(false)} />
          <aside className="config-drawer" onClick={(event) => event.stopPropagation()}>
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.runtimeEyebrow}</p>
                <h2>{t.forcedSubtask}</h2>
              </div>
              <button type="button" className="ghost-button" onClick={() => setOverrideOpen(false)}>
                {t.close}
              </button>
            </div>
            <div className="config-field">
              <span>{t.forcedSubtask}</span>
              <div className="quick-tasks subtask-override-grid">
                <button
                  type="button"
                  className={`quick-task ${uiConfig.forcedLowLevelSubtask === null ? 'active' : ''}`}
                  onClick={() =>
                    setUiConfig((current) => {
                      const nextConfig = {
                        ...current,
                        forcedLowLevelSubtask: null,
                      }
                      void pushRuntimeConfig(nextConfig)
                      return nextConfig
                    })
                  }
                >
                  {t.forcedSubtaskAuto}
                </button>
                {lowLevelSubtaskOptions.map((subtask) => (
                <button
                  key={subtask}
                  type="button"
                  className={`quick-task subtask-override-button ${uiConfig.forcedLowLevelSubtask === subtask ? 'active' : ''}`}
                  onClick={() =>
                    setUiConfig((current) => {
                      const nextConfig = {
                        ...current,
                        forcedLowLevelSubtask: subtask,
                      }
                      void pushRuntimeConfig(nextConfig)
                      return nextConfig
                    })
                  }
                  >
                    {subtask}
                  </button>
                ))}
              </div>
            </div>
          </aside>
        </>
      ) : null}

      {runtimeOpen ? (
        <>
          <div className="drawer-backdrop runtime-drawer-backdrop" onClick={() => setRuntimeOpen(false)} />
          <aside className="runtime-drawer" onClick={(event) => event.stopPropagation()}>
            <div className="panel-header">
              <div>
                <p className="eyebrow">{t.runtimeEyebrow}</p>
                <h2>{t.runtimeTitle}</h2>
              </div>
              <div className="drawer-actions">
                <span className="robot-task-badge" title={state.robot.current_task || t.noActiveTask}>
                  {state.robot.current_task || t.noActiveTask}
                </span>
                <button type="button" className="ghost-button" onClick={() => setRuntimeOpen(false)}>
                  {t.close}
                </button>
              </div>
            </div>
            <div className="runtime-grid">
              <div className="info-block">
                <span className="info-label">{t.bottleDescription}</span>
                <pre>{hierarchical.bottle_description || 'N/A'}</pre>
              </div>
              <div className="info-block">
                <span className="info-label">{t.effectiveBottleDescription}</span>
                <pre>{hierarchical.locked_bottle_description || 'N/A'}</pre>
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
              <div className="info-block wide">
                <span className="info-label">{t.stabilizer}</span>
                <pre>{JSON.stringify({
                  pending_subtask: hierarchical.pending_subtask ?? null,
                  pending_subtask_count: hierarchical.pending_subtask_count ?? 0,
                }, null, 2)}</pre>
              </div>
              <div className="info-block wide">
                <span className="info-label">{t.subtaskHistory}</span>
                <div className="history-list">
                  {history.length ? history.map((entry) => (
                    <button
                      key={entry.id}
                      type="button"
                      className={`history-item ${selectedHistory?.id === entry.id ? 'active' : ''}`}
                      onClick={() => setSelectedHistoryId(entry.id)}
                    >
                      <strong>{entry.subtask || 'N/A'}</strong>
                      <span>{entry.bottle_state || 'N/A'}</span>
                    </button>
                  )) : <pre>N/A</pre>}
                </div>
              </div>
              <div className="info-block wide">
                <span className="info-label">{t.selectedHighLevelOutput}</span>
                <pre>{selectedHistory?.high_level_text || 'N/A'}</pre>
              </div>
              <div className="info-block wide">
                <span className="info-label">{t.selectedHighLevelImages}</span>
                <div className="history-image-grid">
                  {selectedHistory?.images && Object.keys(selectedHistory.images).length ? Object.entries(selectedHistory.images).map(([name, data]) => (
                    <figure key={name} className="history-image-card">
                      <img src={`data:image/jpeg;base64,${data}`} alt={name} />
                      <figcaption>{t.cameraLabels[name as keyof typeof t.cameraLabels] || name}</figcaption>
                    </figure>
                  )) : <pre>N/A</pre>}
                </div>
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
                <pre>{formatArray(state.robot.runtime_qpos.length ? state.robot.runtime_qpos : state.robot.qpos)}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.rosQpos}</span>
                <pre>{formatArray(state.robot.ros_qpos)}</pre>
              </div>
              <div className="info-block compact">
                <span className="info-label">{t.latestAction}</span>
                <pre>{formatArray(state.robot.latest_action)}</pre>
              </div>
            </div>
          </aside>
        </>
      ) : null}
    </main>
  )
}
