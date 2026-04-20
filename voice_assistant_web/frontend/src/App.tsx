import { FormEvent, useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react'
import { AppLanguage, translations } from './i18n'

const defaultHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const defaultHttpOrigin = typeof window !== 'undefined' ? window.location.origin : `http://${defaultHost}`
const defaultWsOrigin =
  typeof window !== 'undefined'
    ? `${window.location.protocol === 'https:' ? 'wss' : 'ws'}://${window.location.host}`
    : `ws://${defaultHost}`
const cameraNames = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist'] as const

export type SubtaskCatalogEntry = {
  subtask: string
  is_start_subtask: boolean
  good_bad_action?: 'good action' | 'bad action' | 'normal' | null
}

export type StateSubtaskPairRow = {
  bottle_state: string
  subtask: string
}

/** 与后端 `low_level_subtask_defaults` 一致；Mongo 空文档时由服务端回填 */
const DEFAULT_SUBTASK_CATALOG: SubtaskCatalogEntry[] = [
  { subtask: 'Rotate so opening faces right', is_start_subtask: true },
  { subtask: 'Pick up with left hand', is_start_subtask: true },
  { subtask: 'Unscrew cap', is_start_subtask: false },
  { subtask: 'Bottle to left trash bin, cap to right trash bin', is_start_subtask: false },
  { subtask: 'Bottle to left trash bin', is_start_subtask: false },
  { subtask: 'Use right hand to remove and place into left trash bin', is_start_subtask: false },
  { subtask: 'Pick up cap and place into right trash bin', is_start_subtask: false },
  { subtask: 'Return to initial pose', is_start_subtask: false },
]

const DEFAULT_STATE_SUBTASK_PAIRS: StateSubtaskPairRow[] = [
  { bottle_state: 'Bottle on table, opening faces left', subtask: 'Rotate so opening faces right' },
  { bottle_state: 'Bottle on table, opening faces right', subtask: 'Pick up with left hand' },
  { bottle_state: 'Bottle in left hand and capped', subtask: 'Unscrew cap' },
  {
    bottle_state: 'Bottle in left hand, cap removed, and cap in right hand',
    subtask: 'Bottle to left trash bin, cap to right trash bin',
  },
  {
    bottle_state: 'Bottle in left hand, cap removed, and cap not in right hand',
    subtask: 'Bottle to left trash bin',
  },
  { bottle_state: 'Bottle in left hand and upside down', subtask: 'Bottle to left trash bin' },
  {
    bottle_state: 'Bottle stuck in left hand',
    subtask: 'Use right hand to remove and place into left trash bin',
  },
  { bottle_state: 'Cap on table', subtask: 'Pick up cap and place into right trash bin' },
  { bottle_state: 'No bottle on table', subtask: 'Return to initial pose' },
]

function parseSubtaskCatalogFromApi(raw: unknown): SubtaskCatalogEntry[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    return DEFAULT_SUBTASK_CATALOG.map((e) => ({ ...e }))
  }
  const out: SubtaskCatalogEntry[] = []
  for (const x of raw) {
    if (!x || typeof x !== 'object') continue
    const o = x as Record<string, unknown>
    const sub = String(o.subtask ?? '').trim()
    if (!sub) continue
    const isStart = Boolean(o.is_start_subtask ?? o.isStartSubtask)
    const gba = o.good_bad_action ?? o.goodBadAction
    let good_bad_action: SubtaskCatalogEntry['good_bad_action'] = null
    if (gba === 'good action' || gba === 'bad action' || gba === 'normal') {
      good_bad_action = gba
    }
    out.push({ subtask: sub, is_start_subtask: isStart, good_bad_action })
  }
  return out.length > 0 ? out : DEFAULT_SUBTASK_CATALOG.map((e) => ({ ...e }))
}

function parseStateSubtaskPairsFromApi(raw: unknown): StateSubtaskPairRow[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    return DEFAULT_STATE_SUBTASK_PAIRS.map((e) => ({ ...e }))
  }
  const out: StateSubtaskPairRow[] = []
  for (const x of raw) {
    if (Array.isArray(x) && x.length >= 2) {
      const bs = String(x[0]).trim()
      const st = String(x[1]).trim()
      if (bs && st) out.push({ bottle_state: bs, subtask: st })
      continue
    }
    if (x && typeof x === 'object') {
      const o = x as Record<string, unknown>
      const bs = String(o.bottle_state ?? o.bottleState ?? '').trim()
      const st = String(o.subtask ?? '').trim()
      if (bs && st) out.push({ bottle_state: bs, subtask: st })
    }
  }
  return out.length > 0 ? out : DEFAULT_STATE_SUBTASK_PAIRS.map((e) => ({ ...e }))
}

/** `low_level_prompt` 为 runtime 下发的 `json.dumps(structured_subtask)` */
function parseStructuredLowLevelPrompt(raw: string | null | undefined): {
  subtask: string | null
  bottle_description: string | null
} | null {
  if (raw == null || !String(raw).trim()) return null
  try {
    const o = JSON.parse(String(raw)) as Record<string, unknown>
    if (!o || typeof o !== 'object') return null
    const sub = o.subtask
    const bd = o.bottle_description
    return {
      subtask: typeof sub === 'string' && sub.trim() ? sub.trim() : null,
      bottle_description: typeof bd === 'string' && bd.trim() ? bd.trim() : null,
    }
  } catch {
    return null
  }
}

function getAnnouncementLocale(language: 'zh' | 'ja') {
  return language === 'ja' ? 'ja-JP' : 'zh-CN'
}

function getAnnouncementTranslation(language: 'zh' | 'ja') {
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

/** Matches `hierarchical` published by runtime (top-level keys only). */
type HierarchicalHistoryEntry = {
  id: string
  timestamp: number
  obs_version: number
  task_prompt?: string
  high_level_text?: string
  images?: Record<string, string>
  /** 该条 high-level 推理时刻观测中的关节角（与 runtime 观测一致）。 */
  qpos?: number[]
}

type HierarchicalState = {
  task_prompt?: string
  low_level_prompt?: string | null
  high_level_text?: string
  high_level_server_timing?: Record<string, unknown>
  low_level_server_timing?: Record<string, unknown>
  history?: HierarchicalHistoryEntry[]
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
  uiLanguage: AppLanguage
  datasetDir: string
  manualDatasetDir: string
  hdf5RecentSeconds: number
  includeBottleDescription: boolean
  lockBottleDescription: boolean
  includeBottlePosition: boolean
  includeBottleState: boolean
  includeSubtask: boolean
  videoMemoryNumFrames: 1 | 4
  highLevelSource: 'gpt' | 'service'
  gptModel: string
  gptImageMode: 'high_only' | 'all_cameras'
  forcedLowLevelSubtask: string | null
  /** 低层子任务目录（Mongo）；含 is_start_subtask、可选 good_bad_action */
  subtaskCatalog: SubtaskCatalogEntry[]
  /** (bottle_state, subtask) 合法对 */
  stateSubtaskPairs: StateSubtaskPairRow[]
  /** 语音播报 / 合成等目标语言（中/日），持久化在服务端 */
  announcementLanguage: 'zh' | 'ja'
}

type RuntimeConfigPayload = {
  datasetDir: string
  manualDatasetDir: string
  hdf5RecentSeconds: number
  includeBottleDescription: boolean
  lockBottleDescription: boolean
  includeBottlePosition: boolean
  includeBottleState: boolean
  includeSubtask: boolean
  videoMemoryNumFrames: 1 | 4
  highLevelSource: 'gpt' | 'service'
  gptModel: string
  gptImageMode: 'high_only' | 'all_cameras'
  forcedLowLevelSubtask: string | null
  subtaskCatalog: SubtaskCatalogEntry[]
  stateSubtaskPairs: StateSubtaskPairRow[]
  announcementLanguage: 'zh' | 'ja'
  apiBase: string
  wsBase: string
  cameraRefreshMs: number
  uiLanguage: AppLanguage
}

/** FastAPI returns snake_case field names; accept camelCase too for robustness. */
function parseRuntimeConfigResponse(data: unknown): RuntimeConfigPayload {
  const r = data as Record<string, unknown>
  const str = (camel: string, snake: string) => {
    const v = r[camel] ?? r[snake]
    return typeof v === 'string' ? v : ''
  }
  const bool = (camel: string, snake: string, fallback: boolean) => {
    const v = r[camel] ?? r[snake]
    if (v === undefined || v === null) return fallback
    return Boolean(v)
  }
  const rawFrames = r['video_memory_num_frames'] ?? r['videoMemoryNumFrames']
  const videoMemoryNumFrames: 1 | 4 = rawFrames === 4 ? 4 : 1
  const rawHdf5RecentSeconds = r['hdf5_recent_seconds'] ?? r['hdf5RecentSeconds']
  let hdf5RecentSeconds = 5
  if (typeof rawHdf5RecentSeconds === 'number' && Number.isFinite(rawHdf5RecentSeconds)) {
    hdf5RecentSeconds = Math.max(0, rawHdf5RecentSeconds)
  } else if (typeof rawHdf5RecentSeconds === 'string' && rawHdf5RecentSeconds.trim()) {
    const n = Number.parseFloat(rawHdf5RecentSeconds)
    if (Number.isFinite(n)) hdf5RecentSeconds = Math.max(0, n)
  }
  const rawSource = r['high_level_source'] ?? r['highLevelSource']
  const highLevelSource: 'gpt' | 'service' = rawSource === 'service' ? 'service' : 'gpt'
  const rawModel = r['gpt_model'] ?? r['gptModel']
  const gptModel = typeof rawModel === 'string' && rawModel.trim() ? rawModel.trim() : 'gpt-5.4'
  const rawImageMode = r['gpt_image_mode'] ?? r['gptImageMode']
  const gptImageMode: 'high_only' | 'all_cameras' =
    rawImageMode === 'high_only' ? 'high_only' : 'all_cameras'
  const forced = r['forced_low_level_subtask'] ?? r['forcedLowLevelSubtask']
  const ann = r['announcement_language'] ?? r['announcementLanguage']
  const announcementLanguage: 'zh' | 'ja' = ann === 'ja' ? 'ja' : 'zh'
  const apiBase = str('apiBase', 'api_base')
  const wsBase = str('wsBase', 'ws_base')
  const rawCam = r['camera_refresh_ms'] ?? r['cameraRefreshMs']
  let cameraRefreshMs = 100
  if (typeof rawCam === 'number' && Number.isFinite(rawCam)) {
    cameraRefreshMs = Math.max(100, Math.floor(rawCam))
  } else if (typeof rawCam === 'string' && rawCam.trim()) {
    const n = Number.parseInt(rawCam, 10)
    if (Number.isFinite(n)) cameraRefreshMs = Math.max(100, n)
  }
  const uiLangRaw = r['ui_language'] ?? r['uiLanguage']
  const uiLanguage: AppLanguage =
    uiLangRaw === 'en' || uiLangRaw === 'ja' || uiLangRaw === 'zh' ? uiLangRaw : 'en'
  return {
    datasetDir: str('datasetDir', 'dataset_dir'),
    manualDatasetDir: str('manualDatasetDir', 'manual_dataset_dir'),
    hdf5RecentSeconds,
    includeBottleDescription: bool('includeBottleDescription', 'include_bottle_description', true),
    lockBottleDescription: bool('lockBottleDescription', 'lock_bottle_description', true),
    includeBottlePosition: bool('includeBottlePosition', 'include_bottle_position', false),
    includeBottleState: bool('includeBottleState', 'include_bottle_state', true),
    includeSubtask: bool('includeSubtask', 'include_subtask', true),
    videoMemoryNumFrames,
    highLevelSource,
    gptModel,
    gptImageMode,
    forcedLowLevelSubtask:
      typeof forced === 'string' && forced.trim() ? forced.trim() : null,
    subtaskCatalog: parseSubtaskCatalogFromApi(r['subtask_catalog'] ?? r['subtaskCatalog']),
    stateSubtaskPairs: parseStateSubtaskPairsFromApi(r['state_subtask_pairs'] ?? r['stateSubtaskPairs']),
    announcementLanguage,
    apiBase,
    wsBase,
    cameraRefreshMs,
    uiLanguage,
  }
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
  apiBase: import.meta.env.VITE_API_BASE || defaultHttpOrigin,
  wsBase: import.meta.env.VITE_WS_BASE || defaultWsOrigin,
  cameraRefreshMs: 100,
  uiLanguage: 'en',
  datasetDir: '',
  manualDatasetDir: '',
  hdf5RecentSeconds: 5,
  includeBottleDescription: true,
  lockBottleDescription: true,
  includeBottlePosition: false,
  includeBottleState: true,
  includeSubtask: true,
  videoMemoryNumFrames: 1,
  highLevelSource: 'gpt',
  gptModel: 'gpt-5.4',
  gptImageMode: 'all_cameras',
  forcedLowLevelSubtask: null,
  subtaskCatalog: DEFAULT_SUBTASK_CATALOG.map((e) => ({ ...e })),
  stateSubtaskPairs: DEFAULT_STATE_SUBTASK_PAIRS.map((e) => ({ ...e })),
  announcementLanguage: 'zh',
}

function createInitialUiConfig(): UiConfig {
  let apiBase = defaultConfig.apiBase
  let wsBase = defaultConfig.wsBase
  if (apiBase === `http://${defaultHost}:8011`) {
    apiBase = defaultHttpOrigin
  }
  if (wsBase === `ws://${defaultHost}:8011`) {
    wsBase = defaultWsOrigin
  }
  return { ...defaultConfig, apiBase, wsBase }
}

function formatArray(values: number[], maxItems = 6) {
  if (!values.length) return '[]'
  const head = values.slice(0, maxItems).map((value) => value.toFixed(3)).join(', ')
  return values.length > maxItems ? `[${head}, ...]` : `[${head}]`
}

function formatHistoryTimestamp(ts: number | undefined) {
  if (ts == null || !Number.isFinite(ts)) return '—'
  return new Date(ts * 1000).toLocaleString()
}

async function drawJpegB64ToCanvas(b64: string, canvas: HTMLCanvasElement | null) {
  if (!canvas || !b64) return
  try {
    const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'image/jpeg' })
    const bmp = await createImageBitmap(blob)
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      bmp.close()
      return
    }
    if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
      canvas.width = bmp.width
      canvas.height = bmp.height
    }
    ctx.drawImage(bmp, 0, 0)
    bmp.close()
  } catch (err) {
    console.error('Failed to decode camera frame', err)
  }
}

async function drawCameraFramesToCanvases(
  frames: Record<string, string>,
  canvases: Record<string, HTMLCanvasElement | null>,
) {
  await Promise.all(
    Object.entries(frames).map(([name, b64]) => drawJpegB64ToCanvas(b64, canvases[name] ?? null)),
  )
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

function formatRuntimeModeLabel(
  mode: string | null | undefined,
  t: (typeof translations)[AppLanguage],
) {
  switch (mode) {
    case 'start_ai':
    case 'policy':
    case 'policy_waiting_subtask':
      return t.modeAiRunning
    case 'human_intervention':
    case 'teleop_prepare':
    case 'human_teleop':
      return t.modeManualControl
    case 'return_home':
      return t.modeHoming
    case 'return_sleep':
    case 'sleep':
      return t.modeSleeping
    case 'waiting':
    default:
      return t.waiting
  }
}

export default function App() {
  const [uiConfig, setUiConfig] = useState<UiConfig>(() => createInitialUiConfig())
  const [state, setState] = useState<RealtimeState>(initialState)
  const [command, setCommand] = useState('')
  const [voiceResponse, setVoiceResponse] = useState<VoiceResponse | null>(null)
  const [voiceStatus, setVoiceStatus] = useState<VoiceStatus>('idle')
  const [sending, setSending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [configOpen, setConfigOpen] = useState(false)
  const [avatarMenuOpen, setAvatarMenuOpen] = useState(false)
  const [consoleView, setConsoleView] = useState<'voice' | 'runtime'>('voice')
  const [selectedHistoryId, setSelectedHistoryId] = useState<string | null>(null)
  const [historyAutoScroll, setHistoryAutoScroll] = useState(true)
  const [cameraView, setCameraView] = useState<'focus' | 'quad'>('focus')
  const cameraCanvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({})
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const mediaChunksRef = useRef<Blob[]>([])
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const vadAnimationRef = useRef<number | null>(null)
  const vadSilenceStartedAtRef = useRef<number | null>(null)
  const historyListRef = useRef<HTMLDivElement>(null)
  const prevHistoryLenRef = useRef(0)
  const prevLastHistoryIdRef = useRef<string | null>(null)
  const prevSubtaskForAnnouncementRef = useRef<string | null>(null)
  const translatedAnnouncementCacheRef = useRef<Record<string, string>>({})
  /** Bumps when a new announce/translate should supersede an in-flight fetch (not on every WS tick). */
  const bottleAnnounceGenerationRef = useRef(0)
  const t = translations[uiConfig.uiLanguage]

  /** Subtask + description only; ignores noisy JSON churn in `low_level_prompt` from realtime pushes. */
  const bottleAnnounceStableKey = useMemo(() => {
    const p = parseStructuredLowLevelPrompt(state.robot.hierarchical?.low_level_prompt)
    if (!p) return ''
    const sub = (p.subtask ?? '').trim()
    const desc = (p.bottle_description ?? '').trim()
    if (!sub && !desc) return ''
    return `${sub}\0${desc}`
  }, [state.robot.hierarchical?.low_level_prompt])

  const bottleStartSubtasksSet = useMemo(() => {
    return new Set(
      uiConfig.subtaskCatalog
        .filter((e) => e.is_start_subtask)
        .map((e) => e.subtask.trim())
        .filter(Boolean),
    )
  }, [uiConfig.subtaskCatalog])

  const forcedSubtaskChoices = useMemo(() => {
    const seen = new Set<string>()
    const out: string[] = []
    for (const e of uiConfig.subtaskCatalog) {
      const s = e.subtask.trim()
      if (!s || seen.has(s)) continue
      seen.add(s)
      out.push(s)
    }
    return out
  }, [uiConfig.subtaskCatalog])

  const cameraCanvasBinders = useMemo(() => {
    const out: Record<string, (el: HTMLCanvasElement | null) => void> = {}
    for (const name of cameraNames) {
      out[name] = (el: HTMLCanvasElement | null) => {
        cameraCanvasRefs.current[name] = el
      }
    }
    return out
  }, [])

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
        const data = JSON.parse(event.data) as {
          robot: RealtimeState['robot']
          camera_status: Record<string, boolean>
          camera_timestamps?: Record<string, number | null>
          camera_jpeg_b64?: Record<string, string>
        }
        setState({
          robot: data.robot,
          camera_status: data.camera_status,
          camera_timestamps: data.camera_timestamps ?? {},
        })
        setError(null)
        const frames = data.camera_jpeg_b64
        if (frames && Object.keys(frames).length > 0) {
          void drawCameraFramesToCanvases(frames, cameraCanvasRefs.current)
        }
      } catch (err) {
        console.error('Failed to parse realtime payload', err)
      }
    }
    ws.onerror = () => setError('Realtime websocket disconnected')
    return () => ws.close()
  }, [uiConfig.wsBase])

  useEffect(() => {
    const log = (level: 'info' | 'debug', phase: string, extra: Record<string, unknown> = {}) => {
      const row = { phase, current_task: state.robot.current_task, ...extra }
      if (level === 'info') console.info('[bottle-announce]', row)
      else console.debug('[bottle-announce]', row)
    }

    if (state.robot.current_task !== 'Process all bottles') {
      log('debug', 'skip_wrong_task', { expected: 'Process all bottles' })
      prevSubtaskForAnnouncementRef.current = null
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel()
      }
      return
    }
    const rawLow = state.robot.hierarchical?.low_level_prompt
    const parsed = parseStructuredLowLevelPrompt(rawLow)
    if (!parsed) {
      const rawLen = rawLow != null ? String(rawLow).length : 0
      log(rawLen > 0 ? 'info' : 'debug', 'skip_unparseable_low_level', {
        raw_len: rawLen,
        raw_head: rawLow != null ? String(rawLow).slice(0, 120) : null,
      })
      return
    }
    const sub = parsed.subtask
    const desc = parsed.bottle_description ?? ''
    const prev = prevSubtaskForAnnouncementRef.current
    const inStart = sub != null && bottleStartSubtasksSet.has(sub)
    const prevInStart = prev != null && bottleStartSubtasksSet.has(prev)
    const enteredStart = inStart && !prevInStart

    // Do not update `prev` when we are in an "entered start" edge but bottle_description is not
    // ready yet. Otherwise the next tick sets prevInStart=true and we never announce (regression).
    if (!enteredStart) {
      log('debug', 'skip_not_entered_start', {
        sub,
        prev,
        inStart,
        prevInStart,
        start_subtask_count: bottleStartSubtasksSet.size,
      })
      prevSubtaskForAnnouncementRef.current = sub
      return
    }
    if (!desc.trim()) {
      log('info', 'wait_bottle_description', { sub, enteredStart: true, desc_len: 0 })
      return
    }
    prevSubtaskForAnnouncementRef.current = sub
    if (!('speechSynthesis' in window)) {
      log('info', 'skip_no_speech_synthesis', { sub, desc_len: desc.length })
      return
    }
    const targetLanguage = uiConfig.announcementLanguage
    const cacheKey = `${targetLanguage}:${desc}`
    const announcementTranslation = getAnnouncementTranslation(targetLanguage)
    const announcementLocale = getAnnouncementLocale(targetLanguage)
    const speak = (translatedDescription: string) => {
      const text = announcementTranslation.announceBottle(translatedDescription)
      log('info', 'speak', { sub, desc_len: desc.length, utterance_len: text.length, locale: announcementLocale })
      const utterance = new SpeechSynthesisUtterance(text)
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
      bottleAnnounceGenerationRef.current += 1
      log('info', 'speak_cached_translation', { sub, cacheKey: cacheKey.slice(0, 80) })
      speak(cached)
      return
    }
    log('info', 'translate_request', { sub, targetLanguage, api: `${uiConfig.apiBase}/api/runtime/translate` })
    const myGen = (bottleAnnounceGenerationRef.current += 1)
    void fetch(`${uiConfig.apiBase}/api/runtime/translate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: desc,
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
        if (myGen !== bottleAnnounceGenerationRef.current) {
          console.info('[bottle-announce]', {
            phase: 'translate_superseded',
            sub,
            myGen,
            currentGen: bottleAnnounceGenerationRef.current,
          })
          return
        }
        const translatedDescription = (payload.translated_text || desc).trim() || desc
        translatedAnnouncementCacheRef.current[cacheKey] = translatedDescription
        console.info('[bottle-announce]', {
          phase: 'translate_ok',
          sub,
          translated_len: translatedDescription.length,
        })
        speak(translatedDescription)
      })
      .catch((err: unknown) => {
        if (myGen !== bottleAnnounceGenerationRef.current) {
          console.info('[bottle-announce]', {
            phase: 'translate_error_superseded',
            sub,
            myGen,
            err: String(err),
          })
          return
        }
        console.info('[bottle-announce]', { phase: 'translate_failed_use_original', sub, err: String(err) })
        speak(desc)
      })
    return undefined
  }, [
    state.robot.current_task,
    bottleAnnounceStableKey,
    uiConfig.announcementLanguage,
    uiConfig.apiBase,
    bottleStartSubtasksSet,
  ])

  useEffect(() => {
    const controller = new AbortController()
    const loadRuntimeConfig = async () => {
      try {
        const response = await fetch(`${uiConfig.apiBase}/api/runtime/config`, { signal: controller.signal })
        if (!response.ok) {
          throw new Error(`HTTP ${response.status}`)
        }
        const runtimeConfig = parseRuntimeConfigResponse(await response.json())
        setUiConfig((current) => ({
          ...current,
          datasetDir: runtimeConfig.datasetDir || '',
          manualDatasetDir: runtimeConfig.manualDatasetDir || '',
          hdf5RecentSeconds: runtimeConfig.hdf5RecentSeconds,
          includeBottleDescription: runtimeConfig.includeBottleDescription,
          lockBottleDescription: runtimeConfig.lockBottleDescription,
          includeBottlePosition: runtimeConfig.includeBottlePosition,
          includeBottleState: runtimeConfig.includeBottleState,
          includeSubtask: runtimeConfig.includeSubtask,
          videoMemoryNumFrames: runtimeConfig.videoMemoryNumFrames,
          highLevelSource: runtimeConfig.highLevelSource,
          gptModel: runtimeConfig.gptModel,
          gptImageMode: runtimeConfig.gptImageMode,
          forcedLowLevelSubtask: runtimeConfig.forcedLowLevelSubtask || null,
          subtaskCatalog: runtimeConfig.subtaskCatalog.map((e) => ({ ...e })),
          stateSubtaskPairs: runtimeConfig.stateSubtaskPairs.map((e) => ({ ...e })),
          announcementLanguage: runtimeConfig.announcementLanguage,
          apiBase: runtimeConfig.apiBase.trim() || defaultHttpOrigin,
          wsBase: runtimeConfig.wsBase.trim() || defaultWsOrigin,
          cameraRefreshMs: runtimeConfig.cameraRefreshMs,
          uiLanguage: runtimeConfig.uiLanguage,
        }))
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          console.error('Failed to load runtime config', err)
        }
      }
    }
    void loadRuntimeConfig()
    return () => controller.abort()
  }, [uiConfig.apiBase])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === ',') {
        event.preventDefault()
        setConfigOpen((current) => !current)
      }
      if (event.key === 'Escape') {
        setConfigOpen(false)
        setAvatarMenuOpen(false)
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

  const freshness = useMemo(() => {
    if (!state.robot.timestamp) return t.waitingForRobot
    const age = Date.now() / 1000 - state.robot.timestamp
    return age < 1 ? t.live : t.stale(age.toFixed(1))
  }, [state.robot.timestamp, t])
  const robotLive = useMemo(() => {
    if (!state.robot.timestamp) return false
    return Date.now() / 1000 - state.robot.timestamp < 1
  }, [state.robot.timestamp])
  const runtimeModeLabel = useMemo(() => formatRuntimeModeLabel(state.robot.mode, t), [state.robot.mode, t])
  const camerasLive = useMemo(() => {
    if (!cameraNames.length) return false
    return cameraNames.every((name) => Boolean(state.camera_status[name]))
  }, [state.camera_status])

  const hierarchical = state.robot.hierarchical || {}
  const history = Array.isArray(hierarchical.history) ? hierarchical.history : []
  const selectedHistory =
    history.find((entry) => entry.id === selectedHistoryId) ||
    history[history.length - 1] ||
    null
  const historyLen = history.length
  const lastHistoryId = historyLen > 0 ? (history[history.length - 1]?.id ?? null) : null

  useLayoutEffect(() => {
    const el = historyListRef.current
    const prevLen = prevHistoryLenRef.current
    const prevLastId = prevLastHistoryIdRef.current
    const grew = historyLen > prevLen
    const lastChanged = lastHistoryId != null && lastHistoryId !== prevLastId

    prevHistoryLenRef.current = historyLen
    prevLastHistoryIdRef.current = lastHistoryId

    if (!el || !historyAutoScroll) return
    if (grew || lastChanged) {
      el.scrollTop = el.scrollHeight
    }
  }, [historyLen, lastHistoryId, historyAutoScroll])

  const primaryCamera = 'cam_high'
  const secondaryCameras = cameraNames.filter((name) => name !== primaryCamera)
  const displayFps = useMemo(() => {
    const intervalMs = Math.max(100, Number(uiConfig.cameraRefreshMs) || defaultConfig.cameraRefreshMs)
    return 1000 / intervalMs
  }, [uiConfig.cameraRefreshMs])
  const primaryCameraAge = formatCameraAge(state.camera_timestamps[primaryCamera])
  const primaryCameraStats = `${displayFps.toFixed(1)} FPS · ${primaryCameraAge}`
  const voiceTaskWithHighLevel = [voiceResponse?.task_name, hierarchical.high_level_text]
    .filter((value) => typeof value === 'string' && value.trim())
    .join('\n')

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
          language: uiConfig.uiLanguage,
          dataset_dir: uiConfig.datasetDir.trim() || undefined,
          manual_dataset_dir: uiConfig.manualDatasetDir.trim() || undefined,
          hdf5_recent_seconds: uiConfig.hdf5RecentSeconds,
          include_bottle_description: uiConfig.includeBottleDescription,
          lock_bottle_description: uiConfig.lockBottleDescription,
          include_bottle_position: uiConfig.includeBottlePosition,
          include_bottle_state: uiConfig.includeBottleState,
          include_subtask: uiConfig.includeSubtask,
          forced_low_level_subtask: uiConfig.forcedLowLevelSubtask || undefined,
          video_memory_num_frames: uiConfig.videoMemoryNumFrames,
          high_level_source: uiConfig.highLevelSource,
          gpt_model: uiConfig.gptModel.trim(),
          gpt_image_mode: uiConfig.gptImageMode,
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
          hdf5_recent_seconds: nextConfig.hdf5RecentSeconds,
          include_bottle_description: nextConfig.includeBottleDescription,
          lock_bottle_description: nextConfig.lockBottleDescription,
          include_bottle_position: nextConfig.includeBottlePosition,
          include_bottle_state: nextConfig.includeBottleState,
          include_subtask: nextConfig.includeSubtask,
          forced_low_level_subtask: nextConfig.forcedLowLevelSubtask,
          video_memory_num_frames: nextConfig.videoMemoryNumFrames,
          high_level_source: nextConfig.highLevelSource,
          gpt_model: nextConfig.gptModel.trim(),
          gpt_image_mode: nextConfig.gptImageMode,
          announcement_language: nextConfig.announcementLanguage,
          api_base: nextConfig.apiBase.trim(),
          ws_base: nextConfig.wsBase.trim(),
          camera_refresh_ms: nextConfig.cameraRefreshMs,
          ui_language: nextConfig.uiLanguage,
          subtask_catalog: nextConfig.subtaskCatalog
            .filter((e) => e.subtask.trim())
            .map((e) => ({
              subtask: e.subtask.trim(),
              is_start_subtask: e.is_start_subtask,
              ...(e.good_bad_action ? { good_bad_action: e.good_bad_action } : {}),
            })),
          state_subtask_pairs: nextConfig.stateSubtaskPairs
            .filter((p) => p.bottle_state.trim() && p.subtask.trim())
            .map((p) => ({
              bottle_state: p.bottle_state.trim(),
              subtask: p.subtask.trim(),
            })),
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
      formData.append('language', uiConfig.uiLanguage)
      if (uiConfig.datasetDir.trim()) {
        formData.append('dataset_dir', uiConfig.datasetDir.trim())
      }
      if (uiConfig.manualDatasetDir.trim()) {
        formData.append('manual_dataset_dir', uiConfig.manualDatasetDir.trim())
      }
      formData.append('include_bottle_description', String(uiConfig.includeBottleDescription))
      formData.append('hdf5_recent_seconds', String(uiConfig.hdf5RecentSeconds))
      formData.append('lock_bottle_description', String(uiConfig.lockBottleDescription))
      formData.append('include_bottle_position', String(uiConfig.includeBottlePosition))
      formData.append('include_bottle_state', String(uiConfig.includeBottleState))
      formData.append('include_subtask', String(uiConfig.includeSubtask))
      formData.append('video_memory_num_frames', String(uiConfig.videoMemoryNumFrames))
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
      <header className="app-header">
        <div className="header-brand">
          <h1>{t.title}</h1>
        </div>
        <div className="header-actions">
          <span className="status-pill mode">{runtimeModeLabel}</span>
          <div className="header-status-lamps">
            <span className="status-lamp-chip" title={freshness}>
              <span className={`lamp ${robotLive ? 'green' : 'red'}`} />
              <span>Robot</span>
            </span>
            <span className="status-lamp-chip" title={cameraNames.map((name) => `${name}:${state.camera_status[name] ? 'on' : 'off'}`).join(' | ')}>
              <span className={`lamp ${camerasLive ? 'green' : 'red'}`} />
              <span>Camera</span>
            </span>
          </div>
          {avatarMenuOpen ? <div className="avatar-menu-backdrop" onClick={() => setAvatarMenuOpen(false)} /> : null}
          <button
            type="button"
            className="avatar-button"
            onClick={() => setAvatarMenuOpen((current) => !current)}
            aria-label={t.menuSettings}
          >
            <span>AV</span>
          </button>
          {avatarMenuOpen ? (
            <div className="avatar-menu">
              <button
                type="button"
                className="avatar-menu-item"
                onClick={() => {
                  setConfigOpen(true)
                  setAvatarMenuOpen(false)
                }}
              >
                {t.openConfig}
              </button>
            </div>
          ) : null}
        </div>
      </header>

      <section className="layout">
        <section className={`stage-panel ${cameraView === 'quad' ? 'quad-mode' : 'focus-mode'}`}>
          <article className="hero-camera">
            <div className="camera-panel-header">
              <div>
                <p className="eyebrow">{t.camerasEyebrow}</p>
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
                <canvas
                  ref={cameraCanvasBinders[primaryCamera]}
                  className="camera-feed-canvas"
                  aria-label={t.cameraLabels[primaryCamera] || primaryCamera}
                />
                <div className="camera-frame-top">{renderCameraOverlay(primaryCamera)}</div>
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
                    <canvas
                      ref={cameraCanvasBinders[cameraName]}
                      className="camera-feed-canvas"
                      aria-label={t.cameraLabels[cameraName] || cameraName}
                    />
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
                  <canvas
                    ref={cameraCanvasBinders[cameraName]}
                    className="camera-feed-canvas"
                    aria-label={t.cameraLabels[cameraName] || cameraName}
                  />
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
              </div>
              <div className="console-switch">
                <button
                  type="button"
                  className={`ghost-button ${consoleView === 'voice' ? 'active' : ''}`}
                  onClick={() => setConsoleView('voice')}
                >
                  {t.consoleVoice}
                </button>
                <button
                  type="button"
                  className={`ghost-button ${consoleView === 'runtime' ? 'active' : ''}`}
                  onClick={() => setConsoleView('runtime')}
                >
                  {t.consoleRuntime}
                </button>
              </div>
            </div>
            {consoleView === 'voice' ? (
              <>
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
                  {[
                    ['1', t.quickTask1],
                    ['2', t.quickTask2],
                    ['3', t.quickTask3],
                    ['4', t.quickTask4],
                  ].map(([taskNum, label]) => (
                    <button key={taskNum} type="button" className="quick-task" onClick={() => void sendCommand(taskNum)}>
                      {label}
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
              </>
            ) : (
              <div className="runtime-grid runtime-inline">
                <div className="info-block wide">
                  <span className="info-label">{t.highLevelText}</span>
                  <pre>{hierarchical.high_level_text || 'N/A'}</pre>
                </div>
                <div className="info-block wide">
                  <span className="info-label">{t.lowLevelPrompt}</span>
                  <pre>{hierarchical.low_level_prompt || 'N/A'}</pre>
                </div>
                <div className="info-block wide">
                  <div className="history-block-header">
                    <span className="info-label">{t.subtaskHistory}</span>
                    <button
                      type="button"
                      className={`history-autoscroll-toggle ${historyAutoScroll ? 'active' : ''}`}
                      aria-pressed={historyAutoScroll}
                      onClick={() => setHistoryAutoScroll((v) => !v)}
                    >
                      {historyAutoScroll ? t.historyAutoScrollOn : t.historyAutoScrollOff}
                    </button>
                  </div>
                  <div className="history-list" ref={historyListRef}>
                    {history.length ? history.map((entry) => (
                      <button
                        key={entry.id}
                        type="button"
                        className={`history-item ${selectedHistory?.id === entry.id ? 'active' : ''}`}
                        onClick={() => setSelectedHistoryId(entry.id)}
                      >
                        <strong>{entry.high_level_text?.trim() || 'N/A'}</strong>
                        <span className="history-item-time">{formatHistoryTimestamp(entry.timestamp)}</span>
                        <span>{entry.task_prompt?.trim() || `obs ${entry.obs_version}`}</span>
                        <span className="history-item-qpos">
                          {t.qpos}: {formatArray(entry.qpos ?? [], 6)}
                        </span>
                      </button>
                    )) : <pre>N/A</pre>}
                  </div>
                </div>
                <div className="info-block wide">
                  <span className="info-label">{t.selectedHighLevelOutput}</span>
                  <pre>{selectedHistory?.high_level_text || 'N/A'}</pre>
                </div>
                <div className="info-block compact">
                  <span className="info-label">{t.historyEntryTime}</span>
                  <pre>{selectedHistory ? formatHistoryTimestamp(selectedHistory.timestamp) : 'N/A'}</pre>
                </div>
                <div className="info-block wide">
                  <span className="info-label">{t.historyEntryQpos}</span>
                  <pre>{selectedHistory ? formatArray(selectedHistory.qpos ?? [], 32) : 'N/A'}</pre>
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
            )}
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
              <select
                value={uiConfig.uiLanguage}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = { ...current, uiLanguage: event.target.value as AppLanguage }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
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
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      announcementLanguage: event.target.value as 'zh' | 'ja',
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
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
                  setUiConfig((current) => {
                    const nextConfig = { ...current, apiBase: event.target.value }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
                placeholder={`http://${defaultHost}:8011`}
              />
            </label>
            <label className="config-field">
              <span>{t.wsBaseLabel}</span>
              <input
                value={uiConfig.wsBase}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = { ...current, wsBase: event.target.value }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
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
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      cameraRefreshMs: Math.max(100, Number(event.target.value) || defaultConfig.cameraRefreshMs),
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
                placeholder="1000"
              />
            </label>
            <label className="config-field">
              <span>{t.highLevelSource}</span>
              <select
                value={uiConfig.highLevelSource}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextHighLevelSource: UiConfig['highLevelSource'] =
                      event.target.value === 'service' ? 'service' : 'gpt'
                    const nextConfig = {
                      ...current,
                      highLevelSource: nextHighLevelSource,
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                <option value="gpt">{t.highLevelSourceGpt}</option>
                <option value="service">{t.highLevelSourceService}</option>
              </select>
            </label>
            <label className="config-field">
              <span>{t.gptModel}</span>
              <input
                value={uiConfig.gptModel}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = { ...current, gptModel: event.target.value }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
                placeholder="gpt-5.4"
              />
            </label>
            <label className="config-field">
              <span>{t.gptImageMode}</span>
              <select
                value={uiConfig.gptImageMode}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextGptImageMode: UiConfig['gptImageMode'] =
                      event.target.value === 'high_only' ? 'high_only' : 'all_cameras'
                    const nextConfig = {
                      ...current,
                      gptImageMode: nextGptImageMode,
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                <option value="all_cameras">{t.gptImageModeAll}</option>
                <option value="high_only">{t.gptImageModeHighOnly}</option>
              </select>
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
              <span>{t.lockBottleDescription}</span>
              <select
                value={uiConfig.lockBottleDescription ? 'true' : 'false'}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextConfig = {
                      ...current,
                      lockBottleDescription: event.target.value === 'true',
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
              <span>{t.videoMemoryNumFrames}</span>
              <select
                value={String(uiConfig.videoMemoryNumFrames)}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const nextVideoMemoryNumFrames: UiConfig['videoMemoryNumFrames'] = event.target.value === '4' ? 4 : 1
                    const nextConfig = {
                      ...current,
                      videoMemoryNumFrames: nextVideoMemoryNumFrames,
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                <option value="1">{t.videoMemoryOneFrame}</option>
                <option value="4">{t.videoMemoryFourFrames}</option>
              </select>
            </label>
            <div className="config-field config-subtask-editor-block">
              <span>{t.subtaskCatalogSection}</span>
              <p className="config-help config-subtask-editor-help">{t.subtaskCatalogHelp}</p>
              <div className="subtask-catalog-rows">
                {uiConfig.subtaskCatalog.map((row, idx) => (
                  <div key={`cat-${idx}`} className="subtask-catalog-row">
                    <input
                      className="subtask-catalog-subtask-input"
                      value={row.subtask}
                      placeholder={t.subtaskTextLabel}
                      onChange={(event) =>
                        setUiConfig((current) => {
                          const nextCatalog = current.subtaskCatalog.map((e, i) =>
                            i === idx ? { ...e, subtask: event.target.value } : e,
                          )
                          const nextConfig = { ...current, subtaskCatalog: nextCatalog }
                          void pushRuntimeConfig(nextConfig)
                          return nextConfig
                        })
                      }
                    />
                    <label className="subtask-catalog-checkbox">
                      <input
                        type="checkbox"
                        checked={row.is_start_subtask}
                        onChange={(event) =>
                          setUiConfig((current) => {
                            const nextCatalog = current.subtaskCatalog.map((e, i) =>
                              i === idx ? { ...e, is_start_subtask: event.target.checked } : e,
                            )
                            const nextConfig = { ...current, subtaskCatalog: nextCatalog }
                            void pushRuntimeConfig(nextConfig)
                            return nextConfig
                          })
                        }
                      />
                      <span>{t.startSubtaskLabel}</span>
                    </label>
                    <select
                      className="subtask-catalog-goodbad"
                      value={row.good_bad_action ?? ''}
                      onChange={(event) =>
                        setUiConfig((current) => {
                          const v = event.target.value
                          const good_bad_action: SubtaskCatalogEntry['good_bad_action'] =
                            v === 'good action' || v === 'bad action' || v === 'normal' ? v : null
                          const nextCatalog = current.subtaskCatalog.map((e, i) =>
                            i === idx ? { ...e, good_bad_action } : e,
                          )
                          const nextConfig = { ...current, subtaskCatalog: nextCatalog }
                          void pushRuntimeConfig(nextConfig)
                          return nextConfig
                        })
                      }
                      aria-label={t.goodBadLabel}
                    >
                      <option value="">{t.goodBadDefault}</option>
                      <option value="good action">good action</option>
                      <option value="bad action">bad action</option>
                      <option value="normal">normal</option>
                    </select>
                    <button
                      type="button"
                      className="subtask-row-remove"
                      onClick={() =>
                        setUiConfig((current) => {
                          const nextCatalog = current.subtaskCatalog.filter((_, i) => i !== idx)
                          const nextConfig = { ...current, subtaskCatalog: nextCatalog }
                          void pushRuntimeConfig(nextConfig)
                          return nextConfig
                        })
                      }
                    >
                      {t.subtaskCatalogRemoveRow}
                    </button>
                  </div>
                ))}
              </div>
              <button
                type="button"
                className="subtask-add-row"
                onClick={() =>
                  setUiConfig((current) => {
                    const nextCatalog = [
                      ...current.subtaskCatalog,
                      { subtask: '', is_start_subtask: false, good_bad_action: null },
                    ]
                    const nextConfig = { ...current, subtaskCatalog: nextCatalog }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                {t.subtaskCatalogAddRow}
              </button>
            </div>
            <div className="config-field config-subtask-editor-block">
              <span>{t.stateSubtaskPairsSection}</span>
              <div className="state-subtask-pair-rows">
                {uiConfig.stateSubtaskPairs.map((row, idx) => (
                  <div key={`pair-${idx}`} className="state-subtask-pair-row">
                    <input
                      className="pair-bottle-state"
                      value={row.bottle_state}
                      placeholder={t.pairBottleState}
                      onChange={(event) =>
                        setUiConfig((current) => {
                          const nextPairs = current.stateSubtaskPairs.map((e, i) =>
                            i === idx ? { ...e, bottle_state: event.target.value } : e,
                          )
                          const nextConfig = { ...current, stateSubtaskPairs: nextPairs }
                          void pushRuntimeConfig(nextConfig)
                          return nextConfig
                        })
                      }
                    />
                    <input
                      className="pair-subtask"
                      value={row.subtask}
                      placeholder={t.pairSubtask}
                      onChange={(event) =>
                        setUiConfig((current) => {
                          const nextPairs = current.stateSubtaskPairs.map((e, i) =>
                            i === idx ? { ...e, subtask: event.target.value } : e,
                          )
                          const nextConfig = { ...current, stateSubtaskPairs: nextPairs }
                          void pushRuntimeConfig(nextConfig)
                          return nextConfig
                        })
                      }
                    />
                    <button
                      type="button"
                      className="subtask-row-remove"
                      onClick={() =>
                        setUiConfig((current) => {
                          const nextPairs = current.stateSubtaskPairs.filter((_, i) => i !== idx)
                          const nextConfig = { ...current, stateSubtaskPairs: nextPairs }
                          void pushRuntimeConfig(nextConfig)
                          return nextConfig
                        })
                      }
                    >
                      {t.pairRemoveRow}
                    </button>
                  </div>
                ))}
              </div>
              <button
                type="button"
                className="subtask-add-row"
                onClick={() =>
                  setUiConfig((current) => {
                    const nextPairs = [
                      ...current.stateSubtaskPairs,
                      { bottle_state: '', subtask: '' },
                    ]
                    const nextConfig = { ...current, stateSubtaskPairs: nextPairs }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              >
                {t.pairAddRow}
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
                {forcedSubtaskChoices.map((subtask) => (
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
            <label className="config-field">
              <span>{t.hdf5RecentSeconds}</span>
              <input
                min="0"
                step="0.5"
                type="number"
                value={String(uiConfig.hdf5RecentSeconds)}
                onChange={(event) =>
                  setUiConfig((current) => {
                    const parsed = Number.parseFloat(event.target.value)
                    const nextConfig = {
                      ...current,
                      hdf5RecentSeconds: Number.isFinite(parsed) ? Math.max(0, parsed) : 0,
                    }
                    void pushRuntimeConfig(nextConfig)
                    return nextConfig
                  })
                }
              />
            </label>
            <p className="config-help">{t.configHelp}</p>
          </aside>
        </>
      ) : null}
    </main>
  )
}
