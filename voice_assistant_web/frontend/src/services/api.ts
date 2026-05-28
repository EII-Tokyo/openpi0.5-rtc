const trimSlash = (value: string) => value.replace(/\/+$/, '')

const browserHost = typeof window !== 'undefined' ? window.location.hostname : 'localhost'
const browserProtocol = typeof window !== 'undefined' ? window.location.protocol : 'http:'
const browserWsProtocol =
  typeof window !== 'undefined' && window.location.protocol === 'https:' ? 'wss:' : 'ws:'

export const apiBase = trimSlash(
  (import.meta.env.VITE_API_BASE as string | undefined) || `${browserProtocol}//${browserHost}:8011`,
)

export const wsBase = (() => {
  const envBase = (import.meta.env.VITE_WS_BASE as string | undefined)?.replace(/\/+$/, '')
  if (envBase) return envBase
  return `${browserWsProtocol}//${browserHost}:8011`
})()

export type RolloutNode = {
  name: string
  path: string
  type: 'directory' | 'file'
  extension?: string
  size?: number
  modified?: number
  children?: RolloutNode[]
}

export const rolloutVideoUrl = (path: string) => `${apiBase}/api/rollouts/video?path=${encodeURIComponent(path)}`

export type RLTEvent = {
  timestamp: number
  event: string
  detail: string
}

export type RLTControlState = {
  phase: 'idle' | 'key_region' | 'await_score' | string
  training_phase: 'warmup' | 'rl' | string
  warmup_target: number
  warmup_count: number
  warmup_success: number
  warmup_failure: number
  auto_rollout_count: number
  auto_rollout_success: number
  auto_rollout_failure: number
  actor_enabled: boolean
  actor_effective: boolean
  actor_locked_reason: string | null
  beta: number
  intervention_scale: number
  max_delta: number
  active_key_region_id: string | null
  score_deadline: number | null
  last_reward: number | null
  last_event: string | null
  wandb_url: string | null
  critic_loss: number | null
  actor_loss: number | null
  replay_size: number | null
  events: RLTEvent[]
}

const postJson = async <T>(path: string, body: unknown = {}): Promise<T> => {
  const response = await fetch(`${apiBase}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!response.ok) {
    const message = await response.text().catch(() => '')
    throw new Error(message || `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const startKeyRegion = () => postJson<RLTControlState>('/api/rlt/key-region/start', { source: 'ui' })
export const endKeyRegion = () => postJson<RLTControlState>('/api/rlt/key-region/end', { source: 'ui' })
export const scoreKeyRegion = (reward: 0 | 1) =>
  postJson<RLTControlState>('/api/rlt/key-region/score', { reward, source: 'ui' })
export const updateRLTConfig = (config: Partial<RLTControlState>) =>
  postJson<RLTControlState>('/api/rlt/config', config)
