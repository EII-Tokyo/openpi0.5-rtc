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
  manifest_summary?: RolloutManifestSummary
  children?: RolloutNode[]
}

export type RolloutManifestSummary = {
  key_region_id?: string
  task?: string
  phase?: string
  reward?: number
  score_timeout?: boolean
  start_time?: number
  end_time?: number
  score_time?: number
  num_frames?: number
  num_replay_transitions?: number
  fps?: number
  duration_seconds?: number
}

export const rolloutTreeUrl = (path?: string) => {
  const query = path ? `?path=${encodeURIComponent(path)}` : ''
  return `${apiBase}/api/rollouts/tree${query}`
}

export const rolloutVideoUrl = (path: string) => `${apiBase}/api/rollouts/video?path=${encodeURIComponent(path)}`

export type RLTEvent = {
  timestamp: number
  event: string
  detail: string
}

export type RLTSegmentRecord = {
  key_region_id: string
  status: string
  phase: string
  reward: number | null
  shard_path: string | null
  num_replay_transitions: number
  invalid_reason: string | null
  created_at: number
  updated_at: number
}

export type RLTKeyRegionReviewRecord = {
  key_region_id: string
  status: string
  trainable: boolean
  incomplete_reason: string | null
  phase: string | null
  reward: number | null
  shard_path: string | null
  npz_exists: boolean
  video_exists: boolean
  manifest_exists: boolean
  rollout_path: string | null
  segment_status: string | null
  train_eligible: boolean | null
  replay_status: string | null
  missing_rlt_metadata: string[]
  voided: boolean | null
  default_video_path: string | null
  video_paths: string[]
  task: string | null
  start_time: number | null
  end_time: number | null
  score_time: number | null
  duration_seconds: number | null
  key_region_duration_seconds: number | null
  key_region_start_sec: number | null
  key_region_end_sec: number | null
  fps: number | null
  num_frames: number | null
  crop_start_sec: number | null
  crop_end_sec: number | null
  crop_start_sample: number | null
  crop_end_sample: number | null
  crop_original_num_replay_transitions: number | null
  num_replay_transitions: number
  updated_at: number | null
}

export type RLTKeyRegionReviewSummary = {
  total: number
  trainable: number
  needs_crop: number
  success: number
  failure: number
  replay_samples: number
}

export type RLTKeyRegionReviewPage = {
  items: RLTKeyRegionReviewRecord[]
  total: number
  limit: number
  offset: number
  next_offset: number | null
  summary: RLTKeyRegionReviewSummary
}

export type RLTKeyRegionReviewQuery = {
  limit?: number
  offset?: number
  status?: 'all' | 'trainable' | 'needsCrop'
  reward?: 'all' | 'success' | 'failure'
}

export type RLTKeyRegionCropResponse = {
  key_region_id: string
  status: string
  trainable: boolean
  shard_path: string
  source_shard_path: string
  crop_start_sec: number
  crop_end_sec: number
  crop_start_sample: number
  crop_end_sample: number
  num_replay_transitions: number
  manifest_path: string
}

export type RLTControlState = {
  phase: 'idle' | 'key_region' | 'await_score' | 'pending_replay' | string
  training_phase: 'warmup' | 'rl' | string
  warmup_target: number
  warmup_count: number
  warmup_success: number
  warmup_failure: number
  warmup_attempts: number
  warmup_invalid: number
  auto_rollout_count: number
  auto_rollout_success: number
  auto_rollout_failure: number
  auto_rollout_attempts: number
  auto_rollout_invalid: number
  trainer_enabled: boolean
  trainer_running: boolean
  actor_enabled: boolean
  actor_effective: boolean
  actor_ready: boolean
  actor_locked_reason: string | null
  beta: number
  auto_beta_enabled: boolean
  auto_beta_target_delta_norm: number | null
  auto_beta_min: number
  auto_beta_max: number
  auto_beta_lr: number
  auto_beta_ema_decay: number
  auto_beta_update_interval: number
  auto_beta_q_margin: number
  auto_beta_delta_norm_ema: number | null
  auto_beta_q_advantage_ema: number | null
  auto_beta_critic_loss_ema: number | null
  auto_beta_reason: string | null
  intervention_scale: number
  max_delta: number
  critic_gate_enabled: boolean
  critic_gate_margin: number
  critic_gate_temperature: number
  critic_ready: boolean
  inference_actor_active: boolean
  inference_delta_norm: number | null
  inference_gate_reason: string | null
  key_region_probability: number | null
  loaded_actor_step: number | null
  inference_reference_q_value: number | null
  inference_actor_q_value: number | null
  inference_q_advantage: number | null
  active_key_region_id: string | null
  score_deadline: number | null
  last_reward: number | null
  last_event: string | null
  wandb_url: string | null
  critic_loss: number | null
  critic_q1_loss: number | null
  critic_q2_loss: number | null
  actor_loss: number | null
  actor_q_value: number | null
  reference_q_value: number | null
  q_advantage: number | null
  actor_delta_norm: number | null
  q1_mean: number | null
  q2_mean: number | null
  target_q_mean: number | null
  q_gap: number | null
  actor_updated: boolean | null
  publish_actor: boolean | null
  trainer_step: number | null
  critic_burn_in_steps: number | null
  target_sync_step: number | null
  steps_per_sec: number | null
  success_episodes: number | null
  failure_episodes: number | null
  replay_action_horizon: number | null
  train_action_horizon: number | null
  rlt_metrics_timestamp: number | null
  replay_size: number | null
  replay_shards: number | null
  bad_shards: number | null
  trainable_replay_count: number
  trainable_replay_success: number
  trainable_replay_failure: number
  trainable_replay_samples: number
  trainable_replay_shards: number
  invalid_replay_shards: number
  actor_checkpoint_path: string | null
  actor_checkpoint_step: number | null
  rl_token_checkpoint_path: string | null
  events: RLTEvent[]
}

export type RLTConfigRequest = {
  warmup_target?: number
  beta?: number
  auto_beta_enabled?: boolean
  auto_beta_target_delta_norm?: number
  auto_beta_min?: number
  auto_beta_max?: number
  auto_beta_lr?: number
  auto_beta_ema_decay?: number
  auto_beta_update_interval?: number
  auto_beta_q_margin?: number
  critic_burn_in_steps?: number
  actor_enabled?: boolean
  trainer_enabled?: boolean
  intervention_scale?: number
  max_delta?: number
  critic_gate_enabled?: boolean
  critic_gate_margin?: number
  critic_gate_temperature?: number
  wandb_url?: string | null
}

const getJson = async <T>(path: string): Promise<T> => {
  const response = await fetch(`${apiBase}${path}`)
  if (!response.ok) {
    const message = await response.text().catch(() => '')
    throw new Error(message || `HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
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

export const fetchRLTSegments = () => getJson<RLTSegmentRecord[]>('/api/rlt/segments')
export const fetchRLTKeyRegionReview = (query: RLTKeyRegionReviewQuery = {}) => {
  const params = new URLSearchParams()
  if (query.limit !== undefined) params.set('limit', String(query.limit))
  if (query.offset !== undefined) params.set('offset', String(query.offset))
  if (query.status && query.status !== 'all') params.set('status', query.status)
  if (query.reward && query.reward !== 'all') params.set('reward', query.reward)
  const suffix = params.toString() ? `?${params.toString()}` : ''
  return getJson<RLTKeyRegionReviewPage>(`/api/rlt/key-regions/review${suffix}`)
}
export const fetchRLTKeyRegionDetail = (keyRegionId: string) =>
  getJson<RLTKeyRegionReviewRecord>(`/api/rlt/key-region/${encodeURIComponent(keyRegionId)}`)
export const startKeyRegion = () => postJson<RLTControlState>('/api/rlt/key-region/start', { source: 'ui' })
export const endKeyRegion = () => postJson<RLTControlState>('/api/rlt/key-region/end', { source: 'ui' })
export const scoreKeyRegion = (reward: 0 | 1) =>
  postJson<RLTControlState>('/api/rlt/key-region/score', { reward, source: 'ui' })
export const confirmKeyRegion = () =>
  postJson<RLTControlState>('/api/rlt/key-region/confirm', { source: 'ui' })
export const discardKeyRegion = (reason = 'operator_discard') =>
  postJson<RLTControlState>('/api/rlt/key-region/discard', { source: 'ui', reason })
export const voidKeyRegion = (keyRegionId: string, reason = 'operator_void') =>
  postJson<RLTControlState>(`/api/rlt/key-region/${encodeURIComponent(keyRegionId)}/void`, { source: 'ui', reason })
export const deleteKeyRegions = (keyRegionIds: string[], reason = 'operator_delete') =>
  postJson<RLTControlState>('/api/rlt/key-regions/delete', { key_region_ids: keyRegionIds, source: 'ui', reason })
export const cropKeyRegion = (keyRegionId: string, startSec: number, endSec: number) =>
  postJson<RLTKeyRegionCropResponse>(`/api/rlt/key-region/${encodeURIComponent(keyRegionId)}/crop`, {
    start_sec: startSec,
    end_sec: endSec,
    source: 'ui',
    reason: 'operator_crop',
  })
export const rescoreKeyRegion = (keyRegionId: string, reward: 0 | 1) =>
  postJson<RLTControlState>(`/api/rlt/key-region/${encodeURIComponent(keyRegionId)}/rescore`, {
    reward,
    source: 'ui',
    reason: 'operator_rescore',
  })
export const updateRLTConfig = (config: RLTConfigRequest) =>
  postJson<RLTControlState>('/api/rlt/config', config)

export const sendRobotTask = (taskNum: '1' | '4' | '5') =>
  postJson<{ status: string; task_num: string; task_name: string }>('/api/robot/task', {
    task_num: taskNum,
    source: 'ui',
  })
