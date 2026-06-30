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
export const cameraStreamUrl = (cameraName: string, fps?: number) => {
  const query = fps ? `?fps=${encodeURIComponent(String(fps))}` : ''
  return `${apiBase}/api/cameras/${encodeURIComponent(cameraName)}/stream.mjpg${query}`
}

export const browserReachableMediaBase = (mediaServiceUrl?: string | null) => {
  const fallback = `${browserProtocol}//${browserHost}:8013`
  if (!mediaServiceUrl) return fallback
  try {
    const url = new URL(mediaServiceUrl)
    if (url.hostname === '127.0.0.1' || url.hostname === 'localhost') {
      url.hostname = browserHost
      url.protocol = browserProtocol
    }
    return trimSlash(url.toString())
  } catch {
    return fallback
  }
}

export const cameraWebrtcOfferUrl = (cameraName: string, mediaServiceUrl?: string | null) =>
  `${browserReachableMediaBase(mediaServiceUrl)}/api/media/aiortc/ros-camera/${encodeURIComponent(cameraName)}/offer`

export const cameraWebrtcSessionUrl = (sessionId: string, mediaServiceUrl?: string | null) =>
  `${browserReachableMediaBase(mediaServiceUrl)}/api/media/aiortc/sessions/${encodeURIComponent(sessionId)}`

export type CameraTransport = 'webrtc' | 'mjpeg' | 'jpeg_ws'

export type CameraCapabilitiesResponse = {
  preferred_transport: CameraTransport
  transports: CameraTransport[]
  cameras: string[]
  include_realtime_frames: boolean
  webrtc: {
    enabled?: boolean
    codec?: string
    ice_servers?: unknown[]
    [key: string]: unknown
  }
}

export const fetchCameraCapabilities = () => getJson<CameraCapabilitiesResponse>('/api/cameras/capabilities')

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
  batch: string | null
  status: string
  trainable: boolean
  needs_crop: boolean
  incomplete_reason: string | null
  phase: string | null
  reward: number | null
  shard_path: string | null
  npz_exists: boolean
  video_exists: boolean
  manifest_exists: boolean
  rollout_path: string | null
  local_rollout_path: string | null
  local_shard_path: string | null
  actor_inference_kind: string | null
  actor_delta_p95: number | null
  actor_delta_max: number | null
  actor_delta_mean: number | null
  has_intervention_metadata: boolean
  has_action_source: boolean
  has_takeover_id: boolean
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
  review_datetime: string | null
  start_datetime: string | null
  score_datetime: string | null
  crop_datetime: string | null
  updated_datetime: string | null
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
  batches: string[]
}

export type RLTKeyRegionReviewQuery = {
  limit?: number
  offset?: number
  status?: 'all' | 'trainable' | 'needsCrop' | 'noActor' | 'actorModified'
  reward?: 'all' | 'success' | 'failure'
  batch?: string
  search?: string
  focusKeyRegionId?: string
}

export type RLTExpertDemoRecord = {
  episode_key: string
  dataset_id: string
  episode_index: number
  fps: number | null
  num_frames: number | null
  duration_seconds: number | null
  video_paths: string[]
  local_video_paths: string[]
  video_start_secs: number[]
  camera_count: number
  missing_cameras: string[]
  camera_complete: boolean
  source_dataset_path: string
  saved_crop_count: number
  saved_crop_start_sec: number | null
  saved_crop_end_sec: number | null
  saved_crop_reward: number | null
}

export type RLTExpertDemoPage = {
  items: RLTExpertDemoRecord[]
  total: number
  limit: number
  offset: number
  next_offset: number | null
  datasets: string[]
  crop_summary: {
    total_episodes: number
    cropped_episodes: number
    remaining_episodes: number
    saved_crops: number
  }
}

export type RLTExpertDemoCropResponse = {
  dataset_id: string
  episode_index: number
  start_sec: number
  end_sec: number
  reward: number
  label: string
  metadata_path: string | null
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

export type RLTKeyRegionCameraMedia = {
  camera: string
  video_path: string
  frame_url: string
}

export type RLTKeyRegionMediaMetadata = {
  key_region_id: string
  fps: number | null
  frame_count: number | null
  duration_seconds: number | null
  cameras: RLTKeyRegionCameraMedia[]
}

export type RLTPreferenceStats = {
  total_preferences: number
  left_wins: number
  right_wins: number
  ties: number
  both_bad: number
  skipped: number
}

export type RLTPreferencePairResponse = {
  left: RLTKeyRegionReviewRecord | null
  right: RLTKeyRegionReviewRecord | null
  stats: RLTPreferenceStats
  remaining_unseen_pairs: number
  pair_type: string | null
  strategy: string
  round_budget: number
  round_labeled: number
  round_remaining: number
}

export type RLTPreferenceRequest = {
  left_key_region_id: string
  right_key_region_id: string
  preference: 'left' | 'right' | 'tie' | 'both_bad' | 'skip'
  reason_tags?: string[]
  notes?: string
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
  online_safety_enabled: boolean
  online_safety_phase: string | null
  online_round_index: number | null
  online_last_committed_shards: number | null
  online_last_committed_success: number | null
  online_last_committed_failure: number | null
  online_round_start_shards: number | null
  online_round_start_success: number | null
  online_round_start_failure: number | null
  online_critic_steps_remaining: number | null
  online_actor_steps_remaining: number | null
  online_best_critic_auc: number | null
  online_best_critic_q_gap: number | null
  online_rejection_reason: string | null
  online_target_delta_norm: number | null
  online_min_new_shards_per_round: number
  online_min_new_success_per_round: number
  online_min_new_failure_per_round: number
  online_critic_updates_per_round: number
  online_actor_updates_per_round: number
  online_critic_auc_min: number
  online_critic_max_auc_drop: number
  online_require_positive_q_gap: boolean
  online_actor_max_delta_norm: number
  online_actor_min_q_advantage: number
  online_beta_initial: number
  online_beta_min: number
  online_beta_max: number
  online_beta_decay_on_actor_accept: number
  online_beta_increase_on_reject: number
  online_target_delta_initial: number
  online_target_delta_max: number
  online_target_delta_increment: number
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
  actor_handoff_steps: number
  actor_delta_ema_alpha: number
  actor_speed_limit_preset: 'off' | '80' | '50' | '20'
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
  online_safety_enabled?: boolean
  online_min_new_shards_per_round?: number
  online_min_new_success_per_round?: number
  online_min_new_failure_per_round?: number
  online_critic_updates_per_round?: number
  online_actor_updates_per_round?: number
  online_critic_auc_min?: number
  online_critic_max_auc_drop?: number
  online_require_positive_q_gap?: boolean
  online_actor_max_delta_norm?: number
  online_actor_min_q_advantage?: number
  online_beta_initial?: number
  online_beta_min?: number
  online_beta_max?: number
  online_beta_decay_on_actor_accept?: number
  online_beta_increase_on_reject?: number
  online_target_delta_initial?: number
  online_target_delta_max?: number
  online_target_delta_increment?: number
  intervention_scale?: number
  max_delta?: number
  actor_handoff_steps?: number
  actor_delta_ema_alpha?: number
  actor_speed_limit_preset?: 'off' | '80' | '50' | '20'
  critic_gate_enabled?: boolean
  critic_gate_margin?: number
  critic_gate_temperature?: number
  wandb_url?: string | null
}

const getJson = async <T>(path: string): Promise<T> => {
  const response = await fetch(`${apiBase}${path}`, { cache: 'no-store' })
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
  if (query.batch && query.batch !== 'all') params.set('batch', query.batch)
  if (query.search?.trim()) params.set('search', query.search.trim())
  if (query.focusKeyRegionId) params.set('focus_key_region_id', query.focusKeyRegionId)
  const suffix = params.toString() ? `?${params.toString()}` : ''
  return getJson<RLTKeyRegionReviewPage>(`/api/rlt/key-regions/review${suffix}`)
}
export const fetchRLTKeyRegionDetail = (keyRegionId: string) =>
  getJson<RLTKeyRegionReviewRecord>(`/api/rlt/key-region/${encodeURIComponent(keyRegionId)}`)
export const fetchRLTExpertDemoReview = (query: { limit?: number; offset?: number; dataset?: string; search?: string; cameraStatus?: string } = {}) => {
  const params = new URLSearchParams()
  if (query.limit !== undefined) params.set('limit', String(query.limit))
  if (query.offset !== undefined) params.set('offset', String(query.offset))
  if (query.dataset && query.dataset !== 'all') params.set('dataset', query.dataset)
  if (query.search?.trim()) params.set('search', query.search.trim())
  if (query.cameraStatus && query.cameraStatus !== 'complete') params.set('camera_status', query.cameraStatus)
  const suffix = params.toString() ? `?${params.toString()}` : ''
  return getJson<RLTExpertDemoPage>(`/api/rlt/expert-demos/review${suffix}`)
}
export const cropExpertDemo = (datasetId: string, episodeIndex: number, startSec: number, endSec: number, reward = 1) =>
  postJson<RLTExpertDemoCropResponse>(
    `/api/rlt/expert-demos/${encodeURIComponent(datasetId)}/${encodeURIComponent(String(episodeIndex))}/crop`,
    {
      start_sec: startSec,
      end_sec: endSec,
      reward,
    },
  )
export const fetchRLTKeyRegionMediaMetadata = (keyRegionId: string) =>
  getJson<RLTKeyRegionMediaMetadata>(`/api/rlt/key-region/${encodeURIComponent(keyRegionId)}/media-metadata`)
export const keyRegionFrameUrl = (keyRegionId: string, camera: string, frame: number) =>
  `${apiBase}/api/rlt/key-region/${encodeURIComponent(keyRegionId)}/frame?camera=${encodeURIComponent(camera)}&frame=${encodeURIComponent(String(frame))}`
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
export const fetchRLTPreferencePair = (query: { batch?: string; reward?: string; pairType?: string } = {}) => {
  const params = new URLSearchParams()
  if (query.batch && query.batch !== 'all') params.set('batch', query.batch)
  if (query.reward && query.reward !== 'all') params.set('reward', query.reward)
  if (query.pairType && query.pairType !== 'auto') params.set('pair_type', query.pairType)
  const suffix = params.toString() ? `?${params.toString()}` : ''
  return getJson<RLTPreferencePairResponse>(`/api/rlt/preferences/next-pair${suffix}`)
}
export const saveRLTPreference = (request: RLTPreferenceRequest) =>
  postJson('/api/rlt/preferences', {
    ...request,
    source: 'ui',
  })
export const updateRLTConfig = (config: RLTConfigRequest) =>
  postJson<RLTControlState>('/api/rlt/config', config)

export const sendRobotTask = (taskNum: '1' | '4' | '5' | '9') =>
  postJson<{ status: string; task_num: string; task_name: string }>('/api/robot/task', {
    task_num: taskNum,
    source: 'ui',
  })
