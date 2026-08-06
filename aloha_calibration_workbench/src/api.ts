export type CameraRole = 'cam_high' | 'cam_low' | 'wrist_left' | 'wrist_right'

export interface PreflightCamera {
  role: CameraRole
  connected: boolean
  identity_match: boolean
  production_profile_supported: boolean
  ownership: 'FREE' | 'ROS_SOURCE' | 'BUSY' | 'UNKNOWN'
}

export interface PreflightSession {
  id: string
  state: 'PREFLIGHT_READY' | 'PREFLIGHT_BLOCKED' | 'PREFLIGHT_FAILED'
  latest_preflight: {
    status: 'READY' | 'BLOCKED' | 'FAILED'
    cameras: PreflightCamera[]
    issues: Array<{ code: string; severity: string; camera_role?: string | null }>
  }
}

export interface FactorySnapshotBundle {
  status: 'FACTORY_INTRINSICS_FROZEN'
  cameras: Array<{ role: CameraRole; serial: string }>
}

export interface TransformRecord {
  source_frame: string
  target_frame: string
  matrix: number[][]
  length_unit?: 'meter'
  matrix_order?: 'row-major'
  vector_convention?: 'column-vector'
  quaternion_order?: 'wxyz'
}

export interface WorldOriginResult {
  status: 'WORLD_ORIGIN_SOLVED'
  world_from_camera: TransformRecord
  accepted_frames: number
  total_frames: number
  median_reprojection_rms_px: number
  p95_reprojection_rms_px: number
  translation_jitter_m: number
  rotation_jitter_deg: number
}

export interface FrozenTableContract {
  status: 'TABLE_POINT_CONTRACT_FROZEN'
  contract_sha256: string
}

export interface TableResult {
  status: 'WORLD_REGISTRATION_VALIDATED'
  validation_scope: 'tabletop-xy-cross-validation'
  held_out_rms_m: number
  held_out_max_m: number
  refinement_translation_m: number
  refinement_rotation_deg: number
}

export interface TableSnapshot {
  blob: Blob
  attemptId: string
  frameNumber: number
  deviceTimestampMs: number
  imageSha256: string
}

export interface FrozenBottleContract {
  status: 'BOTTLE_FIXTURE_CONTRACT_FROZEN'
  contract_sha256: string
}

export interface BottleCaptureResult {
  observation: { id: 'B-A' | 'B-B' | 'B-C'; camera_from_tag: TransformRecord }
  stability: { accepted_frames: number; translation_jitter_m: number; rotation_jitter_deg: number }
}

export interface BottleValidationResult {
  status: 'TAGGED_FIXTURE_TRANSFER_PASS'
  claim_scope: 'tagged-rigid-fixture-transfer-only'
  center_rms_m: number
  long_axis_rms_deg: number
  support_max_abs_m: number
}

export interface ExportResult {
  calibration_json: string
  calibration_layer: string
  review_stage: string
  source_stage_sha256: string
}

async function postJson<T>(url: string, body?: unknown): Promise<T> {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    ...(body === undefined ? {} : { body: JSON.stringify(body) }),
  })
  if (!response.ok) {
    const message = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(message?.detail ?? `Request failed: HTTP ${response.status}`)
  }
  return response.json() as Promise<T>
}

export const runPreflightSession = () => postJson<PreflightSession>('/api/preflight-session')

export const freezeFactoryIntrinsics = (sessionId: string) =>
  postJson<FactorySnapshotBundle>(`/api/sessions/${sessionId}/actions/factory/freeze`)

export const captureAndSolveWorldOrigin = (
  sessionId: string,
  input: { tag_size_m: number; tag_plane_height_m: number; frame_count: number },
) => postJson<WorldOriginResult>(`/api/sessions/${sessionId}/actions/world-origin/capture-solve`, input)

export const freezeTableContract = (sessionId: string, input: unknown) =>
  postJson<FrozenTableContract>(`/api/sessions/${sessionId}/actions/table-contract/freeze`, input)

export async function captureTableSnapshot(sessionId: string): Promise<TableSnapshot> {
  const response = await fetch(`/api/sessions/${sessionId}/actions/table/snapshot`, { method: 'POST' })
  if (!response.ok) {
    const message = await response.json().catch(() => null) as { detail?: string } | null
    throw new Error(message?.detail ?? `Request failed: HTTP ${response.status}`)
  }
  return {
    blob: await response.blob(),
    attemptId: response.headers.get('X-Attempt-Id') ?? 'UNKNOWN',
    frameNumber: Number(response.headers.get('X-Frame-Number') ?? 0),
    deviceTimestampMs: Number(response.headers.get('X-Device-Timestamp-Ms') ?? 0),
    imageSha256: response.headers.get('X-Image-Sha256') ?? '',
  }
}

export const solveTableRegistration = (sessionId: string, input: unknown) =>
  postJson<TableResult>(`/api/sessions/${sessionId}/actions/table/solve`, input)

export const freezeBottleContract = (sessionId: string, input: unknown) =>
  postJson<FrozenBottleContract>(`/api/sessions/${sessionId}/actions/bottle-contract/freeze`, input)

export const captureBottleTrial = (
  sessionId: string,
  trialId: 'B-A' | 'B-B' | 'B-C',
  input: { tag_size_m: number; frame_count: number },
) => postJson<BottleCaptureResult>(`/api/sessions/${sessionId}/actions/bottle/${trialId}/capture`, input)

export const validateBottleTrials = (sessionId: string) =>
  postJson<BottleValidationResult>(`/api/sessions/${sessionId}/actions/bottle/validate`)

export const exportCalibrationBundle = (sessionId: string, input: unknown) =>
  postJson<ExportResult>(`/api/sessions/${sessionId}/actions/export`, input)
