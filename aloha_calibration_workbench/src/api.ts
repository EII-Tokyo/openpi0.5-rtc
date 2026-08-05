export type OwnershipState = 'FREE' | 'BUSY' | 'UNKNOWN'
export type PreflightStatus = 'READY' | 'BLOCKED' | 'FAILED'

export interface PreflightCamera {
  role: string
  connected: boolean
  identity_match: boolean
  production_profile_supported: boolean
  ownership: OwnershipState
}

export interface PreflightIssue {
  code: string
  severity: 'WARNING' | 'BLOCKING' | 'ERROR'
  message: string
  camera_role?: string | null
}

export interface PreflightReport {
  status: PreflightStatus
  cameras: PreflightCamera[]
  issues: PreflightIssue[]
}

export interface PreflightSession {
  id: string
  state: 'PREFLIGHT_READY' | 'PREFLIGHT_BLOCKED' | 'PREFLIGHT_FAILED'
  latest_preflight: PreflightReport
}

export interface FactoryIntrinsics {
  width: number
  height: number
  fx: number
  fy: number
  cx: number
  cy: number
  distortion_model: string
  distortion_coefficients: number[]
}

export interface CharucoObservation {
  board_detected: boolean
  marker_count: number
  charuco_corner_count: number
  blur_variance: number
  black_clip_percent: number
  white_clip_percent: number
  centroid_x?: number | null
  centroid_y?: number | null
  board_area_percent?: number | null
  reprojection_rms_px?: number | null
  frame_number: number
  device_timestamp_ms: number
}

export interface CaptureStatus {
  state: 'IDLE' | 'STREAMING' | 'UNAVAILABLE'
  session_id?: string | null
  role?: string | null
  serial?: string | null
  profile?: { stream: string; width: number; height: number; fps: number; format: string } | null
  factory_intrinsics?: FactoryIntrinsics | null
  latest_observation?: CharucoObservation | null
  pipeline_started: boolean
  depth_stream_started: boolean
  robot_command_api: boolean
}

export interface SampleRecord {
  id: string
  partition: 'SOLVE' | 'HELD_OUT'
  accepted: boolean
  reason: string
  observation: CharucoObservation
}

export async function runPreflightSession(): Promise<PreflightSession> {
  const response = await fetch('/api/preflight-session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
  })
  if (!response.ok) {
    throw new Error(`Preflight request failed: HTTP ${response.status}`)
  }
  return response.json() as Promise<PreflightSession>
}

export async function startIntrinsicsCapture(sessionId: string, role = 'cam_high'): Promise<CaptureStatus> {
  const response = await fetch(`/api/sessions/${sessionId}/actions/intrinsics/start`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ role }),
  })
  if (!response.ok) throw new Error(`Intrinsics start failed: HTTP ${response.status}`)
  return response.json() as Promise<CaptureStatus>
}

export async function getIntrinsicsStatus(): Promise<CaptureStatus> {
  const response = await fetch('/api/intrinsics/status', { cache: 'no-store' })
  if (!response.ok) throw new Error(`Intrinsics status failed: HTTP ${response.status}`)
  return response.json() as Promise<CaptureStatus>
}

export async function captureIntrinsicsSample(): Promise<SampleRecord> {
  const response = await fetch('/api/intrinsics/sample', { method: 'POST' })
  if (!response.ok) throw new Error(`Sample capture failed: HTTP ${response.status}`)
  return response.json() as Promise<SampleRecord>
}

export async function stopIntrinsicsCapture(): Promise<CaptureStatus> {
  const response = await fetch('/api/intrinsics/stop', { method: 'POST' })
  if (!response.ok) throw new Error(`Intrinsics stop failed: HTTP ${response.status}`)
  return response.json() as Promise<CaptureStatus>
}
