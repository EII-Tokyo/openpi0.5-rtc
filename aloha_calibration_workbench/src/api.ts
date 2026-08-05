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
