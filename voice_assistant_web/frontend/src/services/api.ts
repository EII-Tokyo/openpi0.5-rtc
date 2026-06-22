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

export type RLTTrajectoryRecord = {
  path: string
  name: string
  terminal_label: string | null
  terminal_success: number | null
  num_steps: number
  num_chunks: number | null
  duration_s: number | null
  fps: number | null
  camera_names: string[]
  trim_start_step: number
  trim_end_step: number
  mtime: number
}

export type RLTTrajectoryListResponse = {
  replay_dir: string
  trajectories: RLTTrajectoryRecord[]
}

export async function fetchRltTrajectories(): Promise<RLTTrajectoryListResponse> {
  const response = await fetch(`${apiBase}/api/rlt/trajectories`)
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return (await response.json()) as RLTTrajectoryListResponse
}

export function rltTrajectoryVideoUrl(path: string, camera: string): string {
  const params = new URLSearchParams({ path, camera })
  return `${apiBase}/api/rlt/trajectories/video?${params.toString()}`
}

export async function saveRltTrajectoryTrim(
  path: string,
  trimStartStep: number,
  trimEndStep: number,
  terminalLabel?: 'success' | 'failure' | 'unlabeled',
): Promise<RLTTrajectoryRecord> {
  const response = await fetch(`${apiBase}/api/rlt/trajectories/trim`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      path,
      trim_start_step: trimStartStep,
      trim_end_step: trimEndStep,
      terminal_label: terminalLabel,
    }),
  })
  if (!response.ok) {
    throw new Error(`HTTP ${response.status}`)
  }
  return (await response.json()) as RLTTrajectoryRecord
}
