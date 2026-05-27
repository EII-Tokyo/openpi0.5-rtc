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
