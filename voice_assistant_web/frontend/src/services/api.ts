const trimSlash = (value: string) => value.replace(/\/+$/, '')

export const apiBase =
  trimSlash((import.meta.env.VITE_API_BASE as string | undefined) || 'http://localhost:8011')

export const wsBase = (() => {
  const envBase = (import.meta.env.VITE_WS_BASE as string | undefined)?.replace(/\/+$/, '')
  if (envBase) return envBase
  return apiBase.replace(/^http/, 'ws')
})()
