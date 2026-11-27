import { useEffect, useRef, useState } from 'react'
import { apiBase } from './services/api'

type Parameters = Record<string, unknown>
type Snapshot = {
  captured_at: string
  runtime_started_at: string
  server_started_at: string
  runtime: Parameters
  policy_server: Parameters
  checkpoint: Parameters
  sol: Parameters
  runtime_environment: Parameters
  code: Parameters
}
type SolStatus = {
  checkpoint_step?: string | number
  request_interval_seconds?: number
  request_schedule?: string
  prompt_contract?: { reply_expiry?: number | null }
}

function valueText(value: unknown) {
  if (value == null) return '—'
  if (typeof value === 'object') return JSON.stringify(value)
  return String(value)
}

export function RuntimeParametersDrawer({ open, onClose, language }: {
  open: boolean
  onClose: () => void
  language: string
}) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [liveSol, setLiveSol] = useState<SolStatus | null>(null)
  const [error, setError] = useState(false)
  const ja = language === 'ja'
  const title = ja ? '起動パラメーター' : 'Runtime parameters'

  useEffect(() => {
    const element = dialog.current
    if (open && !element?.open) element?.showModal()
    else if (!open && element?.open) element.close()
  }, [open])

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    setSnapshot(null)
    setLiveSol(null)
    setError(false)
    const read = async (url: string) => {
      const response = await fetch(url, { signal: controller.signal, cache: 'no-store' })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      return response.json()
    }
    void read(`${import.meta.env.BASE_URL}runtime-parameters.json`).then((data: Snapshot) => {
      if (!controller.signal.aborted) setSnapshot(data)
    }).catch(() => {
      if (!controller.signal.aborted) setError(true)
    })
    void read(`${apiBase}/api/high-level/status`).then((data: SolStatus) => {
      if (!controller.signal.aborted) setLiveSol(data)
    }).catch(() => {})
    return () => controller.abort()
  }, [open])

  const matching = snapshot && liveSol?.checkpoint_step != null &&
    String(snapshot.checkpoint.checkpoint_step) === String(liveSol.checkpoint_step)
  const stale = snapshot && liveSol?.checkpoint_step != null && !matching
  const sol = snapshot && {
    ...snapshot.sol,
    ...(matching ? {
      request_interval_seconds: liveSol?.request_interval_seconds ?? snapshot.sol.request_interval_seconds,
      reply_expiry_seconds: liveSol?.prompt_contract?.reply_expiry === undefined
        ? snapshot.sol.reply_expiry_seconds : liveSol.prompt_contract.reply_expiry,
      request_schedule: liveSol?.request_schedule ?? snapshot.sol.request_schedule,
    } : {}),
  }
  const sections: [string, Parameters][] = snapshot && sol ? [
    [ja ? 'モデルとカメラ' : 'Checkpoint and cameras', snapshot.checkpoint],
    ['Sol', sol],
    ['Runtime', snapshot.runtime],
    [ja ? '推論サーバー' : 'Policy server', snapshot.policy_server],
    [ja ? '実行時の環境変数' : 'Runtime environment', snapshot.runtime_environment],
    [ja ? 'コード' : 'Code checkouts', snapshot.code],
  ] : []

  const launchTime = (iso: string) => new Intl.DateTimeFormat(ja ? 'ja-JP' : 'en-GB', {
    dateStyle: 'medium', timeStyle: 'medium', timeZone: 'Asia/Tokyo',
  }).format(new Date(iso))

  return (
    <dialog ref={dialog} id="runtime-parameters" className="inference-settings-drawer runtime-parameters-drawer"
      aria-labelledby="runtime-parameters-title" onClose={onClose}
      onCancel={event => { event.preventDefault(); onClose() }}
      onClick={event => {
        if (event.target !== event.currentTarget) return
        const bounds = event.currentTarget.getBoundingClientRect()
        if (event.clientX < bounds.left || event.clientX > bounds.right ||
          event.clientY < bounds.top || event.clientY > bounds.bottom) onClose()
      }}>
      <div className="motor-drawer-header">
        <h2 id="runtime-parameters-title">{title}</h2>
        <button className="conditioning-close" type="button" aria-label={ja ? '閉じる' : 'Close runtime parameters'}
          onClick={onClose}>×</button>
      </div>
      {error && <p role="alert">{ja ? '起動パラメーターを読み込めませんでした。' : 'No runtime startup snapshot available.'}</p>}
      {!snapshot && !error && <p role="status">{ja ? '読み込み中…' : 'Loading runtime parameters…'}</p>}
      {snapshot && <>
        <div className="runtime-parameters-meta">
          <p>{ja ? 'Runtime 起動' : 'Runtime started'} · {launchTime(snapshot.runtime_started_at)} JST</p>
          <p>{ja ? '推論サーバー起動' : 'Policy server started'} · {launchTime(snapshot.server_started_at)} JST</p>
          <p>{ja ? '取得時刻' : 'Snapshot captured'} · {launchTime(snapshot.captured_at)} JST</p>
        </div>
        {stale && <p role="alert">{ja ? '現在のチェックポイントと起動時の記録が異なります。' : 'The live checkpoint differs from this startup snapshot.'}</p>}
        {sections.map(([heading, values]) => (
          <section className="runtime-parameters-section" key={heading}>
            <h3>{heading}</h3>
            <dl>
              {Object.entries(values).map(([key, value]) => (
                <div key={key}><dt>{key}</dt><dd>{valueText(value)}</dd></div>
              ))}
            </dl>
          </section>
        ))}
      </>}
    </dialog>
  )
}
