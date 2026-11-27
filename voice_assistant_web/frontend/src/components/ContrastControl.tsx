import { useEffect, useState } from 'react'
import { translations, type AppLanguage } from '../i18n'
import { apiBase } from '../services/api'

type Reading = { min: number; max: number; step: number; value: number; default: number; writable: boolean }
const cameraLabels = { cam_high: 'high', cam_low: 'low', cam_left_wrist: 'leftWrist', cam_right_wrist: 'rightWrist' } as const
const text = {
  en: { contrast: 'Contrast', all: 'All cameras', apply: 'Apply', reset: 'Default', mixed: 'Mixed', unavailable: 'Camera contrast unavailable' },
  ja: { contrast: 'コントラスト', all: '全カメラ', apply: '適用', reset: '初期値', mixed: '複数の値', unavailable: 'コントラストを取得できません' },
  zh: { contrast: '对比度', all: '全部相机', apply: '应用', reset: '默认值', mixed: '各路不同', unavailable: '无法读取相机对比度' },
}

export function ContrastControl({ language }: { language: AppLanguage }) {
  const t = text[language]
  const [cameras, setCameras] = useState<Record<string, Reading>>({})
  const [target, setTarget] = useState('all')
  const [draft, setDraft] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  useEffect(() => {
    const controller = new AbortController()
    void (async () => {
      try {
        const response = await fetch(`${apiBase}/api/cameras/contrast`, { signal: controller.signal, cache: 'no-store' })
        if (!response.ok) throw new Error()
        setCameras((await response.json()).cameras)
        setError('')
      } catch { if (!controller.signal.aborted) setError(t.unavailable) }
    })()
    return () => controller.abort()
  }, [t.unavailable])

  const names = target === 'all' ? Object.keys(cameraLabels) : [target]
  const readings = names.map(name => cameras[name]).filter(Boolean)
  const current = readings[0]
  const mixed = current && readings.some(reading => reading.value !== current.value)
  const enabled = readings.length === names.length && readings.every(reading => reading.writable) && !busy
  const apply = async (value: number) => {
    setBusy(true)
    setError('')
    try {
      const response = await fetch(`${apiBase}/api/cameras/contrast`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ camera: target, value }),
      })
      const data = await response.json()
      if (!response.ok) throw new Error(typeof data.detail === 'string' ? data.detail : t.unavailable)
      setCameras(data.cameras)
      setDraft(null)
    } catch (error) { setError(error instanceof Error ? error.message : t.unavailable) }
    finally { setBusy(false) }
  }

  return <div className="contrast-control">
    <div className="exposure-inputs contrast-inputs">
      <label htmlFor="contrast-value">{t.contrast}</label>
      <select aria-label={t.contrast + ' · ' + t.all} value={target} disabled={busy}
        onChange={event => { setTarget(event.target.value); setDraft(null) }}>
        <option value="all">{t.all}</option>
        {Object.entries(cameraLabels).map(([camera, label]) => <option value={camera} key={camera}>{translations[language][label]}</option>)}
      </select>
      <input id="contrast-value" type="range" min={current?.min ?? 0} max={current?.max ?? 100} step={current?.step || 1}
        value={draft ?? current?.value ?? 50} disabled={!enabled}
        onChange={event => setDraft(Number(event.target.value))} />
      <output>{draft ?? (mixed ? t.mixed : current?.value ?? '—')}</output>
      <button type="button" className="ghost-button" disabled={!enabled || draft === null}
        onClick={() => { if (draft !== null) void apply(draft) }}>{t.apply}</button>
      <button type="button" className="ghost-button"
        disabled={!enabled || !current || readings.some(reading => reading.default !== current.default)}
        onClick={() => void apply(current.default)}>{t.reset}</button>
    </div>
    {error && <small className="error-text" role="alert">{error}</small>}
  </div>
}
