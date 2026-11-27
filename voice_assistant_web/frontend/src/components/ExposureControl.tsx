import { useCallback, useEffect, useRef, useState } from 'react'
import { apiBase } from '../services/api'
import { AppLanguage, translations } from '../i18n'

type CameraExposure = { exposure_ms: number; auto_exposure: boolean; gain: number; min_ms: number; max_ms: number; writable: boolean; auto_exposure_writable: boolean }
type Snapshot = { cameras: Record<string, CameraExposure> }
type ActualFrame = { actual_exposure_ms: number | null; frame_number: number | null; received_unix: number; fresh: boolean; auto_exposure: boolean | null; error: string | null }
type ActualSnapshot = { cameras: Record<string, ActualFrame> }
type Change = { camera: string; exposure_ms?: number; auto_exposure?: boolean }
const cameras = [
  ['cam_high', 'high'], ['cam_low', 'low'],
  ['cam_left_wrist', 'leftWrist'], ['cam_right_wrist', 'rightWrist'],
] as const

export function ExposureControl({ language }: { language: AppLanguage }) {
  const ja = language === 'ja'
  const t = translations[language]
  const [target, setTarget] = useState('all')
  const [snapshot, setSnapshot] = useState<Snapshot | null>(null)
  const [actual, setActual] = useState<ActualSnapshot | null>(null)
  const [draft, setDraft] = useState(30)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const pending = useRef<Change | null>(null)
  const busy = useRef(false)
  const alive = useRef(false)
  const dragging = useRef(false)
  const timer = useRef<number>()
  const generation = useRef(0)
  const targetRef = useRef(target)
  const autoToggle = useRef<HTMLInputElement>(null)
  targetRef.current = target

  const accept = useCallback((data: Snapshot) => {
    if (!alive.current) return
    setSnapshot(data)
    if (!pending.current && !dragging.current) {
      const name = targetRef.current === 'all' ? 'cam_high' : targetRef.current
      if (data.cameras[name]) setDraft(data.cameras[name].exposure_ms)
    }
  }, [])

  const flush = useCallback(async () => {
    window.clearTimeout(timer.current)
    if (busy.current || !pending.current) return
    busy.current = true
    try {
      while (alive.current && pending.current) {
        const change = pending.current
        pending.current = null
        generation.current++
        const response = await fetch(`${apiBase}/api/cameras/exposure`, {
          method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(change),
          signal: AbortSignal.timeout(10000),
        })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`)
        accept(data)
        if (alive.current) setError('')
      }
    } catch (err) {
      pending.current = null
      if (alive.current) setError(String(err))
    } finally {
      busy.current = false
      if (alive.current) setSaving(false)
    }
  }, [accept])

  useEffect(() => {
    let active = true
    let reading = false
    const refreshActual = async () => {
      if (reading) return
      reading = true
      try {
        const response = await fetch(`${apiBase}/api/cameras/exposure/actual`, { cache: 'no-store', signal: AbortSignal.timeout(2500) })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        if (active) setActual(data)
      } catch { if (active) setActual(null) }
      finally { reading = false }
    }
    void refreshActual()
    const interval = window.setInterval(() => { void refreshActual() }, 500)
    return () => { active = false; window.clearInterval(interval) }
  }, [])

  useEffect(() => {
    alive.current = true
    const refresh = async () => {
      if (busy.current || pending.current || dragging.current) return
      const version = generation.current
      try {
        const response = await fetch(`${apiBase}/api/cameras/exposure`, { cache: 'no-store', signal: AbortSignal.timeout(10000) })
        const data = await response.json()
        if (!response.ok) throw new Error(data.detail || `HTTP ${response.status}`)
        if (alive.current && version === generation.current && !pending.current && !busy.current) {
          accept(data)
          setError('')
        }
      } catch (err) { if (alive.current && version === generation.current) setError(String(err)) }
    }
    void refresh()
    const interval = window.setInterval(() => { void refresh() }, 10000)
    return () => { alive.current = false; pending.current = null; window.clearInterval(interval); window.clearTimeout(timer.current) }
  }, [accept])

  const selected = snapshot ? (target === 'all' ? Object.values(snapshot.cameras) : [snapshot.cameras[target]].filter(Boolean)) : []
  const min = selected.length ? Math.max(...selected.map(c => c.min_ms)) : 1
  const max = selected.length ? Math.min(...selected.map(c => c.max_ms)) : 165
  const anyAuto = selected.some(c => c.auto_exposure)
  const allAuto = selected.length > 0 && selected.every(c => c.auto_exposure)
  const mixed = selected.some(c => c.auto_exposure !== selected[0]?.auto_exposure || (!anyAuto && c.exposure_ms !== selected[0]?.exposure_ms))
  useEffect(() => { if (autoToggle.current) autoToggle.current.indeterminate = anyAuto && !allAuto }, [anyAuto, allAuto])
  const disabled = !selected.length || selected.some(c => !c.writable)
  const autoDisabled = !selected.length || saving || selected.some(c => !c.auto_exposure_writable)
  const changed = (value: number) => {
    setDraft(value); setSaving(true); setError('')
    pending.current = { camera: target, exposure_ms: value }
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(() => { void flush() }, 250)
  }
  const changeMode = (automatic: boolean) => {
    dragging.current = false
    setSaving(true); setError('')
    pending.current = { camera: target, auto_exposure: automatic }
    void flush()
  }
  const actualNames = target === 'all' ? cameras.map(([key]) => key) : [target]
  const shortNames: Record<string, string> = { cam_high: 'H', cam_low: 'L', cam_left_wrist: 'LW', cam_right_wrist: 'RW' }
  const help = anyAuto ? (ja ? '撮影フレームの実露光時間を表示（毎秒2回更新）。—は未取得または古い値です。チェックを外すと手動に戻ります。' : language === 'zh' ? '显示实拍帧的曝光时间，每秒更新两次。— 表示暂未获取。关闭自动曝光后可手动调节。' : 'Actual exposure from captured frames, refreshed twice per second. — means missing or stale metadata. Uncheck to use the manual slider.') :
    (ja ? 'ドラッグで即時反映・ゲインは変更しません。16.7 ms超では60 fpsを維持できません。' : language === 'zh' ? '拖动滑块调整曝光。超过 16.7 毫秒时无法维持 60 fps。' : 'Drag to apply; gain unchanged. Above 16.7 ms, exposure cannot sustain 60 fps.')
  return <div className="exposure-control">
    <div className="exposure-inputs" title={help}>
      <label htmlFor="exposure-camera">{ja ? '露光時間' : language === 'zh' ? '曝光' : 'Exposure'}</label>
      <select id="exposure-camera" value={target} disabled={saving} onChange={event => {
        setTarget(event.target.value)
        const camera = event.target.value === 'all' ? 'cam_high' : event.target.value
        if (snapshot?.cameras[camera]) setDraft(snapshot.cameras[camera].exposure_ms)
      }}>
        <option value="all">{ja ? '全カメラ' : language === 'zh' ? '全部相机' : 'All cameras'}</option>
        {cameras.map(([key, label]) => <option key={key} value={key}>{t[label]}</option>)}
      </select>
      <label className="exposure-auto">
        <input ref={autoToggle} type="checkbox" checked={allAuto} disabled={autoDisabled}
          onChange={event => changeMode(event.target.checked)} />
        {ja ? '自動露光' : language === 'zh' ? '自动曝光' : 'Auto exposure'}
      </label>
      <input aria-label={ja ? '露光時間（ミリ秒）' : 'Exposure time in milliseconds'} type="range" min={min} max={max} step="1" value={draft}
        disabled={disabled || anyAuto} onPointerDown={() => { dragging.current = true }} onPointerUp={() => { dragging.current = false; void flush() }}
        onPointerCancel={() => { dragging.current = false; void flush() }} onBlur={() => { dragging.current = false; void flush() }}
        onChange={event => changed(Number(event.target.value))} />
      <output className="exposure-value" aria-label={anyAuto ? 'Actual frame exposure' : undefined}>
        {anyAuto ? actualNames.map(name => {
          const frame = actual?.cameras[name]
          const valid = frame?.fresh && frame.actual_exposure_ms !== null
          const label = cameras.find(([key]) => key === name)?.[1]
          return <span className="exposure-reading" key={name} title={`${label ? t[label] : name} · frame ${frame?.frame_number ?? '—'} · ${frame?.received_unix ? new Date(frame.received_unix * 1000).toLocaleTimeString() : '—'}${frame?.error ? ` · ${frame.error}` : ''}`}>
            {target === 'all' && <span className="exposure-camera-name">{shortNames[name]}</span>}<strong>{valid ? `${frame.actual_exposure_ms!.toFixed(2)}ms` : '—'}</strong>
          </span>
        }) : `${draft.toFixed(0)} ms`}
      </output>
      {(saving || !snapshot || mixed) && <span role="status" className="exposure-status">{saving ? (ja ? '適用中…' : language === 'zh' ? '应用中…' : 'Applying…') : !snapshot ? (ja ? '読込中…' : language === 'zh' ? '读取中…' : 'Reading…') : (ja ? '混在' : language === 'zh' ? '各路不同' : 'Mixed')}</span>}
    </div>
    {error && <small className="exposure-error" role="alert">{error}</small>}
  </div>
}
