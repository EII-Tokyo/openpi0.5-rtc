import { useEffect, useRef, useState } from 'react'
import type { MouseEvent } from 'react'
import { translations, type AppLanguage } from '../i18n'
import { apiBase } from '../services/api'

export const nozzleCameras = ['cam_high', 'cam_low'] as const
type NozzleCamera = typeof nozzleCameras[number]
type Point = { x: number; y: number }
export type NozzleTips = Partial<Record<NozzleCamera, Point | null>>
export type Calibration = { saved_tips: NozzleTips; run_tips: NozzleTips | null; busy: boolean }

const text = {
  en: { title: 'Nozzle calibration', instruction: 'Click the nozzle tip in each full camera image. Save before starting rinse.', clear: 'Clear point', save: 'Save calibration', close: 'Close', reload: 'Refresh images', active: 'Calibrate after stopping the current run.', error: 'Could not save calibration.' },
  ja: { title: 'ノズル位置の調整', instruction: '各カメラ画像でノズル先端をクリックし、洗浄開始前に保存してください。', clear: '座標を削除', save: '保存', close: '閉じる', reload: '画像を更新', active: '洗浄を停止してから調整してください。', error: 'ノズル位置を保存できませんでした。' },
  zh: { title: '标定喷嘴', instruction: '在两路原图上分别点一下喷嘴顶端，再保存并开始冲洗。', clear: '清除该点', save: '保存标定', close: '关闭', reload: '刷新图像', active: '请先停止当前冲洗，再修改标定。', error: '保存标定失败。' },
}

export function NozzleMarker({ point, label }: { point: Point; label: string }) {
  return <svg className="nozzle-overlay" viewBox="0 0 640 480" preserveAspectRatio="xMidYMid slice" aria-label={label} role="img">
    <circle cx={point.x * 640} cy={point.y * 480} r="8" fill="none" stroke="#181818" strokeWidth="4" vectorEffect="non-scaling-stroke" />
    <circle cx={point.x * 640} cy={point.y * 480} r="8" fill="none" stroke="#ffdd36" strokeWidth="2" vectorEffect="non-scaling-stroke" />
  </svg>
}

export function NozzleCalibration({ language, calibration, onSave }: {
  language: AppLanguage; calibration: Calibration | null; onSave: (value: Calibration) => void
}) {
  const t = text[language]
  const dialog = useRef<HTMLDialogElement>(null)
  const [draft, setDraft] = useState<NozzleTips>({})
  const [stamp, setStamp] = useState(0)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')
  const open = () => {
    setDraft(calibration?.saved_tips ?? {})
    setStamp(Date.now())
    setError('')
    dialog.current?.showModal()
  }
  const save = async () => {
    setSaving(true)
    setError('')
    try {
      const response = await fetch(`${apiBase}/api/nozzle-calibration`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cam_high: draft.cam_high ?? null, cam_low: draft.cam_low ?? null }),
      })
      const result = await response.json()
      if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : t.error)
      onSave(result)
      dialog.current?.close()
    } catch (error) {
      setError(error instanceof Error ? error.message : t.error)
    } finally { setSaving(false) }
  }
  const pick = (event: MouseEvent<HTMLImageElement>, camera: NozzleCamera) => {
    const box = event.currentTarget.getBoundingClientRect()
    setDraft({ ...draft, [camera]: { x: (event.clientX - box.left) / box.width, y: (event.clientY - box.top) / box.height } })
  }
  return <>
    <button type="button" className="ghost-button" onClick={open} aria-haspopup="dialog">{t.title}</button>
    <dialog ref={dialog} className="nozzle-dialog" aria-labelledby="nozzle-calibration-title"
      onClose={() => setStamp(0)} onKeyDown={event => event.stopPropagation()}>
      <header><h2 id="nozzle-calibration-title">{t.title}</h2>
        <button type="button" className="ghost-button" onClick={() => dialog.current?.close()}>{t.close}</button>
      </header>
      <p>{calibration?.busy ? t.active : t.instruction}</p>
      {!!stamp && <div className="nozzle-grid">
        {nozzleCameras.map(camera => {
          const label = translations[language][camera === 'cam_high' ? 'high' : 'low']
          const tip = draft[camera]
          return <section key={camera}>
            <strong>{label}</strong>
            <div className="nozzle-reference">
              <img src={`${apiBase}/api/cameras/${camera}/latest.png?calibration=${stamp}`} alt={label}
                onClick={event => { if (!calibration?.busy && !saving) pick(event, camera) }} />
              {tip && <NozzleMarker point={tip} label={t.title} />}
            </div>
            <button type="button" className="ghost-button" disabled={calibration?.busy || saving || !tip}
              onClick={() => setDraft({ ...draft, [camera]: null })}>{t.clear}</button>
          </section>
        })}
      </div>}
      <footer>
        <button type="button" className="ghost-button" onClick={() => setStamp(Date.now())}>{t.reload}</button>
        <button type="button" className="ghost-button" disabled={calibration?.busy || saving || !calibration}
          onClick={() => void save()}>{t.save}</button>
      </footer>
      {error && <p className="error-text" role="alert">{error}</p>}
    </dialog>
  </>
}
