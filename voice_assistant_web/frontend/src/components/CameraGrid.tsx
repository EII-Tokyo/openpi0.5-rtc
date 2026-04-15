import { useEffect, useRef } from 'react'
import { AppLanguage, translations } from '../i18n'

const CAMERAS = [
  { key: 'cam_high', labelKey: 'high' },
  { key: 'cam_low', labelKey: 'low' },
  { key: 'cam_left_wrist', labelKey: 'leftWrist' },
  { key: 'cam_right_wrist', labelKey: 'rightWrist' },
] as const

type Props = {
  cameraStatus: Record<string, boolean>
  cameraTimestamps: Record<string, number | null>
  cameraFrames: Record<string, string>
  language: AppLanguage
  highOnly: boolean
  currentTask: string | null
  onToggleHighOnly: () => void
}

async function drawJpegB64ToCanvas(b64: string, canvas: HTMLCanvasElement | null) {
  if (!canvas || !b64) return
  try {
    const bytes = Uint8Array.from(atob(b64), (char) => char.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'image/jpeg' })
    const bitmap = await createImageBitmap(blob)
    const context = canvas.getContext('2d')
    if (!context) {
      bitmap.close()
      return
    }
    if (canvas.width !== bitmap.width || canvas.height !== bitmap.height) {
      canvas.width = bitmap.width
      canvas.height = bitmap.height
    }
    context.drawImage(bitmap, 0, 0)
    bitmap.close()
  } catch (error) {
    console.error('Failed to decode camera frame', error)
  }
}

function formatCameraAge(timestamp: number | null | undefined) {
  if (!timestamp) return 'N/A'
  const age = Math.max(0, Date.now() / 1000 - timestamp)
  return `${age.toFixed(age < 1 ? 2 : 1)}s`
}

export function CameraGrid({
  cameraStatus,
  cameraTimestamps,
  cameraFrames,
  language,
  highOnly,
  currentTask,
  onToggleHighOnly,
}: Props) {
  const t = translations[language]
  const heroCamera = CAMERAS[0]
  const stripCameras = highOnly ? [] : CAMERAS.slice(1)
  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({})
  useEffect(() => {
    void Promise.all(
      Object.entries(cameraFrames).map(([name, b64]) => drawJpegB64ToCanvas(b64, canvasRefs.current[name] ?? null)),
    )
  }, [cameraFrames])

  const bindCanvas = (name: string) => (element: HTMLCanvasElement | null) => {
    canvasRefs.current[name] = element
  }

  const renderOverlay = (cameraKey: (typeof CAMERAS)[number]['key'], label: string) => (
    <div className="camera-overlay-chip">
      <div className="camera-overlay-head">
        <span>{label}</span>
        <span className={cameraStatus[cameraKey] ? 'dot live' : 'dot offline'} />
      </div>
      <div className="camera-overlay-meta">
        <span>{cameraStatus[cameraKey] ? t.live : t.offline}</span>
        <span>{formatCameraAge(cameraTimestamps[cameraKey])}</span>
      </div>
    </div>
  )

  return (
    <section className={`stage-panel${highOnly ? ' quad-mode' : ''}`}>
      <article className="hero-camera">
        <div className="camera-panel-header">
          <div className="hero-camera-copy">
            <p className="eyebrow">{t.camerasEyebrow}</p>
            <h2>{t.camerasTitle}</h2>
          </div>
          <div className="camera-controls">
            <span className={cameraStatus[heroCamera.key] ? 'status-pill live' : 'status-pill offline'}>
              {cameraStatus[heroCamera.key] ? t.live : t.waiting}
            </span>
            <button type="button" className="ghost-button" onClick={onToggleHighOnly}>
              {highOnly ? t.showAllCameras : t.showHighCameraOnly}
            </button>
          </div>
        </div>
        <div className="camera-stage-frame">
          <canvas ref={bindCanvas(heroCamera.key)} className="camera-feed-canvas" aria-label={t[heroCamera.labelKey]} />
          <div className="camera-frame-top">{renderOverlay(heroCamera.key, t[heroCamera.labelKey])}</div>
          {currentTask ? (
            <div className="camera-stage-overlay">
              <div className="stage-task">
                <span>{t.latestDispatch}</span>
                <strong>{currentTask}</strong>
              </div>
            </div>
          ) : null}
        </div>
      </article>
      {highOnly ? null : (
        <div className="camera-strip">
          {stripCameras.map((camera) => (
            <article key={camera.key} className="mini-camera-card">
              <canvas ref={bindCanvas(camera.key)} className="camera-feed-canvas" aria-label={t[camera.labelKey]} />
              <div className="camera-frame-top">{renderOverlay(camera.key, t[camera.labelKey])}</div>
            </article>
          ))}
        </div>
      )}
    </section>
  )
}
