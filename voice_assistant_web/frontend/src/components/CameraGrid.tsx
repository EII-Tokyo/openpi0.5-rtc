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
  currentTask: string | null
  cameraView: 'focus' | 'quad'
  onCameraViewChange: (view: 'focus' | 'quad') => void
}

async function drawJpegB64ToCanvas(b64: string, canvas: HTMLCanvasElement | null) {
  if (!canvas || !b64) return
  try {
    const bytes = Uint8Array.from(atob(b64), (char) => char.charCodeAt(0))
    const blob = new Blob([bytes], { type: 'image/jpeg' })
    const bmp = await createImageBitmap(blob)
    const ctx = canvas.getContext('2d')
    if (!ctx) {
      bmp.close()
      return
    }
    if (canvas.width !== bmp.width || canvas.height !== bmp.height) {
      canvas.width = bmp.width
      canvas.height = bmp.height
    }
    ctx.drawImage(bmp, 0, 0)
    bmp.close()
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
  currentTask,
  cameraView,
  onCameraViewChange,
}: Props) {
  const t = translations[language]
  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({})
  const primaryCamera = 'cam_high'
  const secondaryCameras = CAMERAS.filter((camera) => camera.key !== primaryCamera)

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
    <section className={`stage-panel ${cameraView === 'quad' ? 'quad-mode' : 'focus-mode'}`}>
      <article className="hero-camera">
        <div className="camera-panel-header">
          <div className="camera-controls">
            <button
              type="button"
              className={`ghost-button ${cameraView === 'focus' ? 'active' : ''}`}
              onClick={() => onCameraViewChange('focus')}
            >
              {t.cameraFocus}
            </button>
            <button
              type="button"
              className={`ghost-button ${cameraView === 'quad' ? 'active' : ''}`}
              onClick={() => onCameraViewChange('quad')}
            >
              {t.cameraQuad}
            </button>
          </div>
        </div>

        {cameraView === 'focus' ? (
          <div className="camera-stage-frame">
            <canvas ref={bindCanvas(primaryCamera)} className="camera-feed-canvas" aria-label={t.high} />
            <div className="camera-frame-top">{renderOverlay(primaryCamera, t.high)}</div>
          </div>
        ) : (
          <div className="camera-grid">
            {CAMERAS.map((camera) => (
              <article key={camera.key} className="mini-camera-card quad-camera-card">
                <canvas ref={bindCanvas(camera.key)} className="camera-feed-canvas" aria-label={t[camera.labelKey]} />
                <div className="camera-frame-top">{renderOverlay(camera.key, t[camera.labelKey])}</div>
              </article>
            ))}
          </div>
        )}
      </article>

      {cameraView === 'focus' ? (
        <div className="camera-strip">
          {secondaryCameras.map((camera) => (
            <article key={camera.key} className="mini-camera-card">
              <canvas ref={bindCanvas(camera.key)} className="camera-feed-canvas" aria-label={t[camera.labelKey]} />
              <div className="camera-frame-top">{renderOverlay(camera.key, t[camera.labelKey])}</div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}
