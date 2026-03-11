import { apiBase } from '../services/api'
import { AppLanguage, translations } from '../i18n'

const CAMERAS = [
  { key: 'cam_high', labelKey: 'high' },
  { key: 'cam_low', labelKey: 'low' },
  { key: 'cam_left_wrist', labelKey: 'leftWrist' },
  { key: 'cam_right_wrist', labelKey: 'rightWrist' },
] as const

type Props = {
  cameraStatus: Record<string, boolean>
  language: AppLanguage
  highOnly: boolean
  onToggleHighOnly: () => void
}

export function CameraGrid({ cameraStatus, language, highOnly, onToggleHighOnly }: Props) {
  const t = translations[language]
  const visibleCameras = highOnly ? CAMERAS.filter((camera) => camera.key === 'cam_high') : CAMERAS

  return (
    <section className="panel camera-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">{t.camerasEyebrow}</p>
          <h2>{t.camerasTitle}</h2>
        </div>
        <button type="button" className="view-toggle" onClick={onToggleHighOnly}>
          {highOnly ? t.showAllCameras : t.showHighCameraOnly}
        </button>
      </div>
      <div className={highOnly ? 'camera-grid single' : 'camera-grid'}>
        {visibleCameras.map((camera) => (
          <article key={camera.key} className="camera-card">
            <div className="camera-frame">
              <div className="camera-overlay">
                <span className="camera-chip">{t[camera.labelKey]}</span>
                <span className={cameraStatus[camera.key] ? 'camera-chip status ok' : 'camera-chip status offline'}>
                  {cameraStatus[camera.key] ? t.live : t.waiting}
                </span>
              </div>
              <img src={`${apiBase}/api/cameras/${camera.key}/stream.mjpg`} alt={t[camera.labelKey]} />
            </div>
          </article>
        ))}
      </div>
    </section>
  )
}
