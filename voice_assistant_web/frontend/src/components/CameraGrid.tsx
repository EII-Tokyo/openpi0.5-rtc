import { useEffect, useRef, useState } from 'react'
import { AppLanguage, translations } from '../i18n'
import { cameraStreamUrl, cameraWebrtcOfferUrl, cameraWebrtcSessionUrl, CameraTransport } from '../services/api'

const CAMERAS = [
  { key: 'cam_high', labelKey: 'high' },
  { key: 'cam_low', labelKey: 'low' },
  { key: 'cam_left_wrist', labelKey: 'leftWrist' },
  { key: 'cam_right_wrist', labelKey: 'rightWrist' },
] as const

const MJPEG_FPS = {
  focusPrimary: 30,
  focusSecondary: 10,
  quad: 30,
}

type Props = {
  cameraStatus: Record<string, boolean>
  cameraTimestamps: Record<string, number | null>
  cameraFrames: Record<string, string>
  language: AppLanguage
  currentTask: string | null
  cameraView: 'focus' | 'quad'
  onCameraViewChange: (view: 'focus' | 'quad') => void
  cameraTransport: CameraTransport | null
  cameraWebrtcMediaUrl: string | null
}

type AiortcCameraFeedProps = {
  cameraKey: string
  label: string
  mediaServiceUrl: string | null
  fps: number
}

type MjpegCameraFeedProps = {
  cameraKey: string
  label: string
  fps: number
}

const activeCameraFeedReleases: Record<string, (() => void) | undefined> = {}
let lifecycleCleanupInstalled = false

function releaseAllCameraFeeds() {
  Object.values(activeCameraFeedReleases).forEach((release) => release?.())
}

function installCameraFeedLifecycleCleanup() {
  if (lifecycleCleanupInstalled || typeof window === 'undefined') return
  lifecycleCleanupInstalled = true
  window.addEventListener('pagehide', releaseAllCameraFeeds)
  window.addEventListener('beforeunload', releaseAllCameraFeeds)
}

function waitForIceGatheringComplete(peerConnection: RTCPeerConnection) {
  if (peerConnection.iceGatheringState === 'complete') return Promise.resolve()
  return new Promise<void>((resolve) => {
    const checkState = () => {
      if (peerConnection.iceGatheringState === 'complete') {
        peerConnection.removeEventListener('icegatheringstatechange', checkState)
        resolve()
      }
    }
    peerConnection.addEventListener('icegatheringstatechange', checkState)
  })
}

function preferH264(transceiver: RTCRtpTransceiver) {
  const capabilities = RTCRtpSender.getCapabilities('video')
  const h264Codecs = capabilities?.codecs.filter((codec) => codec.mimeType.toLowerCase() === 'video/h264') || []
  if (h264Codecs.length > 0) {
    transceiver.setCodecPreferences(h264Codecs)
  }
}

function AiortcCameraFeed({ cameraKey, label, mediaServiceUrl, fps }: AiortcCameraFeedProps) {
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const sessionIdRef = useRef<string | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    installCameraFeedLifecycleCleanup()
    let isActive = true
    const pc = new RTCPeerConnection({ iceServers: [] })
    const release = () => {
      if (!isActive) return
      isActive = false
      if (videoRef.current) {
        videoRef.current.srcObject = null
      }
      pc.close()
      const sessionId = sessionIdRef.current
      sessionIdRef.current = null
      if (sessionId) {
        void fetch(cameraWebrtcSessionUrl(sessionId, mediaServiceUrl), { method: 'DELETE', keepalive: true })
      }
    }
    activeCameraFeedReleases[cameraKey]?.()
    activeCameraFeedReleases[cameraKey] = release

    const start = async () => {
      setError(null)
      const transceiver = pc.addTransceiver('video', { direction: 'recvonly' })
      preferH264(transceiver)
      pc.ontrack = (event) => {
        if (videoRef.current) {
          videoRef.current.srcObject = event.streams[0]
        }
      }
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      await waitForIceGatheringComplete(pc)
      if (!isActive || !pc.localDescription) return
      const response = await fetch(cameraWebrtcOfferUrl(cameraKey, mediaServiceUrl), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          sdp: pc.localDescription.sdp,
          type: pc.localDescription.type,
          fps,
        }),
      })
      if (!response.ok) {
        throw new Error(`WebRTC ${cameraKey} offer failed: ${response.status}`)
      }
      const answer = await response.json()
      sessionIdRef.current = typeof answer.session_id === 'string' ? answer.session_id : null
      if (!isActive) {
        if (sessionIdRef.current) {
          void fetch(cameraWebrtcSessionUrl(sessionIdRef.current, mediaServiceUrl), { method: 'DELETE', keepalive: true })
        }
        sessionIdRef.current = null
        return
      }
      await pc.setRemoteDescription(answer)
    }

    start().catch((cause) => {
      if (!isActive) return
      setError(cause instanceof Error ? cause.message : String(cause))
    })

    return () => {
      release()
      if (activeCameraFeedReleases[cameraKey] === release) {
        delete activeCameraFeedReleases[cameraKey]
      }
    }
  }, [cameraKey, fps, mediaServiceUrl])

  return (
    <>
      <video ref={videoRef} className="camera-feed-media" aria-label={label} autoPlay playsInline muted />
      {error ? <div className="camera-feed-error">{error}</div> : null}
    </>
  )
}

function MjpegCameraFeed({ cameraKey, label, fps }: MjpegCameraFeedProps) {
  const imageRef = useRef<HTMLImageElement | null>(null)

  useEffect(() => {
    const image = imageRef.current
    if (!image) return
    image.src = cameraStreamUrl(cameraKey, fps)
    return () => {
      image.removeAttribute('src')
      image.src = ''
    }
  }, [cameraKey, fps])

  return <img ref={imageRef} className="camera-feed-media" alt={label} />
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
  cameraTransport,
  cameraWebrtcMediaUrl,
}: Props) {
  const t = translations[language]
  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({})
  const primaryCamera = 'cam_high'
  const secondaryCameras = CAMERAS.filter((camera) => camera.key !== primaryCamera)

  useEffect(() => {
    if (cameraTransport !== 'jpeg_ws') return
    void Promise.all(
      Object.entries(cameraFrames).map(([name, b64]) => drawJpegB64ToCanvas(b64, canvasRefs.current[name] ?? null)),
    )
  }, [cameraFrames, cameraTransport])

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

  const mjpegFpsFor = (cameraKey: (typeof CAMERAS)[number]['key']) => {
    if (cameraView === 'quad') return MJPEG_FPS.quad
    return cameraKey === primaryCamera ? MJPEG_FPS.focusPrimary : MJPEG_FPS.focusSecondary
  }

  const renderCameraMedia = (cameraKey: (typeof CAMERAS)[number]['key'], label: string) => {
    if (!cameraTransport) {
      return <div className="camera-feed-placeholder">Loading camera transport</div>
    }
    if (cameraTransport === 'webrtc') {
      return (
        <AiortcCameraFeed
          cameraKey={cameraKey}
          label={label}
          mediaServiceUrl={cameraWebrtcMediaUrl}
          fps={mjpegFpsFor(cameraKey)}
        />
      )
    }
    if (cameraTransport === 'mjpeg') {
      return <MjpegCameraFeed cameraKey={cameraKey} label={label} fps={mjpegFpsFor(cameraKey)} />
    }
    return <canvas ref={bindCanvas(cameraKey)} className="camera-feed-canvas" aria-label={label} />
  }

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
            {renderCameraMedia(primaryCamera, t.high)}
            <div className="camera-frame-top">{renderOverlay(primaryCamera, t.high)}</div>
          </div>
        ) : (
          <div className="camera-grid">
            {CAMERAS.map((camera) => (
              <article key={camera.key} className="mini-camera-card quad-camera-card">
                {renderCameraMedia(camera.key, t[camera.labelKey])}
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
              {renderCameraMedia(camera.key, t[camera.labelKey])}
              <div className="camera-frame-top">{renderOverlay(camera.key, t[camera.labelKey])}</div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}
