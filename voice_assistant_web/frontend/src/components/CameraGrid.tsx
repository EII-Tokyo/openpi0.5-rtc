import { useEffect, useRef, useState } from 'react'
import { apiBase, wsBase } from '../services/api'
import { AppLanguage, translations } from '../i18n'
import { ExposureControl } from './ExposureControl'
import { ContrastControl } from './ContrastControl'
import { NozzleCalibration, NozzleMarker, type Calibration } from './NozzleCalibration'

const CAMERAS = [
  { key: 'cam_high', labelKey: 'high' },
  { key: 'cam_low', labelKey: 'low' },
  { key: 'cam_left_wrist', labelKey: 'leftWrist' },
  { key: 'cam_right_wrist', labelKey: 'rightWrist' },
] as const

type Props = {
  cameraStatus: Record<string, boolean>
  cameraTimestamps: Record<string, number | null>
  language: AppLanguage
  currentTask: string | null
  cameraView: 'focus' | 'quad'
  onCameraViewChange: (view: 'focus' | 'quad') => void
}

type FrameStats = { width: number; height: number; received: number; fps: number; ageMs: number | null; decodeMs: number; p50: number | null; p95: number | null }
type Sample = { times: number[]; ages: number[]; decodes: number[]; received: number; width: number; height: number }
const quantile = (values: number[], q: number) => values.length ? [...values].sort((a,b) => a-b)[Math.floor((values.length-1)*q)] : null

export function CameraGrid({
  language,
  cameraView,
  onCameraViewChange,
}: Props) {
  const t = translations[language]
  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({})
  const primaryCamera = 'cam_high'
  const secondaryCameras = CAMERAS.filter((camera) => camera.key !== primaryCamera)

  const [stats, setStats] = useState<Record<string, FrameStats>>({})
  const [previewError, setPreviewError] = useState('Connecting lossless PNG preview…')
  useEffect(() => {
    let active = true
    let socket: WebSocket | null = null
    let retry: number | undefined
    let clockOffset: number | null = null
    let clockRtt = Infinity
    const samples: Record<string, Sample> = {}
    const calibrate = async () => {
      for (let i = 0; i < 5 && active; i++) {
        try {
          const start = Date.now()
          const response = await fetch(`${apiBase}/api/cameras/preview-metrics`, {cache:'no-store'})
          if (!response.ok) return
          const data = await response.json()
          const end = Date.now()
          if (end-start < clockRtt) { clockRtt=end-start; clockOffset=data.server_unix*1000-(start+end)/2 }
        } catch { return }
      }
    }
    const connect = () => {
      if (!active) return
      const ws = new WebSocket(`${wsBase}/ws/cameras`)
      ws.binaryType = 'arraybuffer'
      socket = ws
      ws.onmessage = async event => {
        if (!active || socket !== ws) return
        try {
          if (!(event.data instanceof ArrayBuffer)) throw new Error('Expected binary PNG preview')
          const packet: ArrayBuffer = event.data
          const size = new DataView(packet).getUint32(0)
          if (size > 16384 || size+4 > packet.byteLength) throw new Error('Invalid preview header')
          const header = JSON.parse(new TextDecoder().decode(new Uint8Array(packet,4,size)))
          if (header.codec !== 'png' || header.cameras.length !== 4) throw new Error('Expected four lossless cameras')
          let offset = 4+size
          const draws = header.cameras.map((frame: {name: string; bytes: number; width: number; height: number; received_unix: number}) => {
            if (!CAMERAS.some(c => c.key === frame.name) || frame.width !== 640 || frame.height !== 480 || frame.bytes <= 0 || offset+frame.bytes > packet.byteLength) throw new Error('Invalid PNG camera frame')
            const blob = new Blob([new Uint8Array(packet,offset,frame.bytes)], {type:'image/png'})
            offset += frame.bytes
            return (async () => {
              const started = performance.now()
              const bitmap = await createImageBitmap(blob)
              try {
                if (!active || socket !== ws) return
                const canvas = canvasRefs.current[frame.name]
                const context = canvas?.getContext('2d')
                if (!canvas || !context) return
                if (bitmap.width !== 640 || bitmap.height !== 480) throw new Error('PNG dimensions changed')
                if (canvas.width !== 640 || canvas.height !== 480) { canvas.width=640; canvas.height=480 }
                context.drawImage(bitmap,0,0)
                const now = Date.now()
                const sample = samples[frame.name] ?? {times:[],ages:[],decodes:[],received:0,width:640,height:480}
                sample.times.push(now);sample.times=sample.times.slice(-120)
                sample.decodes.push(performance.now()-started);sample.decodes=sample.decodes.slice(-120)
                if (clockOffset !== null) { sample.ages.push(Math.max(0,now+clockOffset-frame.received_unix*1000));sample.ages=sample.ages.slice(-120) }
                sample.received=frame.received_unix;samples[frame.name]=sample
              } finally { bitmap.close() }
            })()
          })
          if (offset !== packet.byteLength) throw new Error('Unexpected preview payload bytes')
          await Promise.all(draws)
          if (active && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({type:'ack',sequence:header.sequence}))
            setPreviewError('')
          }
        } catch (error) { if (active) setPreviewError(String(error));ws.close() }
      }
      ws.onclose = () => { if (active) { setPreviewError('PNG preview reconnecting…');retry=window.setTimeout(connect,1000) } }
      ws.onerror = () => ws.close()
    }
    void calibrate();connect()
    const clockTimer = window.setInterval(() => { clockRtt=Infinity;void calibrate() },10000)
    const timer = window.setInterval(() => {
      const next: Record<string, FrameStats> = {}
      for (const [name,s] of Object.entries(samples)) {
        const span = s.times[s.times.length-1]-s.times[0]
        next[name]={width:s.width,height:s.height,received:s.received,
          fps: Date.now()-s.times[s.times.length-1]>2000 ? 0 : span>0 ? (s.times.length-1)*1000/span : 0,
          ageMs:s.ages.length ? s.ages[s.ages.length-1] : null,
          decodeMs:quantile(s.decodes,.5) ?? 0,p50:quantile(s.ages,.5),p95:quantile(s.ages,.95)}
      }
      if (active) setStats(next)
    },500)
    return () => { active=false;window.clearInterval(timer);window.clearInterval(clockTimer);window.clearTimeout(retry);socket?.close() }
  }, [])

  const bindCanvas = (name: string) => (element: HTMLCanvasElement | null) => {
    canvasRefs.current[name] = element
  }

  const [calibration, setCalibration] = useState<Calibration | null>(null)
  useEffect(() => {
    let active = true
    const controller = new AbortController()
    const poll = async () => {
      try {
        const response = await fetch(`${apiBase}/api/nozzle-calibration`, { signal: controller.signal, cache: 'no-store' })
        if (response.ok && active) setCalibration(await response.json())
      } catch { /* Keep the last calibration when a poll is interrupted. */ }
    }
    void poll()
    const timer = window.setInterval(poll, 1000)
    return () => { active = false; controller.abort(); window.clearInterval(timer) }
  }, [])
  const tips = calibration?.run_tips
  const renderTip = (name: string) => {
    const tip = name === 'cam_high' || name === 'cam_low' ? tips?.[name] : null
    return tip ? <NozzleMarker point={tip} label={language === 'ja' ? 'ノズル先端' : language === 'zh' ? '喷嘴顶端' : 'Nozzle tip'} /> : null
  }

  const renderOverlay = (label: string, name: string) => {
    const missing = (name === 'cam_high' || name === 'cam_low') && tips?.[name] === null
    const missedLabel = language === 'ja' ? 'ノズル未検出' : language === 'zh' ? '喷嘴未识别' : 'Nozzle not detected'
    return <span className="camera-name-label">{label}{missing ? ` · ${missedLabel}` : ''}</span>
  }

  return (
    <section className={`stage-panel ${cameraView === 'quad' ? 'quad-mode' : 'focus-mode'}`}>
      <article className="hero-camera">
        <div className="camera-panel-header">
          {previewError && <small role="status">{previewError}</small>}
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
          <NozzleCalibration language={language} calibration={calibration} onSave={setCalibration} />
          <ExposureControl language={language} />
          <ContrastControl language={language} />
        </div>

        {cameraView === 'focus' ? (
          <div className="camera-stage-frame">
            <canvas ref={bindCanvas(primaryCamera)} className="camera-feed-canvas" aria-label={t.high} data-preview-codec="png" data-preview-fps={stats[primaryCamera]?.fps} data-preview-p50-ms={stats[primaryCamera]?.p50} data-preview-p95-ms={stats[primaryCamera]?.p95} data-preview-decode-ms={stats[primaryCamera]?.decodeMs} />
            {renderTip(primaryCamera)}
            <div className="camera-frame-top">{renderOverlay(t.high, primaryCamera)}</div>
          </div>
        ) : (
          <div className="camera-grid">
            {CAMERAS.map((camera) => (
              <article key={camera.key} className="mini-camera-card quad-camera-card">
                <canvas ref={bindCanvas(camera.key)} className="camera-feed-canvas" aria-label={t[camera.labelKey]} data-preview-codec="png" data-preview-fps={stats[camera.key]?.fps} data-preview-p50-ms={stats[camera.key]?.p50} data-preview-p95-ms={stats[camera.key]?.p95} data-preview-decode-ms={stats[camera.key]?.decodeMs} />
                {renderTip(camera.key)}
                <div className="camera-frame-top">{renderOverlay(t[camera.labelKey], camera.key)}</div>
              </article>
            ))}
          </div>
        )}
      </article>

      {cameraView === 'focus' ? (
        <div className="camera-strip">
          {secondaryCameras.map((camera) => (
            <article key={camera.key} className="mini-camera-card">
              <canvas ref={bindCanvas(camera.key)} className="camera-feed-canvas" aria-label={t[camera.labelKey]} data-preview-codec="png" data-preview-fps={stats[camera.key]?.fps} data-preview-p50-ms={stats[camera.key]?.p50} data-preview-p95-ms={stats[camera.key]?.p95} data-preview-decode-ms={stats[camera.key]?.decodeMs} />
              {renderTip(camera.key)}
                <div className="camera-frame-top">{renderOverlay(t[camera.labelKey], camera.key)}</div>
            </article>
          ))}
        </div>
      ) : null}
    </section>
  )
}
