import { useEffect, useMemo, useRef, useState } from 'react'
import { AppLanguage, translations } from '../i18n'
import { apiBase } from '../services/api'

type Props = {
  mode: string
  language: AppLanguage
  dispatchTask: (taskNumber: string) => Promise<void>
  dispatchError: string
}

const TASK_NUMBERS = ['1', '2', '3', '4', '5'] as const

export function VoicePanel({ mode, language, dispatchTask, dispatchError }: Props) {
  const t = translations[language]
  const [status, setStatus] = useState<'idle' | 'recording' | 'thinking' | 'speaking'>('idle')
  const [errorText, setErrorText] = useState('')
  const [textCommand, setTextCommand] = useState('')
  const [level, setLevel] = useState(0)
  const [orbStyle, setOrbStyle] = useState<'pulse' | 'halo' | 'ripple'>('pulse')
  const [latestExchange, setLatestExchange] = useState<{ transcript: string; reply: string } | null>(null)
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const audioRef = useRef<HTMLAudioElement | null>(null)
  const mediaStreamRef = useRef<MediaStream | null>(null)
  const audioContextRef = useRef<AudioContext | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const silenceTimerRef = useRef<number | null>(null)
  const rafRef = useRef<number | null>(null)
  const speechStartedRef = useRef(false)
  const waitingAudioContextRef = useRef<AudioContext | null>(null)
  const waitingIntervalRef = useRef<number | null>(null)

  const canRecord = useMemo(
    () => typeof navigator !== 'undefined' && !!navigator.mediaDevices?.getUserMedia,
    [],
  )

  const cleanupRecordingResources = () => {
    if (rafRef.current !== null) {
      cancelAnimationFrame(rafRef.current)
      rafRef.current = null
    }
    if (silenceTimerRef.current !== null) {
      window.clearTimeout(silenceTimerRef.current)
      silenceTimerRef.current = null
    }
    mediaStreamRef.current?.getTracks().forEach((track) => track.stop())
    mediaStreamRef.current = null
    audioContextRef.current?.close().catch(() => {})
    audioContextRef.current = null
    analyserRef.current = null
    speechStartedRef.current = false
    setLevel(0)
  }

  const stopWaitingMusic = () => {
    if (waitingIntervalRef.current !== null) {
      window.clearInterval(waitingIntervalRef.current)
      waitingIntervalRef.current = null
    }
    waitingAudioContextRef.current?.close().catch(() => {})
    waitingAudioContextRef.current = null
  }

  const startWaitingMusic = () => {
    stopWaitingMusic()
    const AudioCtx = window.AudioContext || (window as typeof window & { webkitAudioContext?: typeof AudioContext }).webkitAudioContext
    if (!AudioCtx) {
      return
    }
    const context = new AudioCtx()
    waitingAudioContextRef.current = context
    const chords = [
      [293.66, 369.99],
      [329.63, 415.3],
      [369.99, 440.0],
      [329.63, 392.0],
    ]
    let index = 0

    const playNote = () => {
      if (!waitingAudioContextRef.current) {
        return
      }
      const now = context.currentTime
      const masterGain = context.createGain()
      masterGain.gain.setValueAtTime(0.0001, now)
      masterGain.gain.exponentialRampToValueAtTime(0.09, now + 0.08)
      masterGain.gain.exponentialRampToValueAtTime(0.0001, now + 0.7)
      masterGain.connect(context.destination)

      const currentChord = chords[index % chords.length]
      currentChord.forEach((frequency, chordIndex) => {
        const oscillator = context.createOscillator()
        const gain = context.createGain()
        oscillator.type = chordIndex === 0 ? 'sine' : 'triangle'
        oscillator.frequency.value = frequency
        gain.gain.value = chordIndex === 0 ? 0.8 : 0.45
        oscillator.connect(gain)
        gain.connect(masterGain)
        oscillator.start(now)
        oscillator.stop(now + 0.72)
      })
      index += 1
    }

    playNote()
    waitingIntervalRef.current = window.setInterval(playNote, 780)
  }

  useEffect(() => {
    return () => {
      cleanupRecordingResources()
      stopWaitingMusic()
      audioRef.current?.pause()
    }
  }, [])

  useEffect(() => {
    if (status === 'thinking') {
      startWaitingMusic()
    } else {
      stopWaitingMusic()
    }
  }, [status])

  const getAudioMimeType = () => {
    if (typeof MediaRecorder === 'undefined') return ''
    const candidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4', '']
    return candidates.find((candidate) => !candidate || MediaRecorder.isTypeSupported(candidate)) || ''
  }

  const handleVoiceResponse = async (response: Response) => {
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`)
    }
    const payload = await response.json()
    setLatestExchange({
      transcript: payload.transcript || '',
      reply: payload.reply_text || '',
    })
    if (payload.audio_base64 && payload.audio_mime_type) {
      setStatus('speaking')
      const audio = new Audio(`data:${payload.audio_mime_type};base64,${payload.audio_base64}`)
      audioRef.current = audio
      audio.onended = () => setStatus('idle')
      try {
        await audio.play()
      } catch {
        setStatus('idle')
      }
    } else {
      setStatus('idle')
    }
  }

  const submitTextCommand = async () => {
    const transcript = textCommand.trim()
    if (!transcript) return
    setErrorText('')
    setStatus('thinking')
    try {
      const response = await fetch(`${apiBase}/api/voice/text`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: transcript, language }),
      })
      await handleVoiceResponse(response)
      setTextCommand('')
    } catch {
      setStatus('idle')
      setErrorText(t.requestFailed)
    }
  }

  const toggleRecording = async () => {
    if (!canRecord) {
      setErrorText(t.micUnavailable)
      return
    }
    if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
      mediaRecorderRef.current.stop()
      return
    }

    setErrorText('')
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      mediaStreamRef.current = stream
      const mimeType = getAudioMimeType()
      const recorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream)
      const audioContext = new AudioContext()
      const sourceNode = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 2048
      sourceNode.connect(analyser)
      audioContextRef.current = audioContext
      analyserRef.current = analyser
      chunksRef.current = []
      recorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          chunksRef.current.push(event.data)
        }
      }
      recorder.onerror = () => {
        setStatus('idle')
        setErrorText(t.requestFailed)
      }
      recorder.onstop = async () => {
        try {
          setStatus('thinking')
          const blobType = mimeType || 'audio/webm'
          const extension = blobType.includes('mp4') ? 'm4a' : 'webm'
          const blob = new Blob(chunksRef.current, { type: blobType })
          if (blob.size === 0) {
            setStatus('idle')
            setErrorText(t.micFailed)
            return
          }
          const form = new FormData()
          form.append('file', blob, `voice.${extension}`)
          form.append('language', language)
          const response = await fetch(`${apiBase}/api/voice/audio`, {
            method: 'POST',
            body: form,
          })
          await handleVoiceResponse(response)
        } catch {
          setStatus('idle')
          setErrorText(t.requestFailed)
        } finally {
          cleanupRecordingResources()
          mediaRecorderRef.current = null
        }
      }
      mediaRecorderRef.current = recorder
      setStatus('recording')
      recorder.start(250)

      const waveform = new Uint8Array(analyser.frequencyBinCount)
      const silenceThreshold = 0.02
      const silenceMs = 1000

      const monitorSilence = () => {
        const currentRecorder = mediaRecorderRef.current
        if (!currentRecorder || currentRecorder.state !== 'recording' || !analyserRef.current) {
          return
        }
        analyserRef.current.getByteTimeDomainData(waveform)
        let sumSquares = 0
        for (let i = 0; i < waveform.length; i += 1) {
          const centered = (waveform[i] - 128) / 128
          sumSquares += centered * centered
        }
        const rms = Math.sqrt(sumSquares / waveform.length)
        setLevel(Math.min(1, rms * 10))
        const hasSpeech = rms > silenceThreshold

        if (hasSpeech) {
          speechStartedRef.current = true
          if (silenceTimerRef.current !== null) {
            window.clearTimeout(silenceTimerRef.current)
            silenceTimerRef.current = null
          }
        } else if (speechStartedRef.current && silenceTimerRef.current === null) {
          silenceTimerRef.current = window.setTimeout(() => {
            if (mediaRecorderRef.current?.state === 'recording') {
              mediaRecorderRef.current.stop()
            }
          }, silenceMs)
        }

        rafRef.current = requestAnimationFrame(monitorSilence)
      }

      rafRef.current = requestAnimationFrame(monitorSilence)
    } catch {
      setStatus('idle')
      cleanupRecordingResources()
      setErrorText(t.micFailed)
    }
  }

  return (
    <section className="panel voice-panel">
      <div className="panel-header">
        <div>
          <h2>{t.voiceTitle}</h2>
        </div>
        <div className="voice-toolbar">
          <div className="voice-style-toggle" aria-label={t.voiceStyleLabel}>
            <button
              type="button"
              className={orbStyle === 'pulse' ? 'active' : ''}
              onClick={() => setOrbStyle('pulse')}
            >
              {t.voiceStylePulse}
            </button>
            <button
              type="button"
              className={orbStyle === 'halo' ? 'active' : ''}
              onClick={() => setOrbStyle('halo')}
            >
              {t.voiceStyleHalo}
            </button>
            <button
              type="button"
              className={orbStyle === 'ripple' ? 'active' : ''}
              onClick={() => setOrbStyle('ripple')}
            >
              {t.voiceStyleRipple}
            </button>
          </div>
          <span className="status-pill mode" title={`${t.runtime}: ${mode}`}>{t.runtime}: {mode}</span>
        </div>
      </div>

      <div className="voice-console-layout">
        <div className="voice-orb-wrap">
        <button
          className={`voice-orb ${status} ${orbStyle}`}
          onClick={toggleRecording}
          disabled={!canRecord}
          style={{ ['--voice-level' as string]: String(level) }}
        >
          {status === 'recording' ? t.stop : t.talk}
        </button>
        <p className="voice-status">{t[status]}</p>
        <p className="voice-hint">{status === 'thinking' ? t.waitingMusic : t.autoSendHint}</p>
        {errorText ? <p className="voice-error">{errorText}</p> : null}
        </div>
        <div className="voice-transcript-panel">
          <div className="voice-shortcuts">
            <div className="task-grid compact">
              {TASK_NUMBERS.map((taskNumber) => (
                <button
                  key={taskNumber}
                  type="button"
                  className="task-card compact"
                  onClick={() => void dispatchTask(taskNumber)}
                  title={t.taskDescriptions[taskNumber]}
                >
                  <span className="task-key">{t.taskKey(taskNumber)}</span>
                  <strong>{t.taskShortLabels[taskNumber]}</strong>
                </button>
              ))}
            </div>
            {dispatchError ? <p className="voice-error inline-error">{dispatchError}</p> : null}
          </div>
          <div className="conversation-card">
            <div className="conversation-line">
              <span>{t.you}</span>
              <p>{latestExchange?.transcript || t.noConversation}</p>
            </div>
            <div className="conversation-line reply">
              <span>{t.aloha}</span>
              <p>{latestExchange?.reply || t.noReply}</p>
            </div>
          </div>
          <div className="voice-composer">
            <textarea
              value={textCommand}
              onChange={(event) => setTextCommand(event.target.value)}
              placeholder={t.commandPlaceholder}
              rows={3}
            />
            <button type="button" onClick={() => void submitTextCommand()}>
              {t.sendCommand}
            </button>
          </div>
        </div>
      </div>
    </section>
  )
}
