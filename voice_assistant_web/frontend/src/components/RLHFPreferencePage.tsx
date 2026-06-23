import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from 'react'
import {
  fetchRLTPreferencePair,
  rolloutVideoUrl,
  saveRLTPreference,
} from '../services/api'
import type {
  RLTKeyRegionReviewRecord,
  RLTPreferencePairResponse,
  RLTPreferenceRequest,
  RLTPreferenceStats,
} from '../services/api'

const CAMERA_ORDER = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']

const emptyStats: RLTPreferenceStats = {
  total_preferences: 0,
  left_wins: 0,
  right_wins: 0,
  ties: 0,
  both_bad: 0,
  skipped: 0,
}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const cameraLabelFromPath = (path: string) => path.split('/').pop()?.replace(/\.mp4$/, '') || 'camera'

const orderedCameraPaths = (record: RLTKeyRegionReviewRecord) => {
  const paths = [...record.video_paths]
  const ordered: string[] = []
  for (const camera of CAMERA_ORDER) {
    const index = paths.findIndex((path) => cameraLabelFromPath(path) === camera)
    if (index >= 0) ordered.push(paths.splice(index, 1)[0])
  }
  ordered.push(...paths)
  return ordered.slice(0, 4)
}

const formatDateTime = (seconds?: number | null) => {
  if (!seconds) return '-'
  return new Date(seconds * 1000).toLocaleString(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

const formatSeconds = (seconds: number) => `${seconds.toFixed(2)}s`

const cropStart = (record: RLTKeyRegionReviewRecord) => Math.max(0, record.crop_start_sec ?? 0)

const cropEnd = (record: RLTKeyRegionReviewRecord) => {
  const fallback = record.duration_seconds || record.crop_end_sec || 0
  return Math.max(cropStart(record), record.crop_end_sec ?? fallback)
}

type PreferenceClipHandle = {
  play: () => Promise<boolean>
  pause: () => void
  isPlaying: () => boolean
}

const PreferenceClip = forwardRef<
  PreferenceClipHandle,
  {
    label: string
    record: RLTKeyRegionReviewRecord | null
    onEdit: (record: RLTKeyRegionReviewRecord) => void
  }
>(
function PreferenceClip({ label, record, onEdit }, ref) {
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([])
  const [isPlaying, setIsPlaying] = useState(false)
  const [playbackTime, setPlaybackTime] = useState(0)

  const startSec = record ? cropStart(record) : 0
  const endSec = record ? cropEnd(record) : 0
  const windowDuration = Math.max(0.001, endSec - startSec)
  const progress = clamp((playbackTime - startSec) / windowDuration, 0, 1)

  const syncVideos = useCallback((timeSec: number) => {
    videoRefs.current.forEach((video) => {
      if (!video) return
      try {
        video.currentTime = timeSec
      } catch {
        // Metadata can still be loading; the next tick retries.
      }
    })
    setPlaybackTime(timeSec)
  }, [])

  const pause = useCallback(() => {
    videoRefs.current.forEach((video) => video?.pause())
    setIsPlaying(false)
  }, [])

  const play = useCallback(async () => {
    if (!record) return false
    const videos = videoRefs.current.filter((video): video is HTMLVideoElement => Boolean(video))
    if (!videos.length) return false
    const nextStart = playbackTime >= endSec ? startSec : clamp(playbackTime || startSec, startSec, endSec)
    syncVideos(nextStart)
    setIsPlaying(true)
    const results = await Promise.allSettled(
      videos.map(async (video) => {
        video.muted = true
        if (video.readyState === 0) video.load()
        await video.play()
      }),
    )
    const started = results.some((result) => result.status === 'fulfilled')
    if (!started) setIsPlaying(false)
    return started
  }, [endSec, playbackTime, record, startSec, syncVideos])

  useImperativeHandle(ref, () => ({
    play,
    pause,
    isPlaying: () => videoRefs.current.some((video) => Boolean(video && !video.paused)),
  }), [isPlaying, pause, play])

  useEffect(() => {
    if (!record) return undefined
    setIsPlaying(false)
    setPlaybackTime(startSec)
    const timer = window.setTimeout(() => syncVideos(startSec), 0)
    return () => window.clearTimeout(timer)
  }, [record, startSec, syncVideos])

  useEffect(() => {
    if (!record || !isPlaying) return undefined
    let timer = 0
    const tick = () => {
      const primary = videoRefs.current.find(Boolean)
      if (!primary) {
        setIsPlaying(false)
        return
      }
      const current = primary.currentTime
      setPlaybackTime(current)
      if (current >= endSec) {
        videoRefs.current.forEach((video) => video?.pause())
        syncVideos(endSec)
        setIsPlaying(false)
        return
      }
      timer = window.setTimeout(tick, 80)
    }
    timer = window.setTimeout(tick, 80)
    return () => window.clearTimeout(timer)
  }, [endSec, isPlaying, record, syncVideos])

  if (!record) {
    return (
      <section className="preference-clip empty">
        <h3>{label}</h3>
        <p>No clean cropped pair is available for the current filters.</p>
      </section>
    )
  }

  const seekToRatio = (ratio: number) => {
    const nextTime = startSec + clamp(ratio, 0, 1) * windowDuration
    videoRefs.current.forEach((video) => video?.pause())
    setIsPlaying(false)
    syncVideos(nextTime)
  }

  return (
    <section className="preference-clip">
      <div className="preference-clip-head">
        <div className="preference-clip-title-row">
          <span>{label}</span>
          <span>{record.batch || '-'}</span>
          <span>{formatDateTime(record.score_time || record.updated_at)}</span>
          <span>{record.num_replay_transitions || 0} samples</span>
          <span>{formatSeconds(startSec)} - {formatSeconds(endSec)}</span>
        </div>
        <strong className={record.reward === 1 ? 'clean-reward success' : 'clean-reward fail'}>
          {record.reward ?? '-'}
        </strong>
        <button className="preference-edit-button" type="button" onClick={() => onEdit(record)}>
          Edit
        </button>
      </div>
      <div className="preference-video-grid">
        {CAMERA_ORDER.map((camera, index) => {
          const path = orderedCameraPaths(record)[index]
          return (
            <div className="preference-video-tile" key={`${record.key_region_id}-${camera}`}>
              <span>{path ? cameraLabelFromPath(path) : camera}</span>
              {path ? (
                <video
                  ref={(element) => {
                    videoRefs.current[index] = element
                  }}
                  src={rolloutVideoUrl(path)}
                  preload="metadata"
                  muted
                  playsInline
                  onLoadedMetadata={() => syncVideos(startSec)}
                />
              ) : (
                <div>No video</div>
              )}
            </div>
          )
        })}
      </div>
      <div className="preference-playback">
        <button
          className="preference-timeline"
          type="button"
          aria-label={`${label} crop playback timeline`}
          onClick={(event) => {
            const rect = event.currentTarget.getBoundingClientRect()
            seekToRatio(rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0)
          }}
        >
          <span className="preference-timeline-window">
            <span className="preference-timeline-progress" style={{ width: `${progress * 100}%` }} />
            <span className="preference-playhead" style={{ left: `${progress * 100}%` }} />
          </span>
        </button>
        <span className="preference-time-readout">
          {formatSeconds(clamp(playbackTime, startSec, endSec))} / {formatSeconds(endSec)}
        </span>
      </div>
    </section>
  )
})

export function RLHFPreferencePage({
  refreshToken = 0,
  onEditKeyRegion,
}: {
  refreshToken?: number
  onEditKeyRegion: (record: RLTKeyRegionReviewRecord) => void
}) {
  const leftClipRef = useRef<PreferenceClipHandle | null>(null)
  const rightClipRef = useRef<PreferenceClipHandle | null>(null)
  const [pair, setPair] = useState<RLTPreferencePairResponse>({
    left: null,
    right: null,
    stats: emptyStats,
    remaining_unseen_pairs: 0,
    pair_type: null,
    strategy: 'budgeted',
    round_budget: 800,
    round_labeled: 0,
    round_remaining: 0,
  })
  const [batchFilter, setBatchFilter] = useState('all')
  const [rewardFilter, setRewardFilter] = useState('all')
  const [pairTypeFilter, setPairTypeFilter] = useState('auto')
  const [notes, setNotes] = useState('')
  const [pending, setPending] = useState('')
  const [error, setError] = useState('')

  const pausePair = useCallback(() => {
    leftClipRef.current?.pause()
    rightClipRef.current?.pause()
  }, [])

  const playPair = useCallback(async () => {
    await Promise.all([
      leftClipRef.current?.play() ?? Promise.resolve(false),
      rightClipRef.current?.play() ?? Promise.resolve(false),
    ])
  }, [])

  const togglePairPlayback = useCallback(() => {
    if (leftClipRef.current?.isPlaying() || rightClipRef.current?.isPlaying()) {
      pausePair()
      return
    }
    void playPair()
  }, [pausePair, playPair])

  const loadPair = useCallback(async () => {
    setError('')
    const nextPair = await fetchRLTPreferencePair({ batch: batchFilter, reward: rewardFilter, pairType: pairTypeFilter })
    setPair(nextPair)
    setNotes('')
    pausePair()
  }, [batchFilter, pairTypeFilter, pausePair, rewardFilter])

  useEffect(() => {
    void loadPair().catch((exc) => {
      setError(exc instanceof Error ? exc.message : 'Preference pair could not be loaded.')
    })
  }, [loadPair, refreshToken])

  const submitPreference = async (preference: RLTPreferenceRequest['preference']) => {
    if (!pair.left || !pair.right) return
    pausePair()
    setPending(preference)
    setError('')
    try {
      await saveRLTPreference({
        left_key_region_id: pair.left.key_region_id,
        right_key_region_id: pair.right.key_region_id,
        preference,
        reason_tags: [],
        notes: notes.trim() || undefined,
      })
      await loadPair()
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Preference could not be saved.')
    } finally {
      setPending('')
    }
  }

  useEffect(() => {
    const isTextEntryTarget = (target: EventTarget | null) => {
      if (!(target instanceof HTMLElement)) return false
      return target.dataset.preferenceTextInput === 'true' || target.isContentEditable
    }
    const onKeyDown = (event: KeyboardEvent) => {
      if (isTextEntryTarget(event.target) || event.altKey || event.ctrlKey || event.metaKey) return
      if (event.code === 'Space') {
        event.preventDefault()
        togglePairPlayback()
        return
      }
      const shortcuts: Record<string, RLTPreferenceRequest['preference']> = {
        '1': 'left',
        '2': 'right',
        '3': 'tie',
        '4': 'both_bad',
        s: 'skip',
        S: 'skip',
      }
      const preference = shortcuts[event.key]
      if (!preference || pending || !pair.left || !pair.right) return
      event.preventDefault()
      void submitPreference(preference)
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [pair.left, pair.right, pending, submitPreference, togglePairPlayback])

  return (
    <section className="preference-workspace">
      <div className="preference-toolbar">
        <div>
          <p className="eyebrow">RLHF</p>
          <h2>Preference Learning</h2>
        </div>
        <div className="preference-controls">
          <select value={rewardFilter} onChange={(event) => setRewardFilter(event.target.value)}>
            <option value="all">Reward any</option>
            <option value="success">Success only</option>
            <option value="failure">Failure only</option>
          </select>
          <select value={pairTypeFilter} onChange={(event) => setPairTypeFilter(event.target.value)}>
            <option value="auto">Budgeted auto</option>
            <option value="success_success">Success vs success</option>
            <option value="success_failure">Success vs failure</option>
            <option value="failure_failure">Failure vs failure</option>
          </select>
          <input
            data-preference-text-input="true"
            value={batchFilter}
            onChange={(event) => setBatchFilter(event.target.value || 'all')}
            placeholder="all or YYYY-MM-DD"
          />
          <button className="ghost-button" type="button" onClick={() => void loadPair()}>
            Refresh
          </button>
        </div>
      </div>

      <div className="preference-stats">
        <div><span>Total labels</span><strong>{pair.stats.total_preferences}</strong></div>
        <div><span>Round progress</span><strong>{pair.round_labeled} / {pair.round_budget}</strong></div>
        <div><span>Round remaining</span><strong>{pair.round_remaining}</strong></div>
        <div><span>Pair type</span><strong>{pair.pair_type?.replace('_', ' / ') || '-'}</strong></div>
        <div><span>Candidate pairs</span><strong>{pair.remaining_unseen_pairs}</strong></div>
      </div>

      {error ? <p className="inline-error">{error}</p> : null}

      <p className="preference-shortcuts">
        Space play/pause both · 1 left better · 2 right better · 3 tie · 4 both bad · S skip
      </p>

      <div className="preference-pair-grid">
        <PreferenceClip ref={leftClipRef} label="Left" record={pair.left} onEdit={onEditKeyRegion} />
        <PreferenceClip ref={rightClipRef} label="Right" record={pair.right} onEdit={onEditKeyRegion} />
      </div>

      <div className="preference-review-bar">
        <input
          data-preference-text-input="true"
          value={notes}
          onChange={(event) => setNotes(event.target.value)}
          placeholder="optional note"
        />
        <div className="preference-actions">
          <button type="button" onClick={() => void submitPreference('left')} disabled={!pair.left || !pair.right || Boolean(pending)}>
            {pending === 'left' ? 'Saving' : '1 Left better'}
          </button>
          <button type="button" onClick={() => void submitPreference('right')} disabled={!pair.left || !pair.right || Boolean(pending)}>
            {pending === 'right' ? 'Saving' : '2 Right better'}
          </button>
          <button type="button" onClick={() => void submitPreference('tie')} disabled={!pair.left || !pair.right || Boolean(pending)}>
            3 Tie
          </button>
          <button type="button" onClick={() => void submitPreference('both_bad')} disabled={!pair.left || !pair.right || Boolean(pending)}>
            4 Both bad
          </button>
          <button type="button" onClick={() => void submitPreference('skip')} disabled={!pair.left || !pair.right || Boolean(pending)}>
            S Skip
          </button>
        </div>
      </div>
    </section>
  )
}
