import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  cropKeyRegion,
  deleteKeyRegions,
  fetchRLTKeyRegionDetail,
  fetchRLTKeyRegionReview,
  rescoreKeyRegion,
  rolloutVideoUrl,
} from '../services/api'
import type {
  RLTKeyRegionReviewRecord,
  RLTKeyRegionReviewSummary,
} from '../services/api'

type CropRange = {
  startSec: number
  endSec: number
}

type KeyRegionInfoRow = {
  label: string
  value: string
}

type KeyRegionFocusTarget = {
  keyRegionId: string
  batch: string | null
}

const KEY_REGION_CAMERA_ORDER = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
const KEY_REGION_PAGE_SIZE = 10

const emptySummary: RLTKeyRegionReviewSummary = {
  total: 0,
  trainable: 0,
  needs_crop: 0,
  success: 0,
  failure: 0,
  replay_samples: 0,
}

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

const formatShortDateTime = (seconds?: number | null) => {
  if (!seconds) return ''
  return new Date(seconds * 1000).toLocaleString(undefined, {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

const formatDuration = (seconds?: number | null) => {
  if (seconds === null || seconds === undefined) return '-'
  return `${seconds.toFixed(1)}s`
}

const formatRewardValue = (reward?: number | null) => {
  if (reward === 1) return 'success'
  if (reward === 0) return 'failure'
  return '-'
}

const formatReviewTime = (record: RLTKeyRegionReviewRecord) => {
  const timestamp = record.score_time || record.end_time || record.start_time || record.updated_at
  if (!timestamp) return record.key_region_id
  return new Date(timestamp * 1000).toLocaleString(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
}

const reviewTimestamp = (record: RLTKeyRegionReviewRecord) =>
  record.score_time || record.end_time || record.start_time || record.updated_at || null

const annotationTimestamp = (record: RLTKeyRegionReviewRecord) =>
  record.score_time || record.updated_at || null

const formatAnnotationTime = (record: RLTKeyRegionReviewRecord) => {
  const timestamp = annotationTimestamp(record)
  return timestamp ? formatShortDateTime(timestamp) : 'not annotated'
}

const reviewTitle = (record: RLTKeyRegionReviewRecord, absoluteIndex: number, total: number) => {
  const time = formatShortDateTime(reviewTimestamp(record)) || record.key_region_id
  const task = record.task || 'key region'
  return `Card ${absoluteIndex + 1} / ${total} - ${time} - ${task} - ${record.key_region_id}`
}

const keyRegionStatusLabel = (record: RLTKeyRegionReviewRecord) => {
  if (record.needs_crop) return 'needs crop'
  if (record.trainable) return record.status || 'committed'
  if (record.incomplete_reason) return 'needs crop'
  return record.status || 'pending'
}

const keyRegionEligibility = (record: RLTKeyRegionReviewRecord) => {
  if (record.needs_crop) return 'needs crop'
  if (record.trainable) return 'trainable'
  if (record.incomplete_reason) return 'pending'
  return record.train_eligible === false ? 'not eligible' : 'pending'
}

const cameraLabelFromPath = (path: string) => {
  const filename = path.split('/').pop()?.replace(/\.mp4$/, '') || 'camera'
  return filename
}

const orderedCameraPaths = (record: RLTKeyRegionReviewRecord) => {
  const paths = [...record.video_paths]
  const ordered: string[] = []
  for (const camera of KEY_REGION_CAMERA_ORDER) {
    const index = paths.findIndex((path) => cameraLabelFromPath(path) === camera)
    if (index >= 0) ordered.push(paths.splice(index, 1)[0])
  }
  ordered.push(...paths)
  return ordered.slice(0, 4)
}

const durationForRecord = (record: RLTKeyRegionReviewRecord) =>
  Math.max(0, record.duration_seconds || record.crop_end_sec || 0)

const defaultCropRange = (record: RLTKeyRegionReviewRecord): CropRange => {
  const duration = durationForRecord(record)
  if (record.crop_start_sec !== null && record.crop_start_sec !== undefined && record.crop_end_sec) {
    return {
      startSec: clamp(record.crop_start_sec, 0, duration),
      endSec: clamp(Math.max(record.crop_start_sec, record.crop_end_sec), 0, duration),
    }
  }
  return {
    startSec: 0,
    endSec: duration,
  }
}

const cropRangeForRecord = (
  record: RLTKeyRegionReviewRecord,
  cropRanges: Record<string, CropRange>,
): CropRange => cropRanges[record.key_region_id] || defaultCropRange(record)

const estimatedCropSamples = (record: RLTKeyRegionReviewRecord, range: CropRange) => {
  const duration = durationForRecord(record)
  const sourceSamples = record.crop_original_num_replay_transitions || record.num_replay_transitions || 0
  if (duration <= 0 || sourceSamples <= 0) return 0
  return Math.max(1, Math.round(((range.endSec - range.startSec) / duration) * sourceSamples))
}

const minCropDuration = (record: RLTKeyRegionReviewRecord) => {
  const duration = durationForRecord(record)
  const sourceSamples = record.crop_original_num_replay_transitions || record.num_replay_transitions || 0
  if (duration <= 0 || sourceSamples <= 0) return 0
  return Math.min(duration, Math.max(0.01, duration / sourceSamples))
}

const cropSaveBlockedReason = (record: RLTKeyRegionReviewRecord, range: CropRange) => {
  if (durationForRecord(record) <= 0) return 'No video duration'
  if (range.endSec <= range.startSec) return 'Crop range is empty'
  if (!record.shard_path || !record.npz_exists) {
    const missing = record.missing_rlt_metadata?.length ? `; missing ${record.missing_rlt_metadata.join(', ')}` : ''
    return `No replay shard${missing}`
  }
  return ''
}

const cropSummary = (record: RLTKeyRegionReviewRecord, range: CropRange) => {
  const samples = estimatedCropSamples(record, range)
  return `selected ${range.startSec.toFixed(2)}s - ${range.endSec.toFixed(2)}s / ${samples} replay samples`
}

const frameSummary = (record: RLTKeyRegionReviewRecord, range: CropRange) => {
  const fps = record.fps || 30
  const startFrame = Math.round(range.startSec * fps)
  const endFrame = Math.round(range.endSec * fps)
  return { startFrame, endFrame }
}

const badgeTone = (tone: 'green' | 'blue' | 'amber' | 'red' | 'slate') => `key-region-badge ${tone}`

const keyRegionInfoRows = (record: RLTKeyRegionReviewRecord): KeyRegionInfoRow[] => [
  { label: 'Batch', value: record.batch || '-' },
  { label: 'Last annotation', value: formatAnnotationTime(record) },
  { label: 'Crop', value: `${formatDuration(record.crop_start_sec)} - ${formatDuration(record.crop_end_sec)}` },
  { label: 'Samples', value: `${record.num_replay_transitions || 0}` },
  { label: 'Reward', value: record.reward === null ? '-' : `${record.reward}` },
  { label: 'Replay', value: record.replay_status || (record.npz_exists ? 'written' : 'pending') },
  { label: 'Eligibility', value: keyRegionEligibility(record) },
]

const unloadVideo = (video: HTMLVideoElement) => {
  video.pause()
  video.currentTime = 0
}

const videoReadyForPlayback = (video: HTMLVideoElement | null) => Boolean(video && video.readyState >= 1)

function KeyRegionVideoGrid({
  record,
  active,
  registerVideo,
  onSelect,
}: {
  record: RLTKeyRegionReviewRecord
  active: boolean
  registerVideo: (index: number, element: HTMLVideoElement | null) => void
  onSelect: () => void
}) {
  const cameras = orderedCameraPaths(record)

  return (
    <div className="key-region-video-grid">
      {KEY_REGION_CAMERA_ORDER.map((camera, cameraIndex) => {
        const path = cameras[cameraIndex]
        return (
          <div className="key-region-video-tile" key={`${record.key_region_id}-${camera}`}>
            <span className="key-region-camera-label">{path ? cameraLabelFromPath(path) : camera}</span>
            {path && active ? (
              <video
                key={path}
                ref={(element) => registerVideo(cameraIndex, element)}
                src={rolloutVideoUrl(path)}
                preload="metadata"
                muted
                playsInline
                tabIndex={-1}
              />
            ) : path ? (
              <button className="key-region-video-placeholder" type="button" onClick={onSelect}>
                <span>Load preview</span>
              </button>
            ) : (
              <div className="key-region-video-missing">No video</div>
            )}
          </div>
        )
      })}
    </div>
  )
}

export function KeyRegionsPage({
  title,
  focusTarget = null,
  onBackToRLHF,
}: {
  title: string
  focusTarget?: KeyRegionFocusTarget | null
  onBackToRLHF?: () => void
}) {
  const [records, setRecords] = useState<RLTKeyRegionReviewRecord[]>([])
  const [summary, setSummary] = useState<RLTKeyRegionReviewSummary>(emptySummary)
  const [total, setTotal] = useState(0)
  const [offset, setOffset] = useState(0)
  const [nextOffset, setNextOffset] = useState<number | null>(null)
  const [statusFilter, setStatusFilter] = useState<'all' | 'trainable' | 'needsCrop'>('all')
  const [rewardFilter, setRewardFilter] = useState<'all' | 'success' | 'failure'>('all')
  const [batchFilter, setBatchFilter] = useState('all')
  const [batches, setBatches] = useState<string[]>([])
  const [selectedReviewId, setSelectedReviewId] = useState('')
  const [selectedDetail, setSelectedDetail] = useState<RLTKeyRegionReviewRecord | null>(null)
  const [pendingFocusTarget, setPendingFocusTarget] = useState<KeyRegionFocusTarget | null>(null)
  const [focusKeyRegionId, setFocusKeyRegionId] = useState('')
  const [selectedKeyRegionIds, setSelectedKeyRegionIds] = useState<Set<string>>(new Set())
  const [cropRanges, setCropRanges] = useState<Record<string, CropRange>>({})
  const [playingKeyRegionId, setPlayingKeyRegionId] = useState('')
  const [pendingPlaybackId, setPendingPlaybackId] = useState('')
  const [playbackTimes, setPlaybackTimes] = useState<Record<string, number>>({})
  const [actionPending, setActionPending] = useState('')
  const [actionError, setActionError] = useState('')
  const [playbackError, setPlaybackError] = useState('')
  const [loading, setLoading] = useState(false)
  const videoRefs = useRef<Array<HTMLVideoElement | null>>([])
  const selectedReviewIdRef = useRef('')

  const activeRecord = useMemo(
    () => selectedDetail || records.find((record) => record.key_region_id === selectedReviewId) || null,
    [records, selectedDetail, selectedReviewId],
  )

  const unloadActiveVideos = useCallback(() => {
    videoRefs.current.forEach((video) => {
      if (video) unloadVideo(video)
    })
    videoRefs.current = []
  }, [])

  const registerVideo = useCallback((index: number, element: HTMLVideoElement | null) => {
    if (!element) {
      videoRefs.current[index] = null
      return
    }
    videoRefs.current[index] = element
  }, [])

  const loadPage = useCallback(async () => {
    setLoading(true)
    setActionError('')
    try {
      const page = await fetchRLTKeyRegionReview({
        limit: KEY_REGION_PAGE_SIZE,
        offset,
        status: statusFilter,
        reward: rewardFilter,
        batch: batchFilter,
        focusKeyRegionId: focusKeyRegionId || undefined,
      })
      setRecords(page.items)
      setSummary(page.summary)
      setTotal(page.total)
      if (page.offset !== offset) setOffset(page.offset)
      setNextOffset(page.next_offset)
      setBatches(page.batches)
      setSelectedKeyRegionIds((current) => {
        const visible = new Set(page.items.map((record) => record.key_region_id))
        return new Set([...current].filter((keyRegionId) => visible.has(keyRegionId)))
      })
      if (selectedReviewIdRef.current && !page.items.some((record) => record.key_region_id === selectedReviewIdRef.current)) {
        setSelectedReviewId('')
      }
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : 'Key regions could not be loaded.')
    } finally {
      setLoading(false)
    }
  }, [batchFilter, focusKeyRegionId, offset, rewardFilter, statusFilter])

  useEffect(() => {
    selectedReviewIdRef.current = selectedReviewId
  }, [selectedReviewId])

  useEffect(() => {
    void loadPage()
  }, [loadPage])

  useEffect(() => {
    if (!focusTarget) return
    setPendingFocusTarget(focusTarget)
    setFocusKeyRegionId(focusTarget.keyRegionId)
    setStatusFilter('all')
    setRewardFilter('all')
    setBatchFilter(focusTarget.batch || 'all')
    setOffset(0)
    setSelectedReviewId('')
    setSelectedDetail(null)
    unloadActiveVideos()
    setPlayingKeyRegionId('')
    setPendingPlaybackId('')
  }, [focusTarget, unloadActiveVideos])

  useEffect(() => {
    setOffset(0)
    setSelectedReviewId('')
    setSelectedDetail(null)
    setSelectedKeyRegionIds(new Set())
    unloadActiveVideos()
    setPlayingKeyRegionId('')
    setPendingPlaybackId('')
  }, [batchFilter, rewardFilter, statusFilter, unloadActiveVideos])

  useEffect(() => {
    if (!selectedReviewId) {
      setSelectedDetail(null)
      unloadActiveVideos()
      setPendingPlaybackId('')
      return undefined
    }
    let ignore = false
    void fetchRLTKeyRegionDetail(selectedReviewId)
      .then((record) => {
        if (!ignore) setSelectedDetail(record)
      })
      .catch((exc) => {
        if (!ignore) setPlaybackError(exc instanceof Error ? exc.message : 'Preview details could not be loaded.')
      })
    return () => {
      ignore = true
    }
  }, [selectedReviewId, unloadActiveVideos])

  useEffect(() => {
    return () => {
      unloadActiveVideos()
    }
  }, [unloadActiveVideos])

  const visibleIds = useMemo(() => new Set(records.map((record) => record.key_region_id)), [records])
  const selectedCount = selectedKeyRegionIds.size
  const pageNumber = Math.floor(offset / KEY_REGION_PAGE_SIZE) + 1
  const pageCount = Math.max(1, Math.ceil(total / KEY_REGION_PAGE_SIZE))

  const getCropRange = (record: RLTKeyRegionReviewRecord) => cropRangeForRecord(record, cropRanges)

  const selectReviewRecord = (record: RLTKeyRegionReviewRecord) => {
    setPlaybackError('')
    if (selectedReviewIdRef.current && selectedReviewIdRef.current !== record.key_region_id) {
      pauseActiveVideos()
      unloadActiveVideos()
      setPlayingKeyRegionId('')
    }
    setSelectedReviewId(record.key_region_id)
  }

  useEffect(() => {
    if (!pendingFocusTarget || loading) return
    const focusedRecord = records.find((record) => record.key_region_id === pendingFocusTarget.keyRegionId)
    if (focusedRecord) {
      selectReviewRecord(focusedRecord)
      setPendingFocusTarget(null)
      setFocusKeyRegionId('')
      return
    }
    setActionError(`Could not find key region ${pendingFocusTarget.keyRegionId} in the current dataset.`)
    setPendingFocusTarget(null)
    setFocusKeyRegionId('')
  }, [loading, pendingFocusTarget, records])

  const syncVideos = (record: RLTKeyRegionReviewRecord, timeSec: number) => {
    videoRefs.current.forEach((video) => {
      if (!video) return
      try {
        if (Number.isFinite(video.duration)) {
          video.currentTime = clamp(timeSec, 0, Math.max(0, video.duration))
        } else {
          video.currentTime = Math.max(0, timeSec)
        }
      } catch {
        // Metadata can still be loading; the next play/drag tick will retry.
      }
    })
    setPlaybackTimes((times) => ({ ...times, [record.key_region_id]: timeSec }))
  }

  const pauseActiveVideos = () => {
    videoRefs.current.forEach((video) => video?.pause())
  }

  const startCropPlayback = async (record: RLTKeyRegionReviewRecord) => {
    const videos = videoRefs.current.filter((video): video is HTMLVideoElement => Boolean(video))
    if (!videos.length) {
      setPlaybackError('')
      return false
    }
    const range = getCropRange(record)
    setPlaybackError('')
    syncVideos(record, range.startSec)
    const results = await Promise.allSettled(
      videos.map(async (video) => {
        video.muted = true
        if (video.readyState === 0) video.load()
        await video.play()
      }),
    )
    if (results.every((result) => result.status === 'rejected')) {
      setPlaybackError('Preview playback failed. Try again after video metadata loads.')
      setPlayingKeyRegionId('')
      return false
    }
    setPlayingKeyRegionId(record.key_region_id)
    return true
  }

  const toggleCropPlayback = async (record: RLTKeyRegionReviewRecord) => {
    if (playingKeyRegionId === record.key_region_id) {
      pauseActiveVideos()
      setPlayingKeyRegionId('')
      setPendingPlaybackId('')
      return
    }
    if (selectedReviewId !== record.key_region_id) {
      setPendingPlaybackId(record.key_region_id)
      selectReviewRecord(record)
      return
    }
    setPendingPlaybackId('')
    await startCropPlayback(record)
  }

  useEffect(() => {
    if (!pendingPlaybackId || !activeRecord || activeRecord.key_region_id !== pendingPlaybackId) return undefined
    let cancelled = false
    const startedAt = Date.now()
    let timer = 0
    const attemptPlayback = () => {
      if (cancelled) return
      const hasReadyVideo = videoRefs.current.some(videoReadyForPlayback)
      if (!hasReadyVideo) {
        if (Date.now() - startedAt > 5000) {
          setPlaybackError('Preview video could not be loaded. Try opening the card again.')
          setPendingPlaybackId('')
          return
        }
        timer = window.setTimeout(attemptPlayback, 120)
        return
      }
      void startCropPlayback(activeRecord).then((started) => {
        if (!cancelled && started) {
          setPendingPlaybackId('')
        } else if (!cancelled) {
          timer = window.setTimeout(attemptPlayback, 160)
        }
      })
    }
    timer = window.setTimeout(attemptPlayback, 0)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [activeRecord, pendingPlaybackId])

  useEffect(() => {
    if (!playingKeyRegionId || !activeRecord || activeRecord.key_region_id !== playingKeyRegionId) return undefined
    let timer = 0
    const tick = () => {
      const primaryVideo = videoRefs.current.find(Boolean)
      if (!primaryVideo) {
        setPlayingKeyRegionId('')
        return
      }
      const range = cropRangeForRecord(activeRecord, cropRanges)
      const currentTime = primaryVideo.currentTime
      setPlaybackTimes((times) => ({ ...times, [playingKeyRegionId]: currentTime }))
      if (currentTime >= range.endSec) {
        pauseActiveVideos()
        syncVideos(activeRecord, range.endSec)
        setPlayingKeyRegionId('')
        return
      }
      timer = window.setTimeout(tick, 100)
    }
    timer = window.setTimeout(tick, 100)
    return () => window.clearTimeout(timer)
  }, [activeRecord, cropRanges, playingKeyRegionId])

  const updateCropFromPointer = (
    record: RLTKeyRegionReviewRecord,
    edge: 'start' | 'end',
    track: HTMLElement,
    clientX: number,
  ) => {
    const duration = durationForRecord(record)
    if (duration <= 0) return
    const rect = track.getBoundingClientRect()
    const ratio = rect.width > 0 ? clamp((clientX - rect.left) / rect.width, 0, 1) : 0
    const nextSec = ratio * duration
    const current = getCropRange(record)
    const minDuration = minCropDuration(record)
    const nextRange =
      edge === 'start'
        ? {
            startSec: clamp(nextSec, 0, Math.max(0, current.endSec - minDuration)),
            endSec: current.endSec,
          }
        : {
            startSec: current.startSec,
            endSec: clamp(nextSec, Math.min(duration, current.startSec + minDuration), duration),
          }
    setCropRanges((ranges) => ({ ...ranges, [record.key_region_id]: nextRange }))
    if (record.key_region_id === selectedReviewId) {
      syncVideos(record, edge === 'start' ? nextRange.startSec : nextRange.endSec)
    }
  }

  const beginCropDrag = (
    event: React.PointerEvent<HTMLButtonElement>,
    record: RLTKeyRegionReviewRecord,
    edge: 'start' | 'end',
  ) => {
    event.preventDefault()
    event.stopPropagation()
    const track = event.currentTarget.closest('.key-region-timeline-track')
    if (!(track instanceof HTMLElement)) return
    pauseActiveVideos()
    setPlayingKeyRegionId('')
    updateCropFromPointer(record, edge, track, event.clientX)
    const onMove = (moveEvent: PointerEvent) => updateCropFromPointer(record, edge, track, moveEvent.clientX)
    const onUp = () => {
      window.removeEventListener('pointermove', onMove)
      window.removeEventListener('pointerup', onUp)
    }
    window.addEventListener('pointermove', onMove)
    window.addEventListener('pointerup', onUp, { once: true })
  }

  const saveCropForQ = async (record: RLTKeyRegionReviewRecord) => {
    const range = getCropRange(record)
    setActionError('')
    setActionPending(`crop-${record.key_region_id}`)
    try {
      await cropKeyRegion(record.key_region_id, range.startSec, range.endSec)
      await loadPage()
      if (record.key_region_id === selectedReviewId) {
        const detail = await fetchRLTKeyRegionDetail(record.key_region_id)
        setSelectedDetail(detail)
      }
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : 'Crop save failed')
    } finally {
      setActionPending('')
    }
  }

  const rescoreForQ = async (record: RLTKeyRegionReviewRecord, reward: 0 | 1) => {
    setActionError('')
    setActionPending(`rescore-${record.key_region_id}-${reward}`)
    try {
      await rescoreKeyRegion(record.key_region_id, reward)
      await loadPage()
      if (record.key_region_id === selectedReviewId) {
        const detail = await fetchRLTKeyRegionDetail(record.key_region_id)
        setSelectedDetail(detail)
      }
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : 'Rescore failed')
    } finally {
      setActionPending('')
    }
  }

  const toggleKeyRegionSelection = (keyRegionId: string) => {
    setSelectedKeyRegionIds((current) => {
      const next = new Set(current)
      if (next.has(keyRegionId)) next.delete(keyRegionId)
      else next.add(keyRegionId)
      return next
    })
  }

  const selectVisibleKeyRegions = () => {
    setSelectedKeyRegionIds(new Set(records.map((record) => record.key_region_id)))
  }

  const deleteSelected = async () => {
    const keyRegionIds = [...selectedKeyRegionIds].filter((keyRegionId) => visibleIds.has(keyRegionId))
    if (!keyRegionIds.length) return
    if (!window.confirm(`Delete ${keyRegionIds.length} selected key region${keyRegionIds.length === 1 ? '' : 's'}?`)) {
      return
    }
    setActionError('')
    setActionPending('delete-key-regions')
    try {
      await deleteKeyRegions(keyRegionIds, keyRegionIds.length > 1 ? 'operator_batch_delete' : 'operator_delete')
      setSelectedKeyRegionIds(new Set())
      await loadPage()
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : 'Delete failed')
    } finally {
      setActionPending('')
    }
  }

  const goToOffset = (next: number) => {
    unloadActiveVideos()
    setPlayingKeyRegionId('')
    setSelectedDetail(null)
    setOffset(Math.max(0, next))
  }

  return (
    <section className="key-regions-workspace">
      <div className="key-region-scroll-indicator">
        Page {pageNumber} / {pageCount}
      </div>
      <div className="key-regions-toolbar">
        <div>
          <p className="eyebrow">Rollouts</p>
          <h2>{title}</h2>
        </div>
        <div className="key-regions-controls">
          <select
            className="key-region-control"
            value={statusFilter}
            onChange={(event) => setStatusFilter(event.target.value as 'all' | 'trainable' | 'needsCrop')}
          >
            <option value="all">All statuses</option>
            <option value="trainable">Trainable</option>
            <option value="needsCrop">Needs crop</option>
          </select>
          <select
            className="key-region-control"
            value={rewardFilter}
            onChange={(event) => setRewardFilter(event.target.value as 'all' | 'success' | 'failure')}
          >
            <option value="all">Reward any</option>
            <option value="success">Success only</option>
            <option value="failure">Failure only</option>
          </select>
          <select
            className="key-region-control"
            value={batchFilter}
            onChange={(event) => setBatchFilter(event.target.value)}
          >
            <option value="all">All batches</option>
            {batches.map((batch) => (
              <option value={batch} key={batch}>{batch}</option>
            ))}
          </select>
          <button className="ghost-button" type="button" onClick={selectVisibleKeyRegions} disabled={!records.length}>
            Select page
          </button>
          <button className="ghost-button" type="button" onClick={() => setSelectedKeyRegionIds(new Set())} disabled={!selectedCount}>
            Clear
          </button>
          <button
            className="ghost-button danger"
            type="button"
            onClick={() => void deleteSelected()}
            disabled={!selectedCount || actionPending === 'delete-key-regions'}
          >
            Delete selected
          </button>
          <button className="ghost-button" type="button" onClick={() => void loadPage()} disabled={loading}>
            {loading ? 'Refreshing' : 'Refresh'}
          </button>
          {onBackToRLHF ? (
            <button
              className="apply-button"
              type="button"
              onClick={onBackToRLHF}
              disabled={Boolean(actionPending) || loading}
            >
              Back to RLHF
            </button>
          ) : null}
        </div>
      </div>

      <div className="key-region-summary-strip">
        <div className="key-region-summary-tile"><span>Total key regions</span><strong>{summary.total}</strong></div>
        <div className="key-region-summary-tile"><span>Trainable key regions</span><strong>{summary.trainable}</strong></div>
        <div className="key-region-summary-tile"><span>Confirmed replay samples</span><strong>{summary.replay_samples}</strong></div>
        <div className="key-region-summary-tile"><span>Success / failure</span><strong>{summary.success} / {summary.failure}</strong></div>
        <div className="key-region-summary-tile"><span>Needs crop review</span><strong>{summary.needs_crop}</strong></div>
        <div className="key-region-summary-tile"><span>Selected</span><strong>{selectedCount}</strong></div>
      </div>

      <div className="key-region-page-controls">
        <button className="ghost-button" type="button" onClick={() => goToOffset(offset - KEY_REGION_PAGE_SIZE)} disabled={offset <= 0 || loading}>
          Previous
        </button>
        <span>
          Showing {total ? offset + 1 : 0}-{Math.min(offset + records.length, total)} / {total}
        </span>
        <button className="ghost-button" type="button" onClick={() => goToOffset(nextOffset ?? offset)} disabled={nextOffset === null || loading}>
          Next
        </button>
      </div>

      {actionError ? <p className="inline-error">{actionError}</p> : null}
      {playbackError ? <p className="inline-error">{playbackError}</p> : null}

      <div className="key-region-card-stack">
        {records.map((record, index) => {
          const activeCard = selectedReviewId === record.key_region_id
          const renderRecord = activeCard && activeRecord ? activeRecord : record
          const checked = selectedKeyRegionIds.has(record.key_region_id)
          const cropRange = getCropRange(renderRecord)
          const duration = durationForRecord(renderRecord)
          const frames = frameSummary(renderRecord, cropRange)
          const clipLeft = duration > 0 ? clamp((cropRange.startSec / duration) * 100, 0, 100) : 0
          const clipWidth =
            duration > 0 ? clamp(((cropRange.endSec - cropRange.startSec) / duration) * 100, 0, 100 - clipLeft) : 0
          const playbackTime = playbackTimes[renderRecord.key_region_id] ?? cropRange.startSec
          const playbackProgress =
            cropRange.endSec > cropRange.startSec
              ? clamp((playbackTime - cropRange.startSec) / (cropRange.endSec - cropRange.startSec), 0, 1)
              : 0
          const cropPending = actionPending === `crop-${renderRecord.key_region_id}`
          const cropBlockedReason = cropSaveBlockedReason(renderRecord, cropRange)
          const rescoreZeroPending = actionPending === `rescore-${renderRecord.key_region_id}-0`
          const rescoreOnePending = actionPending === `rescore-${renderRecord.key_region_id}-1`
          const isPlaying = playingKeyRegionId === renderRecord.key_region_id
          const isPlaybackLoading = pendingPlaybackId === renderRecord.key_region_id
          return (
            <article key={record.key_region_id} className={`key-region-card ${activeCard ? 'active' : ''}`}>
              <div className="key-region-card-head">
                <input
                  className="key-region-checkbox"
                  type="checkbox"
                  checked={checked}
                  onChange={() => toggleKeyRegionSelection(record.key_region_id)}
                  aria-label={`Select ${record.key_region_id}`}
                />
                <button className="key-region-title-button" type="button" onClick={() => selectReviewRecord(record)}>
                  <strong>{reviewTitle(record, offset + index, total)}</strong>
                  <span>{formatReviewTime(record)} · Last annotation: {formatAnnotationTime(record)}</span>
                </button>
              </div>

              <div className="key-region-card-main">
                <KeyRegionVideoGrid
                  record={renderRecord}
                  active={activeCard}
                  registerVideo={activeCard ? registerVideo : () => undefined}
                  onSelect={() => selectReviewRecord(record)}
                />

                <aside className="key-region-info-panel">
                  <div className="key-region-panel-title">
                    <h3>Replay Buffer</h3>
                    <span>{renderRecord.trainable ? 'ready' : keyRegionStatusLabel(renderRecord)}</span>
                  </div>
                  <div className="key-region-kv-grid">
                    {keyRegionInfoRows(renderRecord).map((row) => (
                      <div className="key-region-kv" key={row.label}>
                        <span>{row.label}</span>
                        <strong>{row.value}</strong>
                      </div>
                    ))}
                  </div>
                  {renderRecord.missing_rlt_metadata?.length ? (
                    <p className="key-region-crop-warning">Missing {renderRecord.missing_rlt_metadata.join(', ')}</p>
                  ) : null}
                  <div className="key-region-paths">
                    <div>
                      <span>Rollout path</span>
                      <code title={renderRecord.local_rollout_path || renderRecord.rollout_path || ''}>
                        {renderRecord.local_rollout_path || renderRecord.rollout_path || '-'}
                      </code>
                    </div>
                    <div>
                      <span>Replay shard</span>
                      <code title={renderRecord.local_shard_path || renderRecord.shard_path || ''}>
                        {renderRecord.local_shard_path || renderRecord.shard_path || '-'}
                      </code>
                    </div>
                  </div>
                </aside>
              </div>

              <div className="key-region-trim-panel">
                <div className="key-region-trim-head">
                  <strong>Crop for Q training</strong>
                  <span>{cropSummary(renderRecord, cropRange)}</span>
                </div>
                <div className="key-region-timeline">
                  <div className="key-region-timeline-track">
                    <div
                      className={`key-region-clip ${renderRecord.reward === 0 ? 'fail' : 'success'}`}
                      style={{ left: `${clipLeft}%`, width: `${clipWidth}%` }}
                    >
                      <span className="key-region-clip-progress" style={{ width: `${playbackProgress * 100}%` }} />
                      <button
                        className="key-region-handle start"
                        type="button"
                        aria-label="Set crop start"
                        onPointerDown={(event) => beginCropDrag(event, renderRecord, 'start')}
                      />
                      <button
                        className="key-region-handle end"
                        type="button"
                        aria-label="Set crop end"
                        onPointerDown={(event) => beginCropDrag(event, renderRecord, 'end')}
                      />
                      <span className="key-region-playhead" style={{ left: `${playbackProgress * 100}%` }} />
                      <span className={`key-region-marker ${renderRecord.reward === 0 ? 'fail' : 'success'}`} style={{ left: '92%' }} />
                    </div>
                  </div>
                  <div className="key-region-ticks">
                    <span>0.0s</span>
                    <span>{formatDuration(duration * 0.25)}</span>
                    <span>{formatDuration(duration * 0.5)}</span>
                    <span>{formatDuration(duration * 0.75)}</span>
                    <span>{formatDuration(duration)}</span>
                  </div>
                </div>
                <div className="key-region-trim-actions">
                  <div className="key-region-range-readout">
                    <span>start frame {frames.startFrame}</span>
                    <span>end frame {frames.endFrame}</span>
                    <span>reward {formatRewardValue(renderRecord.reward)}</span>
                    <span className="key-region-score-actions" aria-label="Rescore key region">
                      <button
                        className={`score-button fail ${renderRecord.reward === 0 ? 'active' : ''}`}
                        type="button"
                        disabled={rescoreZeroPending || renderRecord.reward === 0 || actionPending === 'delete-key-regions'}
                        onClick={() => void rescoreForQ(renderRecord, 0)}
                      >
                        {rescoreZeroPending ? 'Saving 0' : 'Score 0'}
                      </button>
                      <button
                        className={`score-button success ${renderRecord.reward === 1 ? 'active' : ''}`}
                        type="button"
                        disabled={rescoreOnePending || renderRecord.reward === 1 || actionPending === 'delete-key-regions'}
                        onClick={() => void rescoreForQ(renderRecord, 1)}
                      >
                        {rescoreOnePending ? 'Saving 1' : 'Score 1'}
                      </button>
                    </span>
                  </div>
                  <div className="key-region-action-group">
                    <button
                      className={`crop-play-button ${isPlaying ? 'active' : ''}`}
                      type="button"
                      aria-pressed={isPlaying}
                      onClick={() => void toggleCropPlayback(renderRecord)}
                    >
                      <span className="crop-play-icon" aria-hidden="true">{isPlaying ? 'II' : isPlaybackLoading ? '...' : '>'}</span>
                      <span>{isPlaying ? 'Pause' : isPlaybackLoading ? 'Loading' : 'Play'}</span>
                    </button>
                    <button className="ghost-button" type="button" onClick={() => selectReviewRecord(record)}>
                      Open video
                    </button>
                    <button
                      className="apply-button"
                      type="button"
                      disabled={cropPending || Boolean(cropBlockedReason)}
                      title={cropBlockedReason || 'Save the selected replay sample range for Q training'}
                      onClick={() => void saveCropForQ(renderRecord)}
                    >
                      {cropPending ? 'Saving' : 'Save crop for Q'}
                    </button>
                  </div>
                  {cropBlockedReason ? <p className="key-region-crop-warning">{cropBlockedReason}</p> : null}
                </div>
              </div>
            </article>
          )
        })}
        {!records.length && !loading ? <p className="rollout-empty">No key regions match the current filters.</p> : null}
      </div>
    </section>
  )
}
