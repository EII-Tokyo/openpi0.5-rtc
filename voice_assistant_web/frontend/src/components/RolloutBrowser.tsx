import { useEffect, useMemo, useRef, useState } from 'react'
import {
  cropKeyRegion,
  deleteKeyRegions,
  fetchRLTKeyRegionReview,
  rescoreKeyRegion,
  rolloutTreeUrl,
  rolloutVideoUrl,
} from '../services/api'
import type {
  RLTKeyRegionReviewRecord,
  RLTKeyRegionReviewSummary,
  RolloutManifestSummary,
  RolloutNode,
} from '../services/api'

type RolloutBrowserProps = {
  title: string
  rootPath?: string
  defaultCamera?: string
  showManifest?: boolean
  enableKeyRegionActions?: boolean
  excludeRootPaths?: string[]
}

type KeyRegionEntry = {
  keyRegionId: string
  path: string
  label: string
  manifest: RolloutManifestSummary
}

type CropRange = {
  startSec: number
  endSec: number
}

const formatBytes = (bytes?: number) => {
  if (bytes === undefined) return ''
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  let value = bytes
  let unit = 0
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024
    unit += 1
  }
  return `${value.toFixed(unit === 0 ? 0 : 1)} ${units[unit]}`
}

const formatModified = (seconds?: number) => {
  if (!seconds) return ''
  return new Date(seconds * 1000).toLocaleString()
}

const formatKeyRegionTime = (manifest?: RolloutManifestSummary) => {
  const timestamp = manifest?.score_time || manifest?.end_time || manifest?.start_time
  if (!timestamp) return ''
  return new Date(timestamp * 1000).toLocaleString(undefined, {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
  })
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

const keyRegionDisplayName = (node: Pick<RolloutNode, 'name' | 'manifest_summary'>) => {
  const time = formatKeyRegionTime(node.manifest_summary)
  if (!time) return node.name || node.manifest_summary?.key_region_id || 'key region'
  return time
}

const filterTree = (node: RolloutNode, excludeRootPaths: string[]): RolloutNode | null => {
  if (excludeRootPaths.includes(node.path)) return null
  if (node.type === 'file') return node
  return {
    ...node,
    children: (node.children || [])
      .map((child) => filterTree(child, excludeRootPaths))
      .filter((child): child is RolloutNode => child !== null),
  }
}

const flattenVideos = (node: RolloutNode): RolloutNode[] => {
  if (node.type === 'file') return node.extension === '.mp4' ? [node] : []
  return (node.children || []).flatMap(flattenVideos)
}

const flattenKeyRegions = (node: RolloutNode): KeyRegionEntry[] => {
  const entries: KeyRegionEntry[] = []
  if (node.type === 'directory' && node.manifest_summary?.key_region_id) {
    entries.push({
      keyRegionId: node.manifest_summary.key_region_id,
      path: node.path,
      label: keyRegionDisplayName(node),
      manifest: node.manifest_summary,
    })
  }
  for (const child of node.children || []) {
    entries.push(...flattenKeyRegions(child))
  }
  return entries
}

const selectDefaultVideo = (videos: RolloutNode[], defaultCamera?: string) => {
  const sorted = [...videos].sort((a, b) => (b.modified || 0) - (a.modified || 0))
  if (!defaultCamera) return sorted[0] || null
  return sorted.find((video) => video.name === defaultCamera) || sorted[0] || null
}

const formatManifestSummary = (manifest?: RolloutManifestSummary) => {
  if (!manifest) return ''
  const parts = []
  if (manifest.phase) parts.push(manifest.phase)
  if (manifest.reward !== undefined) parts.push(`reward ${manifest.reward}`)
  if (manifest.score_timeout) parts.push('timeout')
  if (manifest.duration_seconds !== undefined) parts.push(`${manifest.duration_seconds.toFixed(1)}s`)
  if (manifest.num_replay_transitions !== undefined) parts.push(`${manifest.num_replay_transitions} samples`)
  return parts.join(' / ')
}

const formatReviewSummary = (record: RLTKeyRegionReviewRecord) => {
  const parts = []
  if (record.status) parts.push(record.status)
  if (record.phase) parts.push(record.phase)
  if (record.reward !== null && record.reward !== undefined) parts.push(`reward ${record.reward}`)
  if (record.duration_seconds !== null && record.duration_seconds !== undefined) {
    parts.push(`${record.duration_seconds.toFixed(1)}s`)
  }
  if (record.num_replay_transitions !== undefined) parts.push(`${record.num_replay_transitions} samples`)
  if (!record.trainable && record.incomplete_reason) parts.push(record.incomplete_reason)
  return parts.join(' / ')
}

const findManifestForPath = (node: RolloutNode, selectedPath: string): RolloutManifestSummary | undefined => {
  if (node.type !== 'directory') return undefined
  const nodePrefix = node.path ? `${node.path}/` : ''
  const isAncestor = node.path === '' || selectedPath === node.path || selectedPath.startsWith(nodePrefix)
  if (!isAncestor) return undefined

  for (const child of node.children || []) {
    const childMatch = findManifestForPath(child, selectedPath)
    if (childMatch) return childMatch
  }
  return node.manifest_summary
}

const defaultExpanded = (node: RolloutNode, selectedPath: string) => {
  const expanded = new Set<string>([''])
  const parts = selectedPath.split('/').slice(0, -1)
  let current = ''
  for (const part of parts) {
    current = current ? `${current}/${part}` : part
    expanded.add(current)
  }
  const expandRecent = (item: RolloutNode, depth = 0) => {
    if (depth > 4 || item.type !== 'directory') return
    expanded.add(item.path)
    for (const child of item.children || []) {
      if (child.type === 'directory') expandRecent(child, depth + 1)
    }
  }
  expandRecent(node)
  return expanded
}

function RolloutTreeNode({
  node,
  selectedPath,
  expanded,
  onToggle,
  onSelect,
  showManifest,
}: {
  node: RolloutNode
  selectedPath: string
  expanded: Set<string>
  onToggle: (path: string) => void
  onSelect: (node: RolloutNode) => void
  showManifest?: boolean
}) {
  const isDirectory = node.type === 'directory'
  const isExpanded = expanded.has(node.path)
  const isSelected = node.path === selectedPath

  if (isDirectory) {
    return (
      <li>
        <button className="tree-row directory" type="button" onClick={() => onToggle(node.path)}>
          <span className="tree-twist">{isExpanded ? 'v' : '>'}</span>
          <span className="tree-name" title={node.manifest_summary?.key_region_id || node.name}>
            {node.manifest_summary?.key_region_id ? keyRegionDisplayName(node) : node.name || 'rollouts'}
          </span>
          {showManifest && node.manifest_summary ? (
            <span className="tree-size">{formatManifestSummary(node.manifest_summary)}</span>
          ) : null}
        </button>
        {isExpanded && node.children?.length ? (
          <ul className="tree-children">
            {node.children.map((child) => (
              <RolloutTreeNode
                key={child.path || 'root'}
                node={child}
                selectedPath={selectedPath}
                expanded={expanded}
                onToggle={onToggle}
                onSelect={onSelect}
                showManifest={showManifest}
              />
            ))}
          </ul>
        ) : null}
      </li>
    )
  }

  return (
    <li>
      <button
        className={`tree-row file ${isSelected ? 'selected' : ''}`}
        type="button"
        onClick={() => onSelect(node)}
      >
        <span className="tree-file-mark">{node.extension === '.mp4' ? 'play' : 'data'}</span>
        <span className="tree-name">{node.name}</span>
        <span className="tree-size">{formatBytes(node.size)}</span>
      </button>
    </li>
  )
}


const KEY_REGION_CAMERA_ORDER = ['cam_high', 'cam_low', 'cam_left_wrist', 'cam_right_wrist']
const KEY_REGION_PAGE_SIZE = 20
const DEFAULT_REPLAY_HORIZON = 50
const DEFAULT_TRAIN_HORIZON = 10
const DEFAULT_CHUNK_STRIDE = 2
const EMPTY_REVIEW_SUMMARY: RLTKeyRegionReviewSummary = {
  total: 0,
  trainable: 0,
  needs_crop: 0,
  formal_replay_ready: 0,
  needs_offline_reencode: 0,
  legacy_unmarked: 0,
  success: 0,
  failure: 0,
  replay_samples: 0,
}

type KeyRegionInfoRow = {
  label: string
  value: string
}

type DeleteKeyRegionDialogProps = {
  open: boolean
  records: RLTKeyRegionReviewRecord[]
  deleting: boolean
  error: string
  onCancel: () => void
  onConfirm: () => void
}

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

const pluralize = (count: number, singular: string, plural = `${singular}s`) => `${count} ${count === 1 ? singular : plural}`

const summarizeDeleteRecords = (records: RLTKeyRegionReviewRecord[]) => {
  const replaySamples = records.reduce((total, record) => total + (record.num_replay_transitions || 0), 0)
  const success = records.filter((record) => record.reward === 1).length
  const failure = records.filter((record) => record.reward === 0).length
  const videos = records.reduce((total, record) => total + (record.video_paths?.length || 0), 0)
  return { replaySamples, success, failure, videos }
}

function DeleteKeyRegionDialog({
  open,
  records,
  deleting,
  error,
  onCancel,
  onConfirm,
}: DeleteKeyRegionDialogProps) {
  const cancelButtonRef = useRef<HTMLButtonElement | null>(null)
  const count = records.length
  const isBatch = count > 1
  const summary = useMemo(() => summarizeDeleteRecords(records), [records])
  const primaryRecord = records[0]

  useEffect(() => {
    if (!open) return undefined
    const previousActiveElement = document.activeElement as HTMLElement | null
    const timer = window.setTimeout(() => cancelButtonRef.current?.focus(), 0)
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape' && !deleting) onCancel()
    }
    window.addEventListener('keydown', onKeyDown)
    return () => {
      window.clearTimeout(timer)
      window.removeEventListener('keydown', onKeyDown)
      previousActiveElement?.focus?.()
    }
  }, [deleting, onCancel, open])

  if (!open || !primaryRecord) return null

  const heading = isBatch ? `Delete ${pluralize(count, 'key region')}?` : 'Delete key region?'
  const confirmLabel = deleting
    ? 'Deleting...'
    : isBatch
      ? `Delete ${count} selected`
      : 'Delete region'
  const recordTitle = isBatch
    ? `${pluralize(count, 'region')} selected`
    : `${formatReviewTime(primaryRecord)} · ${primaryRecord.key_region_id.slice(0, 8)}`
  const statusLine = isBatch
    ? `${summary.replaySamples} replay samples · ${summary.success} success / ${summary.failure} failure`
    : [
        primaryRecord.reward === null ? 'reward -' : `reward ${primaryRecord.reward}`,
        `${primaryRecord.num_replay_transitions || 0} samples`,
        primaryRecord.phase || 'phase -',
      ].join(' · ')

  return (
    <div
      className="modal-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !deleting) onCancel()
      }}
    >
      <section
        className="delete-region-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="delete-region-title"
        aria-describedby="delete-region-description"
      >
        <div className="delete-dialog-icon" aria-hidden="true">
          !
        </div>
        <div className="delete-dialog-copy">
          <h2 id="delete-region-title">{heading}</h2>
          <p id="delete-region-description">
            This removes the selected key-region files from local review data and excludes them from replay training.
          </p>
        </div>

        <div className="delete-dialog-summary">
          <span>{isBatch ? 'Selected' : 'Region'}</span>
          <strong>{recordTitle}</strong>
          <small>{statusLine}</small>
        </div>

        <div className="delete-dialog-impact" aria-label="Affected data">
          <div>
            <span>Videos</span>
            <strong>{summary.videos}</strong>
          </div>
          <div>
            <span>Replay samples</span>
            <strong>{summary.replaySamples}</strong>
          </div>
          <div>
            <span>Success / failure</span>
            <strong>{summary.success} / {summary.failure}</strong>
          </div>
        </div>

        <ul className="delete-dialog-list">
          <li>rollout videos and manifest metadata</li>
          <li>episode HDF5 files when present</li>
          <li>replay shard and segment ledger references</li>
        </ul>

        {error ? <div className="delete-dialog-error">{error}</div> : null}

        <div className="delete-dialog-actions">
          <button ref={cancelButtonRef} className="ghost-button" type="button" onClick={onCancel} disabled={deleting}>
            Cancel
          </button>
          <button className="danger-button" type="button" onClick={onConfirm} disabled={deleting}>
            {confirmLabel}
          </button>
        </div>
      </section>
    </div>
  )
}

const formatRewardValue = (reward?: number | null) => {
  if (reward === 1) return 'success'
  if (reward === 0) return 'failure'
  return '-'
}

const keyRegionStatusLabel = (record: RLTKeyRegionReviewRecord) => {
  if (record.trainable) return record.status || 'committed'
  if (record.incomplete_reason) return 'needs crop'
  return record.status || 'pending'
}

const keyRegionEligibility = (record: RLTKeyRegionReviewRecord) => {
  if (record.trainable) return 'trainable'
  if (record.incomplete_reason) return 'pending'
  return record.train_eligible === false ? 'not eligible' : 'pending'
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

const keyRegionInfoRows = (record: RLTKeyRegionReviewRecord): KeyRegionInfoRow[] => [
  { label: 'Status', value: keyRegionStatusLabel(record) },
  { label: 'Reward', value: formatRewardValue(record.reward) },
  { label: 'Phase', value: record.phase || '-' },
  { label: 'Video duration', value: formatDuration(record.duration_seconds) },
  { label: 'Key duration', value: formatDuration(record.key_region_duration_seconds) },
  { label: 'Transitions', value: `${record.num_replay_transitions || 0} samples` },
  { label: 'Full horizon', value: `${DEFAULT_REPLAY_HORIZON} actions` },
  { label: 'Train horizon', value: `${DEFAULT_TRAIN_HORIZON} actions` },
  { label: 'Chunk stride', value: `${DEFAULT_CHUNK_STRIDE} frames` },
  { label: 'Replay status', value: record.replay_status || (record.npz_exists ? 'written' : 'pending crop') },
  { label: 'Missing metadata', value: record.missing_rlt_metadata?.length ? record.missing_rlt_metadata.join(', ') : '-' },
  { label: 'Eligibility', value: keyRegionEligibility(record) },
]

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

const reviewTimestamp = (record: RLTKeyRegionReviewRecord) =>
  record.score_time || record.end_time || record.start_time || record.updated_at || null

const reviewTitle = (record: RLTKeyRegionReviewRecord, index: number, total: number) => {
  const time = formatShortDateTime(reviewTimestamp(record)) || record.key_region_id
  const task = record.task || 'key region'
  return `Card ${index + 1} / ${total} - ${time} - ${task} - ${record.key_region_id}`
}

const badgeTone = (tone: 'green' | 'blue' | 'amber' | 'red' | 'slate') => `key-region-badge ${tone}`

const artifactBadgeTone = (exists: boolean) => (exists ? 'green' : 'amber')

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

const clamp = (value: number, min: number, max: number) => Math.min(Math.max(value, min), max)

export function RolloutBrowser({
  title,
  rootPath,
  defaultCamera,
  showManifest = false,
  enableKeyRegionActions = false,
  excludeRootPaths = [],
}: RolloutBrowserProps) {
  const [tree, setTree] = useState<RolloutNode | null>(null)
  const [selected, setSelected] = useState<RolloutNode | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['']))
  const [error, setError] = useState('')
  const [reviewRecords, setReviewRecords] = useState<RLTKeyRegionReviewRecord[]>([])
  const [reviewStatusFilter, setReviewStatusFilter] = useState<'all' | 'trainable' | 'needsCrop'>('all')
  const [reviewRewardFilter, setReviewRewardFilter] = useState<'all' | 'success' | 'failure'>('all')
  const [reviewBatchFilter, setReviewBatchFilter] = useState('latest')
  const [reviewBatches, setReviewBatches] = useState<string[]>([])
  const [reviewNextOffset, setReviewNextOffset] = useState<number | null>(null)
  const [reviewTotal, setReviewTotal] = useState(0)
  const [reviewSummary, setReviewSummary] = useState<RLTKeyRegionReviewSummary>(EMPTY_REVIEW_SUMMARY)
  const [activeReviewIndex, setActiveReviewIndex] = useState(1)
  const [selectedReviewId, setSelectedReviewId] = useState('')
  const [selectedKeyRegionIds, setSelectedKeyRegionIds] = useState<Set<string>>(new Set())
  const [actionPending, setActionPending] = useState('')
  const [actionError, setActionError] = useState('')
  const [deleteDialogRecords, setDeleteDialogRecords] = useState<RLTKeyRegionReviewRecord[]>([])
  const [deleteDialogError, setDeleteDialogError] = useState('')
  const [cropRanges, setCropRanges] = useState<Record<string, CropRange>>({})
  const [playingKeyRegionId, setPlayingKeyRegionId] = useState('')
  const [pendingPlaybackKeyRegionId, setPendingPlaybackKeyRegionId] = useState('')
  const [playbackTimes, setPlaybackTimes] = useState<Record<string, number>>({})
  const [playbackError, setPlaybackError] = useState('')
  const keyRegionVideoRefs = useRef<Record<string, Array<HTMLVideoElement | null>>>({})
  const keyRegionCardRefs = useRef<Record<string, HTMLElement | null>>({})
  const keyRegionsWorkspaceRef = useRef<HTMLElement | null>(null)

  const getCropRange = (record: RLTKeyRegionReviewRecord) => cropRangeForRecord(record, cropRanges)

  const setKeyRegionVideoRef = (keyRegionId: string, index: number, element: HTMLVideoElement | null) => {
    if (!element) {
      const refs = keyRegionVideoRefs.current[keyRegionId]
      if (!refs) return
      refs[index] = null
      if (!refs.some(Boolean)) delete keyRegionVideoRefs.current[keyRegionId]
      return
    }
    const refs = keyRegionVideoRefs.current[keyRegionId] || []
    refs[index] = element
    keyRegionVideoRefs.current[keyRegionId] = refs
  }

  const keyRegionVideos = (keyRegionId: string) =>
    (keyRegionVideoRefs.current[keyRegionId] || []).filter((video): video is HTMLVideoElement => Boolean(video))

  const pauseKeyRegionVideos = (keyRegionId: string) => {
    keyRegionVideos(keyRegionId).forEach((video) => video.pause())
  }

  const pauseAllKeyRegionVideos = () => {
    Object.keys(keyRegionVideoRefs.current).forEach(pauseKeyRegionVideos)
  }

  const syncKeyRegionVideos = (keyRegionId: string, timeSec: number) => {
    keyRegionVideos(keyRegionId).forEach((video) => {
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
    setPlaybackTimes((current) => ({ ...current, [keyRegionId]: timeSec }))
  }

  const playCropPreview = async (record: RLTKeyRegionReviewRecord) => {
    const keyRegionId = record.key_region_id
    const videos = keyRegionVideos(keyRegionId)
    if (!videos.length) {
      setPlaybackError('Preview is still loading. Try again in a moment.')
      return
    }
    const range = getCropRange(record)
    pauseAllKeyRegionVideos()
    setPlaybackError('')
    syncKeyRegionVideos(keyRegionId, range.startSec)
    const results = await Promise.allSettled(videos.map((video) => video.play()))
    if (results.every((result) => result.status === 'rejected')) {
      setPlaybackError('Preview playback failed. Try again after video metadata loads.')
      setPlayingKeyRegionId('')
      return
    }
    setPlayingKeyRegionId(keyRegionId)
  }

  const toggleCropPlayback = async (record: RLTKeyRegionReviewRecord) => {
    const keyRegionId = record.key_region_id
    if (playingKeyRegionId === keyRegionId) {
      pauseKeyRegionVideos(keyRegionId)
      setPlayingKeyRegionId('')
      return
    }
    if (selectedReviewId !== keyRegionId) {
      setPendingPlaybackKeyRegionId(keyRegionId)
      selectReviewRecord(record)
      return
    }
    await playCropPreview(record)
  }

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
    syncKeyRegionVideos(record.key_region_id, edge === 'start' ? nextRange.startSec : nextRange.endSec)
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
    pauseKeyRegionVideos(record.key_region_id)
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
      await loadTree()
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
      await loadReviewRecords()
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : 'Rescore failed')
    } finally {
      setActionPending('')
    }
  }

  const selectReviewRecord = (record: RLTKeyRegionReviewRecord) => {
    setSelectedReviewId(record.key_region_id)
    if (!record.default_video_path) {
      setSelected(null)
      return
    }
    const name = record.default_video_path.split('/').pop() || 'cam.mp4'
    setSelected({
      name,
      path: record.default_video_path,
      type: 'file',
      extension: '.mp4',
      modified: record.score_time || record.end_time || record.updated_at || undefined,
    })
  }

  const loadReviewRecords = async ({
    append = false,
    offset = 0,
    status = reviewStatusFilter,
    reward = reviewRewardFilter,
    batch = reviewBatchFilter,
  }: {
    append?: boolean
    offset?: number
    status?: 'all' | 'trainable' | 'needsCrop'
    reward?: 'all' | 'success' | 'failure'
    batch?: string
  } = {}) => {
    if (!enableKeyRegionActions) return
    const page = await fetchRLTKeyRegionReview({
      limit: KEY_REGION_PAGE_SIZE,
      offset,
      status,
      reward,
      batch,
    })
    const records = page.items.filter(
      (record) => !record.voided && record.status.toLowerCase() !== 'voided',
    )
    const nextRecords = append ? [...reviewRecords, ...records] : records
    setReviewRecords(nextRecords)
    setReviewBatches(page.batches)
    setReviewNextOffset(page.next_offset)
    setReviewTotal(page.total)
    setReviewSummary(page.summary || EMPTY_REVIEW_SUMMARY)
    const selectedStillVisible = nextRecords.some((record) => record.key_region_id === selectedReviewId)
    if (!selectedStillVisible) {
      const next = nextRecords[0]
      if (next) selectReviewRecord(next)
      else {
        setSelectedReviewId('')
        setSelected(null)
      }
    }
  }

  const loadTree = async () => {
    setError('')
    if (enableKeyRegionActions) {
      setTree(null)
      setExpanded(new Set(['']))
      await loadReviewRecords()
      return
    }
    const response = await fetch(rolloutTreeUrl(rootPath))
    if (!response.ok) throw new Error(`HTTP ${response.status}`)
    const rawPayload = (await response.json()) as RolloutNode
    const payload = filterTree(rawPayload, excludeRootPaths) || rawPayload
	    const videos = flattenVideos(payload)
	    const defaultVideo = selectDefaultVideo(videos, defaultCamera)
	    setTree(payload)
	    if (!enableKeyRegionActions) setSelected(defaultVideo)
	    setExpanded(defaultVideo ? defaultExpanded(payload, defaultVideo.path) : new Set([payload.path || '']))
	    await loadReviewRecords()
	  }

  useEffect(() => {
    let ignore = false
    const loadCurrentTree = async () => {
      setError('')
      try {
        if (enableKeyRegionActions) {
          setTree(null)
          setExpanded(new Set(['']))
          await loadReviewRecords()
          return
        }
        const response = await fetch(rolloutTreeUrl(rootPath))
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const rawPayload = (await response.json()) as RolloutNode
        if (ignore) return
        const payload = filterTree(rawPayload, excludeRootPaths) || rawPayload
        const videos = flattenVideos(payload)
        const newest = selectDefaultVideo(videos, defaultCamera)
        setTree(payload)
	        if (!enableKeyRegionActions) setSelected(newest)
	        setExpanded(newest ? defaultExpanded(payload, newest.path) : new Set([payload.path || '']))
	        await loadReviewRecords()
      } catch {
        if (!ignore) setError('Rollouts could not be loaded.')
      }
    }
    void loadCurrentTree()
    return () => {
      ignore = true
    }
  }, [rootPath, defaultCamera, enableKeyRegionActions, excludeRootPaths.join('|')])

  const videoSrc = useMemo(() => (selected?.extension === '.mp4' ? rolloutVideoUrl(selected.path) : ''), [selected])
  const selectedManifest = useMemo(
    () => (tree && selected ? findManifestForPath(tree, selected.path) : undefined),
    [tree, selected],
  )
  const visibleReviewRecords = reviewRecords
  const renderedReviewRecords = reviewRecords
  const hasMoreReviewRecords = reviewNextOffset !== null
  const trainableReviewCount = reviewSummary.trainable
  const incompleteReviewCount = reviewSummary.needs_crop
  const successReviewCount = reviewSummary.success
  const failureReviewCount = reviewSummary.failure
  const replaySampleCount = reviewSummary.replay_samples
  const selectedCount = selectedKeyRegionIds.size

  useEffect(() => {
    if (!enableKeyRegionActions) return undefined
    return () => {
      pauseAllKeyRegionVideos()
      keyRegionVideoRefs.current = {}
      keyRegionCardRefs.current = {}
    }
  }, [enableKeyRegionActions])

  useEffect(() => {
    if (!enableKeyRegionActions) return
    const visibleIds = new Set(renderedReviewRecords.map((record) => record.key_region_id))
    Object.keys(keyRegionVideoRefs.current).forEach((keyRegionId) => {
      if (!visibleIds.has(keyRegionId) || keyRegionId !== selectedReviewId) delete keyRegionVideoRefs.current[keyRegionId]
    })
  }, [enableKeyRegionActions, renderedReviewRecords, selectedReviewId])

  useEffect(() => {
    pauseAllKeyRegionVideos()
    setPlayingKeyRegionId('')
    setPendingPlaybackKeyRegionId('')
    keyRegionVideoRefs.current = {}
    if (enableKeyRegionActions) void loadReviewRecords()
  }, [reviewRewardFilter, reviewStatusFilter, reviewBatchFilter])

  useEffect(() => {
    if (!enableKeyRegionActions) return
    if (!renderedReviewRecords.length) {
      setSelectedReviewId('')
      setSelected(null)
      return
    }
    if (!renderedReviewRecords.some((record) => record.key_region_id === selectedReviewId)) {
      selectReviewRecord(renderedReviewRecords[0])
    }
  }, [enableKeyRegionActions, renderedReviewRecords, selectedReviewId])

  useEffect(() => {
    if (!enableKeyRegionActions || !playingKeyRegionId) return undefined
    let frame = 0
    const tick = () => {
      const record = reviewRecords.find((item) => item.key_region_id === playingKeyRegionId)
      const primaryVideo = keyRegionVideos(playingKeyRegionId)[0]
      if (!record || !primaryVideo) {
        setPlayingKeyRegionId('')
        return
      }
      const range = cropRangeForRecord(record, cropRanges)
      const currentTime = primaryVideo.currentTime
      setPlaybackTimes((times) => ({ ...times, [playingKeyRegionId]: currentTime }))
      if (currentTime >= range.endSec) {
        pauseKeyRegionVideos(playingKeyRegionId)
        syncKeyRegionVideos(playingKeyRegionId, range.endSec)
        setPlayingKeyRegionId('')
        return
      }
      frame = window.requestAnimationFrame(tick)
    }
    frame = window.requestAnimationFrame(tick)
    return () => window.cancelAnimationFrame(frame)
  }, [enableKeyRegionActions, playingKeyRegionId, cropRanges, reviewRecords])

  useEffect(() => {
    if (!enableKeyRegionActions || !pendingPlaybackKeyRegionId) return undefined
    if (selectedReviewId !== pendingPlaybackKeyRegionId) return undefined
    const record = renderedReviewRecords.find((item) => item.key_region_id === pendingPlaybackKeyRegionId)
    if (!record) {
      setPendingPlaybackKeyRegionId('')
      return undefined
    }
    let cancelled = false
    let attempts = 0
    let timer = 0
    const attemptPlayback = () => {
      if (cancelled) return
      if (keyRegionVideos(pendingPlaybackKeyRegionId).length) {
        setPendingPlaybackKeyRegionId('')
        void playCropPreview(record)
        return
      }
      attempts += 1
      if (attempts >= 20) {
        setPendingPlaybackKeyRegionId('')
        setPlaybackError('Preview is still loading. Try again in a moment.')
        return
      }
      timer = window.setTimeout(attemptPlayback, 50)
    }
    timer = window.setTimeout(attemptPlayback, 0)
    return () => {
      cancelled = true
      window.clearTimeout(timer)
    }
  }, [enableKeyRegionActions, pendingPlaybackKeyRegionId, renderedReviewRecords, selectedReviewId])

  const toggleKeyRegionSelection = (keyRegionId: string) => {
    setSelectedKeyRegionIds((current) => {
      const next = new Set(current)
      if (next.has(keyRegionId)) next.delete(keyRegionId)
      else next.add(keyRegionId)
      return next
    })
  }

  const clearKeyRegionSelection = () => {
    setSelectedKeyRegionIds(new Set())
  }

  const selectVisibleKeyRegions = () => {
    setSelectedKeyRegionIds(new Set(renderedReviewRecords.map((record) => record.key_region_id)))
  }

  const openDeleteDialog = (records: RLTKeyRegionReviewRecord[]) => {
    setActionError('')
    setDeleteDialogError('')
    setDeleteDialogRecords(records)
  }

  const closeDeleteDialog = () => {
    if (actionPending === 'delete-key-regions') return
    setDeleteDialogRecords([])
    setDeleteDialogError('')
  }

  const openDeleteSelectedDialog = () => {
    if (!selectedKeyRegionIds.size) return
    const selectedRecords = visibleReviewRecords.filter((record) => selectedKeyRegionIds.has(record.key_region_id))
    if (selectedRecords.length) openDeleteDialog(selectedRecords)
  }

  const confirmDeleteKeyRegions = async () => {
    const keyRegionIds = deleteDialogRecords.map((record) => record.key_region_id)
    if (!keyRegionIds.length) return
    setActionError('')
    setDeleteDialogError('')
    setActionPending('delete-key-regions')
    try {
      await deleteKeyRegions(keyRegionIds, keyRegionIds.length > 1 ? 'operator_batch_delete' : 'operator_delete')
      setSelectedKeyRegionIds((current) => {
        const next = new Set(current)
        keyRegionIds.forEach((keyRegionId) => next.delete(keyRegionId))
        return next
      })
      setDeleteDialogRecords([])
      await loadTree()
    } catch (exc) {
      const message = exc instanceof Error ? exc.message : 'Delete failed'
      setDeleteDialogError(message)
      setActionError(message)
    } finally {
      setActionPending('')
    }
  }


  const toggle = (path: string) => {
    setExpanded((current) => {
      const next = new Set(current)
      if (next.has(path)) next.delete(path)
      else next.add(path)
      return next
    })
  }

  useEffect(() => {
    if (!enableKeyRegionActions) return undefined
    const updateActiveCard = () => {
      let nextIndex = 1
      let bestDistance = Number.POSITIVE_INFINITY
      renderedReviewRecords.forEach((record, index) => {
        const element = keyRegionCardRefs.current[record.key_region_id]
        if (!element) return
        const rect = element.getBoundingClientRect()
        if (rect.bottom < 80) return
        const distance = Math.abs(rect.top - 96)
        if (distance < bestDistance) {
          bestDistance = distance
          nextIndex = index + 1
        }
      })
      setActiveReviewIndex(nextIndex)
    }
    const scrollRoot = keyRegionsWorkspaceRef.current || window
    updateActiveCard()
    scrollRoot.addEventListener('scroll', updateActiveCard, { passive: true })
    window.addEventListener('keydown', updateActiveCard)
    window.addEventListener('resize', updateActiveCard)
    return () => {
      scrollRoot.removeEventListener('scroll', updateActiveCard)
      window.removeEventListener('keydown', updateActiveCard)
      window.removeEventListener('resize', updateActiveCard)
    }
  }, [enableKeyRegionActions, renderedReviewRecords])

  if (enableKeyRegionActions) {
    const totalVisible = reviewTotal
    return (
      <section className="key-regions-workspace" ref={keyRegionsWorkspaceRef}>
        <div className="key-region-scroll-indicator">Card {totalVisible ? activeReviewIndex : 0} / {totalVisible}</div>
        <div className="key-regions-toolbar">
          <div>
            <p className="eyebrow">Rollouts</p>
            <h2>{title}</h2>
          </div>
          <div className="key-regions-controls">
            <select
              className="key-region-control"
              value={reviewBatchFilter}
              onChange={(event) => setReviewBatchFilter(event.target.value)}
            >
              <option value="latest">Latest batch</option>
              <option value="all">All batches</option>
              {reviewBatches.map((batch) => (
                <option value={batch} key={batch}>
                  {batch}
                </option>
              ))}
            </select>
            <select
              className="key-region-control"
              value={reviewStatusFilter}
              onChange={(event) => setReviewStatusFilter(event.target.value as 'all' | 'trainable' | 'needsCrop')}
            >
              <option value="all">All statuses</option>
              <option value="trainable">Trainable</option>
              <option value="needsCrop">Needs crop</option>
            </select>
            <select
              className="key-region-control"
              value={reviewRewardFilter}
              onChange={(event) => setReviewRewardFilter(event.target.value as 'all' | 'success' | 'failure')}
            >
              <option value="all">Reward any</option>
              <option value="success">Success only</option>
              <option value="failure">Failure only</option>
            </select>
            <button className="ghost-button" type="button" onClick={selectVisibleKeyRegions} disabled={!totalVisible}>
              Select visible
            </button>
            <button className="ghost-button" type="button" onClick={clearKeyRegionSelection} disabled={!selectedCount}>
              Clear
            </button>
            <button
              className="ghost-button danger"
              type="button"
              onClick={openDeleteSelectedDialog}
              disabled={!selectedCount || actionPending === 'delete-key-regions'}
            >
              Delete selected
            </button>
            <button
              className="ghost-button"
              type="button"
              onClick={() => {
                setTree(null)
                void loadReviewRecords().catch(() => setError('Key regions could not be loaded.'))
              }}
            >
              Refresh
            </button>
          </div>
        </div>

        <div className="key-region-summary-strip">
          <div className="key-region-summary-tile"><span>Trainable key regions</span><strong>{trainableReviewCount}</strong></div>
          <div className="key-region-summary-tile"><span>Confirmed replay samples</span><strong>{replaySampleCount}</strong></div>
          <div className="key-region-summary-tile"><span>Success / failure</span><strong>{successReviewCount} / {failureReviewCount}</strong></div>
          <div className="key-region-summary-tile"><span>Needs crop review</span><strong>{incompleteReviewCount}</strong></div>
          <div className="key-region-summary-tile"><span>Train horizon</span><strong>{DEFAULT_TRAIN_HORIZON} / {DEFAULT_REPLAY_HORIZON}</strong></div>
          <div className="key-region-summary-tile"><span>Selected</span><strong>{selectedCount}</strong></div>
        </div>

        {error ? <p className="inline-error">{error}</p> : null}
        {actionError ? <p className="inline-error">{actionError}</p> : null}
        {playbackError ? <p className="inline-error">{playbackError}</p> : null}

        <div className="key-region-card-stack">
          {renderedReviewRecords.map((record, index) => {
            const checked = selectedKeyRegionIds.has(record.key_region_id)
            const cameras = orderedCameraPaths(record)
            const cropRange = getCropRange(record)
            const duration = durationForRecord(record)
            const frames = frameSummary(record, cropRange)
            const clipLeft = duration > 0 ? clamp((cropRange.startSec / duration) * 100, 0, 100) : 0
            const clipWidth =
              duration > 0 ? clamp(((cropRange.endSec - cropRange.startSec) / duration) * 100, 0, 100 - clipLeft) : 0
            const playbackTime = playbackTimes[record.key_region_id] ?? cropRange.startSec
            const playbackProgress =
              cropRange.endSec > cropRange.startSec
                ? clamp((playbackTime - cropRange.startSec) / (cropRange.endSec - cropRange.startSec), 0, 1)
                : 0
            const cropPending = actionPending === `crop-${record.key_region_id}`
            const cropBlockedReason = cropSaveBlockedReason(record, cropRange)
            const rescoreZeroPending = actionPending === `rescore-${record.key_region_id}-0`
            const rescoreOnePending = actionPending === `rescore-${record.key_region_id}-1`
            const isPlaying = playingKeyRegionId === record.key_region_id
            const isActiveCard = selectedReviewId === record.key_region_id
            const rewardTone = record.reward === 0 ? 'red' : record.reward === 1 ? 'green' : 'slate'
            return (
              <article
                key={record.key_region_id}
                className={`key-region-card ${isActiveCard ? 'active' : ''}`}
                ref={(element) => {
                  keyRegionCardRefs.current[record.key_region_id] = element
                }}
              >
                <div className="key-region-card-head">
                  <input
                    className="key-region-checkbox"
                    type="checkbox"
                    checked={checked}
                    onChange={() => toggleKeyRegionSelection(record.key_region_id)}
                    aria-label={`Select ${record.key_region_id}`}
                  />
                  <button className="key-region-title-button" type="button" onClick={() => selectReviewRecord(record)}>
                    <strong>{reviewTitle(record, index, totalVisible)}</strong>
                    <span>{formatReviewTime(record)}</span>
                  </button>
                  <div className="key-region-badges">
                    <span className={badgeTone(record.trainable ? 'green' : 'amber')}>
                      {record.trainable ? 'trainable' : 'needs crop'}
                    </span>
                    <span className={badgeTone(rewardTone)}>{record.reward === null ? 'reward -' : `reward ${record.reward}`}</span>
                    <span className={badgeTone('blue')}>{record.phase || 'phase -'}</span>
                    <span className={badgeTone(record.trainable ? 'slate' : 'amber')}>{keyRegionStatusLabel(record)}</span>
                  </div>
                </div>

                <div className="key-region-card-main">
                  <div className="key-region-video-grid">
                    {KEY_REGION_CAMERA_ORDER.map((camera, cameraIndex) => {
                      const path = cameras[cameraIndex]
                      return (
                        <div className="key-region-video-tile" key={`${record.key_region_id}-${camera}`}>
                          <span className="key-region-camera-label">{path ? cameraLabelFromPath(path) : camera}</span>
                          {path && isActiveCard ? (
                            <video
                              ref={(element) => setKeyRegionVideoRef(record.key_region_id, cameraIndex, element)}
                              src={rolloutVideoUrl(path)}
                              preload="none"
                              muted
                              playsInline
                              tabIndex={-1}
                            />
                          ) : path ? (
                            <button
                              className="key-region-video-placeholder"
                              type="button"
                              onClick={() => selectReviewRecord(record)}
                            >
                              <span>Load preview</span>
                            </button>
                          ) : (
                            <div className="key-region-video-missing">No video</div>
                          )}
                        </div>
                      )
                    })}
                  </div>

                  <aside className="key-region-info-panel">
                    <div className="key-region-panel-title">
                      <h3>Replay Buffer</h3>
                      <span className={badgeTone(record.trainable ? 'green' : 'amber')}>
                        {record.trainable ? 'eligible for Q' : 'not eligible yet'}
                      </span>
                    </div>
                    <div className="key-region-kv-grid">
                      {keyRegionInfoRows(record).map((row) => (
                        <div className="key-region-kv" key={row.label}>
                          <span>{row.label}</span>
                          <strong>{row.value}</strong>
                        </div>
                      ))}
                    </div>
                    <div className="key-region-artifacts">
                      <span className={badgeTone(artifactBadgeTone(record.video_exists))}>video</span>
                      <span className={badgeTone(artifactBadgeTone(record.manifest_exists))}>manifest</span>
                      <span className={badgeTone(artifactBadgeTone(record.npz_exists))}>npz</span>
                      <span className={badgeTone('slate')}>{record.video_paths.length || 0} cameras</span>
                    </div>
                  </aside>
                </div>

                <div className="key-region-trim-panel">
                  <div className="key-region-trim-head">
                    <strong>Crop for Q training</strong>
                    <span>{cropSummary(record, cropRange)}</span>
                  </div>
                  <div className="key-region-timeline">
                    <div className="key-region-timeline-track">
                      <div
                        className={`key-region-clip ${record.reward === 0 ? 'fail' : 'success'}`}
                        style={{ left: `${clipLeft}%`, width: `${clipWidth}%` }}
                      >
                        <span className="key-region-clip-progress" style={{ width: `${playbackProgress * 100}%` }} />
                        <button
                          className="key-region-handle start"
                          type="button"
                          aria-label="Set crop start"
                          onPointerDown={(event) => beginCropDrag(event, record, 'start')}
                        />
                        <button
                          className="key-region-handle end"
                          type="button"
                          aria-label="Set crop end"
                          onPointerDown={(event) => beginCropDrag(event, record, 'end')}
                        />
                        <span className="key-region-playhead" style={{ left: `${playbackProgress * 100}%` }} />
                        <span className={`key-region-marker ${record.reward === 0 ? 'fail' : 'success'}`} style={{ left: '92%' }} />
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
                      <span>reward {formatRewardValue(record.reward)}</span>
                      <span className="key-region-score-actions" aria-label="Rescore key region">
                        <button
                          className={`score-button fail ${record.reward === 0 ? 'active' : ''}`}
                          type="button"
                          disabled={rescoreZeroPending || record.reward === 0 || actionPending === 'delete-key-regions'}
                          onClick={() => void rescoreForQ(record, 0)}
                        >
                          {rescoreZeroPending ? 'Saving 0' : 'Score 0'}
                        </button>
                        <button
                          className={`score-button success ${record.reward === 1 ? 'active' : ''}`}
                          type="button"
                          disabled={rescoreOnePending || record.reward === 1 || actionPending === 'delete-key-regions'}
                          onClick={() => void rescoreForQ(record, 1)}
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
                        onClick={() => void toggleCropPlayback(record)}
                      >
                        <span className="crop-play-icon" aria-hidden="true">{isPlaying ? 'II' : '>'}</span>
                        <span>{isPlaying ? 'Pause' : 'Play'}</span>
                      </button>
                      <button className="ghost-button" type="button" onClick={() => selectReviewRecord(record)}>
                        Open video
                      </button>
                      <button
                        className="apply-button"
                        type="button"
                        disabled={cropPending || Boolean(cropBlockedReason)}
                        title={cropBlockedReason || 'Save the selected replay sample range for Q training'}
                        onClick={() => void saveCropForQ(record)}
                      >
                        {cropPending ? 'Saving crop' : 'Save crop for Q'}
                      </button>
                      <button
                        className="apply-button delete"
                        type="button"
                        disabled={actionPending === 'delete-key-regions'}
                        onClick={() => openDeleteDialog([record])}
                      >
                        Delete region
                      </button>
                    </div>
                    {cropBlockedReason ? <p className="key-region-crop-warning">{cropBlockedReason}</p> : null}
                  </div>
                </div>
              </article>
            )
          })}
          {hasMoreReviewRecords ? (
            <div className="key-region-load-more">
              <button
                className="ghost-button"
                type="button"
                onClick={() => {
                  if (reviewNextOffset !== null) void loadReviewRecords({ append: true, offset: reviewNextOffset })
                }}
              >
                Load {Math.min(KEY_REGION_PAGE_SIZE, Math.max(0, reviewTotal - renderedReviewRecords.length))} more
                ({renderedReviewRecords.length} / {reviewTotal})
              </button>
            </div>
          ) : null}
          {!visibleReviewRecords.length ? <p className="rollout-empty">No key regions match the current filters.</p> : null}
        </div>
        <DeleteKeyRegionDialog
          open={deleteDialogRecords.length > 0}
          records={deleteDialogRecords}
          deleting={actionPending === 'delete-key-regions'}
          error={deleteDialogError}
          onCancel={closeDeleteDialog}
          onConfirm={() => void confirmDeleteKeyRegions()}
        />
      </section>
    )
  }

  return (
    <section className="rollouts-page">
      <aside className="rollouts-tree-panel">
        <div className="rollouts-panel-header">
          <div>
            <p className="eyebrow">Rollouts</p>
            <h2>{title}</h2>
          </div>
          <button
            className="ghost-button"
            type="button"
            onClick={() => {
              setTree(null)
              setSelected(null)
              setExpanded(new Set([rootPath || '']))
              void loadTree()
                .catch(() => setError('Rollouts could not be loaded.'))
            }}
          >
            Refresh
          </button>
        </div>
        <div className="rollouts-tree-scroll">
          {error ? <p className="inline-error">{error}</p> : null}
          {tree ? (
            <ul className="rollouts-tree">
              <RolloutTreeNode
                node={tree}
                selectedPath={selected?.path || ''}
                expanded={expanded}
                onToggle={toggle}
                onSelect={setSelected}
                showManifest={showManifest}
              />
            </ul>
          ) : (
            <p className="rollout-empty">Loading rollouts...</p>
          )}
        </div>
      </aside>

      <section className="rollouts-player-panel">
        <div className="rollouts-player-header">
          <div>
            <p className="eyebrow">Playback</p>
            <h2>{selected?.name || 'Select an MP4 file'}</h2>
          </div>
          {selected ? <span className="status-pill mode">{formatBytes(selected.size)}</span> : null}
        </div>
        <div className="rollout-video-frame">
          {videoSrc ? (
            <video key={selected?.path} controls preload="metadata" src={videoSrc} />
          ) : (
            <div className="rollout-empty-state">Select an MP4 file from the tree.</div>
          )}
        </div>
        <div className="rollout-file-meta">
          <span>{selected?.path || 'No file selected'}</span>
          <span>{formatModified(selected?.modified)}</span>
        </div>
        {showManifest && selectedManifest ? (
          <div className="rollout-file-meta">
            <span>{formatManifestSummary(selectedManifest)}</span>
            <span>{selectedManifest.task || selectedManifest.key_region_id || ''}</span>
          </div>
        ) : null}
      </section>
    </section>
  )
}
