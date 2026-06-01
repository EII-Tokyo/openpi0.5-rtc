import { useEffect, useMemo, useState } from 'react'
import {
  deleteKeyRegions,
  fetchRLTKeyRegionReview,
  rolloutTreeUrl,
  rolloutVideoUrl,
} from '../services/api'
import type { RLTKeyRegionReviewRecord, RolloutManifestSummary, RolloutNode } from '../services/api'

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
  const [reviewTab, setReviewTab] = useState<'trainable' | 'incomplete'>('trainable')
  const [selectedReviewId, setSelectedReviewId] = useState('')
  const [selectedKeyRegionIds, setSelectedKeyRegionIds] = useState<Set<string>>(new Set())
  const [actionPending, setActionPending] = useState('')
  const [actionError, setActionError] = useState('')

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

  const loadReviewRecords = async () => {
    if (!enableKeyRegionActions) return
    const records = await fetchRLTKeyRegionReview()
    setReviewRecords(records)
    const visible = records.filter((record) => (reviewTab === 'trainable' ? record.trainable : !record.trainable))
    const selectedStillVisible = visible.some((record) => record.key_region_id === selectedReviewId)
    if (!selectedStillVisible) {
      const next = visible[0] || records.find((record) => record.trainable) || records[0]
      if (next) selectReviewRecord(next)
      else {
        setSelectedReviewId('')
        setSelected(null)
      }
    }
  }

  const loadTree = async () => {
    setError('')
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
  }, [rootPath, defaultCamera, enableKeyRegionActions, excludeRootPaths.join('|'), reviewTab])

  const videoSrc = useMemo(() => (selected?.extension === '.mp4' ? rolloutVideoUrl(selected.path) : ''), [selected])
  const selectedManifest = useMemo(
    () => (tree && selected ? findManifestForPath(tree, selected.path) : undefined),
    [tree, selected],
  )
  const visibleReviewRecords = useMemo(
    () => reviewRecords.filter((record) => (reviewTab === 'trainable' ? record.trainable : !record.trainable)),
    [reviewRecords, reviewTab],
  )
  const trainableReviewCount = useMemo(() => reviewRecords.filter((record) => record.trainable).length, [reviewRecords])
  const incompleteReviewCount = reviewRecords.length - trainableReviewCount
  const selectedCount = selectedKeyRegionIds.size

  const toggleKeyRegionSelection = (keyRegionId: string) => {
    setSelectedKeyRegionIds((current) => {
      const next = new Set(current)
      if (next.has(keyRegionId)) next.delete(keyRegionId)
      else next.add(keyRegionId)
      return next
    })
  }

  const selectOnlyKeyRegion = (keyRegionId: string) => {
    setSelectedKeyRegionIds(new Set([keyRegionId]))
  }

  const clearKeyRegionSelection = () => {
    setSelectedKeyRegionIds(new Set())
  }

  const runBatchAction = async (name: string, action: (ids: string[]) => Promise<unknown>) => {
    const ids = [...selectedKeyRegionIds]
    if (!ids.length) return
    setActionError('')
    setActionPending(name)
    try {
      await action(ids)
      setSelectedKeyRegionIds(new Set())
      await loadTree()
    } catch (exc) {
      setActionError(exc instanceof Error ? exc.message : 'Batch action failed')
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
	          {enableKeyRegionActions ? (
	            <>
	              <div className="tab-strip compact">
	                <button
	                  className={reviewTab === 'trainable' ? 'active' : ''}
	                  type="button"
	                  onClick={() => setReviewTab('trainable')}
	                >
	                  Trainable {trainableReviewCount}
	                </button>
	                <button
	                  className={reviewTab === 'incomplete' ? 'active' : ''}
	                  type="button"
	                  onClick={() => setReviewTab('incomplete')}
	                >
	                  Incomplete {incompleteReviewCount}
	                </button>
	              </div>
	              <div className="key-region-review-list primary">
	                {visibleReviewRecords.map((record) => {
	                  const checked = selectedKeyRegionIds.has(record.key_region_id)
	                  const active = selectedReviewId === record.key_region_id
	                  return (
	                    <div
	                      key={record.key_region_id}
	                      className={`key-region-review-row ${checked ? 'selected' : ''} ${active ? 'active' : ''}`}
	                    >
	                      <label>
	                        <input
	                          type="checkbox"
	                          checked={checked}
	                          onChange={() => toggleKeyRegionSelection(record.key_region_id)}
	                        />
	                        <button className="link-button" type="button" onClick={() => selectReviewRecord(record)}>
	                          {formatReviewTime(record)}
	                        </button>
	                      </label>
	                      <small>{formatReviewSummary(record)}</small>
	                    </div>
	                  )
	                })}
	                {!visibleReviewRecords.length ? (
	                  <p className="rollout-empty">
	                    {reviewTab === 'trainable' ? 'No trainable key regions.' : 'No incomplete key regions.'}
	                  </p>
	                ) : null}
	              </div>
	            </>
	          ) : tree ? (
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
        {enableKeyRegionActions ? (
          <div className="key-region-review-panel">
            <div className="key-region-review-header">
              <strong>{selectedCount} selected</strong>
              <button className="ghost-button" type="button" disabled={!selectedCount} onClick={clearKeyRegionSelection}>
                Clear
              </button>
            </div>
            <div className="key-region-review-actions">
              <button
                className="apply-button danger"
                type="button"
                disabled={!selectedCount || !!actionPending}
                onClick={() => {
                  if (window.confirm(`Delete ${selectedCount} key region(s) and their files?`)) {
                    void runBatchAction('delete', (ids) => deleteKeyRegions(ids, 'operator_delete'))
                  }
                }}
              >
                Delete selected
              </button>
            </div>
            {actionError ? <p className="inline-error">{actionError}</p> : null}
          </div>
        ) : null}
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
