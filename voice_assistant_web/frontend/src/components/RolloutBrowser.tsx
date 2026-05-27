import { useEffect, useMemo, useState } from 'react'
import { apiBase, rolloutVideoUrl } from '../services/api'
import type { RolloutNode } from '../services/api'

type RolloutBrowserProps = {
  title: string
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

const flattenVideos = (node: RolloutNode): RolloutNode[] => {
  if (node.type === 'file') return node.extension === '.mp4' ? [node] : []
  return (node.children || []).flatMap(flattenVideos)
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
}: {
  node: RolloutNode
  selectedPath: string
  expanded: Set<string>
  onToggle: (path: string) => void
  onSelect: (node: RolloutNode) => void
}) {
  const isDirectory = node.type === 'directory'
  const isExpanded = expanded.has(node.path)
  const isSelected = node.path === selectedPath

  if (isDirectory) {
    return (
      <li>
        <button className="tree-row directory" type="button" onClick={() => onToggle(node.path)}>
          <span className="tree-twist">{isExpanded ? 'v' : '>'}</span>
          <span className="tree-name">{node.name || 'rollouts'}</span>
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

export function RolloutBrowser({ title }: RolloutBrowserProps) {
  const [tree, setTree] = useState<RolloutNode | null>(null)
  const [selected, setSelected] = useState<RolloutNode | null>(null)
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['']))
  const [error, setError] = useState('')

  useEffect(() => {
    let ignore = false
    const loadTree = async () => {
      setError('')
      try {
        const response = await fetch(`${apiBase}/api/rollouts/tree`)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const payload = (await response.json()) as RolloutNode
        if (ignore) return
        const videos = flattenVideos(payload)
        const newest = videos.sort((a, b) => (b.modified || 0) - (a.modified || 0))[0] || null
        setTree(payload)
        setSelected(newest)
        setExpanded(newest ? defaultExpanded(payload, newest.path) : new Set(['']))
      } catch {
        if (!ignore) setError('Rollouts could not be loaded.')
      }
    }
    void loadTree()
    return () => {
      ignore = true
    }
  }, [])

  const videoSrc = useMemo(() => (selected?.extension === '.mp4' ? rolloutVideoUrl(selected.path) : ''), [selected])

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
              setExpanded(new Set(['']))
              setError('')
              void fetch(`${apiBase}/api/rollouts/tree`)
                .then((response) => {
                  if (!response.ok) throw new Error()
                  return response.json()
                })
                .then((payload: RolloutNode) => {
                  const videos = flattenVideos(payload)
                  const newest = videos.sort((a, b) => (b.modified || 0) - (a.modified || 0))[0] || null
                  setTree(payload)
                  setSelected(newest)
                  setExpanded(newest ? defaultExpanded(payload, newest.path) : new Set(['']))
                })
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
      </section>
    </section>
  )
}
