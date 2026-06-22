import { useEffect, useMemo, useState } from 'react'
import {
  fetchRltTrajectories,
  rltTrajectoryVideoUrl,
  RLTTrajectoryRecord,
  saveRltTrajectoryTrim,
} from '../services/api'

type TrimDraft = {
  start: number
  end: number
}

function clampStep(value: number, min: number, max: number) {
  if (!Number.isFinite(value)) return min
  return Math.max(min, Math.min(Math.trunc(value), max))
}

function labelText(record: RLTTrajectoryRecord) {
  if (record.terminal_label) return record.terminal_label
  if (record.terminal_success === 1) return 'success'
  if (record.terminal_success === 0) return 'failure'
  return 'unlabeled'
}

export function RLTTrajectoryReview({ onClose }: { onClose: () => void }) {
  const [records, setRecords] = useState<RLTTrajectoryRecord[]>([])
  const [selectedPath, setSelectedPath] = useState('')
  const [camera, setCamera] = useState('cam_high')
  const [draft, setDraft] = useState<TrimDraft>({ start: 0, end: 1 })
  const [terminalLabel, setTerminalLabel] = useState<'success' | 'failure' | 'unlabeled'>('unlabeled')
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState('')

  const selected = useMemo(
    () => records.find((record) => record.path === selectedPath) ?? records[0] ?? null,
    [records, selectedPath],
  )

  const load = async () => {
    setError('')
    setLoading(true)
    try {
      const payload = await fetchRltTrajectories()
      setRecords(payload.trajectories)
      if (!selectedPath && payload.trajectories.length) {
        setSelectedPath(payload.trajectories[0].path)
      }
    } catch {
      setError('Failed to load RLT trajectories.')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void load()
  }, [])

  useEffect(() => {
    if (!selected) return
    setDraft({ start: selected.trim_start_step, end: selected.trim_end_step })
    setTerminalLabel(labelText(selected) as 'success' | 'failure' | 'unlabeled')
    if (!selected.camera_names.includes(camera)) {
      setCamera(selected.camera_names[0] ?? 'cam_high')
    }
  }, [selected?.path])

  const saveTrim = async () => {
    if (!selected) return
    setSaving(true)
    setError('')
    try {
      const maxStep = Math.max(selected.num_steps, 1)
      const start = clampStep(draft.start, 0, maxStep - 1)
      const end = clampStep(draft.end, start + 1, maxStep)
      const updated = await saveRltTrajectoryTrim(selected.path, start, end, terminalLabel)
      setRecords((current) => current.map((record) => (record.path === updated.path ? updated : record)))
      setDraft({ start: updated.trim_start_step, end: updated.trim_end_step })
    } catch {
      setError('Failed to save crop.')
    } finally {
      setSaving(false)
    }
  }

  const videoSrc = selected ? rltTrajectoryVideoUrl(selected.path, camera) : ''
  const trimDuration =
    selected && selected.fps && selected.fps > 0 ? (draft.end - draft.start) / selected.fps : null

  return (
    <div className="trajectory-review">
      <header className="trajectory-review-header">
        <div>
          <p className="eyebrow">RLT Dataset</p>
          <h2>Trajectory Review</h2>
        </div>
        <div className="header-actions">
          <button className="ghost-button" type="button" onClick={() => void load()}>
            Refresh
          </button>
          <button className="ghost-button active" type="button" onClick={onClose}>
            Live
          </button>
        </div>
      </header>

      <section className="trajectory-review-layout">
        <aside className="trajectory-list panel">
          {loading ? <p className="voice-hint">Loading trajectories...</p> : null}
          {!loading && records.length === 0 ? <p className="voice-hint">No episode npz files found.</p> : null}
          {records.map((record) => (
            <button
              key={record.path}
              className={`trajectory-row ${record.path === selected?.path ? 'active' : ''}`}
              type="button"
              onClick={() => setSelectedPath(record.path)}
            >
              <strong>{record.name}</strong>
              <span>{labelText(record)}</span>
              <small>
                {record.num_steps} steps
                {record.duration_s !== null ? ` / ${record.duration_s.toFixed(1)}s` : ''}
              </small>
            </button>
          ))}
        </aside>

        <section className="trajectory-detail panel">
          {selected ? (
            <>
              <div className="panel-header">
                <div>
                  <p className="eyebrow">{labelText(selected)}</p>
                  <h2>{selected.name}</h2>
                </div>
                <select value={camera} onChange={(event) => setCamera(event.target.value)}>
                  {selected.camera_names.map((name) => (
                    <option key={name} value={name}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>

              <video className="trajectory-video" src={videoSrc} controls preload="metadata" />

              <div className="trim-controls">
                <label>
                  <span>Start step</span>
                  <input
                    type="number"
                    min={0}
                    max={Math.max(selected.num_steps - 1, 0)}
                    value={draft.start}
                    onChange={(event) => setDraft((current) => ({ ...current, start: Number(event.target.value) }))}
                  />
                </label>
                <label>
                  <span>End step</span>
                  <input
                    type="number"
                    min={1}
                    max={selected.num_steps}
                    value={draft.end}
                    onChange={(event) => setDraft((current) => ({ ...current, end: Number(event.target.value) }))}
                  />
                </label>
                <label>
                  <span>Label</span>
                  <select value={terminalLabel} onChange={(event) => setTerminalLabel(event.target.value as typeof terminalLabel)}>
                    <option value="unlabeled">unlabeled</option>
                    <option value="success">success</option>
                    <option value="failure">failure</option>
                  </select>
                </label>
                <button className="primary-command start" type="button" disabled={saving} onClick={() => void saveTrim()}>
                  {saving ? 'Saving...' : 'Save Crop'}
                </button>
              </div>

              <div className="rlt-meta">
                <span>
                  Crop: {draft.start} to {draft.end}
                  {trimDuration !== null ? ` (${trimDuration.toFixed(2)}s)` : ''}
                </span>
                <span>Raw path: {selected.path}</span>
              </div>
            </>
          ) : (
            <p className="voice-hint">Select a trajectory.</p>
          )}
          {error ? <p className="voice-error inline-error">{error}</p> : null}
        </section>
      </section>
    </div>
  )
}
