import { useEffect, useRef, useState } from 'react'
import { apiBase } from './services/api'

type Settings = { operator: string | null; rating: number | null }
type ResponseSettings = Settings & { operators: string[] }

type Props = Settings & { open: boolean; onClose: () => void }

export function InferenceSettingsPanel({ operator, rating, open, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const element = dialog.current
    if (open && !element?.open) element?.showModal()
    else if (!open && element?.open) element.close()
  }, [open])
  const [value, setValue] = useState<Settings>({ operator: null, rating: null })
  const [saved, setSaved] = useState<Settings | null>(null)
  const [operators, setOperators] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('Loading inference settings…')
  const [failed, setFailed] = useState(false)

  const accept = (data: ResponseSettings) => {
    const settings = { operator: data.operator, rating: data.rating }
    setOperators(data.operators)
    setValue(settings)
    setSaved(settings)
  }

  const load = async () => {
    setBusy(true)
    try {
      const response = await fetch(`${apiBase}/api/inference-settings`)
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      accept(await response.json())
      setMessage('Save to apply to the current task immediately.')
      setFailed(false)
    } catch {
      setFailed(true)
      setMessage('Could not load inference settings. Retry to edit.')
    } finally {
      setBusy(false)
    }
  }

  useEffect(() => { void load() }, [])

  const save = async () => {
    setBusy(true)
    try {
      const response = await fetch(`${apiBase}/api/inference-settings`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(value),
      })
      if (!response.ok) throw new Error(`HTTP ${response.status}`)
      accept(await response.json())
      setMessage('Saved. Applying to the current task’s next inference.')
      setFailed(false)
    } catch {
      setFailed(true)
      setMessage('Could not save. The previous settings are still in use.')
    } finally {
      setBusy(false)
    }
  }

  const dirty = saved !== null && (saved.operator !== value.operator || saved.rating !== value.rating)
  return (
    <dialog ref={dialog} id="inference-conditioning" className="inference-settings-drawer"
      aria-labelledby="inference-conditioning-title" onClose={onClose}
      onCancel={event => { event.preventDefault(); onClose() }}
      onClick={event => {
        if (event.target !== event.currentTarget) return
        const rect = event.currentTarget.getBoundingClientRect()
        if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose()
      }}>
      <div className="motor-drawer-header">
        <div>
          <p className="conditioning-eyebrow">MODEL INPUT</p>
          <h2 id="inference-conditioning-title">Inference conditioning</h2>
        </div>
        <button className="conditioning-close" type="button" aria-label="Close inference conditioning" onClick={onClose}>×</button>
      </div>
      <p className="conditioning-description">Update the current task’s conditions in real time.</p>
      <div className="conditioning-body">
        <fieldset className="conditioning-fieldset" disabled={!saved || busy}>
          <legend>Operator</legend>
          <div className="conditioning-operators">
            {[null, ...operators].map(name => (
              <label className="conditioning-option" key={name ?? 'unspecified'}>
                <input className="conditioning-radio" type="radio" name="inference-operator"
                  value={name ?? ''} checked={value.operator === name}
                  onChange={() => setValue({ ...value, operator: name })} />
                <span className="conditioning-person">
                  <span className={`conditioning-avatar${name ? '' : ' unspecified'}`} aria-hidden="true">
                    {name ? name.split(' ').map(part => part[0]).join('') : '—'}
                  </span>
                  <span className="conditioning-person-text">
                    <strong>{name ?? 'Unspecified'}</strong>
                    {!name && <small>No operator condition</small>}
                  </span>
                  <span className="conditioning-check" aria-hidden="true" />
                </span>
              </label>
            ))}
          </div>
        </fieldset>
        <fieldset className="conditioning-fieldset" disabled={!saved || busy}>
          <legend>Rating</legend>
          <div className="conditioning-rating-header">
            <span>{value.rating === null ? 'No rating selected' : `Selected: ${value.rating} / 5`}</span>
            <button className="conditioning-clear" type="button" disabled={value.rating === null}
              onClick={() => setValue({ ...value, rating: null })}>Clear</button>
          </div>
          <div className="conditioning-rating">
            {[1, 2, 3, 4, 5].map(score => (
              <label className="conditioning-option" key={score}>
                <input className="conditioning-radio" type="radio" name="inference-rating" aria-label={`Rating ${score}`}
                  value={score} checked={value.rating === score}
                  onChange={() => setValue({ ...value, rating: score })} />
                <span className="conditioning-score">{score}</span>
              </label>
            ))}
          </div>
          <div className="conditioning-scale"><span>Lower</span><span>Higher</span></div>
        </fieldset>
      </div>
      <div className="conditioning-footer">
        <button className="conditioning-save" type="button" disabled={!saved || busy || !dirty}
          onClick={() => void save()}>{busy ? 'Saving…' : 'Apply now'}</button>
        {failed && !saved && <button className="ghost-button" type="button" disabled={busy}
          onClick={() => void load()}>Retry</button>}
        <p className={`conditioning-status${failed ? ' error' : ''}`} role={failed ? 'alert' : 'status'}>
          {failed ? message : dirty ? 'Unsaved changes. Apply to update the running task.' : message}
        </p>
        <div className="conditioning-active">
          <span className="conditioning-eyebrow">ACTIVE TASK</span>
          <span>{operator ?? 'Unspecified operator'} <span aria-hidden="true">·</span> {rating === null ? 'No rating' : `Rating ${rating}`}</span>
        </div>
      </div>
    </dialog>
  )
}
