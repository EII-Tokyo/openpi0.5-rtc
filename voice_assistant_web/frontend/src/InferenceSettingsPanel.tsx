import { useEffect, useRef, useState } from 'react'
import { apiBase } from './services/api'

type Settings = { operator: string | null; rating: number | null; camera_low_position: 'front' | 'rear' | null }
type ResponseSettings = Settings & { operators: string[] }
type PromptContract = {
  version: string; sent_language: string; system_prompt_en: string; system_prompt_zh: string
  response_schema: unknown; user_text_template: unknown; camera_keys: string[]
}

type Props = Settings & { open: boolean; onClose: () => void }

export function InferenceSettingsPanel({ operator, rating, camera_low_position, open, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  useEffect(() => {
    const element = dialog.current
    if (open && !element?.open) element?.showModal()
    else if (!open && element?.open) element.close()
  }, [open])
  const [value, setValue] = useState<Settings>({ operator: null, rating: null, camera_low_position: null })
  const [saved, setSaved] = useState<Settings | null>(null)
  const [operators, setOperators] = useState<string[]>([])
  const [busy, setBusy] = useState(false)
  const [message, setMessage] = useState('Loading inference settings…')
  const [failed, setFailed] = useState(false)
  const [prompt, setPrompt] = useState<PromptContract | null>(null)
  const [promptError, setPromptError] = useState(false)
  const [promptLanguage, setPromptLanguage] = useState<'en' | 'zh'>('en')

  useEffect(() => {
    if (!open) return
    const controller = new AbortController()
    void (async () => {
      try {
        const response = await fetch(`${apiBase}/api/high-level/status`, { signal: controller.signal })
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const data = await response.json()
        if (!data.prompt_contract?.system_prompt_en) throw new Error('Prompt unavailable')
        if (!controller.signal.aborted) { setPrompt(data.prompt_contract); setPromptError(false) }
      } catch {
        if (!controller.signal.aborted) { setPrompt(null); setPromptError(true) }
      }
    })()
    return () => controller.abort()
  }, [open])

  const accept = (data: ResponseSettings) => {
    const settings = { operator: data.operator, rating: data.rating, camera_low_position: data.camera_low_position ?? null }
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

  const dirty = saved !== null && (saved.operator !== value.operator || saved.rating !== value.rating || saved.camera_low_position !== value.camera_low_position)
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
        <section className="conditioning-prompt" aria-labelledby="sol-prompt-title">
          <h3 id="sol-prompt-title">Sol prompt · eight tasks</h3>
          {prompt ? <>
            <p className="conditioning-prompt-version">{prompt.version}</p>
            <p className="conditioning-description">Active policy prompt. Sent in English; Chinese is the reviewed translation. No time-based reply expiry.</p>
            <div className="conditioning-prompt-languages" role="group" aria-label="Prompt language">
              <button type="button" aria-pressed={promptLanguage === 'en'} onClick={() => setPromptLanguage('en')}>English · sent to Sol</button>
              <button type="button" aria-pressed={promptLanguage === 'zh'} onClick={() => setPromptLanguage('zh')}>中文对照</button>
            </div>
            <pre className="conditioning-prompt-text" aria-label={promptLanguage === 'en' ? 'Active English system prompt' : 'Chinese prompt translation'}>{promptLanguage === 'en' ? prompt.system_prompt_en : prompt.system_prompt_zh}</pre>
            <details className="conditioning-prompt-protocol">
              <summary>Input and output protocol</summary>
              <p>Four current images: {prompt.camera_keys.join(', ')}. Each call receives the previous complete state and returns the new complete state. Memory and current task stay separate.</p>
              <h4>User text template</h4>
              <pre>{JSON.stringify(prompt.user_text_template, null, 2)}</pre>
              <h4>Exact output schema</h4>
              <pre>{JSON.stringify(prompt.response_schema, null, 2)}</pre>
            </details>
          </> : <p role="status">{promptError ? 'Active prompt is unavailable. Reopen after the policy server is ready.' : 'Loading active prompt…'}</p>}
        </section>
        <fieldset className="conditioning-fieldset" disabled={!saved || busy}>
          <legend>Camera low position</legend>
          <div className="conditioning-operators">
            {([null, 'front', 'rear'] as const).map(position => (
              <label className="conditioning-option" key={position ?? 'unset'}>
                <input className="conditioning-radio" type="radio" name="camera-low-position"
                  checked={value.camera_low_position === position}
                  onChange={() => setValue({ ...value, camera_low_position: position })} />
                <span className="conditioning-person">
                  <span className="conditioning-person-text">
                    <strong>{position === 'front' ? 'Near side' : position === 'rear' ? 'Far side' : 'Unspecified'}</strong>
                    <small>{position ? `${position === 'front' ? 'near' : 'far'} side of the workbench` : 'No camera position override'}</small>
                  </span>
                  <span className="conditioning-check" aria-hidden="true" />
                </span>
              </label>
            ))}
          </div>
        </fieldset>
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
          <span>Camera low: {camera_low_position === 'front' ? 'near side of the workbench' : camera_low_position === 'rear' ? 'far side of the workbench' : 'unspecified'}</span>
        </div>
      </div>
    </dialog>
  )
}
