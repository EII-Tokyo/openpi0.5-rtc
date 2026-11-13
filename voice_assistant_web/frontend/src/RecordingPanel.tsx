import { useEffect, useRef, useState } from 'react'
import { apiBase } from './services/api'

type State = { status: string; frames: number; path?: string; error?: string; paused: boolean;
  online?: boolean; mode?: string; command_id?: string; command_error?: string; batch_name?: string }
type Props = { open: boolean; onClose: () => void }

// No task or actuator command is sent when opening the drawer or starting a recording.
export function RecordingPanel({ open, onClose }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [state, setState] = useState<State>({ status: 'idle', frames: 0, paused: false })
  const [rating, setRating] = useState<number | null>(null)
  const [isError, setIsError] = useState(false)
  const [note, setNote] = useState('')
  const [batchName, setBatchName] = useState(() => localStorage.getItem('aloha-recording-batch') ?? '')
  const [pending, setPending] = useState<{ id: string; since: number } | null>(null)
  const [error, setError] = useState('')
  const [folders, setFolders] = useState<{ name: string; recordable: boolean }[]>([])
  const [folderError, setFolderError] = useState('')
  const [folderBusy, setFolderBusy] = useState(false)
  const [newFolder, setNewFolder] = useState('')
  const [folderRoot, setFolderRoot] = useState('')

  const loadFolders = async (name?: string) => {
    setFolderBusy(true)
    setFolderError('')
    try {
      const response = await fetch(`${apiBase}/api/recording/folders`, name === undefined ? {} : {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ name }) })
      const result = await response.json()
      if (!response.ok) throw new Error(result.detail ?? `HTTP ${response.status}`)
      setFolders(result.folders)
      setFolderRoot(result.root)
      if (name !== undefined) {
        setBatchName(name)
        localStorage.setItem('aloha-recording-batch', name)
        setNewFolder('')
      }
    } catch (reason) {
      setFolderError(reason instanceof Error ? reason.message : '目录读取失败')
    } finally { setFolderBusy(false) }
  }

  useEffect(() => { if (open) void loadFolders() }, [open])

  useEffect(() => {
    const element = dialog.current
    if (open && !element?.open) element?.showModal()
    else if (!open && element?.open) element.close()
  }, [open])

  useEffect(() => {
    let active = true
    const refresh = async () => {
      try {
        const response = await fetch(`${apiBase}/api/recording`)
        if (!response.ok) throw new Error(`HTTP ${response.status}`)
        const next: State = await response.json()
        if (active) setState(next)
      } catch {
        if (active) setState(previous => ({ ...previous, online: false }))
      }
    }
    void refresh()
    const timer = window.setInterval(() => void refresh(), 500)
    return () => { active = false; window.clearInterval(timer) }
  }, [])

  useEffect(() => {
    if (!pending) return
    if (state.command_id === pending.id) {
      setPending(null)
      setError(state.command_error ?? '')
    } else if (Date.now() - pending.since > 5000) {
      setPending(null)
      setError('No runtime acknowledgement. Check robot state before retrying; this is not an emergency stop.')
    }
  }, [state, pending])

  const send = async (action: 'start' | 'stop' | 'resume') => {
    setError('')
    setPending({ id: 'sending', since: Date.now() })
    try {
      const response = await fetch(`${apiBase}/api/recording`, { method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ action, batch_name: batchName || null, annotations: { rating, is_error: isError, note } }) })
      if (!response.ok) {
        const detail = await response.json()
        throw new Error(detail.detail ?? `HTTP ${response.status}`)
      }
      const result = await response.json()
      if (action === 'start') localStorage.setItem('aloha-recording-batch', batchName)
      setPending({ id: result.command_id, since: Date.now() })
    } catch (reason) {
      setPending(null)
      setError(reason instanceof Error ? reason.message : 'Command failed')
    }
  }

  const active = state.status === 'recording'
  return <dialog ref={dialog} className="inference-settings-drawer recording-drawer" aria-labelledby="recording-title"
    onClose={onClose} onCancel={event => { event.preventDefault(); onClose() }}
    onClick={event => {
      if (event.target !== event.currentTarget) return
      const rect = event.currentTarget.getBoundingClientRect()
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose()
    }}>
    <div className="motor-drawer-header">
      <div><p className="conditioning-eyebrow">POLICY ROLLOUT DATA</p><h2 id="recording-title">录制 / Recording</h2></div>
      <button className="conditioning-close" aria-label="Close recording" onClick={onClose}>×</button>
    </div>
    <p className="conditioning-description">只录制按钮之间的动作。结束后保持当前位置，不回 home；继续运行需单独点击。</p>
    <div className="conditioning-body">
      <a href={`http://${window.location.hostname}:8765/`} target="_blank" rel="noopener noreferrer">数据采集页面 ↗</a>
      <fieldset className="conditioning-fieldset" disabled={active || state.status === 'saving' || !!pending || folderBusy}>
        <legend>保存文件夹 / Batch folder</legend>
        <div className="recording-folder-row">
        <label className="recording-note">选择已有采集文件夹
          <select value={folders.some(folder => folder.name === batchName) ? batchName : ''}
            onChange={event => { setBatchName(event.target.value); localStorage.setItem('aloha-recording-batch', event.target.value) }}>
            <option value="">请选择文件夹</option>
            {folders.map(folder => <option key={folder.name} value={folder.name} disabled={!folder.recordable}>
              {folder.name}{folder.recordable ? '' : '（名称暂不支持录制）'}
            </option>)}
          </select>
        </label>
        <button type="button" onClick={() => void loadFolders()}>刷新</button>
        </div>
        <div className="recording-folder-row">
        <label className="recording-note">创建新文件夹
          <input value={newFolder} maxLength={96} placeholder="例如：2026-10-19-rinse"
            onChange={event => setNewFolder(event.target.value)} />
        </label>
        <button type="button" disabled={!newFolder.trim()} onClick={() => void loadFolders(newFolder.trim())}>创建并选择</button>
        </div>
      </fieldset>
      {folderBusy && <p role="status">正在读取或创建文件夹…</p>}
      {folderError && <p role="alert" className="conditioning-status error">{folderError}</p>}
      {folderRoot && <p className="conditioning-description">保存根目录：{folderRoot}</p>}
      <p className="conditioning-description">每批一个目录；批内 episode_0、episode_1 各含四路 MP4 和 episode.hdf5。换批修改名称；同名批次继续编号，不覆盖。</p>
      <div className="conditioning-active" aria-live="polite">
        <strong>{state.online ? state.status : 'Runtime offline'} {state.paused ? ' · 已暂停 / Paused' : ''}</strong>
        <span>{state.frames} frames</span>
      </div>
      <fieldset className="conditioning-fieldset" disabled={state.status === 'saving' || !!pending}>
        <legend>采集结果评分 · 不是推理条件</legend>
        <p className="conditioning-description">可不评分。0 和 1 保留为原始采集标注，不会改写推理 Rating。</p>
        <div className="recording-scores">
          {[null, 0, 1, 2, 3, 4, 5].map(score => <label className="conditioning-option" key={score ?? 'none'}>
            <input className="conditioning-radio" type="radio" name="recording-rating"
              checked={rating === score} onChange={() => setRating(score)} />
            <span className="conditioning-score">{score ?? '未评分'}</span>
          </label>)}
        </div>
        <label className="recording-error-label"><input type="checkbox" checked={isError}
          onChange={event => setIsError(event.target.checked)} />错误片段 / Failed behavior</label>
        <label className="recording-note">备注 / Notes<textarea value={note} maxLength={2000}
          onChange={event => setNote(event.target.value)} /></label>
      </fieldset>
      <button className="conditioning-save" disabled={folderBusy || !!folderError || !folders.some(folder => folder.name === batchName && folder.recordable) || !state.online || !!pending || active || state.status === 'saving' || state.paused || state.mode !== 'policy'}
        onClick={() => void send('start')}>开始录制 / Start recording</button>
      <button className="recording-stop" disabled={!state.online || !active}
        onClick={() => void send('stop')}>结束录制并暂停 / Finish &amp; pause</button>
      <button className="conditioning-save" disabled={!state.online || !!pending || !state.paused || state.status !== 'saved'}
        onClick={() => void send('resume')}>继续运行 / Resume execution</button>
      <p className="conditioning-description">先启动任务，再开始录制。停止不是硬件急停；危险时请使用现场急停。</p>
      {pending && <p role="status">等待 runtime 确认…</p>}
      {(error || state.error) && <p className="conditioning-status error" role="alert">{error || state.error}</p>}
      {state.path && <div className="recording-path"><strong>{state.status === 'saved' ? '已保存 EP：四路 MP4 + HDF5' : 'EP 输出位置'}</strong><code>{state.path}</code></div>}
    </div>
  </dialog>
}
