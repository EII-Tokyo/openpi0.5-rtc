import { useEffect, useRef, useState } from 'react'
import { apiBase } from './services/api'
import './high-level.css'
import type { NozzleTips } from './components/NozzleCalibration'

type Result = { status?: string; error?: string; requested_unix?: number | null; returned_unix?: number | null; id?: string; seq: number; applied: boolean | null; current_task: string; bottle_description: string; memory: string; latency_seconds: number; observation_unix: number | null }
type Snapshot = { nozzle_tips?: NozzleTips | null; dwell_token?: string | null; request_started_unix?: number | null; active_request_seq?: number | null; result_from_previous_episode?: boolean; applied?: boolean | null; request_interval_seconds?: number; enabled?: boolean; status?: string; model?: string; service_tier?: string; checkpoint_step?: number | string; current_task_index?: number | null; current_task?: string; bottle_description?: string; memory?: string; observation_unix?: number | null; latency_seconds?: number | null; submitted?: number; completed?: number; inflight?: number; discarded?: number; error?: string | null; recent_results?: Result[]; updated_unix?: number; status_age_seconds?: number }
const texts = {
 en: { request: 'Request', returned: 'Returned', result: 'Result', failed: 'Failed', requestLatency: 'Sol latency per request', recent: 'Latest 20 requests · complete records retained on 103', elapsed: 'Current request elapsed', noRequest: 'No request in flight', previous: 'Last returned result from an earlier episode · not an active robot command', pendingDecision: 'Application not confirmed', interval: 'Minimum request interval', title: 'High-level results', task: 'Current task', bottle: 'Bottle description', memory: 'Memory · completed actions', empty: 'Empty', none: 'Waiting for the first result', frequency: 'Four current cameras · no image history · one request at a time', latency: 'Response latency', age: 'Image age', pending: 'In flight', discarded: 'Discarded replies', history: 'Recent replies', applied: 'Applied', ignored: 'Ignored', unavailable: 'High-level status is unavailable', modes: { waiting_result: 'Waiting for a fresh Sol reply · holding position', error: 'Error', ready: 'Ready · starts with inference', running: 'Running', idle: 'Idle', waiting_gpu: 'Prepared · waiting for GPU', downloading: 'Preparing checkpoint', not_configured: 'Not configured', stopped: 'Stopped' } },
 ja: { request: 'リクエスト', returned: '応答時刻', result: '結果', failed: '失敗', requestLatency: 'Sol リクエストごとの遅延', recent: '直近20件 · 完全な記録は103に保存', elapsed: '処理中の経過時間', noRequest: '処理中のリクエストなし', previous: '以前のエピソードの最後の応答 · 現在の動作指示ではありません', pendingDecision: '反映未確認', interval: '最小リクエスト間隔', title: '上位モデルの結果', task: '現在のタスク', bottle: 'ボトルの説明', memory: 'Memory · 完了した動作', empty: '空', none: '最初の結果を待っています', frequency: '現在の4カメラ · 画像履歴なし · 応答後に次を更新', latency: '応答遅延', age: '画像の経過時間', pending: '処理中', discarded: '破棄した応答', history: '最近の応答', applied: '反映', ignored: '破棄', unavailable: '上位モデルの状態を取得できません', modes: { waiting_result: 'Sol の最新応答を待機中 · 現在位置を保持', error: 'エラー', ready: '準備完了 · 推論開始待ち', running: '実行中', idle: '待機中', waiting_gpu: '準備済み · GPU待ち', downloading: 'モデル準備中', not_configured: '未設定', stopped: '停止' } },
 zh: { request: '请求', returned: '返回时间', result: '结果', failed: '失败', requestLatency: '每次 Sol 请求延迟', recent: '最近20次请求 · 完整记录保留在103', elapsed: '当前请求已等待', noRequest: '当前没有进行中的请求', previous: '上次任务的最后返回，仅供查看，不是当前执行指令', pendingDecision: '未确认采用', interval: '最短请求间隔', title: 'High level · 高层结果', task: '当前任务', bottle: '瓶子描述', memory: 'Memory · 已完成的任务', empty: '空', none: '等待第一次返回', frequency: '四路当前图片 · 无历史图片 · 等待返回后继续更新', latency: '请求返回耗时', age: '图像距今', pending: '正在请求', discarded: '已丢弃的返回', history: '最近的返回', applied: '已采用', ignored: '已丢弃', unavailable: '暂时无法读取高层状态', modes: { waiting_result: '等待新的 Sol 回复，机械臂保持当前位置', error: '发生错误', ready: '已就绪，等待开始推理', running: '运行中', idle: '空闲', waiting_gpu: '已准备，等待GPU空闲', downloading: '正在准备模型', not_configured: '尚未配置', stopped: '已停止' } },
}
export function HighLevelPanel({ language, open, onClose, inferencePrompt, runtimeMode }: { language: string; open: boolean; onClose: () => void; inferencePrompt: string | null; runtimeMode: string }) {
 const dialog = useRef<HTMLDialogElement>(null)
 useEffect(() => {
  if (open && !dialog.current?.open) dialog.current?.showModal()
  else if (!open && dialog.current?.open) dialog.current.close()
 }, [open])
 const t = texts[language as keyof typeof texts] ?? texts.en
 const [snapshot, setSnapshot] = useState<Snapshot>({ status: 'not_configured' })
 const [promptLanguage, setPromptLanguage] = useState<'zh' | 'en'>('zh')
 const [prompts, setPrompts] = useState<{ zh: string; en: string } | null>(null)
 const [promptError, setPromptError] = useState(false)
 const [dwell, setDwell] = useState({ token: '', message: '' })
 const confirmDwell = async (token: string) => {
  try {
   const response = await fetch(`${apiBase}/api/high-level/confirm-dwell/${token}`, { method: 'POST' })
   if (!response.ok) throw new Error('dwell unavailable')
   setDwell({ token, message: '已确认 / Confirmed' })
  } catch { setDwell({ token, message: '确认未记录 / Confirmation unavailable' }) }
 }
 const loadPrompts = async () => {
  if (prompts) return
  try {
   const response = await fetch(`${apiBase}/api/high-level/prompts`)
   if (!response.ok) throw new Error('prompt unavailable')
   setPrompts(await response.json())
   setPromptError(false)
  } catch { setPromptError(true) }
 }
 const [networkError, setNetworkError] = useState(false)
 const [now, setNow] = useState(Date.now() / 1000)
 useEffect(() => {
  if (!open) return
  let active = true
  const controller = new AbortController()
  const poll = async () => {
   try {
    const response = await fetch(`${apiBase}/api/high-level/status`, { signal: controller.signal, cache: 'no-store' })
    if (!response.ok) throw new Error('status unavailable')
    const value = await response.json()
    if (active) { setSnapshot(value); setNetworkError(false); setNow(Date.now() / 1000) }
   } catch { if (active) setNetworkError(true) }
  }
  void poll()
  const timer = window.setInterval(() => { setNow(Date.now() / 1000); void poll() }, 1000)
  return () => { active = false; controller.abort(); window.clearInterval(timer) }
 }, [open])
 const age = snapshot.observation_unix ? Math.max(0, now - snapshot.observation_unix) : null
 const displayStatus = t.modes[snapshot.status as keyof typeof t.modes] ?? snapshot.status ?? t.none
 return <dialog ref={dialog} id="high-level-results" className="inference-settings-drawer high-level-drawer" aria-labelledby="high-level-title"
  onClose={onClose} onCancel={event => { event.preventDefault(); onClose() }}
  onClick={event => { if (event.target === event.currentTarget) { const rect = event.currentTarget.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose() } }}>
  <div className="motor-drawer-header"><div><p className="eyebrow">{snapshot.model ?? 'High level'} · {snapshot.service_tier ?? '—'} · checkpoint {snapshot.checkpoint_step ?? '—'}</p><h2 id="high-level-title">{t.title}</h2></div><button className="conditioning-close" type="button" aria-label="Close high-level results" onClick={onClose}>×</button></div>
  <div className="high-level-panel high-level-drawer-body"><span className="status-pill">{displayStatus}</span>
  <p className="high-level-note">{t.frequency}</p>
  {snapshot.result_from_previous_episode && <p className="high-level-note">{t.previous}</p>}
  {networkError && <p role="alert" className="error-text">{t.unavailable}</p>}
  {snapshot.error && <p role="alert" className="error-text">{snapshot.error}</p>}
  <dl className="high-level-fields"><div><dt>{t.task}</dt><dd>{snapshot.current_task || t.none}</dd></div><div><dt>{t.bottle}</dt><dd>{snapshot.bottle_description || t.empty}</dd></div><div><dt>{t.memory}</dt><dd>{snapshot.memory || t.empty}</dd></div></dl>
  <div className="high-level-metrics"><span>{t.interval}: {snapshot.request_interval_seconds ?? '—'}s</span>{snapshot.current_task && <span>{snapshot.applied === true ? t.applied : snapshot.applied === false ? t.ignored : t.pendingDecision}</span>}<span>{t.latency}: {typeof snapshot.latency_seconds === 'number' ? `${snapshot.latency_seconds.toFixed(2)}s` : '—'}</span><span>{t.age}: {age === null ? '—' : `${age.toFixed(1)}s`}</span><span>{t.pending}: {snapshot.inflight ?? 0}</span><span>{t.discarded}: {snapshot.discarded ?? 0}</span></div>
  <section className="high-level-low-prompt" aria-labelledby="high-level-full-prompt">
   <p className="eyebrow">ACTUAL LOW-LEVEL INPUT · {runtimeMode}</p>
   <h3 id="high-level-full-prompt">Full prompt</h3>
   <p className="high-level-note">Latest prompt actually used by the low-level policy. A new high-level reply appears here after it is used by inference.</p>
   <pre className="inference-prompt-text">{inferencePrompt || 'Waiting for the first inference of the task…'}</pre>
  </section>
  {snapshot.nozzle_tips && Object.values(snapshot.nozzle_tips).some(Boolean) && <div className="high-level-metrics">
   <span>{language === 'zh' ? '本次运行喷嘴坐标 · 原图像素' : language === 'ja' ? '実行中のノズル座標 · 元画像' : 'Run nozzle coordinates · source pixels'}</span>
   {Object.entries(snapshot.nozzle_tips).map(([camera, point]) => point &&
    <span key={camera}>{camera}: ({Math.round(point.x * 640)}, {Math.round(point.y * 480)})</span>)}
  </div>}
  {!!snapshot.dwell_token && <div className="high-level-metrics">
   <button className="ghost-button" type="button" onClick={() => void confirmDwell(snapshot.dwell_token!)}>
    确认已插到底并连续停留 1.5 秒 / Confirm full insertion for 1.5 s
   </button>
   {dwell.token === snapshot.dwell_token && <span role="status">{dwell.message}</span>}
  </div>}
  <details onToggle={event => { if (event.currentTarget.open) void loadPrompts() }}>
   <summary>冲洗 Prompt / Rinse prompt</summary>
   <label className="language-switch">
    <select aria-label="Prompt language" value={promptLanguage}
     onChange={event => setPromptLanguage(event.target.value as 'zh' | 'en')}>
     <option value="zh">中文</option><option value="en">English</option>
    </select>
   </label>
   {promptError && <p role="alert">{t.unavailable}</p>}
   {prompts && <pre className="rinse-prompt-text">{prompts[promptLanguage]}</pre>}
  </details>
  <section className="sol-request-latencies" aria-label={t.requestLatency}>
   <h3>{t.requestLatency}</h3>
   <p className="high-level-note">{t.recent}</p>
   <p className="sol-request-live" role="status">{snapshot.inflight && snapshot.request_started_unix
     ? `#${snapshot.active_request_seq ?? '—'} · ${t.elapsed}: ${Math.max(0, now - snapshot.request_started_unix).toFixed(1)}s`
     : t.noRequest}</p>
   {!!snapshot.recent_results?.length && <div className="sol-request-table-scroll"><table>
    <thead><tr><th>{t.request}</th><th>{t.returned}</th><th>{t.latency}</th><th>{t.result}</th></tr></thead>
    <tbody>{[...snapshot.recent_results].reverse().map(result => <tr key={result.id ?? result.seq}>
     <td>#{result.seq}</td>
     <td>{result.returned_unix ? new Date(result.returned_unix * 1000).toLocaleTimeString() : '—'}</td>
     <td className="sol-latency-value">{typeof result.latency_seconds === 'number' ? `${result.latency_seconds.toFixed(2)}s` : '—'}</td>
     <td><details><summary>{result.status === 'error' ? t.failed : result.applied === true ? t.applied : result.applied === false ? t.ignored : t.pendingDecision}</summary>
      {result.error && <p className="error-text">{result.error}</p>}
      <p>{result.current_task}</p><p>{result.bottle_description || t.empty}</p><p>{result.memory || t.empty}</p>
     </details></td>
    </tr>)}</tbody>
   </table></div>}
  </section>
 </div>
 </dialog>
}
