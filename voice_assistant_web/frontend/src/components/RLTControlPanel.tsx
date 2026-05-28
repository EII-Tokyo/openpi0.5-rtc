import { useEffect, useMemo, useState } from 'react'
import {
  endKeyRegion,
  RLTControlState,
  scoreKeyRegion,
  startKeyRegion,
  updateRLTConfig,
} from '../services/api'

type Props = {
  rlt: RLTControlState
  onState: (state: RLTControlState) => void
}

function formatCountdown(deadline: number | null) {
  if (!deadline) return ''
  return `${Math.max(0, deadline - Date.now() / 1000).toFixed(1)}s`
}

export function RLTControlPanel({ rlt, onState }: Props) {
  const [error, setError] = useState('')
  const [pending, setPending] = useState('')
  const [countdown, setCountdown] = useState(formatCountdown(rlt.score_deadline))

  useEffect(() => {
    const timer = window.setInterval(() => setCountdown(formatCountdown(rlt.score_deadline)), 100)
    return () => window.clearInterval(timer)
  }, [rlt.score_deadline])

  const run = async (name: string, fn: () => Promise<RLTControlState>) => {
    setError('')
    setPending(name)
    try {
      onState(await fn())
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Request failed')
    } finally {
      setPending('')
    }
  }

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.repeat) return
      const active = document.activeElement
      if (
        active instanceof HTMLInputElement ||
        active instanceof HTMLTextAreaElement ||
        active instanceof HTMLSelectElement ||
        active?.getAttribute('contenteditable') === 'true'
      ) {
        return
      }

      const key = event.key.toLowerCase()
      if (key === 's') {
        event.preventDefault()
        if (rlt.phase === 'idle') void run('start', startKeyRegion)
      } else if (key === 'e') {
        event.preventDefault()
        if (rlt.phase === 'key_region') void run('end', endKeyRegion)
      } else if (event.key === '1') {
        event.preventDefault()
        if (rlt.phase === 'await_score') void run('score1', () => scoreKeyRegion(1))
      } else if (event.key === '0') {
        event.preventDefault()
        if (rlt.phase === 'await_score') void run('score0', () => scoreKeyRegion(0))
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [rlt.phase])

  const phaseLabel = useMemo(() => {
    if (rlt.phase === 'key_region') return 'Recording key region'
    if (rlt.phase === 'await_score') return `Awaiting score ${countdown}`
    return 'Idle'
  }, [rlt.phase, countdown])

  return (
    <section className="panel rlt-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">RLT Controls</p>
          <h2>{phaseLabel}</h2>
        </div>
        <span className={`status-pill ${rlt.phase === 'idle' ? 'mode' : 'live'}`}>{rlt.training_phase}</span>
      </div>

      <div className="rlt-control-grid">
        <button
          className="rlt-key-button start"
          type="button"
          disabled={rlt.phase !== 'idle' || !!pending}
          onClick={() => void run('start', startKeyRegion)}
        >
          <span>S</span>
          <small>Start</small>
        </button>
        <button
          className="rlt-key-button end"
          type="button"
          disabled={rlt.phase !== 'key_region' || !!pending}
          onClick={() => void run('end', endKeyRegion)}
        >
          <span>E</span>
          <small>End</small>
        </button>
        <button
          className="rlt-key-button success"
          type="button"
          disabled={rlt.phase !== 'await_score' || !!pending}
          onClick={() => void run('score1', () => scoreKeyRegion(1))}
        >
          <span>1</span>
          <small>Success</small>
        </button>
        <button
          className="rlt-key-button fail"
          type="button"
          disabled={rlt.phase !== 'await_score' || !!pending}
          onClick={() => void run('score0', () => scoreKeyRegion(0))}
        >
          <span>0</span>
          <small>Fail</small>
        </button>
      </div>

      {rlt.phase === 'await_score' ? (
        <p className="rlt-warning">Score within 10 seconds. No score defaults to failure.</p>
      ) : null}
      {error ? <p className="rlt-error">{error}</p> : null}
    </section>
  )
}

export function RLTStatsPanel({ rlt }: { rlt: RLTControlState }) {
  const warmupPct = Math.min(100, (rlt.warmup_count / Math.max(1, rlt.warmup_target)) * 100)
  return (
    <section className="panel rlt-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Collection</p>
          <h2>Warmup {rlt.warmup_count} / {rlt.warmup_target}</h2>
        </div>
        <span className={`status-pill ${rlt.actor_effective ? 'live' : 'offline'}`}>
          {rlt.actor_effective ? 'Actor active' : 'Actor locked'}
        </span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${warmupPct}%` }} />
      </div>
      <div className="metric-grid">
        <Metric label="Warmup Success" value={rlt.warmup_success} />
        <Metric label="Warmup Failure" value={rlt.warmup_failure} />
        <Metric label="Auto Rollouts" value={rlt.auto_rollout_count} />
        <Metric label="Last Reward" value={rlt.last_reward ?? '-'} />
      </div>
      <div className="event-log">
        {rlt.events.slice(-5).reverse().map((event) => (
          <div key={`${event.timestamp}-${event.event}`} className="event-row">
            <span>{new Date(event.timestamp * 1000).toLocaleTimeString()}</span>
            <strong>{event.event}</strong>
            <small>{event.detail}</small>
          </div>
        ))}
      </div>
    </section>
  )
}

export function RLTConfigPanel({ rlt, onState }: Props) {
  const [draft, setDraft] = useState({
    warmup_target: rlt.warmup_target,
    beta: rlt.beta,
    intervention_scale: rlt.intervention_scale,
    max_delta: rlt.max_delta,
    actor_enabled: rlt.actor_enabled,
  })
  const [error, setError] = useState('')

  useEffect(() => {
    setDraft({
      warmup_target: rlt.warmup_target,
      beta: rlt.beta,
      intervention_scale: rlt.intervention_scale,
      max_delta: rlt.max_delta,
      actor_enabled: rlt.actor_enabled,
    })
  }, [rlt.warmup_target, rlt.beta, rlt.intervention_scale, rlt.max_delta, rlt.actor_enabled])

  const submit = async () => {
    setError('')
    try {
      onState(await updateRLTConfig(draft))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Config update failed')
    }
  }

  return (
    <section className="panel rlt-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Actor Settings</p>
          <h2>Beta {draft.beta.toFixed(2)}</h2>
        </div>
        <button className="apply-button" type="button" onClick={() => void submit()}>
          Apply
        </button>
      </div>

      <label className="rlt-field">
        <span>Warmup Target</span>
        <input
          type="number"
          min={1}
          value={draft.warmup_target}
          onChange={(event) => setDraft({ ...draft, warmup_target: Number(event.target.value) })}
        />
      </label>
      <label className="rlt-field">
        <span>Beta</span>
        <input
          type="range"
          min={0}
          max={20}
          step={0.1}
          value={draft.beta}
          onChange={(event) => setDraft({ ...draft, beta: Number(event.target.value) })}
        />
      </label>
      <label className="rlt-field">
        <span>Intervention Scale {draft.intervention_scale.toFixed(2)}</span>
        <input
          type="range"
          min={0}
          max={1}
          step={0.01}
          value={draft.intervention_scale}
          onChange={(event) => setDraft({ ...draft, intervention_scale: Number(event.target.value) })}
        />
      </label>
      <label className="rlt-field">
        <span>Max Delta</span>
        <input
          type="number"
          min={0}
          step={0.01}
          value={draft.max_delta}
          onChange={(event) => setDraft({ ...draft, max_delta: Number(event.target.value) })}
        />
      </label>
      <label className="rlt-toggle">
        <input
          type="checkbox"
          checked={draft.actor_enabled}
          disabled={rlt.warmup_count < rlt.warmup_target}
          onChange={(event) => setDraft({ ...draft, actor_enabled: event.target.checked })}
        />
        <span>Enable Actor</span>
      </label>
      {rlt.actor_locked_reason ? <p className="rlt-warning">Actor locked: {rlt.actor_locked_reason}</p> : null}
      {rlt.wandb_url ? (
        <a className="wandb-link" href={rlt.wandb_url} target="_blank" rel="noreferrer">
          Open wandb run
        </a>
      ) : null}
      <div className="metric-grid">
        <Metric label="Actor Loss" value={formatMetric(rlt.actor_loss)} />
        <Metric label="Critic Loss" value={formatMetric(rlt.critic_loss)} />
      </div>
      {error ? <p className="rlt-error">{error}</p> : null}
    </section>
  )
}

function Metric({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="metric-tile">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function formatMetric(value: number | null) {
  return value === null ? '-' : value.toFixed(4)
}

