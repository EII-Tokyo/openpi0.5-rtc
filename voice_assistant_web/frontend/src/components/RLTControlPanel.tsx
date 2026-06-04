import { useEffect, useMemo, useState } from 'react'
import {
  confirmKeyRegion,
  discardKeyRegion,
  endKeyRegion,
  RLTControlState,
  scoreKeyRegion,
  startKeyRegion,
  updateRLTConfig,
  voidKeyRegion,
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
  const [flashKey, setFlashKey] = useState('')
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

  const flash = (key: string) => {
    setFlashKey('')
    window.setTimeout(() => setFlashKey(key), 0)
    window.setTimeout(() => setFlashKey(''), 260)
  }

  const normalizeHotkey = (event: KeyboardEvent) => {
    if (event.key === 'ArrowLeft' || event.code === 'ArrowLeft' || event.keyCode === 37) return 'start'
    if (event.key === 'ArrowRight' || event.code === 'ArrowRight' || event.keyCode === 39) return 'end'
    if (event.key === 'ArrowUp' || event.code === 'ArrowUp' || event.keyCode === 38) return 'score1'
    if (event.key === 'ArrowDown' || event.code === 'ArrowDown' || event.keyCode === 40) return 'score0'
    if (event.key === 'Enter' || event.key.toLowerCase() === 'c') return 'confirm'
    if (event.key === 'Backspace' || event.key.toLowerCase() === 'd') return 'discard'
    if (event.key.toLowerCase() === 'v') return 'void'
    return ''
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

      const hotkey = normalizeHotkey(event)
      if (!hotkey) return
      event.preventDefault()
      event.stopPropagation()
      flash(hotkey)

      if (pending) return
      if (hotkey === 'start') {
        if (rlt.phase === 'idle') void run('start', startKeyRegion)
      } else if (hotkey === 'end') {
        if (rlt.phase === 'key_region') void run('end', endKeyRegion)
      } else if (hotkey === 'score1') {
        if (rlt.phase === 'await_score') void run('score1', () => scoreKeyRegion(1))
      } else if (hotkey === 'score0') {
        if (rlt.phase === 'await_score') void run('score0', () => scoreKeyRegion(0))
      } else if (hotkey === 'confirm') {
        if (rlt.phase === 'pending_replay') void run('confirm', confirmKeyRegion)
      } else if (hotkey === 'discard') {
        if (['key_region', 'await_score', 'pending_replay'].includes(rlt.phase)) {
          void run('discard', () => discardKeyRegion('operator_discard'))
        }
      } else if (hotkey === 'void') {
        if (['key_region', 'await_score'].includes(rlt.phase) && rlt.active_key_region_id) {
          void run('void', () => voidKeyRegion(rlt.active_key_region_id as string, 'operator_void'))
        }
      }
    }
    window.addEventListener('keydown', onKeyDown, true)
    return () => window.removeEventListener('keydown', onKeyDown, true)
  }, [rlt.phase, rlt.active_key_region_id, pending])

  const phaseLabel = useMemo(() => {
    if (rlt.phase === 'key_region') return 'Recording key region'
    if (rlt.phase === 'await_score') return `Awaiting score ${countdown}`
    if (rlt.phase === 'pending_replay') return 'Review scored replay'
    return 'Idle'
  }, [rlt.phase, countdown])

  return (
    <section className="panel rlt-panel rlt-control-compact">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Key Region</p>
          <h2>{phaseLabel}</h2>
        </div>
        <span className={`status-pill ${rlt.phase === 'idle' ? 'mode' : 'live'}`}>{rlt.training_phase}</span>
      </div>

      <div className="rlt-control-grid">
        <button
          className={`rlt-key-button start ${flashKey === 'start' ? 'key-flash' : ''}`}
          type="button"
          disabled={rlt.phase !== 'idle' || !!pending}
          onClick={() => {
            flash('start')
            void run('start', startKeyRegion)
          }}
        >
          <span>←</span>
          <small>Start</small>
        </button>
        <button
          className={`rlt-key-button end ${flashKey === 'end' ? 'key-flash' : ''}`}
          type="button"
          disabled={rlt.phase !== 'key_region' || !!pending}
          onClick={() => {
            flash('end')
            void run('end', endKeyRegion)
          }}
        >
          <span>→</span>
          <small>End</small>
        </button>
        <button
          className={`rlt-key-button success ${flashKey === 'score1' ? 'key-flash' : ''}`}
          type="button"
          disabled={rlt.phase !== 'await_score' || !!pending}
          onClick={() => {
            flash('score1')
            void run('score1', () => scoreKeyRegion(1))
          }}
        >
          <span>↑</span>
          <small>Success 1</small>
        </button>
        <button
          className={`rlt-key-button fail ${flashKey === 'score0' ? 'key-flash' : ''}`}
          type="button"
          disabled={rlt.phase !== 'await_score' || !!pending}
          onClick={() => {
            flash('score0')
            void run('score0', () => scoreKeyRegion(0))
          }}
        >
          <span>↓</span>
          <small>Fail 0</small>
        </button>
      </div>

      {rlt.phase === 'pending_replay' ? (
        <div className="rlt-review-actions">
          <button
            className={`apply-button ${flashKey === 'confirm' ? 'key-flash' : ''}`}
            type="button"
            disabled={!!pending}
            onClick={() => {
              flash('confirm')
              void run('confirm', confirmKeyRegion)
            }}
          >
            Confirm
          </button>
          <button
            className={`apply-button danger ${flashKey === 'discard' ? 'key-flash' : ''}`}
            type="button"
            disabled={!!pending}
            onClick={() => {
              flash('discard')
              void run('discard', () => discardKeyRegion('operator_discard'))
            }}
          >
            Discard
          </button>
        </div>
      ) : null}
      {['key_region', 'await_score'].includes(rlt.phase) ? (
        <div className="rlt-review-actions">
          <button
            className={`apply-button danger ${flashKey === 'discard' ? 'key-flash' : ''}`}
            type="button"
            disabled={!!pending}
            onClick={() => {
              flash('discard')
              void run('discard', () => discardKeyRegion('operator_discard'))
            }}
          >
            Discard
          </button>
          <button
            className={`apply-button danger ${flashKey === 'void' ? 'key-flash' : ''}`}
            type="button"
            disabled={!!pending || !rlt.active_key_region_id}
            onClick={() => {
              if (!rlt.active_key_region_id) return
              flash('void')
              void run('void', () => voidKeyRegion(rlt.active_key_region_id as string, 'operator_void'))
            }}
          >
            Void
          </button>
        </div>
      ) : null}
      {rlt.phase === 'await_score' ? (
        <p className="rlt-warning">Choose the reward. Enter/C confirms, Backspace/D discards, V voids.</p>
      ) : null}
      {rlt.phase === 'pending_replay' ? (
        <p className="rlt-warning">Enter/C confirms. Backspace/D discards before replay is committed.</p>
      ) : null}
      {error ? <p className="rlt-error">{error}</p> : null}
    </section>
  )
}

export function RLTStatsPanel({ rlt }: { rlt: RLTControlState }) {
  const trainablePct = Math.min(100, (rlt.trainable_replay_count / Math.max(1, rlt.warmup_target)) * 100)
  return (
    <section className="panel rlt-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Collection</p>
          <h2>Trainable Replay {rlt.trainable_replay_count} / {rlt.warmup_target}</h2>
        </div>
        <span className={`status-pill ${rlt.actor_effective ? 'live' : 'offline'}`}>
          {rlt.actor_effective ? 'Actor active' : 'Actor locked'}
        </span>
      </div>
      <div className="progress-track">
        <div className="progress-fill" style={{ width: `${trainablePct}%` }} />
      </div>
      <div className="metric-grid">
        <Metric label="Trainable Episodes" value={rlt.trainable_replay_count} />
        <Metric label="Trainable Samples" value={rlt.trainable_replay_samples} />
        <Metric label="Trainable Success" value={rlt.trainable_replay_success} />
        <Metric label="Trainable Failure" value={rlt.trainable_replay_failure} />
        <Metric label="Trainable Shards" value={rlt.trainable_replay_shards} />
        <Metric label="Invalid Shards" value={rlt.invalid_replay_shards} />
        <Metric label="Attempts" value={rlt.warmup_attempts ?? 0} />
        <Metric label="Last Reward" value={rlt.last_reward ?? '-'} />
      </div>
      <div className="event-log latest-event-log">
        {rlt.events.slice(-1).reverse().map((event) => (
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
    intervention_scale: rlt.intervention_scale,
    max_delta: rlt.max_delta,
    actor_enabled: rlt.actor_enabled,
    critic_gate_enabled: rlt.critic_gate_enabled,
    critic_gate_margin: rlt.critic_gate_margin,
    critic_gate_temperature: rlt.critic_gate_temperature,
  })
  const [error, setError] = useState('')
  const [trainerPending, setTrainerPending] = useState(false)
  const [activeTab, setActiveTab] = useState<'settings' | 'readiness' | 'q' | 'actor' | 'inference'>('readiness')

  useEffect(() => {
    setDraft({
      warmup_target: rlt.warmup_target,
      intervention_scale: rlt.intervention_scale,
      max_delta: rlt.max_delta,
      actor_enabled: rlt.actor_enabled,
      critic_gate_enabled: rlt.critic_gate_enabled,
      critic_gate_margin: rlt.critic_gate_margin,
      critic_gate_temperature: rlt.critic_gate_temperature,
    })
  }, [
    rlt.warmup_target,
    rlt.intervention_scale,
    rlt.max_delta,
    rlt.actor_enabled,
    rlt.critic_gate_enabled,
    rlt.critic_gate_margin,
    rlt.critic_gate_temperature,
  ])

  const submit = async () => {
    setError('')
    try {
      onState(await updateRLTConfig(draft))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Config update failed')
    }
  }

  const setTrainerEnabled = async (enabled: boolean) => {
    setError('')
    setTrainerPending(true)
    try {
      onState(await updateRLTConfig({ trainer_enabled: enabled }))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Training control update failed')
    } finally {
      setTrainerPending(false)
    }
  }

  const metricsAge = metricAgeSeconds(rlt.rlt_metrics_timestamp)
  const trainerTone = metricsAge !== null && metricsAge <= 10 ? 'ok' : metricsAge !== null && metricsAge <= 30 ? 'watch' : 'danger'
  const qGapTone = rlt.q_gap !== null && rlt.q_gap > Math.max(1, Math.abs(rlt.target_q_mean ?? 0) * 0.5) ? 'danger' : undefined
  const actorDeltaTone = rlt.actor_delta_norm !== null && rlt.actor_delta_norm > Math.max(0.05, rlt.max_delta * 0.8) ? 'watch' : undefined
  const trainableBalanced = rlt.trainable_replay_success > 0 && rlt.trainable_replay_failure > 0
  const trainerLabel = rlt.trainer_running ? 'Training running' : rlt.trainer_enabled ? 'Training requested' : 'Training stopped'
  const inferenceTone = !rlt.critic_gate_enabled
    ? 'offline'
    : !rlt.critic_ready
      ? 'danger'
      : rlt.inference_actor_active
        ? 'live'
        : 'watch'
  const inferenceLabel = !rlt.critic_gate_enabled
    ? 'Critic gate off'
    : !rlt.critic_ready
      ? 'Critic not loaded'
      : rlt.inference_actor_active
        ? 'Actor taking over'
        : 'Watching region'

  return (
    <section className="panel rlt-panel training-panel">
      <div className="panel-header">
        <div>
          <p className="eyebrow">Actor Critic</p>
          <h2>{trainerLabel}</h2>
          <span className={`status-pill ${rlt.trainer_running ? 'live' : rlt.trainer_enabled ? 'mode' : ''}`}>
            {rlt.trainer_running ? 'running' : rlt.trainer_enabled ? 'waiting' : 'manual'}
          </span>
        </div>
        <div className="training-actions">
          <button
            className="apply-button"
            type="button"
            disabled={trainerPending || rlt.trainer_enabled}
            onClick={() => void setTrainerEnabled(true)}
          >
            Start training
          </button>
          <button
            className="secondary-button danger-button"
            type="button"
            disabled={trainerPending || !rlt.trainer_enabled}
            onClick={() => void setTrainerEnabled(false)}
          >
            Stop training
          </button>
          <button className="secondary-button" type="button" onClick={() => void submit()}>
            Apply
          </button>
        </div>
      </div>

      <div className="training-tabs" role="tablist" aria-label="Actor critic metrics">
        <button className={activeTab === 'readiness' ? 'active' : ''} type="button" onClick={() => setActiveTab('readiness')}>
          Readiness
        </button>
        <button className={activeTab === 'q' ? 'active' : ''} type="button" onClick={() => setActiveTab('q')}>
          Q Network
        </button>
        <button className={activeTab === 'actor' ? 'active' : ''} type="button" onClick={() => setActiveTab('actor')}>
          Actor
        </button>
        <button className={activeTab === 'inference' ? 'active' : ''} type="button" onClick={() => setActiveTab('inference')}>
          Inference
        </button>
        <button className={activeTab === 'settings' ? 'active' : ''} type="button" onClick={() => setActiveTab('settings')}>
          Settings
        </button>
      </div>

      {activeTab === 'settings' ? (
        <div className="training-tab-panel settings-grid">
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
              disabled={rlt.trainable_replay_count < rlt.warmup_target}
              onChange={(event) => setDraft({ ...draft, actor_enabled: event.target.checked })}
            />
            <span>Enable Actor</span>
          </label>
          <label className="rlt-toggle">
            <input
              type="checkbox"
              checked={draft.critic_gate_enabled}
              onChange={(event) => setDraft({ ...draft, critic_gate_enabled: event.target.checked })}
            />
            <span>Critic Gate</span>
          </label>
          <label className="rlt-field">
            <span>Q Advantage Margin</span>
            <input
              type="number"
              step={0.01}
              value={draft.critic_gate_margin}
              onChange={(event) => setDraft({ ...draft, critic_gate_margin: Number(event.target.value) })}
            />
          </label>
          <label className="rlt-field">
            <span>Gate Temperature</span>
            <input
              type="number"
              min={0.001}
              step={0.001}
              value={draft.critic_gate_temperature}
              onChange={(event) => setDraft({ ...draft, critic_gate_temperature: Number(event.target.value) })}
            />
          </label>
          {rlt.wandb_url ? (
            <a className="wandb-link" href={rlt.wandb_url} target="_blank" rel="noreferrer">
              Open wandb run
            </a>
          ) : null}
        </div>
      ) : null}

      {activeTab === 'readiness' ? (
        <div className="training-tab-panel">
          <div className="metric-grid">
            <Metric label="Trainer Age" value={formatAge(metricsAge)} tone={trainerTone} />
            <Metric label="Trainer Step" value={formatOptionalInt(rlt.trainer_step)} />
            <Metric label="Training Requested" value={formatBool(rlt.trainer_enabled)} tone={rlt.trainer_enabled ? 'ok' : 'watch'} />
            <Metric label="Training Running" value={formatBool(rlt.trainer_running)} tone={rlt.trainer_running ? 'ok' : undefined} />
            <Metric label="Trainable" value={`${rlt.trainable_replay_count} / ${rlt.warmup_target}`} tone={rlt.trainable_replay_count >= rlt.warmup_target ? 'ok' : 'watch'} />
            <Metric label="Balance" value={trainableBalanced ? 'ready' : 'need both'} tone={trainableBalanced ? 'ok' : 'watch'} />
            <Metric label="Success" value={rlt.trainable_replay_success} />
            <Metric label="Failure" value={rlt.trainable_replay_failure} />
            <Metric label="Replay Samples" value={formatOptionalInt(rlt.replay_size ?? rlt.trainable_replay_samples)} />
            <Metric label="Bad Shards" value={formatOptionalInt(rlt.bad_shards ?? rlt.invalid_replay_shards)} tone={(rlt.bad_shards ?? rlt.invalid_replay_shards) > 0 ? 'watch' : undefined} />
          </div>
          {rlt.actor_locked_reason ? <p className="rlt-warning">Actor locked: {rlt.actor_locked_reason}</p> : null}
        </div>
      ) : null}

      {activeTab === 'q' ? (
        <div className="training-tab-panel">
          <div className="metric-grid">
            <Metric label="Critic Loss" value={formatMetric(rlt.critic_loss)} />
            <Metric label="Q Gap" value={formatMetric(rlt.q_gap)} tone={qGapTone} />
            <Metric label="Q1 Loss" value={formatMetric(rlt.critic_q1_loss)} />
            <Metric label="Q2 Loss" value={formatMetric(rlt.critic_q2_loss)} />
            <Metric label="Q1 Mean" value={formatMetric(rlt.q1_mean)} />
            <Metric label="Q2 Mean" value={formatMetric(rlt.q2_mean)} />
            <Metric label="Target Q" value={formatMetric(rlt.target_q_mean)} />
            <Metric label="Reference Q" value={formatMetric(rlt.reference_q_value)} />
            <Metric label="Q Advantage" value={formatMetric(rlt.q_advantage)} tone={rlt.q_advantage !== null && rlt.q_advantage < 0 ? 'watch' : undefined} />
            <Metric label="Steps / Sec" value={formatMetric(rlt.steps_per_sec)} tone={rlt.steps_per_sec !== null && rlt.steps_per_sec < 0.2 ? 'watch' : undefined} />
          </div>
        </div>
      ) : null}

      {activeTab === 'actor' ? (
        <div className="training-tab-panel">
          <div className="metric-grid">
            <Metric label="Auto Beta" value={formatMetric(rlt.beta)} tone={rlt.auto_beta_enabled ? 'ok' : 'watch'} />
            <Metric label="Beta Reason" value={rlt.auto_beta_reason || '-'} />
            <Metric label="Target Delta" value={formatMetric(rlt.auto_beta_target_delta_norm)} />
            <Metric label="Delta EMA" value={formatMetric(rlt.auto_beta_delta_norm_ema)} tone={actorDeltaTone} />
            <Metric label="Q Advantage EMA" value={formatMetric(rlt.auto_beta_q_advantage_ema)} tone={rlt.auto_beta_q_advantage_ema !== null && rlt.auto_beta_q_advantage_ema < 0 ? 'watch' : undefined} />
            <Metric label="Critic Loss EMA" value={formatMetric(rlt.auto_beta_critic_loss_ema)} />
            <Metric label="Actor Loss" value={formatMetric(rlt.actor_loss)} />
            <Metric label="Actor Q" value={formatMetric(rlt.actor_q_value)} />
            <Metric label="Delta Norm" value={formatMetric(rlt.actor_delta_norm)} tone={actorDeltaTone} />
            <Metric label="Actor Updated" value={formatBool(rlt.actor_updated)} />
            <Metric label="Publish Actor" value={formatBool(rlt.publish_actor)} />
            <Metric label="Published Step" value={formatOptionalInt(rlt.actor_checkpoint_step)} />
            <Metric label="Replay Horizon" value={formatOptionalInt(rlt.replay_action_horizon)} />
            <Metric label="Train Horizon" value={formatOptionalInt(rlt.train_action_horizon)} />
          </div>
        </div>
      ) : null}

      {activeTab === 'inference' ? (
        <div className="training-tab-panel">
          <div className={`inference-status inference-${inferenceTone}`}>
            <strong>{inferenceLabel}</strong>
            <span>{rlt.inference_gate_reason || '-'}</span>
          </div>
          <div className="metric-grid inference-metrics">
            <Metric label="Inference Actor Active" value={formatBool(rlt.inference_actor_active)} tone={rlt.inference_actor_active ? 'ok' : undefined} />
            <Metric label="Inference Delta Norm" value={formatMetric(rlt.inference_delta_norm)} />
            <Metric label="Inference Gate Reason" value={rlt.inference_gate_reason || '-'} />
            <Metric label="Key Region Probability" value={formatProbability(rlt.key_region_probability)} tone={rlt.key_region_probability !== null && rlt.key_region_probability >= 0.5 ? 'ok' : undefined} />
            <Metric label="Loaded Actor Step" value={formatOptionalInt(rlt.loaded_actor_step)} />
            <Metric label="Critic Ready" value={formatBool(rlt.critic_ready)} tone={rlt.critic_ready ? 'ok' : 'danger'} />
            <Metric label="Inference Actor Q" value={formatMetric(rlt.inference_actor_q_value)} />
            <Metric label="Inference Reference Q" value={formatMetric(rlt.inference_reference_q_value)} />
            <Metric label="Inference Q Advantage" value={formatMetric(rlt.inference_q_advantage)} tone={rlt.inference_q_advantage !== null && rlt.inference_q_advantage >= rlt.critic_gate_margin ? 'ok' : 'watch'} />
            <Metric label="Critic Gate" value={rlt.critic_gate_enabled ? 'on' : 'off'} tone={rlt.critic_gate_enabled ? 'ok' : undefined} />
          </div>
        </div>
      ) : null}

      {error ? <p className="rlt-error">{error}</p> : null}
    </section>
  )
}

function Metric({ label, value, tone }: { label: string; value: number | string; tone?: 'ok' | 'watch' | 'danger' }) {
  return (
    <div className={`metric-tile ${tone ? `metric-${tone}` : ''}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}

function formatMetric(value: number | null | undefined) {
  return value === null || value === undefined || !Number.isFinite(value) ? '-' : value.toFixed(4)
}

function formatOptionalInt(value: number | null | undefined) {
  return value === null || value === undefined || !Number.isFinite(value) ? '-' : Math.round(value).toString()
}

function formatBool(value: boolean | null | undefined) {
  if (value === null || value === undefined) return '-'
  return value ? 'yes' : 'no'
}

function formatProbability(value: number | null | undefined) {
  if (value === null || value === undefined || !Number.isFinite(value)) return '-'
  return `${(value * 100).toFixed(1)}%`
}

function metricAgeSeconds(timestamp: number | null | undefined) {
  if (!timestamp) return null
  return Math.max(0, Date.now() / 1000 - timestamp)
}

function formatAge(age: number | null) {
  if (age === null) return 'no data'
  if (age < 60) return `${age.toFixed(1)}s`
  return `${Math.floor(age / 60)}m ${Math.floor(age % 60)}s`
}
