import { fetchRLTCriticReport, RLTCriticReportSummary, RLTControlState, updateRLTConfig } from '../services/api'
import type { ReactNode } from 'react'
import { useEffect, useState } from 'react'

type Props = {
  rlt: RLTControlState
  onState: (state: RLTControlState) => void
  wsConnected: boolean
  cameraStatus: Record<string, boolean>
  cameraTimestamps: Record<string, number | null>
}

export function SystemPage({ rlt, onState, wsConnected, cameraStatus, cameraTimestamps }: Props) {
  const [burnInDraft, setBurnInDraft] = useState(rlt.critic_burn_in_steps ?? 1000)
  const [burnInError, setBurnInError] = useState('')
  const [burnInPending, setBurnInPending] = useState(false)
  const [criticReport, setCriticReport] = useState<RLTCriticReportSummary | null>(null)
  const [criticReportError, setCriticReportError] = useState('')
  const metricsAge = ageSeconds(rlt.rlt_metrics_timestamp)
  const liveCameras = Object.values(cameraStatus).filter(Boolean).length
  const knownCameras = Object.keys(cameraStatus).length
  const actorBurnInRemaining =
    rlt.critic_burn_in_steps === null || rlt.trainer_step === null
      ? null
      : Math.max(0, rlt.critic_burn_in_steps - rlt.trainer_step)
  const burnInComplete = actorBurnInRemaining !== null && actorBurnInRemaining === 0

  useEffect(() => {
    if (rlt.critic_burn_in_steps !== null) {
      setBurnInDraft(rlt.critic_burn_in_steps)
    }
  }, [rlt.critic_burn_in_steps])

  useEffect(() => {
    let cancelled = false
    fetchRLTCriticReport()
      .then((report) => {
        if (!cancelled) {
          setCriticReport(report)
          setCriticReportError('')
        }
      })
      .catch((exc) => {
        if (!cancelled) setCriticReportError(exc instanceof Error ? exc.message : 'Critic report unavailable')
      })
    return () => {
      cancelled = true
    }
  }, [rlt.online_round_index, rlt.online_rejection_reason, rlt.online_best_critic_auc])

  const applyBurnIn = async () => {
    setBurnInError('')
    setBurnInPending(true)
    try {
      onState(await updateRLTConfig({ critic_burn_in_steps: burnInDraft }))
    } catch (exc) {
      setBurnInError(exc instanceof Error ? exc.message : 'Burn-in update failed')
    } finally {
      setBurnInPending(false)
    }
  }

  return (
    <section className="page-panel system-page">
      <div className="panel-header">
        <div>
          <p className="eyebrow">System</p>
          <h2>Runtime Status</h2>
        </div>
        <span className={`status-pill ${wsConnected ? 'live' : 'offline'}`}>
          {wsConnected ? 'websocket subscribed' : 'websocket offline'}
        </span>
      </div>

      <div className="system-grid">
        <SystemSection title="Training">
          <Metric label="Trainer Requested" value={formatBool(rlt.trainer_enabled)} tone={rlt.trainer_enabled ? 'ok' : 'watch'} />
          <Metric label="Trainer Running" value={formatBool(rlt.trainer_running)} tone={rlt.trainer_running ? 'ok' : undefined} />
          <Metric label="Trainer Step" value={formatInt(rlt.trainer_step)} />
          <Metric label="Steps / Sec" value={formatMetric(rlt.steps_per_sec)} />
          <Metric label="Metrics Age" value={formatAge(metricsAge)} tone={metricsAge !== null && metricsAge <= 10 ? 'ok' : 'watch'} />
          <Metric label="Wandb" value={rlt.wandb_url ? 'linked' : '-'} tone={rlt.wandb_url ? 'ok' : undefined} />
        </SystemSection>

        <SystemSection title="Replay">
          <Metric label="Trainable Count" value={`${rlt.trainable_replay_count} / ${rlt.warmup_target}`} />
          <Metric label="Trainable Samples" value={rlt.trainable_replay_samples} />
          <Metric label="Success Episodes" value={rlt.trainable_replay_success} />
          <Metric label="Failure Episodes" value={rlt.trainable_replay_failure} />
          <Metric label="Replay Shards" value={formatInt(rlt.replay_shards ?? rlt.trainable_replay_shards)} />
          <Metric label="Bad Shards" value={formatInt(rlt.bad_shards ?? rlt.invalid_replay_shards)} tone={(rlt.bad_shards ?? rlt.invalid_replay_shards) > 0 ? 'watch' : undefined} />
        </SystemSection>

        <SystemSection title="Actor Critic">
          <Metric label="Actor Enabled" value={formatBool(rlt.actor_enabled)} />
          <Metric label="Actor Ready" value={formatBool(rlt.actor_ready)} tone={rlt.actor_ready ? 'ok' : 'watch'} />
          <Metric label="Actor Effective" value={formatBool(rlt.actor_effective)} tone={rlt.actor_effective ? 'ok' : 'watch'} />
          <Metric label="Critic Burn-in" value={formatInt(rlt.critic_burn_in_steps)} />
          <Metric label="Actor Wait" value={formatInt(actorBurnInRemaining)} tone={burnInComplete ? 'ok' : 'watch'} />
          <Metric label="Target Sync Step" value={formatInt(rlt.target_sync_step)} tone={rlt.target_sync_step !== null ? 'ok' : 'watch'} />
          <Metric label="Critic Ready" value={formatBool(rlt.critic_ready)} tone={rlt.critic_ready ? 'ok' : 'watch'} />
          <Metric label="Critic Gate" value={rlt.critic_gate_enabled ? 'on' : 'off'} />
          <Metric label="Gate Reason" value={rlt.inference_gate_reason || rlt.actor_locked_reason || '-'} />
          <label className="rlt-field">
            <span>Critic Burn-in Steps</span>
            <input
              type="number"
              min={0}
              max={1000000}
              value={burnInDraft}
              onChange={(event) => setBurnInDraft(Number(event.target.value))}
            />
          </label>
          <button className="apply-button" type="button" disabled={burnInPending} onClick={() => void applyBurnIn()}>
            Apply burn-in
          </button>
          <p className="rlt-warning">
            Critic trains first. At the first actor update, online actor/critic are hard-copied to target networks, then
            target networks continue with slow Polyak updates.
          </p>
          {burnInError ? <p className="rlt-error">{burnInError}</p> : null}
        </SystemSection>

        <SystemSection title="Q Network">
          <Metric label="Critic Loss" value={formatMetric(rlt.critic_loss)} />
          <Metric label="Q1 Loss" value={formatMetric(rlt.critic_q1_loss)} />
          <Metric label="Q2 Loss" value={formatMetric(rlt.critic_q2_loss)} />
          <Metric label="Q Gap" value={formatMetric(rlt.q_gap)} />
          <Metric label="Q1 Mean" value={formatMetric(rlt.q1_mean)} />
          <Metric label="Q2 Mean" value={formatMetric(rlt.q2_mean)} />
          <Metric label="Target Q" value={formatMetric(rlt.target_q_mean)} />
          <Metric label="Reference Q" value={formatMetric(rlt.reference_q_value)} />
          <Metric label="Actor Q" value={formatMetric(rlt.actor_q_value)} />
          <Metric label="Q Advantage" value={formatMetric(rlt.q_advantage)} tone={rlt.q_advantage !== null && rlt.q_advantage < 0 ? 'watch' : undefined} />
        </SystemSection>

        <SystemSection title="Critic Report" wide>
          {criticReport?.exists ? (
            <>
              <Metric label="Round" value={criticReport.round_id ?? '-'} />
              <Metric label="Step" value={formatInt(criticReport.step)} />
              <Metric label="AUC" value={formatMetric(criticReport.auc)} tone={criticReport.auc !== null && criticReport.auc >= rlt.online_critic_auc_min ? 'ok' : 'watch'} />
              <Metric label="Q Gap" value={formatMetric(criticReport.q_gap)} tone={criticReport.q_gap !== null && criticReport.q_gap > 0 ? 'ok' : 'watch'} />
              <Metric label="Usable" value={formatBool(criticReport.is_critic_usable)} tone={criticReport.is_critic_usable ? 'ok' : 'danger'} />
              <Metric label="Success Q" value={formatMetric(criticReport.success_q_mean)} />
              <Metric label="Failure Q" value={formatMetric(criticReport.failure_q_mean)} />
              <Metric label="Bellman Loss" value={formatMetric(criticReport.holdout_bellman_loss)} />
              <PathRow label="Warning" value={criticReport.warning_reason || '-'} />
              <PathRow label="Report" value={criticReport.report_path} />
            </>
          ) : (
            <p className="rlt-warning">{criticReportError || 'No critic holdout report found.'}</p>
          )}
        </SystemSection>

        <SystemSection title="Beta">
          <Metric label="Mode" value={rlt.auto_beta_enabled ? 'auto' : 'manual'} tone={rlt.auto_beta_enabled ? 'ok' : 'watch'} />
          <Metric label="Beta" value={formatMetric(rlt.beta)} />
          <Metric label="Target Delta" value={formatMetric(rlt.auto_beta_target_delta_norm)} />
          <Metric label="Delta EMA" value={formatMetric(rlt.auto_beta_delta_norm_ema)} />
          <Metric label="Q Adv EMA" value={formatMetric(rlt.auto_beta_q_advantage_ema)} />
          <Metric label="Reason" value={rlt.auto_beta_reason || '-'} />
        </SystemSection>

        <SystemSection title="Inference">
          <Metric label="Actor Active" value={formatBool(rlt.inference_actor_active)} tone={rlt.inference_actor_active ? 'ok' : undefined} />
          <Metric label="Loaded Actor Step" value={formatInt(rlt.loaded_actor_step)} />
          <Metric label="Inference Delta" value={formatMetric(rlt.inference_delta_norm)} />
          <Metric label="Key Region Prob" value={formatProbability(rlt.key_region_probability)} />
          <Metric label="Actor Q" value={formatMetric(rlt.inference_actor_q_value)} />
          <Metric label="Reference Q" value={formatMetric(rlt.inference_reference_q_value)} />
        </SystemSection>

        <SystemSection title="Cameras">
          <Metric label="Live Cameras" value={`${liveCameras} / ${knownCameras}`} tone={liveCameras === knownCameras && knownCameras > 0 ? 'ok' : 'watch'} />
          {Object.entries(cameraTimestamps).map(([name, timestamp]) => (
            <Metric key={name} label={name} value={formatAge(ageSeconds(timestamp))} tone={cameraStatus[name] ? 'ok' : 'watch'} />
          ))}
        </SystemSection>

        <SystemSection title="Checkpoints" wide>
          <PathRow label="RL Token" value={rlt.rl_token_checkpoint_path} />
          <PathRow label="Actor" value={rlt.actor_checkpoint_path} />
          <PathRow label="Actor Step" value={formatInt(rlt.actor_checkpoint_step)} />
        </SystemSection>

        <SystemSection title="Latest Events" wide>
          <div className="system-events">
            {rlt.events.slice(-8).reverse().map((event) => (
              <div key={`${event.timestamp}-${event.event}`} className="event-row">
                <span>{new Date(event.timestamp * 1000).toLocaleTimeString()}</span>
                <strong>{event.event}</strong>
                <small>{event.detail}</small>
              </div>
            ))}
            {rlt.events.length === 0 ? <p className="rlt-warning">No RLT events yet.</p> : null}
          </div>
        </SystemSection>
      </div>
    </section>
  )
}

function SystemSection({ title, wide, children }: { title: string; wide?: boolean; children: ReactNode }) {
  return (
    <section className={`config-section ${wide ? 'wide' : ''}`}>
      <div className="section-head">
        <h3>{title}</h3>
      </div>
      <div className="metric-grid">{children}</div>
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

function PathRow({ label, value }: { label: string; value: number | string | null }) {
  return (
    <div className="path-row">
      <span>{label}</span>
      <strong>{value ?? '-'}</strong>
    </div>
  )
}

function formatMetric(value: number | null | undefined) {
  return value === null || value === undefined || !Number.isFinite(value) ? '-' : value.toFixed(4)
}

function formatInt(value: number | null | undefined) {
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

function ageSeconds(timestamp: number | null | undefined) {
  if (!timestamp) return null
  return Math.max(0, Date.now() / 1000 - timestamp)
}

function formatAge(age: number | null) {
  if (age === null) return 'no data'
  if (age < 60) return `${age.toFixed(1)}s`
  return `${Math.floor(age / 60)}m ${Math.floor(age % 60)}s`
}
