import { useEffect, useState } from 'react'
import type { ReactNode } from 'react'
import { RLTConfigRequest, RLTControlState, updateRLTConfig } from '../services/api'

type Props = {
  rlt: RLTControlState
  onState: (state: RLTControlState) => void
}

type Draft = Required<
  Pick<
    RLTConfigRequest,
    | 'warmup_target'
    | 'beta'
    | 'auto_beta_enabled'
    | 'auto_beta_target_delta_norm'
    | 'auto_beta_min'
    | 'auto_beta_max'
    | 'auto_beta_lr'
    | 'auto_beta_ema_decay'
    | 'auto_beta_update_interval'
    | 'auto_beta_q_margin'
    | 'critic_burn_in_steps'
    | 'actor_enabled'
    | 'trainer_enabled'
    | 'intervention_scale'
    | 'max_delta'
    | 'rlt_blend_mode'
    | 'rlt_blend_preset'
    | 'lambda_push'
    | 'lambda_vla_align'
    | 'lambda_actor'
    | 'push_joint_indices'
    | 'push_axis'
    | 'critic_gate_enabled'
    | 'critic_gate_margin'
    | 'critic_gate_temperature'
  >
> & { wandb_url: string }

const blendPresets: Record<string, Pick<Draft, 'lambda_push' | 'lambda_vla_align' | 'lambda_actor'>> = {
  conservative: { lambda_push: 0.1, lambda_vla_align: 0.5, lambda_actor: 0.2 },
  align: { lambda_push: 0.2, lambda_vla_align: 0.3, lambda_actor: 0.5 },
  actor_align: { lambda_push: 0.15, lambda_vla_align: 0.1, lambda_actor: 0.7 },
  insert: { lambda_push: 0.8, lambda_vla_align: 0.7, lambda_actor: 0.1 },
}

const parseNumberList = (value: string) =>
  value
    .split(',')
    .map((item) => Number(item.trim()))
    .filter((item) => Number.isFinite(item))

const makeDraft = (rlt: RLTControlState): Draft => ({
  warmup_target: rlt.warmup_target,
  beta: rlt.beta,
  auto_beta_enabled: rlt.auto_beta_enabled,
  auto_beta_target_delta_norm: rlt.auto_beta_target_delta_norm ?? 0.06,
  auto_beta_min: rlt.auto_beta_min,
  auto_beta_max: rlt.auto_beta_max,
  auto_beta_lr: rlt.auto_beta_lr,
  auto_beta_ema_decay: rlt.auto_beta_ema_decay,
  auto_beta_update_interval: rlt.auto_beta_update_interval,
  auto_beta_q_margin: rlt.auto_beta_q_margin,
  critic_burn_in_steps: rlt.critic_burn_in_steps ?? 1000,
  actor_enabled: rlt.actor_enabled,
  trainer_enabled: rlt.trainer_enabled,
  intervention_scale: rlt.intervention_scale,
  max_delta: rlt.max_delta,
  rlt_blend_mode: rlt.rlt_blend_mode ?? 'projected_slow_push',
  rlt_blend_preset: rlt.rlt_blend_preset ?? 'conservative',
  lambda_push: rlt.lambda_push ?? 0.1,
  lambda_vla_align: rlt.lambda_vla_align ?? 0.5,
  lambda_actor: rlt.lambda_actor ?? 0.2,
  push_joint_indices: rlt.push_joint_indices ?? [0, 1, 2, 3, 4, 5],
  push_axis: rlt.push_axis ?? [-0.53, 0.2, -0.78, 0.23, -0.08, 0.06],
  critic_gate_enabled: rlt.critic_gate_enabled,
  critic_gate_margin: rlt.critic_gate_margin,
  critic_gate_temperature: rlt.critic_gate_temperature,
  wandb_url: rlt.wandb_url ?? '',
})

export function RLTConfigPage({ rlt, onState }: Props) {
  const [draft, setDraft] = useState<Draft>(() => makeDraft(rlt))
  const [error, setError] = useState('')
  const [pending, setPending] = useState(false)

  useEffect(() => {
    setDraft(makeDraft(rlt))
  }, [
    rlt.warmup_target,
    rlt.beta,
    rlt.auto_beta_enabled,
    rlt.auto_beta_target_delta_norm,
    rlt.auto_beta_min,
    rlt.auto_beta_max,
    rlt.auto_beta_lr,
    rlt.auto_beta_ema_decay,
    rlt.auto_beta_update_interval,
    rlt.auto_beta_q_margin,
    rlt.critic_burn_in_steps,
    rlt.actor_enabled,
    rlt.trainer_enabled,
    rlt.intervention_scale,
    rlt.max_delta,
    rlt.rlt_blend_mode,
    rlt.rlt_blend_preset,
    rlt.lambda_push,
    rlt.lambda_vla_align,
    rlt.lambda_actor,
    rlt.push_joint_indices,
    rlt.push_axis,
    rlt.critic_gate_enabled,
    rlt.critic_gate_margin,
    rlt.critic_gate_temperature,
    rlt.wandb_url,
  ])

  const apply = async (patch: RLTConfigRequest = draft) => {
    setError('')
    setPending(true)
    try {
      onState(await updateRLTConfig({ ...patch, wandb_url: patch.wandb_url === '' ? null : patch.wandb_url }))
    } catch (exc) {
      setError(exc instanceof Error ? exc.message : 'Config update failed')
    } finally {
      setPending(false)
    }
  }

  return (
    <section className="page-panel config-page">
      <div className="panel-header">
        <div>
          <p className="eyebrow">RLT Config</p>
          <h2>Runtime Parameters</h2>
        </div>
        <div className="training-actions">
          <button
            className="apply-button"
            type="button"
            disabled={pending || rlt.trainer_enabled}
            onClick={() => void apply({ trainer_enabled: true })}
          >
            Start training
          </button>
          <button
            className="secondary-button danger-button"
            type="button"
            disabled={pending || !rlt.trainer_enabled}
            onClick={() => void apply({ trainer_enabled: false })}
          >
            Stop training
          </button>
          <button className="secondary-button" type="button" disabled={pending} onClick={() => void apply()}>
            Apply
          </button>
        </div>
      </div>

      <div className="config-grid">
        <ConfigSection title="Training">
          <Field label="Warmup Target">
            <input
              type="number"
              min={1}
              value={draft.warmup_target}
              onChange={(event) => setDraft({ ...draft, warmup_target: Number(event.target.value) })}
            />
          </Field>
          <Toggle
            label="Trainer Requested"
            checked={draft.trainer_enabled}
            onChange={(trainer_enabled) => setDraft({ ...draft, trainer_enabled })}
          />
          <Toggle
            label="Actor Enabled"
            checked={draft.actor_enabled}
            onChange={(actor_enabled) => setDraft({ ...draft, actor_enabled })}
          />
          <Field label="Critic Burn-in Steps">
            <input
              type="number"
              min={0}
              max={1000000}
              value={draft.critic_burn_in_steps}
              onChange={(event) => setDraft({ ...draft, critic_burn_in_steps: Number(event.target.value) })}
            />
          </Field>
          <Field label="Wandb URL">
            <input
              type="text"
              value={draft.wandb_url}
              onChange={(event) => setDraft({ ...draft, wandb_url: event.target.value })}
            />
          </Field>
        </ConfigSection>

        <ConfigSection title="Beta Control">
          <Field label="Beta Mode">
            <select
              value={draft.auto_beta_enabled ? 'auto' : 'manual'}
              onChange={(event) => setDraft({ ...draft, auto_beta_enabled: event.target.value === 'auto' })}
            >
              <option value="auto">Auto beta</option>
              <option value="manual">Manual beta</option>
            </select>
          </Field>
          <Field label="Manual Beta">
            <input
              type="number"
              min={0}
              step={0.1}
              disabled={draft.auto_beta_enabled}
              value={draft.beta}
              onChange={(event) => setDraft({ ...draft, beta: Number(event.target.value) })}
            />
          </Field>
          <Field label="Target Delta Norm">
            <input
              type="number"
              min={0.001}
              step={0.001}
              value={draft.auto_beta_target_delta_norm}
              onChange={(event) => setDraft({ ...draft, auto_beta_target_delta_norm: Number(event.target.value) })}
            />
          </Field>
          <Field label="Auto Beta Min">
            <input
              type="number"
              min={0.001}
              step={0.1}
              value={draft.auto_beta_min}
              onChange={(event) => setDraft({ ...draft, auto_beta_min: Number(event.target.value) })}
            />
          </Field>
          <Field label="Auto Beta Max">
            <input
              type="number"
              min={0.001}
              step={0.1}
              value={draft.auto_beta_max}
              onChange={(event) => setDraft({ ...draft, auto_beta_max: Number(event.target.value) })}
            />
          </Field>
          <Field label="Auto Beta LR">
            <input
              type="number"
              min={0}
              step={0.001}
              value={draft.auto_beta_lr}
              onChange={(event) => setDraft({ ...draft, auto_beta_lr: Number(event.target.value) })}
            />
          </Field>
          <Field label="EMA Decay">
            <input
              type="number"
              min={0}
              max={0.999}
              step={0.01}
              value={draft.auto_beta_ema_decay}
              onChange={(event) => setDraft({ ...draft, auto_beta_ema_decay: Number(event.target.value) })}
            />
          </Field>
          <Field label="Update Interval">
            <input
              type="number"
              min={1}
              value={draft.auto_beta_update_interval}
              onChange={(event) => setDraft({ ...draft, auto_beta_update_interval: Number(event.target.value) })}
            />
          </Field>
          <Field label="Q Margin">
            <input
              type="number"
              step={0.001}
              value={draft.auto_beta_q_margin}
              onChange={(event) => setDraft({ ...draft, auto_beta_q_margin: Number(event.target.value) })}
            />
          </Field>
        </ConfigSection>

        <ConfigSection title="Actor Intervention">
          <Field label={`Intervention Scale ${draft.intervention_scale.toFixed(2)}`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.01}
              value={draft.intervention_scale}
              onChange={(event) => setDraft({ ...draft, intervention_scale: Number(event.target.value) })}
            />
          </Field>
          <Field label="Max Delta">
            <input
              type="number"
              min={0}
              step={0.001}
              value={draft.max_delta}
              onChange={(event) => setDraft({ ...draft, max_delta: Number(event.target.value) })}
            />
          </Field>
          <Toggle
            label="Critic Gate"
            checked={draft.critic_gate_enabled}
            onChange={(critic_gate_enabled) => setDraft({ ...draft, critic_gate_enabled })}
          />
          <Field label="Critic Gate Margin">
            <input
              type="number"
              step={0.001}
              value={draft.critic_gate_margin}
              onChange={(event) => setDraft({ ...draft, critic_gate_margin: Number(event.target.value) })}
            />
          </Field>
          <Field label="Gate Temperature">
            <input
              type="number"
              min={0.001}
              step={0.001}
              value={draft.critic_gate_temperature}
              onChange={(event) => setDraft({ ...draft, critic_gate_temperature: Number(event.target.value) })}
            />
          </Field>
        </ConfigSection>

        <ConfigSection title="Actor Blend">
          <Field label="Blend Mode">
            <select
              value={draft.rlt_blend_mode}
              onChange={(event) => setDraft({ ...draft, rlt_blend_mode: event.target.value })}
            >
              <option value="projected_slow_push">Projected slow push</option>
              <option value="full">Full actor action</option>
            </select>
          </Field>
          <Field label="Blend Preset">
            <select
              value={draft.rlt_blend_preset}
              onChange={(event) => {
                const preset = event.target.value
                setDraft({ ...draft, rlt_blend_preset: preset, ...(blendPresets[preset] ?? {}) })
              }}
            >
              <option value="conservative">Conservative</option>
              <option value="align">Align</option>
              <option value="actor_align">Actor Align</option>
              <option value="insert">Insert</option>
              <option value="custom">Custom</option>
            </select>
          </Field>
          <Field label="Lambda Push">
            <input
              type="number"
              min={0}
              max={2}
              step={0.01}
              value={draft.lambda_push}
              onChange={(event) =>
                setDraft({ ...draft, rlt_blend_preset: 'custom', lambda_push: Number(event.target.value) })
              }
            />
          </Field>
          <Field label="Lambda VLA Align">
            <input
              type="number"
              min={0}
              max={2}
              step={0.01}
              value={draft.lambda_vla_align}
              onChange={(event) =>
                setDraft({ ...draft, rlt_blend_preset: 'custom', lambda_vla_align: Number(event.target.value) })
              }
            />
          </Field>
          <Field label="Lambda Actor">
            <input
              type="number"
              min={0}
              max={2}
              step={0.01}
              value={draft.lambda_actor}
              onChange={(event) =>
                setDraft({ ...draft, rlt_blend_preset: 'custom', lambda_actor: Number(event.target.value) })
              }
            />
          </Field>
          <Field label="Push Joint Indices">
            <input
              type="text"
              value={draft.push_joint_indices.join(', ')}
              onChange={(event) =>
                setDraft({ ...draft, rlt_blend_preset: 'custom', push_joint_indices: parseNumberList(event.target.value) })
              }
            />
          </Field>
          <Field label="Push Axis">
            <input
              type="text"
              value={draft.push_axis.join(', ')}
              onChange={(event) =>
                setDraft({ ...draft, rlt_blend_preset: 'custom', push_axis: parseNumberList(event.target.value) })
              }
            />
          </Field>
        </ConfigSection>

        <ConfigSection title="Annotation">
          <ReadOnly label="Key Region Reward" value="Only binary reward is stored: 1=usable success, 0=usable failure" />
          <ReadOnly label="Crop Save" value="Save crop for Q writes the selected replay range and preserves the binary reward" />
          <ReadOnly label="Preference Learning" value="RLHF labels are stored as pairwise comparisons, separate from replay reward" />
          <ReadOnly label="Reward Model Target" value="left/right/tie/both_bad labels can train a Bradley-Terry style reward model later" />
        </ConfigSection>

        <ConfigSection title="Restart Required">
          <ReadOnly label="Batch Size" value="RLT_BATCH_SIZE=64" />
          <ReadOnly label="Actor LR" value="RLT_ACTOR_LR=1e-4" />
          <ReadOnly label="Critic LR" value="RLT_CRITIC_LR=3e-4" />
          <ReadOnly label="Min Replay Samples" value="RLT_MIN_REPLAY_SAMPLES=2048" />
          <ReadOnly label="Min Replay Shards" value="RLT_MIN_REPLAY_SHARDS=40" />
          <ReadOnly label="Actor Min Shards" value="RLT_ACTOR_MIN_REPLAY_SHARDS=40" />
          <ReadOnly label="Save Interval" value="RLT_SAVE_INTERVAL=1000" />
          <ReadOnly label="Actor Publish Interval" value="RLT_ACTOR_PUBLISH_INTERVAL=1000" />
          <ReadOnly label="Replay Horizon" value={rlt.replay_action_horizon ?? 'unknown'} />
          <ReadOnly label="Train Horizon" value={rlt.train_action_horizon ?? 'unknown'} />
        </ConfigSection>
      </div>
      {error ? <p className="rlt-error">{error}</p> : null}
    </section>
  )
}

function ConfigSection({ title, children }: { title: string; children: ReactNode }) {
  return (
    <section className="config-section">
      <div className="section-head">
        <h3>{title}</h3>
      </div>
      <div className="field-grid">{children}</div>
    </section>
  )
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="rlt-field">
      <span>{label}</span>
      {children}
    </label>
  )
}

function Toggle({ label, checked, onChange }: { label: string; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <label className="rlt-toggle">
      <input type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span>{label}</span>
    </label>
  )
}

function ReadOnly({ label, value }: { label: string; value: number | string }) {
  return (
    <div className="readonly-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  )
}
