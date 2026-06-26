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
    | 'actor_handoff_steps'
    | 'actor_delta_ema_alpha'
    | 'actor_speed_limit_preset'
    | 'critic_gate_enabled'
    | 'critic_gate_margin'
    | 'critic_gate_temperature'
  >
> & { wandb_url: string }

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
  actor_handoff_steps: rlt.actor_handoff_steps,
  actor_delta_ema_alpha: rlt.actor_delta_ema_alpha,
  actor_speed_limit_preset: rlt.actor_speed_limit_preset,
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
    rlt.actor_handoff_steps,
    rlt.actor_delta_ema_alpha,
    rlt.actor_speed_limit_preset,
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
            <small className="field-hint">Actor 残差整体强度：0 只用 RTC/VLA，1 使用 actor 给出的完整修正。</small>
          </Field>
          <Field label="Max Delta">
            <input
              type="number"
              min={0}
              step={0.001}
              value={draft.max_delta}
              onChange={(event) => setDraft({ ...draft, max_delta: Number(event.target.value) })}
            />
            <small className="field-hint">单个关节每个 actor chunk 的最大修正幅度；0 表示不裁剪。</small>
          </Field>
          <Field label={`Handoff Steps ${draft.actor_handoff_steps}`}>
            <input
              type="range"
              min={0}
              max={20}
              step={1}
              value={draft.actor_handoff_steps}
              onChange={(event) => setDraft({ ...draft, actor_handoff_steps: Number(event.target.value) })}
            />
            <small className="field-hint">新 actor chunk 开头用多少步从当前左臂状态过渡到 actor 动作；0 或 1 表示关闭。</small>
          </Field>
          <Field label={`Delta EMA Alpha ${draft.actor_delta_ema_alpha.toFixed(2)}`}>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={draft.actor_delta_ema_alpha}
              onChange={(event) => setDraft({ ...draft, actor_delta_ema_alpha: Number(event.target.value) })}
            />
            <small className="field-hint">跨 RTC chunk 的 actor 残差平滑系数；越小越平滑，越大越跟手，1 表示不平滑。</small>
          </Field>
          <Field label="Actor Speed Limit">
            <select
              value={draft.actor_speed_limit_preset}
              onChange={(event) =>
                setDraft({
                  ...draft,
                  actor_speed_limit_preset: event.target.value as Draft['actor_speed_limit_preset'],
                })
              }
            >
              <option value="80">80% limit</option>
              <option value="50">50% limit</option>
              <option value="20">20% limit</option>
              <option value="off">No limit</option>
            </select>
            <small className="field-hint">Actor/key region 期间限制左臂关节目标相对当前状态的单步 delta；默认 No limit，不做左臂速度限幅。</small>
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
