import { useEffect, useState } from 'react'

type JointSeries = {
  names?: string[]
  values?: number[]
}

type Props = {
  open: boolean
  onClose: () => void
  jointEffort: Record<string, JointSeries>
  jointTemperature: Record<string, JointSeries>
}

type MotorHistoryPoint = {
  time: number
  effort: Record<string, number[]>
  temperature: Record<string, number[]>
}

const ROBOTS = [
  ['puppet_left', 'Puppet Left'],
  ['puppet_right', 'Puppet Right'],
] as const

const DEFAULT_JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
const JOINT_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#ca8a04']
const JOINT_MODEL_BY_NAME: Record<string, 'XM540-W270' | 'XM430-W350'> = {
  waist: 'XM540-W270',
  shoulder: 'XM540-W270',
  elbow: 'XM540-W270',
  forearm_roll: 'XM540-W270',
  wrist_angle: 'XM540-W270',
  wrist_rotate: 'XM430-W350',
  gripper: 'XM430-W350',
}
const TORQUE_NM_PER_MA = {
  'XM540-W270': 10.6 / 4400,
  'XM430-W350': 4.1 / 2300,
} as const
const CHART_WIDTH = 640
const CHART_HEIGHT = 260
const PLOT = { left: 54, right: 88, top: 20, bottom: 36 }

export function MotorDrawer({ open, onClose, jointEffort, jointTemperature }: Props) {
  const [motorHistory, setMotorHistory] = useState<MotorHistoryPoint[]>([])
  const [visibleJoints, setVisibleJoints] = useState<Record<string, boolean>>(
    Object.fromEntries(DEFAULT_JOINT_NAMES.map((name) => [name, true])),
  )

  useEffect(() => {
    const now = Date.now()
    const effortSnapshot: Record<string, number[]> = {}
    const temperatureSnapshot: Record<string, number[]> = {}
    ROBOTS.forEach(([key]) => {
      effortSnapshot[key] = [...(jointEffort[key]?.values ?? [])]
      temperatureSnapshot[key] = [...(jointTemperature[key]?.values ?? [])]
    })
    setMotorHistory((history) =>
      [...history, { time: now, effort: effortSnapshot, temperature: temperatureSnapshot }].filter(
        (point) => now - point.time <= 60_000,
      ),
    )
  }, [jointEffort, jointTemperature])

  const renderChart = (
    robotKey: string,
    metric: 'effort' | 'temperature',
    title: string,
    unit: string,
    names: string[],
  ) => {
    const visibleJointEntries = names
      .map((name, jointIndex) => ({ name, jointIndex }))
      .filter(({ name }) => visibleJoints[name] ?? true)
    const series = visibleJointEntries.map(({ jointIndex }) =>
      motorHistory
        .map((point) => ({ time: point.time, value: point[metric][robotKey]?.[jointIndex] }))
        .filter((point): point is { time: number; value: number } => Number.isFinite(point.value)),
    )
    const values = series.flatMap((line) => line.map((point) => point.value))
    const now = Date.now()
    const minTime = now - 60_000
    const minValue = values.length ? Math.min(...values) : 0
    const maxValue = values.length ? Math.max(...values) : 1
    const padding = Math.max((maxValue - minValue) * 0.12, metric === 'temperature' ? 1 : 0.5)
    const yMin = minValue - padding
    const yMax = maxValue + padding
    const yRange = yMax - yMin || 1
    const plotWidth = CHART_WIDTH - PLOT.left - PLOT.right
    const plotHeight = CHART_HEIGHT - PLOT.top - PLOT.bottom
    const xForTime = (time: number) => PLOT.left + ((time - minTime) / 60_000) * plotWidth
    const yForValue = (value: number) => PLOT.top + (1 - (value - yMin) / yRange) * plotHeight
    const yTicks = [yMax, yMin + yRange * 0.75, yMin + yRange * 0.5, yMin + yRange * 0.25, yMin]
    const selectedModels = Array.from(
      new Set(visibleJointEntries.map(({ name }) => JOINT_MODEL_BY_NAME[name]).filter(Boolean)),
    )
    const xTicks = [
      { label: '-60s', x: PLOT.left },
      { label: '-30s', x: PLOT.left + plotWidth / 2 },
      { label: 'now', x: PLOT.left + plotWidth },
    ]

    return (
      <section className="motor-chart-card" key={`${robotKey}-${metric}`}>
        <div className="motor-chart-card-head">
          <h3>{title}</h3>
          <span>{unit}</span>
        </div>
        <svg className="motor-axis-chart" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label={title}>
          <rect className="motor-chart-bg" x={PLOT.left} y={PLOT.top} width={plotWidth} height={plotHeight} />
          {yTicks.map((tick) => {
            const y = yForValue(tick)
            return (
              <g key={`y-${tick.toFixed(3)}`}>
                <line className="motor-grid-line" x1={PLOT.left} x2={CHART_WIDTH - PLOT.right} y1={y} y2={y} />
                <text className="motor-axis-label" x={PLOT.left - 8} y={y + 4} textAnchor="end">
                  {tick.toFixed(1)}
                </text>
                {metric === 'effort' && selectedModels.length > 0 && (
                  <text className="motor-axis-label torque-axis-label" x={CHART_WIDTH - PLOT.right + 10} y={y + 4}>
                    {selectedModels
                      .map((model) => (tick * TORQUE_NM_PER_MA[model]).toFixed(2))
                      .join(' / ')}
                  </text>
                )}
              </g>
            )
          })}
          {xTicks.map((tick) => (
            <g key={tick.label}>
              <line className="motor-grid-line" x1={tick.x} x2={tick.x} y1={PLOT.top} y2={CHART_HEIGHT - PLOT.bottom} />
              <text className="motor-axis-label" x={tick.x} y={CHART_HEIGHT - 12} textAnchor="middle">
                {tick.label}
              </text>
            </g>
          ))}
          <line className="motor-axis-line" x1={PLOT.left} x2={PLOT.left} y1={PLOT.top} y2={CHART_HEIGHT - PLOT.bottom} />
          {metric === 'effort' && selectedModels.length > 0 && (
            <text
              className="motor-axis-title"
              x={CHART_WIDTH - PLOT.right + 10}
              y={PLOT.top - 6}
              textAnchor="start"
            >
              {selectedModels.map((model) => (model.includes('540') ? '540' : '430')).join(' / ')} Nm
            </text>
          )}
          <text className="motor-axis-title" x={PLOT.left - 8} y={PLOT.top - 6} textAnchor="end">
            {unit}
          </text>
          <line
            className="motor-axis-line"
            x1={PLOT.left}
            x2={CHART_WIDTH - PLOT.right}
            y1={CHART_HEIGHT - PLOT.bottom}
            y2={CHART_HEIGHT - PLOT.bottom}
          />
          {series.map((line, visibleIndex) => {
            if (line.length < 2) return null
            const points = line.map((point) => `${xForTime(point.time).toFixed(1)},${yForValue(point.value).toFixed(1)}`).join(' ')
            const { name, jointIndex } = visibleJointEntries[visibleIndex]
            return (
              <polyline
                key={name}
                points={points}
                fill="none"
                stroke={JOINT_COLORS[jointIndex]}
                strokeWidth="2"
                vectorEffect="non-scaling-stroke"
              />
            )
          })}
        </svg>
      </section>
    )
  }

  return (
    <>
      {open && <button className="drawer-backdrop" type="button" aria-label="Close motor drawer" onClick={onClose} />}
      <aside className={open ? 'motor-drawer open' : 'motor-drawer'} aria-hidden={!open}>
        <div className="motor-drawer-header">
          <div>
            <p className="eyebrow">Motors</p>
            <h2>Puppet Motor Curves</h2>
          </div>
          <button className="ghost-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="motor-legend">
          {DEFAULT_JOINT_NAMES.map((name, index) => (
            <label key={name}>
              <input
                type="checkbox"
                checked={visibleJoints[name] ?? true}
                onChange={(event) => setVisibleJoints((current) => ({ ...current, [name]: event.target.checked }))}
              />
              <i style={{ background: JOINT_COLORS[index] }} />
              <span>{name}</span>
              <small>{JOINT_MODEL_BY_NAME[name].replace('XM', '')}</small>
            </label>
          ))}
        </div>
        <div className="motor-drawer-scroll">
          {ROBOTS.map(([robotKey, label]) => {
            const names =
              (jointEffort[robotKey]?.names?.length ? jointEffort[robotKey]?.names : jointTemperature[robotKey]?.names) ??
              DEFAULT_JOINT_NAMES
            return (
              <div className="motor-robot-section" key={robotKey}>
                <h3>{label}</h3>
                {renderChart(robotKey, 'effort', `${label} Current`, 'mA', names.slice(0, 7))}
                {renderChart(robotKey, 'temperature', `${label} Temperature`, 'C', names.slice(0, 7))}
              </div>
            )
          })}
        </div>
      </aside>
    </>
  )
}
