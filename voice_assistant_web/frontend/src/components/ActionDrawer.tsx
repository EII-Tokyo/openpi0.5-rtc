import { useEffect, useState } from 'react'

type Props = {
  open: boolean
  onClose: () => void
  latestAction: number[]
}

type ActionHistoryPoint = {
  time: number
  left: number[]
  right: number[]
}

const ROBOTS = [
  ['left', 'Left Action'],
  ['right', 'Right Action'],
] as const

const JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate', 'gripper']
const JOINT_COLORS = ['#2563eb', '#dc2626', '#16a34a', '#9333ea', '#ea580c', '#0891b2', '#ca8a04']
const CHART_WIDTH = 640
const CHART_HEIGHT = 260
const PLOT = { left: 54, right: 18, top: 20, bottom: 36 }

export function ActionDrawer({ open, onClose, latestAction }: Props) {
  const [history, setHistory] = useState<ActionHistoryPoint[]>([])
  const [visibleJoints, setVisibleJoints] = useState<Record<string, boolean>>(
    Object.fromEntries(JOINT_NAMES.map((name) => [name, true])),
  )

  useEffect(() => {
    if (latestAction.length < 14) return
    const now = Date.now()
    const left = latestAction.slice(0, 7)
    const right = latestAction.slice(7, 14)
    setHistory((current) => [...current, { time: now, left, right }].filter((point) => now - point.time <= 60_000))
  }, [latestAction])

  const renderChart = (side: 'left' | 'right', title: string) => {
    const visibleJointEntries = JOINT_NAMES.map((name, jointIndex) => ({ name, jointIndex })).filter(
      ({ name }) => visibleJoints[name] ?? true,
    )
    const series = visibleJointEntries.map(({ jointIndex }) =>
      history
        .map((point) => ({ time: point.time, value: point[side][jointIndex] }))
        .filter((point): point is { time: number; value: number } => Number.isFinite(point.value)),
    )
    const values = series.flatMap((line) => line.map((point) => point.value))
    const now = Date.now()
    const minTime = now - 60_000
    const minValue = values.length ? Math.min(...values) : 0
    const maxValue = values.length ? Math.max(...values) : 1
    const padding = Math.max((maxValue - minValue) * 0.12, 0.05)
    const yMin = minValue - padding
    const yMax = maxValue + padding
    const yRange = yMax - yMin || 1
    const plotWidth = CHART_WIDTH - PLOT.left - PLOT.right
    const plotHeight = CHART_HEIGHT - PLOT.top - PLOT.bottom
    const xForTime = (time: number) => PLOT.left + ((time - minTime) / 60_000) * plotWidth
    const yForValue = (value: number) => PLOT.top + (1 - (value - yMin) / yRange) * plotHeight
    const yTicks = [yMax, yMin + yRange * 0.75, yMin + yRange * 0.5, yMin + yRange * 0.25, yMin]
    const xTicks = [
      { label: '-60s', x: PLOT.left },
      { label: '-30s', x: PLOT.left + plotWidth / 2 },
      { label: 'now', x: PLOT.left + plotWidth },
    ]

    return (
      <section className="motor-chart-card" key={side}>
        <div className="motor-chart-card-head">
          <h3>{title}</h3>
          <span>action</span>
        </div>
        <svg className="motor-axis-chart" viewBox={`0 0 ${CHART_WIDTH} ${CHART_HEIGHT}`} role="img" aria-label={title}>
          <rect className="motor-chart-bg" x={PLOT.left} y={PLOT.top} width={plotWidth} height={plotHeight} />
          {yTicks.map((tick) => {
            const y = yForValue(tick)
            return (
              <g key={`y-${tick.toFixed(3)}`}>
                <line className="motor-grid-line" x1={PLOT.left} x2={CHART_WIDTH - PLOT.right} y1={y} y2={y} />
                <text className="motor-axis-label" x={PLOT.left - 8} y={y + 4} textAnchor="end">
                  {tick.toFixed(2)}
                </text>
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
          <text className="motor-axis-title" x={PLOT.left - 8} y={PLOT.top - 6} textAnchor="end">
            action
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
      {open && <button className="drawer-backdrop" type="button" aria-label="Close action drawer" onClick={onClose} />}
      <aside className={open ? 'motor-drawer open' : 'motor-drawer'} aria-hidden={!open}>
        <div className="motor-drawer-header">
          <div>
            <p className="eyebrow">Actions</p>
            <h2>Policy Action Curves</h2>
          </div>
          <button className="ghost-button" type="button" onClick={onClose}>
            Close
          </button>
        </div>
        <div className="motor-legend">
          {JOINT_NAMES.map((name, index) => (
            <label key={name}>
              <input
                type="checkbox"
                checked={visibleJoints[name] ?? true}
                onChange={(event) => setVisibleJoints((current) => ({ ...current, [name]: event.target.checked }))}
              />
              <i style={{ background: JOINT_COLORS[index] }} />
              <span>{name}</span>
            </label>
          ))}
        </div>
        <div className="motor-drawer-scroll">
          {ROBOTS.map(([side, label]) => renderChart(side, label))}
        </div>
      </aside>
    </>
  )
}
