import { useEffect, useRef, useState, type ComponentProps, type ReactNode } from 'react'
import { HighLevelPanel } from './HighLevelPanel'
import { InferenceSettingsPanel } from './InferenceSettingsPanel'
import './inference-monitor.css'

type View = 'results' | 'conditions'
type Props = Omit<ComponentProps<typeof InferenceSettingsPanel>, 'open'> & {
  open: boolean
  onClose: () => void
  language: string
  inferencePrompt: string | null
  runtimeMode: string
  controls?: ReactNode
}

export function inferenceMonitorTitle(language: string) {
  return language === 'ja' ? '推論モニター' : language === 'zh' ? '推理监视' : 'Inference monitor'
}

export function InferenceMonitorDrawer({ open, onClose, language, inferencePrompt, runtimeMode, controls, ...conditions }: Props) {
  const dialog = useRef<HTMLDialogElement>(null)
  const [view, setView] = useState<View>('results')
  const title = inferenceMonitorTitle(language)
  const labels = language === 'ja' ? { results: 'Sol の結果', conditions: '推論条件' }
    : language === 'zh' ? { results: 'Sol 结果', conditions: '推理条件' }
    : { results: 'Sol results', conditions: 'Inference conditions' }

  useEffect(() => {
    const element = dialog.current
    if (open && !element?.open) element?.showModal()
    else if (!open && element?.open) element.close()
  }, [open])

  return <dialog ref={dialog} id="inference-monitor" className="inference-settings-drawer inference-monitor-drawer"
    aria-labelledby="inference-monitor-title" onClose={onClose}
    onCancel={event => { event.preventDefault(); onClose() }}
    onClick={event => {
      if (event.target !== event.currentTarget) return
      const rect = event.currentTarget.getBoundingClientRect()
      if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) onClose()
    }}>
    <div className="motor-drawer-header">
      <h2 id="inference-monitor-title">{title}</h2>
      <button className="conditioning-close" type="button" aria-label={`Close ${title}`} onClick={onClose}>×</button>
    </div>
    <div className="inference-monitor-views" role="group" aria-label={title}>
      {(['results', 'conditions'] as const).map(item => (
        <button className="ghost-button" type="button" key={item} aria-pressed={view === item}
          onClick={() => setView(item)}>{labels[item]}</button>
      ))}
    </div>
    <div hidden={view !== 'results'}>
      <HighLevelPanel open={open && view === 'results'} language={language} inferencePrompt={inferencePrompt}
        runtimeMode={runtimeMode} controls={open && view === 'results' ? controls : undefined} />
    </div>
    <div hidden={view !== 'conditions'}>
      <InferenceSettingsPanel {...conditions} open={open && view === 'conditions'} />
    </div>
  </dialog>
}
