import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App from './App'

describe('ALOHA calibration preview workbench', () => {
  it('renders as a device-safe preview without making network calls', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    render(<App />)

    expect(screen.getByRole('heading', { name: 'ALOHA 四相机标定工作台' })).toBeInTheDocument()
    expect(screen.getByText('PREVIEW · 不执行设备操作')).toBeInTheDocument()
    expect(screen.getByText('103 Capture: OFF / EXCLUSIVE')).toBeInTheDocument()
    expect(screen.getByText('Robot command APIs: NONE')).toBeInTheDocument()
    expect(screen.getByText('Isaac: DISCONNECTED')).toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()

    fetchSpy.mockRestore()
  })

  it('shows the six reviewed calibration stages and four independent cameras', () => {
    render(<App />)

    for (const stage of [
      '预检与相机身份',
      '内参与正式 Profile',
      '世界原点板与固定相机',
      '左腕 / 右腕手眼',
      '独立三维检查点',
      'Isaac 叠加与带标夹具',
    ]) {
      expect(screen.getByText(stage)).toBeInTheDocument()
    }

    for (const camera of ['cam_high', 'cam_low', 'wrist_left', 'wrist_right']) {
      expect(screen.getByTestId(`camera-${camera}`)).toBeInTheDocument()
    }
  })

  it('keeps device actions disabled and explains the current physical step', () => {
    render(<App />)

    expect(screen.getByRole('button', { name: '开始检测（预览模式）' })).toBeDisabled()
    expect(screen.getByText('现在放什么 / 怎么摆')).toBeInTheDocument()
    expect(screen.getByText('系统现在检查什么')).toBeInTheDocument()
    expect(screen.getByText('通过后会得到什么')).toBeInTheDocument()
  })

  it('switches between preview-only workspace tabs', async () => {
    const user = userEvent.setup()
    render(<App />)

    await user.click(screen.getByRole('tab', { name: 'Dataset' }))
    expect(screen.getByRole('heading', { name: '数据集审查' })).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Solve' }))
    expect(screen.getByRole('heading', { name: '求解结果' })).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Validate' }))
    expect(screen.getByRole('heading', { name: '独立验证' })).toBeInTheDocument()

    await user.click(screen.getByRole('tab', { name: 'Export' }))
    expect(screen.getByRole('heading', { name: '导出与提交' })).toBeInTheDocument()
  })

  it('does not collapse distinct calibration outcomes into one success badge', () => {
    render(<App />)

    expect(screen.getByText('INTRINSICS_VALIDATED')).toBeInTheDocument()
    expect(screen.getByText('WORLD_REGISTRATION_VALIDATED')).toBeInTheDocument()
    expect(screen.getByText('HAND_EYE_NONCONTACT_VALIDATED')).toBeInTheDocument()
    expect(screen.getByText('ISAAC_IMPORT_PASS')).toBeInTheDocument()
    expect(screen.getByText('TAGGED_FIXTURE_TRANSFER_PASS')).toBeInTheDocument()
  })
})
