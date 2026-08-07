import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it, vi } from 'vitest'
import App, { BOTTLE500_TASK_FROM_ASSET, DEFAULT_BOTTLE_TAG_ID } from './App'

describe('ALOHA table and Bottle500 calibration workbench', () => {
  it('is device-safe in preview mode and explains the three experiments', () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
    render(<App />)

    expect(screen.getByRole('heading', { name: 'ALOHA 桌面与瓶子标定工作台' })).toBeInTheDocument()
    expect(screen.getByText('PREVIEW · 不执行设备操作')).toBeInTheDocument()
    expect(screen.getAllByText('实验一 · 世界锚点').length).toBeGreaterThan(0)
    expect(screen.getAllByText('实验二 · 9 点桌面').length).toBeGreaterThan(0)
    expect(screen.getAllByText('实验三 · Bottle500').length).toBeGreaterThan(0)
    expect(screen.getByText('DEPTH OFF · ROBOT API NONE')).toBeInTheDocument()
    expect(fetchSpy).not.toHaveBeenCalled()
    fetchSpy.mockRestore()
  })

  it('keeps all four identities visible while only camera_high is active', () => {
    render(<App />)
    for (const role of ['cam_high', 'cam_low', 'wrist_left', 'wrist_right']) {
      expect(screen.getByTestId(`camera-${role}`)).toBeInTheDocument()
    }
    expect(screen.getByText('仅 camera_high 参与三个实验')).toBeInTheDocument()
  })

  it('shows non-collinear held-out dots and does not call them 3D validation', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Table Dots' }))

    expect(screen.getByText('P11 · HELD_OUT')).toBeInTheDocument()
    expect(screen.getByText('P23 · HELD_OUT')).toBeInTheDocument()
    expect(screen.getByText('P32 · HELD_OUT')).toBeInTheDocument()
    expect(screen.getByText('范围：桌面平面 XY 交叉验证，不声明离面 Z 已验证')).toBeInTheDocument()
  })

  it('labels the bottle result as tagged rigid fixture transfer only', async () => {
    const user = userEvent.setup()
    render(<App />)
    await user.click(screen.getByRole('tab', { name: 'Bottle' }))

    expect(screen.getByText('TAGGED_FIXTURE_TRANSFER_PASS')).toBeInTheDocument()
    expect(screen.getByText('不代表无标签透明瓶识别、碰撞或动力学通过')).toBeInTheDocument()
    expect(screen.getByLabelText('瓶夹具 Tag ID')).toHaveValue(DEFAULT_BOTTLE_TAG_ID)
  })

  it('maps the Bottle500 asset midpoint to the bottle task origin', () => {
    const matrix = BOTTLE500_TASK_FROM_ASSET.matrix
    const assetMidpoint = [0, 0, 0.103, 1]
    const mapped = matrix.map((row) => row.reduce(
      (total, value, index) => total + value * assetMidpoint[index],
      0,
    ))

    expect(mapped).toEqual([0, 0, 0, 1])
    expect(matrix[0][3]).toBe(-0.103)
  })

  it('gates factory snapshots behind a successful live preflight', async () => {
    const user = userEvent.setup()
    const fetchSpy = vi.spyOn(globalThis, 'fetch')
      .mockResolvedValueOnce(new Response(JSON.stringify({
        id: 'cal-20260805T120000-1234abcd',
        state: 'PREFLIGHT_READY',
        latest_preflight: {
          status: 'READY', cameras: [
            { role: 'cam_high', connected: true, identity_match: true, production_profile_supported: true, ownership: 'FREE' },
            { role: 'cam_low', connected: true, identity_match: true, production_profile_supported: true, ownership: 'FREE' },
            { role: 'wrist_left', connected: true, identity_match: true, production_profile_supported: true, ownership: 'FREE' },
            { role: 'wrist_right', connected: true, identity_match: true, production_profile_supported: true, ownership: 'FREE' },
          ], issues: [],
        },
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))
      .mockResolvedValueOnce(new Response(JSON.stringify({
        status: 'FACTORY_INTRINSICS_FROZEN', cameras: [
          { role: 'cam_high', serial: '130322270656' },
          { role: 'cam_low', serial: '218622270440' },
          { role: 'wrist_left', serial: '130322272542' },
          { role: 'wrist_right', serial: '218622278936' },
        ],
      }), { status: 200, headers: { 'Content-Type': 'application/json' } }))

    render(<App mode="live" />)
    expect(screen.queryByRole('button', { name: '冻结四台出厂 K/D' })).not.toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: '运行只读预检' }))
    expect((await screen.findAllByText('PREFLIGHT_READY')).length).toBeGreaterThan(0)
    await user.click(screen.getByRole('button', { name: '冻结四台出厂 K/D' }))

    expect(fetchSpy).toHaveBeenNthCalledWith(1, '/api/preflight-session', expect.objectContaining({ method: 'POST' }))
    expect(fetchSpy).toHaveBeenNthCalledWith(
      2,
      '/api/sessions/cal-20260805T120000-1234abcd/actions/factory/freeze',
      expect.objectContaining({ method: 'POST' }),
    )
    expect((await screen.findAllByText('FACTORY_INTRINSICS_FROZEN')).length).toBeGreaterThan(0)
    fetchSpy.mockRestore()
  })
})
