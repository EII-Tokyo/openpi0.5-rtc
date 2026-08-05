import { afterEach, describe, expect, it, vi } from 'vitest'

import { captureTableSnapshot } from './api'

describe('table snapshot evidence API', () => {
  afterEach(() => vi.restoreAllMocks())

  it('preserves immutable-attempt evidence headers with the exact JPEG blob', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue(new Response(
      new Blob(['jpeg-evidence'], { type: 'image/jpeg' }),
      {
        status: 200,
        headers: {
          'Content-Type': 'image/jpeg',
          'X-Attempt-Id': 'A-003',
          'X-Frame-Number': '451',
          'X-Device-Timestamp-Ms': '12345.6',
          'X-Image-Sha256': 'c'.repeat(64),
        },
      },
    ))

    const result = await captureTableSnapshot('cal-20260805T120000-1234abcd')

    expect(fetchSpy).toHaveBeenCalledWith(
      '/api/sessions/cal-20260805T120000-1234abcd/actions/table/snapshot',
      { method: 'POST' },
    )
    expect(result.blob.size).toBe('jpeg-evidence'.length)
    expect(result.blob.type).toBe('image/jpeg')
    expect(result.attemptId).toBe('A-003')
    expect(result.frameNumber).toBe(451)
    expect(result.deviceTimestampMs).toBe(12345.6)
    expect(result.imageSha256).toBe('c'.repeat(64))
  })
})
