# ALOHA ViperX Task 5 runtime screenshot blocker

- Historical blocker status: `RESOLVED_WITH_ALTERNATE_VIEWPORT_BACKEND`
- Code: `HARD_BLOCKER_RUNTIME_CAMERA_EMPTY_BUFFER_ON_ROOT_FRAME_DIAGNOSTIC`
- Static structure screenshots: `PASS` (12 captures)
- Runtime readback replay screenshots: `PASS_AUXILIARY_RUNTIME_READBACK_REPLAY`

- Attempt 1: `ZERO_SIZE_CAMERA_BUFFER_HELPER_ERROR`; accepted PNGs = `0`; log: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-063308_aloha-task5-root-frame-drive-replay-capture/stderr.log`
- Attempt 2: `CAMERA_REMAINED_EMPTY_AFTER_RENDER_POLLING`; accepted PNGs = `0`; log: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-063427_aloha-task5-root-frame-drive-replay-capture-attempt2/stderr.log`
- Attempt 3: `CAMERA_REMAINED_EMPTY_AFTER_RENDER_POLLING`; accepted PNGs = `0`; log: `/home/eii/project/openpi0.5-rtc-reward-learning/.codex/artifacts/20260729-063604_aloha-task5-root-frame-drive-replay-capture-attempt3/stderr.log`

The Sensor Camera attempts remain rejected. The separate Isaac viewport backend resolved image acquisition after the camera target was recomputed from runtime finger geometry.
