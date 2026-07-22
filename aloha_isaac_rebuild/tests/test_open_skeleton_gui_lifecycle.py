from __future__ import annotations

from aloha_isaac_rebuild.scripts.open_skeleton_gui import _viewer_should_continue


class _FakeApp:
    def __init__(self, running: bool) -> None:
        self._running = running

    def is_running(self) -> bool:
        return self._running


def test_viewer_lifecycle_masks_early_false_running_state() -> None:
    assert _viewer_should_continue(_FakeApp(False), elapsed_s=1.0, min_alive_s=3.0) is True


def test_viewer_lifecycle_exits_after_grace_period_when_app_is_not_running() -> None:
    assert _viewer_should_continue(_FakeApp(False), elapsed_s=4.0, min_alive_s=3.0) is False


def test_viewer_lifecycle_keeps_running_after_grace_period_when_app_is_running() -> None:
    assert _viewer_should_continue(_FakeApp(True), elapsed_s=4.0, min_alive_s=3.0) is True
