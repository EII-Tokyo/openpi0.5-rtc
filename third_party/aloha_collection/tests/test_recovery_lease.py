import json

import pytest

from aloha.recovery_lease import (
    RecoveryLease,
    RecoveryLeaseBusy,
)


def test_only_one_process_can_hold_recovery_lease(tmp_path):
    path = tmp_path / "safe-sleep.lock"
    first = RecoveryLease.acquire(
        path=path,
        source="recorder",
        robot="aloha_stationary",
        recovery_id="first",
    )
    try:
        with pytest.raises(RecoveryLeaseBusy, match="first"):
            RecoveryLease.acquire(
                path=path,
                source="standalone",
                robot="aloha_stationary",
                recovery_id="second",
            )
    finally:
        first.release()


def test_release_allows_the_next_owner(tmp_path):
    path = tmp_path / "safe-sleep.lock"
    first = RecoveryLease.acquire(
        path=path,
        source="recorder",
        robot="aloha_stationary",
        recovery_id="first",
    )
    first.release()

    second = RecoveryLease.acquire(
        path=path,
        source="standalone",
        robot="aloha_stationary",
        recovery_id="second",
    )
    try:
        assert second.metadata.recovery_id == "second"
        assert second.metadata.source == "standalone"
        assert second.metadata.robot == "aloha_stationary"
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["recovery_id"] == "second"
        assert payload["owner_pid"] == second.metadata.owner_pid
    finally:
        second.release()


def test_release_is_idempotent(tmp_path):
    lease = RecoveryLease.acquire(
        path=tmp_path / "safe-sleep.lock",
        source="recorder",
        robot="aloha_stationary",
    )

    lease.release()
    lease.release()
