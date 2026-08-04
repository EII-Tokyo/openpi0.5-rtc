import os
from pathlib import Path
import stat
import threading

import pytest

from aloha.episode_storage import (
    EpisodeClaim,
    EpisodeClaimCollision,
    EpisodePublishCollision,
    StagedEpisode,
)


def test_claim_acquire_is_exclusive_and_writes_owner_token(tmp_path):
    claim = EpisodeClaim.acquire(tmp_path, 4)

    assert claim.claim_path.read_text(encoding="utf-8") == claim.owner_token
    with pytest.raises(EpisodeClaimCollision):
        EpisodeClaim.acquire(tmp_path, 4)

    claim.release()


def test_claim_release_preserves_entry_when_owner_token_changed(tmp_path):
    claim = EpisodeClaim.acquire(tmp_path, 4)
    claim.claim_path.write_text("another-owner", encoding="utf-8")

    assert claim.release() is False
    assert claim.claim_path.read_text(encoding="utf-8") == "another-owner"


def test_claim_release_is_idempotent_for_owned_entry(tmp_path):
    claim = EpisodeClaim.acquire(tmp_path, 4)

    assert claim.release() is True
    assert claim.release() is True
    assert not claim.claim_path.exists()


def test_claim_release_serializes_with_a_successor_acquire(tmp_path, monkeypatch):
    claim = EpisodeClaim.acquire(tmp_path, 4)
    token_read = threading.Event()
    allow_release = threading.Event()
    successor_acquired = threading.Event()
    original_read_text = Path.read_text

    def blocking_read_text(path, *args, **kwargs):
        token = original_read_text(path, *args, **kwargs)
        if path == claim.claim_path and threading.current_thread().name == "release":
            token_read.set()
            assert allow_release.wait(timeout=2)
        return token

    monkeypatch.setattr(Path, "read_text", blocking_read_text)
    release_result = []
    successor = []
    release_thread = threading.Thread(
        target=lambda: release_result.append(claim.release()),
        name="release",
    )
    acquire_thread = threading.Thread(
        target=lambda: (
            successor.append(EpisodeClaim.acquire(tmp_path, 4)),
            successor_acquired.set(),
        ),
        name="successor",
    )

    release_thread.start()
    assert token_read.wait(timeout=2)
    claim.claim_path.unlink()
    acquire_thread.start()
    try:
        assert not successor_acquired.wait(timeout=0.2)
    finally:
        allow_release.set()
        release_thread.join(timeout=2)
        acquire_thread.join(timeout=2)

    assert release_result == [False]
    assert successor_acquired.is_set()
    assert successor[0].claim_path.read_text(encoding="utf-8") == successor[0].owner_token
    successor[0].release()


@pytest.mark.parametrize("entry_kind", ["dangling", "directory"])
def test_claim_acquire_treats_any_lexical_claim_entry_as_collision(
    tmp_path,
    entry_kind,
):
    claim_path = tmp_path / ".episode_4.claim"
    if entry_kind == "dangling":
        claim_path.symlink_to(tmp_path / "missing")
    else:
        claim_path.mkdir()

    with pytest.raises(EpisodeClaimCollision):
        EpisodeClaim.acquire(tmp_path, 4)


def test_claim_context_manager_releases_owned_entry(tmp_path):
    with EpisodeClaim.acquire(tmp_path, 4) as claim:
        assert claim.claim_path.exists()

    assert not claim.claim_path.exists()


def test_staged_episode_discard_removes_only_owned_staging_and_claim(tmp_path):
    unrelated = tmp_path / ".episode_4.staging-not-owned"
    unrelated.mkdir()
    sentinel = unrelated / "keep"
    sentinel.write_text("keep", encoding="utf-8")
    staged = StagedEpisode.create(tmp_path, 4)
    owned_path = staged.staging_path

    staged.discard()
    staged.discard()

    assert not owned_path.exists()
    assert not staged.claim.claim_path.exists()
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_publish_success_renames_staging_and_releases_claim(tmp_path):
    staged = StagedEpisode.create(tmp_path, 4)
    payload = staged.staging_path / "payload.txt"
    payload.write_text("accepted", encoding="utf-8")

    published = staged.publish()

    assert published == tmp_path / "episode_4"
    assert (published / "payload.txt").read_text(encoding="utf-8") == "accepted"
    assert not staged.claim.claim_path.exists()
    staged.discard()
    assert published.is_dir()


def test_publish_changes_private_staging_to_traversable_completed_directory(
    tmp_path,
):
    staged = StagedEpisode.create(tmp_path, 4)

    assert stat.S_IMODE(staged.staging_path.stat().st_mode) == 0o700

    published = staged.publish()

    assert stat.S_IMODE(published.stat().st_mode) == 0o755


def test_confirmed_overwrite_publishes_traversable_directory(tmp_path):
    destination = tmp_path / "episode_4"
    destination.mkdir()
    staged = StagedEpisode.create(tmp_path, 4)

    published = staged.publish(allow_existing_destination=True)

    assert stat.S_IMODE(published.stat().st_mode) == 0o755


@pytest.mark.parametrize(
    "occupied_name,entry_kind",
    [
        ("episode_4", "directory"),
        ("episode_4", "dangling"),
        ("episode_4", "file"),
        ("episode_4.hdf5", "file"),
        ("episode_4.hdf5", "dangling"),
        ("episode_4.hdf5", "directory"),
    ],
)
def test_publish_rejects_any_lexical_final_or_legacy_collision(
    tmp_path,
    occupied_name,
    entry_kind,
):
    occupied = tmp_path / occupied_name
    if entry_kind == "directory":
        occupied.mkdir()
    elif entry_kind == "dangling":
        occupied.symlink_to(tmp_path / "missing")
    else:
        occupied.write_text("keep", encoding="utf-8")
    staged = StagedEpisode.create(tmp_path, 4)
    owned_staging = staged.staging_path

    with pytest.raises(EpisodePublishCollision):
        staged.publish()

    assert os.path.lexists(occupied)
    assert owned_staging.is_dir()
    assert staged.claim.claim_path.exists()


def test_publish_rechecks_destination_after_taking_dataset_lock(
    tmp_path,
    monkeypatch,
):
    staged = StagedEpisode.create(tmp_path, 4)
    destination = tmp_path / "episode_4"
    real_flock = __import__("fcntl").flock
    created = False

    def create_collision_on_lock(lock_fd, operation):
        nonlocal created
        real_flock(lock_fd, operation)
        if not created and operation == __import__("fcntl").LOCK_EX:
            destination.mkdir()
            created = True

    monkeypatch.setattr(
        "aloha.episode_storage.fcntl.flock",
        create_collision_on_lock,
    )

    with pytest.raises(EpisodePublishCollision):
        staged.publish()

    assert destination.is_dir()
    assert staged.staging_path.is_dir()
    assert staged.claim.claim_path.exists()


def test_publish_rejects_legacy_even_with_directory_overwrite_authority(tmp_path):
    legacy = tmp_path / "episode_4.hdf5"
    legacy.write_text("legacy", encoding="utf-8")
    staged = StagedEpisode.create(tmp_path, 4)

    with pytest.raises(EpisodePublishCollision):
        staged.publish(allow_existing_destination=True)

    assert legacy.read_text(encoding="utf-8") == "legacy"
    assert staged.staging_path.is_dir()
    assert staged.claim.claim_path.exists()


def test_confirmed_overwrite_replaces_existing_directory_atomically(tmp_path):
    destination = tmp_path / "episode_4"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    staged = StagedEpisode.create(tmp_path, 4)
    (staged.staging_path / "new.txt").write_text("new", encoding="utf-8")

    published = staged.publish(allow_existing_destination=True)

    assert published == destination
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
    assert not (destination / "old.txt").exists()
    assert not list(tmp_path.glob(".episode_4.backup-*"))
    assert not staged.claim.claim_path.exists()


def test_failed_confirmed_overwrite_restores_existing_directory(
    tmp_path,
    monkeypatch,
):
    destination = tmp_path / "episode_4"
    destination.mkdir()
    sentinel = destination / "old.txt"
    sentinel.write_text("old", encoding="utf-8")
    staged = StagedEpisode.create(tmp_path, 4)
    owned_staging = staged.staging_path
    (owned_staging / "new.txt").write_text("new", encoding="utf-8")
    real_rename = os.rename

    def fail_staging_publish(source, target):
        if Path(source) == owned_staging and Path(target) == destination:
            raise OSError("injected rename failure")
        return real_rename(source, target)

    monkeypatch.setattr("aloha.episode_storage.os.rename", fail_staging_publish)

    with pytest.raises(OSError, match="injected rename failure"):
        staged.publish(allow_existing_destination=True)

    assert sentinel.read_text(encoding="utf-8") == "old"
    assert owned_staging.is_dir()
    assert (owned_staging / "new.txt").read_text(encoding="utf-8") == "new"
    assert staged.claim.claim_path.exists()
    assert not list(tmp_path.glob(".episode_4.backup-*"))


def test_confirmed_overwrite_rejects_wrong_kind_final_destination(tmp_path):
    destination = tmp_path / "episode_4"
    destination.write_text("keep", encoding="utf-8")
    staged = StagedEpisode.create(tmp_path, 4)

    with pytest.raises(EpisodePublishCollision):
        staged.publish(allow_existing_destination=True)

    assert destination.read_text(encoding="utf-8") == "keep"
    assert staged.staging_path.is_dir()
    assert staged.claim.claim_path.exists()
