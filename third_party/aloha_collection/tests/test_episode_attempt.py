from pathlib import Path

import pytest

from aloha.episode_attempt import (
    AttemptArtifact,
    check_episode_index,
    find_next_available_episode_index,
    guarded_teleop_step,
)


def test_discard_removes_owned_diagnostic_without_creating_episode_dir(tmp_path):
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.diagnostic_path.write_text("attempt-data", encoding="utf-8")
    diagnostic_path = artifact.diagnostic_path

    artifact.discard()

    assert not diagnostic_path.exists()
    assert not (tmp_path / "episode_4").exists()


def test_commit_moves_diagnostic_into_final_episode_directory(tmp_path):
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.diagnostic_path.write_text("accepted", encoding="utf-8")
    diagnostic_path = artifact.diagnostic_path

    committed = artifact.commit(tmp_path / "episode_4")

    assert committed == tmp_path / "episode_4" / "motor6_diagnostics.jsonl"
    assert committed.read_text(encoding="utf-8") == "accepted"
    assert not diagnostic_path.exists()


def test_repeated_discard_is_idempotent(tmp_path):
    artifact = AttemptArtifact.create(tmp_path, "episode_4")

    artifact.discard()
    artifact.discard()

    assert not artifact.diagnostic_path.exists()


def test_commit_twice_is_rejected_clearly(tmp_path):
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.commit(tmp_path / "episode_4")

    with pytest.raises(RuntimeError, match="already committed"):
        artifact.commit(tmp_path / "episode_4")


def test_commit_after_discard_is_rejected_clearly(tmp_path):
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.discard()

    with pytest.raises(RuntimeError, match="discarded"):
        artifact.commit(tmp_path / "episode_4")


def test_discard_preserves_unrelated_neighbor_file(tmp_path):
    unrelated = tmp_path / "operator-notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    artifact = AttemptArtifact.create(tmp_path, "episode_4")

    artifact.discard()

    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_create_allocates_unique_hidden_artifacts(tmp_path):
    first = AttemptArtifact.create(tmp_path, "episode_4")
    second = AttemptArtifact.create(tmp_path, "episode_4")

    assert first.diagnostic_path != second.diagnostic_path
    assert first.diagnostic_path.parent == Path(tmp_path)
    assert second.diagnostic_path.parent == Path(tmp_path)
    assert first.diagnostic_path.name.startswith(".episode_4.attempt-")
    assert second.diagnostic_path.name.startswith(".episode_4.attempt-")
    assert first.diagnostic_path.exists()
    assert second.diagnostic_path.exists()


def test_guarded_teleop_step_checks_health_before_action_and_command():
    calls = []
    action = object()

    result = guarded_teleop_step(
        health_check=lambda: calls.append("health"),
        read_action=lambda: calls.append("action") or action,
        command=lambda value: calls.append(("command", value)) or "ts",
        clock=lambda: calls.append("clock") or 123.0,
    )

    assert calls == [
        "health",
        "action",
        "clock",
        ("command", action),
    ]
    assert result == (action, "ts", 123.0)


def test_guarded_teleop_step_never_commands_after_stale_health():
    def stale():
        raise RuntimeError("leader_left stale")

    with pytest.raises(RuntimeError, match="leader_left stale"):
        guarded_teleop_step(
            health_check=stale,
            read_action=lambda: pytest.fail(
                "must not read stale leader cache"
            ),
            command=lambda _action: pytest.fail(
                "must not command follower"
            ),
            clock=lambda: pytest.fail(
                "must not timestamp a rejected action"
            ),
        )


def test_commit_refuses_preexisting_episode_directory_without_overwriting(tmp_path):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    sentinel = episode_dir / "motor6_diagnostics.jsonl"
    sentinel.write_text("existing-data", encoding="utf-8")
    unrelated = episode_dir / "operator-notes.txt"
    unrelated.write_text("keep", encoding="utf-8")
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.diagnostic_path.write_text("new-data", encoding="utf-8")

    with pytest.raises(FileExistsError):
        artifact.commit(episode_dir)

    assert sentinel.read_text(encoding="utf-8") == "existing-data"
    assert unrelated.read_text(encoding="utf-8") == "keep"
    assert artifact.diagnostic_path.read_text(encoding="utf-8") == "new-data"


def test_commit_can_replace_diagnostic_after_explicit_overwrite_confirmation(tmp_path):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    diagnostic = episode_dir / "motor6_diagnostics.jsonl"
    diagnostic.write_text("old-data", encoding="utf-8")
    neighbor = episode_dir / "operator-notes.txt"
    neighbor.write_text("keep", encoding="utf-8")
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.diagnostic_path.write_text("new-data", encoding="utf-8")

    committed = artifact.commit(episode_dir, allow_existing=True)

    assert committed == diagnostic
    assert diagnostic.read_text(encoding="utf-8") == "new-data"
    assert neighbor.read_text(encoding="utf-8") == "keep"


def test_failed_overwrite_never_removes_preexisting_episode_directory(
    tmp_path,
    monkeypatch,
):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    sentinel = episode_dir / "operator-notes.txt"
    sentinel.write_text("keep", encoding="utf-8")
    artifact = AttemptArtifact.create(tmp_path, "episode_4")

    def fail_replace(_source, _destination):
        raise OSError("injected overwrite failure")

    monkeypatch.setattr("aloha.episode_attempt.os.replace", fail_replace)

    with pytest.raises(OSError, match="injected overwrite failure"):
        artifact.commit(episode_dir, allow_existing=True)

    assert episode_dir.is_dir()
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert artifact.diagnostic_path.exists()


def test_commit_into_existing_rejects_existing_diagnostic_without_authority(tmp_path):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    diagnostic = episode_dir / "motor6_diagnostics.jsonl"
    diagnostic.write_text("prior", encoding="utf-8")
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.diagnostic_path.write_text("new", encoding="utf-8")

    with pytest.raises(FileExistsError):
        artifact.commit_into_existing(episode_dir)

    assert diagnostic.read_text(encoding="utf-8") == "prior"
    assert artifact.diagnostic_path.read_text(encoding="utf-8") == "new"


def test_commit_into_existing_replaces_only_after_explicit_authority(tmp_path):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    diagnostic = episode_dir / "motor6_diagnostics.jsonl"
    diagnostic.write_text("prior", encoding="utf-8")
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.diagnostic_path.write_text("new", encoding="utf-8")

    committed = artifact.commit_into_existing(
        episode_dir,
        allow_existing_destination=True,
    )

    assert committed == diagnostic
    assert diagnostic.read_text(encoding="utf-8") == "new"
    assert not artifact.diagnostic_path.exists()


def test_save_failure_before_commit_keeps_prior_diagnostic_and_discards_temp(tmp_path):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    diagnostic = episode_dir / "motor6_diagnostics.jsonl"
    diagnostic.write_text("prior", encoding="utf-8")
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    temp_path = artifact.diagnostic_path
    temp_path.write_text("new", encoding="utf-8")

    try:
        raise RuntimeError("injected HDF5 failure")
    except RuntimeError:
        artifact.discard()

    assert diagnostic.read_text(encoding="utf-8") == "prior"
    assert not temp_path.exists()


def test_failed_replace_releases_claim_and_keeps_artifact_discardable(
    tmp_path,
    monkeypatch,
):
    episode_dir = tmp_path / "episode_4"
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    artifact.diagnostic_path.write_text("attempt-data", encoding="utf-8")
    diagnostic_path = artifact.diagnostic_path

    def fail_replace(_source, _destination):
        raise OSError("injected replace failure")

    monkeypatch.setattr("aloha.episode_attempt.os.replace", fail_replace)

    with pytest.raises(OSError, match="injected replace failure"):
        artifact.commit(episode_dir)

    assert not episode_dir.exists()
    assert diagnostic_path.read_text(encoding="utf-8") == "attempt-data"

    artifact.discard()

    assert not diagnostic_path.exists()


def test_attempt_artifact_requires_factory_construction(tmp_path):
    with pytest.raises(TypeError, match=r"AttemptArtifact\.create"):
        AttemptArtifact(tmp_path / "not-owned.jsonl")


def test_diagnostic_path_is_read_only_and_cannot_be_retargeted(tmp_path):
    artifact = AttemptArtifact.create(tmp_path, "episode_4")
    original_path = artifact.diagnostic_path
    unrelated = tmp_path / "unrelated.jsonl"
    unrelated.write_text("keep", encoding="utf-8")

    with pytest.raises(AttributeError):
        artifact.diagnostic_path = unrelated

    artifact.discard()

    assert not original_path.exists()
    assert unrelated.read_text(encoding="utf-8") == "keep"


def test_fresh_episode_path_proceeds_without_overwrite_authority(tmp_path):
    prompts = []

    decision = check_episode_index(
        tmp_path,
        4,
        input_fn=lambda prompt: prompts.append(prompt) or "unexpected",
        logger=lambda _message: None,
    )

    assert decision.proceed is True
    assert decision.allow_existing is False
    assert prompts == []


def test_next_available_episode_index_returns_free_start_without_logging(tmp_path):
    logs = []

    episode_idx = find_next_available_episode_index(
        tmp_path,
        start_index=10,
        logger=logs.append,
    )

    assert episode_idx == 10
    assert logs == []


def test_next_available_episode_index_skips_directory_and_legacy_file(tmp_path):
    (tmp_path / "episode_11").mkdir()
    (tmp_path / "episode_12.hdf5").touch()
    logs = []

    episode_idx = find_next_available_episode_index(
        tmp_path,
        start_index=11,
        logger=logs.append,
    )

    assert episode_idx == 13
    assert logs == [
        "[episode-index] episode_11 already exists; skipping.",
        "[episode-index] episode_12 already exists; skipping.",
    ]


@pytest.mark.parametrize("entry_kind", ["file", "dangling", "directory"])
def test_next_available_episode_index_skips_any_live_claim_entry_once(
    tmp_path,
    entry_kind,
):
    claim_path = tmp_path / ".episode_11.claim"
    if entry_kind == "file":
        claim_path.write_text("owner", encoding="utf-8")
    elif entry_kind == "dangling":
        claim_path.symlink_to(tmp_path / "missing-target")
    else:
        claim_path.mkdir()
    logs = []

    episode_idx = find_next_available_episode_index(
        tmp_path,
        start_index=11,
        logger=logs.append,
    )

    assert episode_idx == 12
    assert logs == ["[episode-index] episode_11 already exists; skipping."]


@pytest.mark.parametrize("occupied_name", ["episode_11", "episode_11.hdf5"])
def test_next_available_episode_index_skips_dangling_symlink(
    tmp_path,
    occupied_name,
):
    (tmp_path / occupied_name).symlink_to(tmp_path / "missing-target")
    logs = []

    episode_idx = find_next_available_episode_index(
        tmp_path,
        start_index=11,
        logger=logs.append,
    )

    assert episode_idx == 12
    assert logs == ["[episode-index] episode_11 already exists; skipping."]


def test_next_available_episode_index_rejects_negative_start(tmp_path):
    with pytest.raises(ValueError, match="non-negative"):
        find_next_available_episode_index(tmp_path, start_index=-1)


def test_existing_episode_confirmed_yes_grants_overwrite_authority(tmp_path):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    prompts = []
    logs = []

    decision = check_episode_index(
        tmp_path,
        4,
        input_fn=lambda prompt: prompts.append(prompt) or "y",
        logger=logs.append,
    )

    assert decision.proceed is True
    assert decision.allow_existing is True
    assert prompts == [
        f"Episode path '{episode_dir}' already exists. "
        "Do you want to overwrite it? (y/n): "
    ]
    assert logs == ["Overwriting episode 4."]


def test_existing_episode_rejected_returns_no_authority(tmp_path):
    episode_dir = tmp_path / "episode_4"
    episode_dir.mkdir()
    logs = []

    decision = check_episode_index(
        tmp_path,
        4,
        input_fn=lambda _prompt: "n",
        logger=logs.append,
    )

    assert decision.proceed is False
    assert decision.allow_existing is False
    assert logs == ["Not overwriting the file. Operation aborted."]
