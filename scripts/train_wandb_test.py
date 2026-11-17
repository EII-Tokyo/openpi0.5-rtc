"""Check W&B destination routing without contacting W&B or starting training."""

from unittest import mock

import pytest

from scripts import train
from openpi.training.config import TrainConfig


@pytest.mark.parametrize("resuming", [False, True])
@pytest.mark.parametrize("entity", [None, "yonglianglu2"])
def test_wandb_destination(tmp_path, resuming, entity):
    config = TrainConfig(
        name="wandb_destination_test",
        exp_name="test",
        checkpoint_base_dir=str(tmp_path),
        wandb_entity=entity,
    )
    config.checkpoint_dir.mkdir(parents=True)
    id_path = config.checkpoint_dir / "wandb_id.txt"
    if resuming:
        id_path.write_text("existing-run")
    with mock.patch.object(train, "wandb") as wandb:
        wandb.run.id = "new-run"
        train.init_wandb(config, resuming=resuming)
        kwargs = wandb.init.call_args.kwargs
        assert kwargs["project"] == config.project_name
        if entity is None:
            assert "entity" not in kwargs  # Preserve environment/default routing.
        else:
            assert kwargs["entity"] == entity
        if resuming:
            assert kwargs["resume"] == "must"
            assert kwargs["id"] == "existing-run"
            assert id_path.read_text() == "existing-run"
        else:
            assert kwargs["name"] == "test"
            assert id_path.read_text() == "new-run"


def test_wandb_disabled_does_not_require_checkpoint(tmp_path):
    config = TrainConfig(
        name="disabled",
        exp_name="test",
        checkpoint_base_dir=str(tmp_path),
        wandb_entity="yonglianglu2",
    )
    with mock.patch.object(train, "wandb") as wandb:
        train.init_wandb(config, resuming=False, enabled=False)
        wandb.init.assert_called_once_with(mode="disabled")
    assert not config.checkpoint_dir.exists()
