import os
from pathlib import Path

from .config import Settings


def test_backend_pytest_uses_isolated_segment_db_path():
    db_path = Path(os.environ["RLT_SEGMENT_DB_PATH"])

    assert db_path.parent.exists()
    assert not str(db_path).startswith("/app/")


def test_default_rl_token_checkpoint_is_lower_right_4layer_checkpoint(monkeypatch):
    monkeypatch.delenv("RLT_RL_TOKEN_CHECKPOINT_PATH", raising=False)

    settings = Settings()

    assert "rlt_lower_right_rl_token_ablation_20260701" in settings.rlt_rl_token_checkpoint_path
    assert "BEST/checkpoint" in settings.rlt_rl_token_checkpoint_path
    assert "cam3" not in settings.rlt_rl_token_checkpoint_path
    assert "without_rinse" not in settings.rlt_rl_token_checkpoint_path
