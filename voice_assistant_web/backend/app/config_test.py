from .config import Settings


def test_default_rl_token_checkpoint_is_cam4_query_checkpoint(monkeypatch):
    monkeypatch.delenv("RLT_RL_TOKEN_CHECKPOINT_PATH", raising=False)

    settings = Settings()

    assert "eii_rinse_11repo_cam4_fullft_rl_token_small_query" in settings.rlt_rl_token_checkpoint_path
    assert "rinse_11repo_rl_token_small_query_512_from_9000_20260615/9999" in settings.rlt_rl_token_checkpoint_path
    assert "cam3" not in settings.rlt_rl_token_checkpoint_path
    assert "without_rinse" not in settings.rlt_rl_token_checkpoint_path
    assert "2048" not in settings.rlt_rl_token_checkpoint_path
