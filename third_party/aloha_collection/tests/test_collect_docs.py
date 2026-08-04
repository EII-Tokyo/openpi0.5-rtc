from pathlib import Path


README = (
    Path(__file__).resolve().parents[1] / "README.md"
).read_text(encoding="utf-8")


def test_readme_recommends_one_command_collection():
    assert "./scripts/collect.sh" in README
    assert "./scripts/collect.sh --status" in README
    assert "./scripts/collect.sh --dry-run" in README
    assert "故障排查备用流程" in README


def test_readme_preserves_manual_bringup_and_recorder_commands():
    assert "ros2 launch aloha aloha_bringup.launch.py" in README
    assert "python3 record_episodes_copy.py" in README


def test_readme_documents_current_parallel_home_and_sleep_contract():
    assert "第二次按 `b` 后，回位过程仍属于当前 episode" in README
    assert "四臂到达并锁定在采集 HOME 后" in README
    assert "left_arm = [0.0, -0.96, 1.16, 1.57, 0.0, -1.57]" in README
    assert "right_arm = [0.0, -0.96, 1.16, 0.0, 0.0, 0.0]" in README
    assert "左右两组并行启动" in README
    assert "四臂独立线程并行进入 sleep" in README
    assert "任一机械臂失败只报告一次，不自动重试" in README
    assert "采集 HOME 与退出时的 safe-sleep 位姿不是同一个概念" in README
