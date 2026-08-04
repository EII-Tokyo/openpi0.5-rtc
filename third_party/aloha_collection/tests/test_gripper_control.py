import pytest

from aloha import gripper_control


configure_follower_gripper_mode = (
    gripper_control.configure_follower_gripper_mode
)


def capture_configuration(follower_name):
    follower = object()
    calls = []

    configure_follower_gripper_mode(
        follower_name,
        follower,
        set_operating_modes=lambda *args, **kwargs: calls.append(
            (args, kwargs)
        ),
    )

    return follower, calls


def test_left_follower_uses_finite_velocity_profile():
    follower, calls = capture_configuration("follower_left")

    assert calls == [
        (
            (
                follower,
                "single",
                "gripper",
                "current_based_position",
            ),
            {
                "profile_type": "velocity",
                "profile_velocity": 50,
                "profile_acceleration": 10,
            },
        )
    ]


def test_right_follower_preserves_existing_zero_profile():
    follower, calls = capture_configuration("follower_right")

    assert calls == [
        (
            (
                follower,
                "single",
                "gripper",
                "current_based_position",
            ),
            {
                "profile_type": "velocity",
                "profile_velocity": 0,
                "profile_acceleration": 0,
            },
        )
    ]


def test_unknown_follower_is_rejected_before_mode_request():
    calls = []

    with pytest.raises(ValueError, match="unsupported follower"):
        configure_follower_gripper_mode(
            "follower_center",
            object(),
            set_operating_modes=lambda *args, **kwargs: calls.append(
                (args, kwargs)
            ),
        )

    assert calls == []


def test_idle_cleanup_restores_followers_and_disables_leader_grippers():
    robots = {
        "leader_left": object(),
        "follower_left": object(),
        "leader_right": object(),
        "follower_right": object(),
    }
    calls = []

    gripper_control.restore_gripper_idle_modes(
        robots,
        configure_follower_gripper=lambda name, bot: calls.append(
            ("configure", name, bot)
        ),
        torque_enable=lambda bot, cmd_type, name, enable: calls.append(
            ("torque", bot, cmd_type, name, enable)
        ),
        logger=lambda message: calls.append(("log", message)),
    )

    assert calls == [
        ("torque", robots["leader_left"], "single", "gripper", False),
        ("configure", "follower_left", robots["follower_left"]),
        ("torque", robots["follower_left"], "single", "gripper", True),
        ("torque", robots["leader_right"], "single", "gripper", False),
        ("configure", "follower_right", robots["follower_right"]),
        ("torque", robots["follower_right"], "single", "gripper", True),
    ]


def test_idle_cleanup_continues_after_one_robot_fails():
    robots = {
        "follower_left": object(),
        "leader_left": object(),
        "follower_right": object(),
        "camera_high": object(),
    }
    calls = []
    logs = []

    def configure(name, bot):
        calls.append(("configure", name, bot))
        if name == "follower_left":
            raise RuntimeError("left service unavailable")

    gripper_control.restore_gripper_idle_modes(
        robots,
        configure_follower_gripper=configure,
        torque_enable=lambda bot, cmd_type, name, enable: calls.append(
            ("torque", bot, cmd_type, name, enable)
        ),
        logger=logs.append,
    )

    assert calls == [
        ("configure", "follower_left", robots["follower_left"]),
        ("torque", robots["leader_left"], "single", "gripper", False),
        ("configure", "follower_right", robots["follower_right"]),
        ("torque", robots["follower_right"], "single", "gripper", True),
    ]
    assert len(logs) == 1
    assert "follower_left" in logs[0]
    assert "left service unavailable" in logs[0]
