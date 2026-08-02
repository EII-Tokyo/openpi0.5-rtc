from tools.audit_aloha_viper_cad_finger_task5_geometry import _path_in_collision_scope


def test_right_follower_collision_scope_is_not_hardcoded_to_left_workcell() -> None:
    root = "/vx300s_right"
    prefixes = (
        "/vx300s_right/follower_right_left_finger_link/",
        "/vx300s_right/follower_right_right_finger_link/",
    )

    assert _path_in_collision_scope(
        "/vx300s_right/follower_right_left_finger_link/collisions/finger/mesh",
        robot_root=root,
        collider_prefixes=prefixes,
    )
    assert not _path_in_collision_scope(
        "/vx300s_left/follower_left_left_finger_link/collisions/finger/mesh",
        robot_root=root,
        collider_prefixes=prefixes,
    )
