import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_PATH = ROOT / "launch" / "aloha_bringup.launch.py"


def load_launch_source():
    return LAUNCH_PATH.read_text(encoding="utf-8")


def load_launch_tree():
    return ast.parse(load_launch_source())


def function_node(name):
    for node in load_launch_tree().body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"missing function {name}")


def declared_launch_arguments():
    generated = function_node("generate_launch_description")
    declarations = {}
    requested = {
        "dynamixel_preflight",
        "dynamixel_preflight_attempts",
        "dynamixel_preflight_auto_reboot_input_voltage_alerts",
        "dynamixel_preflight_reboot_delay",
        "dynamixel_preflight_retry_delay",
    }
    for call in ast.walk(generated):
        if not isinstance(call, ast.Call):
            continue
        if not isinstance(call.func, ast.Name):
            continue
        if call.func.id != "DeclareLaunchArgument" or not call.args:
            continue
        name = ast.literal_eval(call.args[0])
        if name not in requested:
            continue
        keywords = {keyword.arg: keyword.value for keyword in call.keywords}
        default = ast.literal_eval(keywords["default_value"])
        declarations[name] = default
    return declarations


def test_preflight_launch_arguments_are_safe_by_default():
    declarations = declared_launch_arguments()

    assert declarations["dynamixel_preflight"] == "true"
    assert declarations["dynamixel_preflight_attempts"] == "3"
    assert declarations[
        "dynamixel_preflight_auto_reboot_input_voltage_alerts"
    ] == "true"
    assert declarations["dynamixel_preflight_reboot_delay"] == "1.0"
    assert declarations["dynamixel_preflight_retry_delay"] == "1.0"


def test_selected_arms_respects_launch_group_flags():
    node = function_node("_selected_arms")
    module = ast.Module(body=[node], type_ignores=[])
    namespace = {}
    exec(compile(module, str(LAUNCH_PATH), "exec"), namespace)
    selected_arms = namespace["_selected_arms"]
    config = {
        "leader_arms": [{"name": "leader_left"}],
        "follower_arms": [{"name": "follower_left"}],
    }

    assert list(
        selected_arms(
            config,
            launch_leaders=True,
            launch_followers=True,
        )
    ) == [
        ("leader", {"name": "leader_left"}),
        ("follower", {"name": "follower_left"}),
    ]
    assert list(
        selected_arms(
            config,
            launch_leaders=False,
            launch_followers=True,
        )
    ) == [("follower", {"name": "follower_left"})]


def test_preflight_runs_before_any_launch_action_is_constructed():
    launch_setup = function_node("launch_setup")
    preflight_calls = [
        call
        for call in ast.walk(launch_setup)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id == "_run_dynamixel_preflight"
    ]
    assert len(preflight_calls) == 1

    launch_action_names = {
        "GroupAction",
        "IncludeLaunchDescription",
        "LogInfo",
        "Node",
    }
    action_calls = [
        call
        for call in ast.walk(launch_setup)
        if isinstance(call, ast.Call)
        and isinstance(call.func, ast.Name)
        and call.func.id in launch_action_names
    ]
    assert action_calls
    assert preflight_calls[0].lineno < min(
        call.lineno for call in action_calls
    )


def test_launch_resolves_mode_and_motor_yaml_and_wraps_failure_context():
    source = load_launch_source()

    assert 'f"{role}_modes_{arm[\'orientation\']}.yaml"' in source
    assert 'f"{arm[\'model\']}.yaml"' in source
    assert "load_bus_expectation(" in source
    assert "run_preflight(" in source
    assert "ALOHA bringup aborted before node startup" in source
    assert "auto_reboot_input_voltage_alerts=" in source
    assert "reboot_delay=" in source


def test_runtime_declares_python_serial_and_dynamixel_sdk():
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    package_xml = (ROOT / "package.xml").read_text(encoding="utf-8")

    assert "python3-serial" in dockerfile
    assert "<exec_depend>python3-serial</exec_depend>" in package_xml
    assert "<exec_depend>dynamixel_sdk</exec_depend>" in package_xml


def test_dockerfile_does_not_source_ros_in_a_transient_default_shell_layer():
    dockerfile_lines = (ROOT / "Dockerfile").read_text(
        encoding="utf-8"
    ).splitlines()

    assert not any(
        line.strip().startswith("RUN source /opt/ros/")
        for line in dockerfile_lines
    )
