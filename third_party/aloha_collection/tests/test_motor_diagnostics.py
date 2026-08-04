from types import SimpleNamespace

from aloha.motor_diagnostics import (
    CURRENT_REGISTERS,
    diagnostic_registers_for_robot,
    read_register_values_with_timeout,
)


def test_leader_omits_current_registers_and_follower_retains_them():
    leader_registers = diagnostic_registers_for_robot("leader_left")
    follower_registers = diagnostic_registers_for_robot("follower_right")

    assert not set(CURRENT_REGISTERS) & set(leader_registers)
    assert set(CURRENT_REGISTERS) <= set(follower_registers)


class PendingFuture:
    def __init__(self):
        self.cancelled = False

    def done(self):
        return False

    def cancel(self):
        self.cancelled = True


class RecordingNode:
    def __init__(self):
        self.calls = []

    def wait_until_future_complete(self, future, timeout_sec=None):
        self.calls.append((future, timeout_sec))


class RecordingClient:
    def __init__(self, future):
        self.future = future
        self.requests = []

    def call_async(self, request):
        self.requests.append(request)
        return self.future


class FakeCore:
    robot_name = "leader_left"

    def __init__(self, future):
        self.node = RecordingNode()
        self.srv_get_reg = RecordingClient(future)

    def get_node(self):
        return self.node


class FakeRobot:
    def __init__(self, future):
        self.core = FakeCore(future)


def test_timed_out_register_future_is_cancelled():
    future = PendingFuture()
    robot = FakeRobot(future)

    result = read_register_values_with_timeout(
        robot,
        "single",
        "wrist_rotate",
        "Operating_Mode",
        timeout_sec=0.02,
        request_factory=SimpleNamespace,
    )

    assert result is None
    assert future.cancelled
    assert robot.core.node.calls == [(future, 0.02)]
