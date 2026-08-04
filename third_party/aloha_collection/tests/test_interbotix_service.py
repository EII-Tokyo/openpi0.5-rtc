import pytest
from types import SimpleNamespace

from aloha.interbotix_service import (
    InterbotixServiceError,
    InterbotixServiceTimeout,
    set_gravity_compensation_with_timeout,
    set_operating_modes_with_timeout,
    torque_enable_with_timeout,
    wait_for_service_future,
)


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


class RaisingNode:
    def wait_until_future_complete(self, future, timeout_sec=None):
        raise RuntimeError("ROS context is not valid")


class CompletedFuture:
    def __init__(self, *, result=None, exception=None):
        self._result = result
        self._exception = exception

    def done(self):
        return True

    def exception(self):
        return self._exception

    def result(self):
        return self._result


class RecordingClient:
    def __init__(self, future, *, available=True):
        self.future = future
        self.available = available
        self.requests = []

    def call_async(self, request):
        self.requests.append(request)
        return self.future

    def wait_for_service(self, timeout_sec=None):
        self.wait_timeout = timeout_sec
        return self.available


class FakeCore:
    def __init__(self, future):
        self.robot_name = "leader_left"
        self.node = RecordingNode()
        self.srv_set_op_modes = RecordingClient(future)
        self.srv_torque = RecordingClient(future)

    def get_node(self):
        return self.node


class FakeRobot:
    def __init__(self, future):
        self.core = FakeCore(future)


class GravityNode(RecordingNode):
    def __init__(self, future, *, available=True, wait_exception=None):
        super().__init__()
        self.client = RecordingClient(future, available=available)
        self.created = []
        self.destroyed = []
        self.wait_exception = wait_exception

    def create_client(self, service_type, service_name):
        self.created.append((service_type, service_name))
        return self.client

    def destroy_client(self, client):
        self.destroyed.append(client)

    def wait_until_future_complete(self, future, timeout_sec=None):
        if self.wait_exception is not None:
            raise self.wait_exception
        super().wait_until_future_complete(future, timeout_sec=timeout_sec)


class GravityCore:
    robot_name = "leader_left"
    ns = "/leader_left"

    def __init__(self, future, **node_kwargs):
        self.node = GravityNode(future, **node_kwargs)

    def get_node(self):
        return self.node


class GravityRobot:
    def __init__(self, future, **node_kwargs):
        self.core = GravityCore(future, **node_kwargs)


def test_unfinished_service_future_is_cancelled_and_times_out():
    node = RecordingNode()
    future = PendingFuture()

    with pytest.raises(
        InterbotixServiceTimeout,
        match="leader_left gripper operating mode.*0.25",
    ):
        wait_for_service_future(
            node,
            future,
            timeout_sec=0.25,
            operation="leader_left gripper operating mode",
        )

    assert node.calls == [(future, 0.25)]
    assert future.cancelled


def test_native_wait_exception_cancels_pending_future_before_propagating():
    future = PendingFuture()

    with pytest.raises(RuntimeError, match="ROS context is not valid"):
        wait_for_service_future(
            RaisingNode(),
            future,
            timeout_sec=0.25,
            operation="leader_left arm operating mode",
        )

    assert future.cancelled


def test_completed_service_exception_is_reported_with_operation_name():
    node = RecordingNode()
    future = CompletedFuture(exception=RuntimeError("sdk rejected request"))

    with pytest.raises(
        InterbotixServiceError,
        match="leader_left arm operating mode.*sdk rejected request",
    ):
        wait_for_service_future(
            node,
            future,
            timeout_sec=0.25,
            operation="leader_left arm operating mode",
        )


def test_bounded_operating_mode_and_torque_calls_build_expected_requests():
    response = object()
    future = CompletedFuture(result=response)
    robot = FakeRobot(future)

    assert set_operating_modes_with_timeout(
        robot,
        "group",
        "arm",
        "position",
        timeout_sec=0.5,
        request_factory=SimpleNamespace,
    ) is response
    assert torque_enable_with_timeout(
        robot,
        "single",
        "gripper",
        True,
        timeout_sec=0.5,
        request_factory=SimpleNamespace,
    ) is response

    mode_request = robot.core.srv_set_op_modes.requests[0]
    assert (mode_request.cmd_type, mode_request.name, mode_request.mode) == (
        "group",
        "arm",
        "position",
    )
    assert (
        mode_request.profile_type,
        mode_request.profile_velocity,
        mode_request.profile_acceleration,
    ) == ("velocity", 0, 0)
    torque_request = robot.core.srv_torque.requests[0]
    assert (
        torque_request.cmd_type,
        torque_request.name,
        torque_request.enable,
    ) == ("single", "gripper", True)
    assert robot.core.node.calls == [(future, 0.5), (future, 0.5)]


def test_gravity_compensation_service_availability_and_future_are_bounded():
    response = object()
    future = CompletedFuture(result=response)
    robot = GravityRobot(future)
    service_type = object()

    assert set_gravity_compensation_with_timeout(
        robot,
        False,
        timeout_sec=0.5,
        service_type=service_type,
        request_factory=SimpleNamespace,
    ) is response

    node = robot.core.node
    assert node.created == [
        (service_type, "/leader_left/gravity_compensation_enable")
    ]
    assert node.client.wait_timeout == 0.5
    assert node.client.requests[0].data is False
    assert node.calls == [(future, 0.5)]
    assert node.destroyed == [node.client]


def test_bounded_operating_mode_forwards_explicit_gripper_profile():
    response = object()
    future = CompletedFuture(result=response)
    robot = FakeRobot(future)

    assert set_operating_modes_with_timeout(
        robot,
        "single",
        "gripper",
        "current_based_position",
        timeout_sec=0.5,
        profile_type="velocity",
        profile_velocity=50,
        profile_acceleration=10,
        request_factory=SimpleNamespace,
    ) is response

    request = robot.core.srv_set_op_modes.requests[0]
    assert (
        request.cmd_type,
        request.name,
        request.mode,
        request.profile_type,
        request.profile_velocity,
        request.profile_acceleration,
    ) == (
        "single",
        "gripper",
        "current_based_position",
        "velocity",
        50,
        10,
    )


def test_gravity_compensation_client_is_destroyed_when_service_is_unavailable():
    robot = GravityRobot(CompletedFuture(result=object()), available=False)

    with pytest.raises(InterbotixServiceTimeout, match="service unavailable"):
        set_gravity_compensation_with_timeout(
            robot,
            True,
            timeout_sec=0.5,
            service_type=object(),
            request_factory=SimpleNamespace,
        )

    assert robot.core.node.destroyed == [robot.core.node.client]


@pytest.mark.parametrize(
    "future,node_kwargs,expected_exception",
    [
        (PendingFuture(), {}, InterbotixServiceTimeout),
        (
            PendingFuture(),
            {"wait_exception": RuntimeError("ROS context is not valid")},
            RuntimeError,
        ),
    ],
)
def test_gravity_compensation_client_is_destroyed_when_future_wait_fails(
    future,
    node_kwargs,
    expected_exception,
):
    robot = GravityRobot(future, **node_kwargs)

    with pytest.raises(expected_exception):
        set_gravity_compensation_with_timeout(
            robot,
            True,
            timeout_sec=0.5,
            service_type=object(),
            request_factory=SimpleNamespace,
        )

    assert future.cancelled
    assert robot.core.node.destroyed == [robot.core.node.client]
