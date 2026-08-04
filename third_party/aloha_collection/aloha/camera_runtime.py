"""Lifecycle owner for camera subscriptions isolated from robot callbacks."""

from __future__ import annotations

from dataclasses import dataclass, field
import threading
from typing import Callable

from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node

from aloha.robot_utils import ImageRecorder


@dataclass
class CameraRuntime:
    """Own one camera-only ROS node, executor, and spin thread."""

    node: object
    executor: object
    thread: threading.Thread
    image_recorder: ImageRecorder
    _closed: bool = False
    _close_lock: threading.Lock = field(
        default_factory=threading.Lock,
        repr=False,
    )
    _shutdown_thread: threading.Thread | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _shutdown_result: bool | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _shutdown_error: BaseException | None = field(
        default=None,
        init=False,
        repr=False,
    )

    @classmethod
    def create(
        cls,
        *,
        config: dict,
        context: object,
        node_factory: Callable[[], object] | None = None,
        executor_factory: Callable[[], object] | None = None,
        image_recorder_factory: Callable[..., ImageRecorder] = ImageRecorder,
        thread_factory: Callable[..., threading.Thread] = threading.Thread,
        logger: Callable[[str], None] = print,
    ) -> "CameraRuntime":
        """Create all resources and roll back partial construction."""
        make_node = node_factory or (
            lambda: Node("aloha_camera_recorder", context=context)
        )
        make_executor = executor_factory or (
            lambda: SingleThreadedExecutor(context=context)
        )
        node = None
        executor = None
        added = False
        try:
            node = make_node()
            executor = make_executor()
            executor.add_node(node)
            added = True
            image_recorder = image_recorder_factory(
                config=config,
                node=node,
            )
            thread = thread_factory(
                target=executor.spin,
                name="aloha-camera-executor",
                daemon=True,
            )
            runtime = cls(
                node=node,
                executor=executor,
                thread=thread,
                image_recorder=image_recorder,
            )
            thread.start()
            try:
                logger(
                    "[camera-runtime] isolated executor started; "
                    "QoS=BEST_EFFORT/KEEP_LAST/depth=1"
                )
            except Exception:
                pass
            return runtime
        except BaseException:
            if executor is not None:
                try:
                    executor.shutdown(timeout_sec=1.0)
                except BaseException as cleanup_error:
                    logger(
                        "[camera-runtime] rollback executor shutdown "
                        f"failed: {cleanup_error}"
                    )
                if added:
                    try:
                        executor.remove_node(node)
                    except BaseException as cleanup_error:
                        logger(
                            "[camera-runtime] rollback remove node "
                            f"failed: {cleanup_error}"
                        )
            if node is not None:
                try:
                    node.destroy_node()
                except BaseException as cleanup_error:
                    logger(
                        "[camera-runtime] rollback destroy node "
                        f"failed: {cleanup_error}"
                    )
            raise

    def close(self) -> None:
        """Stop the executor and release the node exactly once."""
        with self._close_lock:
            if self._closed:
                return

            if self._shutdown_thread is None:
                def shutdown_executor() -> None:
                    try:
                        self._shutdown_result = self.executor.shutdown(
                            timeout_sec=None,
                        )
                    except BaseException as exc:
                        self._shutdown_error = exc

                self._shutdown_thread = threading.Thread(
                    target=shutdown_executor,
                    name="aloha-camera-executor-shutdown",
                    daemon=True,
                )
                self._shutdown_thread.start()

            self._shutdown_thread.join(timeout=1.0)
            if self._shutdown_thread.is_alive():
                raise RuntimeError(
                    "camera executor shutdown did not complete"
                )
            if self._shutdown_error is not None:
                raise self._shutdown_error
            if self._shutdown_result is not True:
                raise RuntimeError(
                    "camera executor shutdown did not complete"
                )

            self.thread.join(timeout=1.0)
            if self.thread.is_alive():
                raise RuntimeError("camera executor thread did not stop")

            first_error = None
            try:
                self.executor.remove_node(self.node)
            except BaseException as exc:
                first_error = exc
            finally:
                try:
                    self.node.destroy_node()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
            self._closed = True

            if first_error is not None:
                raise first_error
