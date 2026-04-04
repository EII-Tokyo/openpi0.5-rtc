import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Supports standard diffusion inference (`infer`) and optional subtask decoding (`infer_subtask`).
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: int | None = None,
        metadata: dict | None = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()

    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))

        prev_total_time = None
        prev_send_ms = None
        while True:
            try:
                start_time = time.monotonic()
                recv_started_at = time.monotonic()
                payload = await websocket.recv()
                recv_ms = (time.monotonic() - recv_started_at) * 1000.0
                unpack_started_at = time.monotonic()
                data = msgpack_numpy.unpackb(payload)
                unpack_ms = (time.monotonic() - unpack_started_at) * 1000.0

                obs = data.get("obs", None)  
                prev_action = data.get("prev_action", None)     
                use_rtc = data.get("use_rtc", False)
                decode_subtask = data.get("decode_subtask", False)
                max_new_tokens = data.get("max_new_tokens", None)
                temperature = data.get("temperature", 0.0)
                infer_started_at = time.monotonic()
                if decode_subtask:
                    if not hasattr(self._policy, "infer_subtask"):
                        raise NotImplementedError("Policy does not support subtask decoding.")
                    action = self._policy.infer_subtask(  # type: ignore[attr-defined]
                        obs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                    )
                else:
                    action = self._policy.infer(obs, prev_action, use_rtc)
                infer_ms = (time.monotonic() - infer_started_at) * 1000.0

                action["server_timing"] = {
                    "recv_ms": recv_ms,
                    "unpack_ms": unpack_ms,
                    "infer_ms": infer_ms,
                }
                if prev_total_time is not None:
                    # We can only record the last total time since we also want to include the send time.
                    action["server_timing"]["prev_total_ms"] = prev_total_time * 1000
                if prev_send_ms is not None:
                    action["server_timing"]["prev_send_ms"] = prev_send_ms

                pack_started_at = time.monotonic()
                packed = packer.pack(action)
                pack_ms = (time.monotonic() - pack_started_at) * 1000.0
                action["server_timing"]["pack_ms"] = pack_ms
                packed = packer.pack(action)
                send_started_at = time.monotonic()
                await websocket.send(packed)
                prev_send_ms = (time.monotonic() - send_started_at) * 1000.0
                prev_total_time = time.monotonic() - start_time

            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise


def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None
