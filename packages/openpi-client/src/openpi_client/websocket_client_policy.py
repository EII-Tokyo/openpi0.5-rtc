import logging
import time
from typing import Dict, Optional, Tuple
import numpy as np

from typing_extensions import override
import websockets.sync.client

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy


def _find_unsupported_numpy(value, path: str = "root"):
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            return path, value.dtype, value.shape
        return None
    if isinstance(value, np.generic):
        if value.dtype.kind in ("V", "O", "c"):
            return path, value.dtype, ()
        return None
    if isinstance(value, dict):
        for key, item in value.items():
            found = _find_unsupported_numpy(item, f"{path}.{key}")
            if found is not None:
                return found
        return None
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            found = _find_unsupported_numpy(item, f"{path}[{i}]")
            if found is not None:
                return found
        return None
    return None


def _collect_numpy_summaries(value, path: str = "root", out=None):
    if out is None:
        out = []
    if isinstance(value, np.ndarray):
        out.append(f"{path}: ndarray dtype={value.dtype} shape={value.shape}")
        return out
    if isinstance(value, np.generic):
        out.append(f"{path}: np.generic dtype={value.dtype}")
        return out
    if isinstance(value, dict):
        for key, item in value.items():
            _collect_numpy_summaries(item, f"{path}.{key}", out)
        return out
    if isinstance(value, (list, tuple)):
        for i, item in enumerate(value):
            _collect_numpy_summaries(item, f"{path}[{i}]", out)
        return out
    return out


class WebsocketClientPolicy(_base_policy.BasePolicy):
    """Implements the Policy interface by communicating with a server over websocket.

    See WebsocketPolicyServer for a corresponding server implementation.
    """

    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> Dict:
        return self._server_metadata

    def _wait_for_server(self) -> Tuple[websockets.sync.client.ClientConnection, Dict]:
        logging.info(f"Waiting for server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except ConnectionRefusedError:
                logging.info("Still waiting for server...")
                time.sleep(5)

    @override
    def infer(self, obs: Dict, prev_action: np.ndarray | None = None, use_rtc: bool = True) -> Dict:  # noqa: UP006
        data = {
            "obs": obs,
            "prev_action": prev_action,
            "use_rtc": use_rtc,
        }
        unsupported = _find_unsupported_numpy(data)
        if unsupported is not None:
            bad_path, bad_dtype, bad_shape = unsupported
            logging.error(
                "Unsupported numpy payload before msgpack: path=%s dtype=%s shape=%s",
                bad_path,
                bad_dtype,
                bad_shape,
            )
        try:
            data = self._packer.pack(data)
        except ValueError:
            summaries = _collect_numpy_summaries(data)
            logging.error("NumPy payload summary before msgpack failure:\n%s", "\n".join(summaries[:200]))
            raise
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we're expecting bytes; if the server sends a string, it's an error.
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    def infer_subtask(
        self,
        obs: Dict,
        *,
        max_new_tokens: int | None = None,
        temperature: float = 0.0,
        max_text_token_id: int = 240000,
    ) -> Dict:
        data = {
            "obs": obs,
            "decode_subtask": True,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "max_text_token_id": max_text_token_id,
        }
        data = self._packer.pack(data)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)

    @override
    def reset(self) -> None:
        pass
