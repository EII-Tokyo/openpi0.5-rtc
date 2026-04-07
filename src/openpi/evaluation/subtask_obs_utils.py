"""Build policy observation dicts from LeRobot rows (shared by training-time eval and scripts)."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch

from openpi import transforms as _transforms

_PROMPT_FROM_TASK = _transforms.PromptFromLeRobotTask()


def subtask_cell_to_str(cell: object) -> str:
    if cell is None:
        return ""
    if isinstance(cell, (bytes, bytearray)):
        return cell.decode("utf-8", errors="replace")
    if hasattr(cell, "item") and not isinstance(cell, (str, bytes)):
        try:
            return subtask_cell_to_str(cell.item())
        except Exception:
            return str(cell)
    return str(cell)


def parse_json_bottle_state_subtask(raw: str | None) -> tuple[str, str] | None:
    """Parse LeRobot `subtask` column JSON; return (bottle_state, subtask) or None."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    try:
        d = json.loads(s)
    except json.JSONDecodeError:
        return None
    if not isinstance(d, dict):
        return None
    bs = d.get("bottle_state")
    st = d.get("subtask")
    if not isinstance(bs, str) or not isinstance(st, str):
        return None
    bs, st = bs.strip(), st.strip()
    if not bs or not st:
        return None
    return bs, st


def expand_action_chunk(raw: dict[str, Any], *, action_horizon: int) -> None:
    """Match training loader: actions are (horizon, dim). Single-frame rows are (dim,)."""
    if "action" not in raw:
        return
    act = np.asarray(raw["action"], dtype=np.float32)
    if act.ndim == 1:
        raw["action"] = np.tile(act, (action_horizon, 1))
    elif act.ndim == 2 and act.shape[0] == 1:
        raw["action"] = np.tile(act[0], (action_horizon, 1))


def lerobot_row_to_subtask_infer_obs(
    row: dict[str, Any],
    *,
    action_horizon: int,
    pop_subtask: bool = True,
) -> dict[str, Any]:
    """Single LeRobot sample dict -> raw obs dict for `Policy.infer_subtask_batch` (matches offline eval)."""
    raw: dict[str, Any] = {}
    for k, v in row.items():
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy()
        raw[k] = v
    if pop_subtask:
        raw.pop("subtask", None)
    expand_action_chunk(raw, action_horizon=action_horizon)
    return _PROMPT_FROM_TASK(raw)
