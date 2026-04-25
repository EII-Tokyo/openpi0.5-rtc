#!/usr/bin/env python3
"""Load Aloha PI05 policy, JIT infer_subtask once, report GPU memory for this process (nvidia-smi)."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import subprocess
import sys
import time

logging.basicConfig(level=logging.INFO, force=True)


def _gpu_mem_mib_for_pid(pid: int) -> int | None:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_gpu_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 2 and parts[0].isdigit() and int(parts[0]) == pid:
            return int(parts[1])
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subtask-max", type=int, required=True, help="Override Pi0Config.subtask_max_token_len")
    p.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/twist_and_static_mixture_full_finetune/"
        "twist_and_static_mixture_full_finetune_vast_20260405_100600/39999",
    )
    args = p.parse_args()

    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.33")

    pid = os.getpid()
    mem0 = _gpu_mem_mib_for_pid(pid)

    from openpi.policies import aloha_policy as _aloha
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    base = _config.get_config("twist_and_static_mixture_full_finetune")
    cfg = dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, subtask_max_token_len=args.subtask_max),
    )
    logging.info("subtask_max_token_len=%s (override)", args.subtask_max)
    policy = _policy_config.create_trained_policy(cfg, args.ckpt)
    mem1 = _gpu_mem_mib_for_pid(pid)
    time.sleep(0.5)
    mem1b = _gpu_mem_mib_for_pid(pid)

    obs = _aloha.make_aloha_example()
    obs["actions_mask"] = False
    policy.infer_subtask(obs)
    time.sleep(0.5)
    mem2 = _gpu_mem_mib_for_pid(pid)

    print(
        f"subtask_max_token_len={args.subtask_max}  "
        f"pid={pid}  "
        f"gpu_mib: before_load={mem0}  after_load={mem1 or mem1b}  after_infer_subtask={mem2}",
        flush=True,
    )
    if mem2 is None or (mem1 is None and mem1b is None):
        print("(nvidia-smi did not list this pid; run on a machine with GPU and driver)", file=sys.stderr)


if __name__ == "__main__":
    main()
