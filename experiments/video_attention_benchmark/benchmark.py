"""Offline inference benchmark; never used as a training launcher."""

import argparse
import dataclasses
import json
from pathlib import Path
import time

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import model as model_lib
from openpi.shared import array_typing as at
from openpi.training.config import get_config


def main(args):
    native = get_config(args.config_name).model
    # An inference-only comparison. Training continues to load named configs directly.
    config = dataclasses.replace(native, vision_attention_backend=args.backend)
    model = config.load(model_lib.restore_params(Path(args.checkpoint_dir) / "params", dtype=jnp.bfloat16))
    model.eval()
    graph, state = nnx.split(model)

    def sample(state, rng, obs, noise, steps):
        return nnx.merge(graph, state).sample_action_chunk(rng, obs, noise=noise, denoising_steps=steps)

    run = jax.jit(sample)
    rng = np.random.default_rng(args.seed)
    height, width = config.image_resolution
    obs = model_lib.Observation(
        images={
            name: jax.device_put(rng.uniform(-1, 1, (1, args.frames, height, width, 3)).astype(np.float32))
            for name in config.image_keys
        },
        image_masks={name: jnp.ones((1,), dtype=bool) for name in config.image_keys},
        state=jnp.zeros((1, config.action_dim), dtype=jnp.float32),
        tokenized_prompt=jnp.asarray((np.arange(config.max_token_len) % 1000 + 100)[None], dtype=jnp.int32),
        tokenized_prompt_mask=jnp.ones((1, config.max_token_len), dtype=bool),
    )
    noise = jax.device_put(rng.normal(size=(1, config.action_horizon, config.action_dim)).astype(np.float32))
    key = jax.random.key(0)
    jax.block_until_ready((obs, noise))
    t = time.perf_counter()
    out = run(state, key, obs, noise, args.denoising_steps)
    jax.block_until_ready(out)
    first = time.perf_counter() - t
    for _ in range(args.warmups):
        jax.block_until_ready(run(state, key, obs, noise, args.denoising_steps))
    times = []
    for _ in range(args.repeats):
        t = time.perf_counter()
        out = run(state, key, obs, noise, args.denoising_steps)
        jax.block_until_ready(out)
        times.append(time.perf_counter() - t)
    actions = np.asarray(out, dtype=np.float32)
    assert np.isfinite(actions).all()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.save(target.with_suffix(".npy"), actions)
    with at.disable_typechecking():
        compiled = run.lower(state, key, obs, noise, args.denoising_steps).compile()
    hlo = compiled.as_text()
    target.with_suffix(".hlo.txt").write_text(hlo)
    result = {
        "status": "COMPLETED",
        "config": args.config_name,
        "backend": args.backend,
        "checkpoint": args.checkpoint_dir,
        "resolution": list(config.image_resolution),
        "frames_per_camera": args.frames,
        "camera_count": len(config.image_keys),
        "image_tokens_per_image": config.image_tokens_per_image,
        "text_tokens": config.max_token_len,
        "denoising_steps": args.denoising_steps,
        "device_kind": jax.devices()[0].device_kind,
        "jax_version": jax.__version__,
        "warmups": args.warmups,
        "repeats": args.repeats,
        "first_compile_and_inference_seconds": first,
        "seconds": times,
        "median_ms": float(np.median(times) * 1000),
        "p95_ms": float(np.percentile(times, 95) * 1000),
        "cudnn_fmha_in_executable": "__cudnn$fmha" in hlo,
        "scope": "native full action-chunk inference; inputs on device; synthetic normalized inputs; excludes preprocessing/network/load/JIT",
    }
    if args.backend == "cudnn":
        assert result["cudnn_fmha_in_executable"]
    target.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config-name", required=True)
    p.add_argument("--checkpoint-dir", required=True)
    p.add_argument("--backend", choices=["native", "cudnn"], default="native")
    p.add_argument("--frames", type=int, default=16)
    p.add_argument("--denoising-steps", type=int, default=10)
    p.add_argument("--warmups", type=int, default=3)
    p.add_argument("--repeats", type=int, default=30)
    p.add_argument("--seed", type=int, default=20261119)
    p.add_argument("--output", required=True)
    main(p.parse_args())
