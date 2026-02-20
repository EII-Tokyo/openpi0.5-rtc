import argparse
import logging
import pathlib

from flax import nnx
import jax
import numpy as np
import safetensors.torch
import torch

from openpi.models.tokenizer import Gemma3Tokenizer
from openpi.serving import websocket_policy_server
from pi_value_function.backbone_type import BACKBONE_QWEN3VL
from pi_value_function.backbone_type import BACKBONE_SIGLIP_GEMMA3
from pi_value_function.config import PiValueConfig
from pi_value_function.pi_value_qwen3vl_torch import PiValueQwen3VLTorch
from pi_value_function.serving.value_policy import ValuePolicy
from pi_value_function.serving.value_policy_torch import ValuePolicyTorch
from pi_value_function.training.checkpoint_downloader import download_gemma_from_kaggle

logging.basicConfig(level=logging.INFO)


def _resolve_step_dir(checkpoint_path: str) -> pathlib.Path:
    path = pathlib.Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint path does not exist: {path}")

    if path.is_dir() and path.name.isdigit():
        return path

    step_dirs = [d for d in path.iterdir() if d.is_dir() and d.name.isdigit()]
    if not step_dirs:
        raise ValueError(
            f"Expected checkpoint step directory or experiment directory with numeric step folders, got: {path}"
        )

    latest = max(step_dirs, key=lambda x: int(x.name))
    print(f"Resolved latest checkpoint step: {latest}")
    return latest


def _detect_backend(step_dir: pathlib.Path, requested: str) -> str:
    if requested in {"jax", "torch"}:
        return requested

    if (step_dir / "model.safetensors").exists() and (step_dir / "metadata.pt").exists():
        return "torch"
    if (step_dir / "state").exists():
        return "jax"

    raise ValueError(
        "Could not auto-detect backend. Expected either JAX files (`state/`) or torch files "
        "(`model.safetensors` and `metadata.pt`)."
    )


def _load_jax_policy(step_dir: pathlib.Path, return_distribution: bool) -> ValuePolicy:
    print(f"Loading JAX checkpoint from {step_dir}")

    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict, unflatten_dict

    ckpt_step_dir = step_dir / "state"
    if not ckpt_step_dir.exists():
        raise FileNotFoundError(f"JAX checkpoint missing state directory: {ckpt_step_dir}")

    mesh = jax.sharding.Mesh(jax.devices(), ("x",))
    sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())

    with ocp.PyTreeCheckpointer() as ckptr:
        metadata = ckptr.metadata(ckpt_step_dir)
        item = {k: metadata[k] for k in metadata}
        restored = ckptr.restore(
            ckpt_step_dir,
            ocp.args.PyTreeRestore(
                item=item,
                restore_args=jax.tree.map(
                    lambda _: ocp.ArrayRestoreArgs(sharding=sharding, restore_type=jax.Array),
                    item,
                ),
            ),
        )

    print(f"Loaded checkpoint from step: {restored['step']}")
    model_state_data = restored["model_state"]

    flat_state = flatten_dict(model_state_data)
    if flat_state and all(kp[-1] == "value" for kp in flat_state):
        print("Unwrapping 'value' keys from checkpoint state...")
        flat_state = {kp[:-1]: v for kp, v in flat_state.items()}
        model_state_data = unflatten_dict(flat_state)

    flat_ckpt = flatten_dict(model_state_data)
    has_value_proj = any("value_proj" in str(k) for k in flat_ckpt)
    has_value_mlp = any("value_mlp" in str(k) for k in flat_ckpt)

    if has_value_proj and not has_value_mlp:
        value_head_layers = 1
        print("Detected OLD checkpoint format (value_proj -> single Linear head)")
    else:
        value_head_layers = 2
        print("Detected NEW checkpoint format (value_mlp -> Linear+GELU+Linear head)")

    config = PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        backbone=BACKBONE_SIGLIP_GEMMA3,
        gemma_variant="gemma-3-270m",
        siglip_variant="siglip2-so400m-patch16-384",
        value_head_layers=value_head_layers,
    )

    rng = jax.random.PRNGKey(0)
    model = config.create(rng)

    _, model_state = nnx.split(model)
    flat_model = flatten_dict(model_state.to_pure_dict())

    def _normalize_key(key_tuple):
        return tuple(int(k) if isinstance(k, str) and k.isdigit() else k for k in key_tuple)

    ckpt_by_normalized = {_normalize_key(k): v for k, v in flat_ckpt.items()}

    for key in flat_model:
        norm_key = _normalize_key(key)
        if norm_key in ckpt_by_normalized:
            flat_model[key] = ckpt_by_normalized[norm_key]
        else:
            print(f"Warning: key {key} not found in checkpoint")
    restored_state = nnx.State(unflatten_dict(flat_model))
    nnx.update(model, restored_state)

    _, tokenizer_path = download_gemma_from_kaggle()
    tokenizer = Gemma3Tokenizer(path=tokenizer_path, max_len=48)

    return ValuePolicy(model, tokenizer, return_distribution=return_distribution)


def _load_torch_policy(step_dir: pathlib.Path, return_distribution: bool) -> ValuePolicyTorch:
    print(f"Loading torch checkpoint from {step_dir}")

    metadata_path = step_dir / "metadata.pt"
    model_path = step_dir / "model.safetensors"

    if not metadata_path.exists() or not model_path.exists():
        raise FileNotFoundError(
            f"Torch checkpoint requires both metadata.pt and model.safetensors, missing in {step_dir}"
        )

    runtime_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load on CPU first to avoid peak VRAM spikes from loading HF weights + checkpoint tensors simultaneously.
    load_device = torch.device("cpu")
    try:
        metadata = torch.load(metadata_path, map_location=load_device, weights_only=False)
    except TypeError:
        metadata = torch.load(metadata_path, map_location=load_device)

    config = PiValueConfig(
        value_dims=int(metadata.get("value_dims", 201)),
        value_min=float(metadata.get("value_min", -1.0)),
        value_max=float(metadata.get("value_max", 0.0)),
        backbone=BACKBONE_QWEN3VL,
        hf_model_id=str(metadata.get("hf_model_id", "Qwen/Qwen3-VL-2B-Instruct")),
        backbone_dtype=str(metadata.get("backbone_dtype", "bfloat16")),
        value_head_layers=int(metadata.get("value_head_layers", 2)),
    )

    model = PiValueQwen3VLTorch(config, device=load_device)
    safetensors.torch.load_model(model, model_path, device=str(load_device))
    if runtime_device.type == "cuda":
        model.to(runtime_device)
        model.device = runtime_device
    model.eval()

    return ValuePolicyTorch(
        model,
        return_distribution=return_distribution,
        device=runtime_device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Start Pi Value Function Server")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to a checkpoint step directory or an experiment checkpoint directory",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument(
        "--return-distribution",
        action="store_true",
        help="Return 201-bin value distribution in addition to scalar expected value.",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["auto", "jax", "torch"],
        default="auto",
        help="Serving backend selection. 'auto' detects from checkpoint format.",
    )
    args = parser.parse_args()

    step_dir = _resolve_step_dir(args.checkpoint)
    backend = _detect_backend(step_dir, args.backend)
    print(f"Selected backend: {backend}")

    if backend == "jax":
        policy = _load_jax_policy(step_dir, args.return_distribution)
    else:
        policy = _load_torch_policy(step_dir, args.return_distribution)

    print("Running warmup inference...")
    warmup_obs = {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/exterior_image_2_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "warmup",
    }
    _ = policy.infer(warmup_obs)
    print("Warmup complete")

    print(f"Starting Value Policy Server on {args.host}:{args.port}")
    server = websocket_policy_server.WebsocketPolicyServer(policy, host=args.host, port=args.port)
    server.serve_forever()


if __name__ == "__main__":
    main()
