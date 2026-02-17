
import argparse
import logging
import pathlib
import sys

# Add src to path if needed (though installation should handle it)
# sys.path.append(str(pathlib.Path(__file__).parent.parent / "src"))

import jax
from flax import nnx
import numpy as np
import optax

from pi_value_function.config import PiValueConfig
from pi_value_function.pi_value import PiValue
from pi_value_function.serving.value_policy import ValuePolicy
from pi_value_function.training import checkpoint_manager as ckpt_manager
from pi_value_function.training.checkpoint_downloader import download_gemma_from_kaggle
from openpi.models.tokenizer import Gemma3Tokenizer
from openpi.serving import websocket_policy_server

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Start Pi Value Function Server")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint directory (containing params.msgpack or similar)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8001, help="Port to bind to")
    parser.add_argument(
        "--return-distribution",
        action="store_true",
        help="Return 201-bin value distribution in addition to scalar expected value.",
    )
    args = parser.parse_args()

    # 1. Load checkpoint first to detect architecture
    ckpt_path = pathlib.Path(args.checkpoint).resolve()
    print(f"Loading checkpoint from {ckpt_path}")

    step_number = int(ckpt_path.name)
    checkpoint_base_dir = ckpt_path.parent.resolve()
    print(f"Restoring model from {checkpoint_base_dir} at step {step_number}")

    import orbax.checkpoint as ocp
    from flax.traverse_util import flatten_dict, unflatten_dict

    ckpt_step_dir = checkpoint_base_dir / str(step_number) / "state"

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

    # NNX saves state with "value" wrappers - unwrap them
    flat_state = flatten_dict(model_state_data)
    if all(kp[-1] == "value" for kp in flat_state):
        print("Unwrapping 'value' keys from checkpoint state...")
        flat_state = {kp[:-1]: v for kp, v in flat_state.items()}
        model_state_data = unflatten_dict(flat_state)

    # Detect checkpoint format: old (value_proj) vs new (value_mlp)
    flat_ckpt = flatten_dict(model_state_data)
    has_value_proj = any('value_proj' in str(k) for k in flat_ckpt)
    has_value_mlp = any('value_mlp' in str(k) for k in flat_ckpt)

    if has_value_proj and not has_value_mlp:
        value_head_layers = 1
        print("Detected OLD checkpoint format (value_proj → single Linear head)")
    else:
        value_head_layers = 2
        print("Detected NEW checkpoint format (value_mlp → Linear+GELU+Linear head)")

    # 2. Create model with matching architecture
    config = PiValueConfig(
        value_dims=201,
        value_min=-1.0,
        value_max=0.0,
        gemma_variant="gemma-3-270m",
        siglip_variant="siglip2-so400m-patch16-384",
        value_head_layers=value_head_layers,
    )

    rng = jax.random.PRNGKey(0)
    print("Initializing model...")
    model = config.create(rng)

    # 3. Load checkpoint parameters into model
    _, model_state = nnx.split(model)
    flat_model = flatten_dict(model_state.to_pure_dict())

    # Normalize keys: nnx.Sequential uses integer list indices (0, 2) but
    # checkpoint restores them as string dict keys ("0", "2").
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

    print(f"Successfully loaded model parameters from step {restored['step']}")

    # 4. Initialize Tokenizer
    # We need the tokenizer model file. 
    # download_gemma_from_kaggle ensures it's in cache and returns path.
    _, tokenizer_path = download_gemma_from_kaggle()
    tokenizer = Gemma3Tokenizer(path=tokenizer_path, max_len=48)
    print("Tokenizer initialized.")

    # 5. Create Policy
    policy = ValuePolicy(model, tokenizer, return_distribution=args.return_distribution)

    # 6. Warmup JIT compilation
    print("Running JIT warmup...")
    warmup_obs = {
        "observation/exterior_image_1_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image_left": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/joint_position": np.zeros(7, dtype=np.float32),
        "observation/gripper_position": np.zeros(1, dtype=np.float32),
        "prompt": "warmup",
    }
    _ = policy.infer(warmup_obs)
    print("JIT warmup complete.")

    # 7. Start Server
    print(f"Starting Value Policy Server on {args.host}:{args.port}")
    server = websocket_policy_server.WebsocketPolicyServer(policy, host=args.host, port=args.port)
    server.serve_forever()

if __name__ == "__main__":
    main()
