import dataclasses
import json
import logging
import pathlib
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import tyro

from openpi.models import rl_token_eval
import openpi.models.model as _model
import openpi.training.config as _config
import openpi.training.data_loader as _data_loader


@dataclasses.dataclass
class Args:
    config_name: str
    checkpoint_dir: pathlib.Path
    num_batches: int = 20
    output_json: pathlib.Path | None = None
    wandb_project: str | None = None
    wandb_run_name: str | None = None


def _checkpoint_step_dir(path: pathlib.Path) -> pathlib.Path:
    if path.name == "params":
        return path.parent
    return path


def _to_float(value: Any) -> float:
    return float(np.asarray(value))


def _average_metrics(metric_dicts: list[dict[str, Any]]) -> dict[str, float]:
    keys = metric_dicts[0].keys()
    return {key: float(np.mean([_to_float(metrics[key]) for metrics in metric_dicts])) for key in keys}


def main(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    config = _config.get_config(args.config_name)
    checkpoint_dir = _checkpoint_step_dir(args.checkpoint_dir)

    logging.info("Loading model from %s", checkpoint_dir)
    model = config.model.load(_model.restore_params(checkpoint_dir / "params", dtype=jnp.bfloat16))
    model.eval()
    if getattr(model, "rl_token_autoencoder", None) is None:
        raise ValueError(f"Config {args.config_name} did not create an RL Token autoencoder.")

    data_loader = _data_loader.create_data_loader(
        config,
        shuffle=True,
        num_batches=args.num_batches,
    )

    @jax.jit
    def eval_batch(observation: _model.Observation):
        observation = _model.preprocess_observation(
            jax.random.key(0),
            observation,
            train=False,
            image_resolution=model.image_resolution,
        )
        h_vla, prefix_mask = model.embed_prefix_hidden(observation, drop_language=True)
        h_vla = jax.lax.stop_gradient(h_vla)
        return rl_token_eval.compute_reconstruction_ablations(model.rl_token_autoencoder, h_vla, prefix_mask).as_dict()

    metric_dicts = []
    for batch_idx, (observation, _) in enumerate(data_loader, start=1):
        metrics = eval_batch(observation)
        metric_dicts.append(metrics)
        logging.info(
            "batch=%d real=%.6f shuffled=%.6f zero=%.6f",
            batch_idx,
            _to_float(metrics["real_loss"]),
            _to_float(metrics["shuffled_loss"]),
            _to_float(metrics["zero_loss"]),
        )

    summary = _average_metrics(metric_dicts)
    print(json.dumps(summary, indent=2, sort_keys=True))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    if args.wandb_project is not None:
        import wandb

        wandb.init(project=args.wandb_project, name=args.wandb_run_name or f"{config.name}_rl_token_eval")
        wandb.log({f"rl_token_eval/{key}": value for key, value in summary.items()})
        wandb.finish()


if __name__ == "__main__":
    main(tyro.cli(Args))
