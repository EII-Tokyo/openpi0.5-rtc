# ALOHA checkpoint entrypoints

Policy server: `scripts/serve_policy.py --model-dir CHECKPOINT --port 8001 --no-warmup-subtask`.
Runtime: `scripts/run_aloha_real.py --model-dir CHECKPOINT --host 127.0.0.1 --port 8001`.

The server selects the config from `manifest.json` or `export_metadata/training-settings.json`, preferring `export_metadata/config.py`. Old checkpoints without a config snapshot use the registered training config; `--config NAME` can identify legacy layouts. Conflicting config names or checkpoint steps fail startup. Norm stats always come from this checkpoint's assets.

Every manifest file is checked for existence, size, and SHA256 when recorded. Runtime source checksums exported by old experiments are provenance records, not immutable runtime requirements: intentional runtime fixes no longer require hardcoded hash allowlists. Weight/config-file hashes remain checked against each manifest.

The server publishes camera keys, image resolution, temporal/mosaic settings, action horizon/dimension and ALOHA adaptation mode. Runtime validates the model identity and installs the matching mosaic adapter automatically. No per-model runtime patches or model-name substring rules are used. Server warmup uses the same camera/time dimensions, with temporal input only for cameras selected for mosaic history.

Each migrated compose overlay declares `x-checkpoint-dir` once and aliases it in both service commands. Edit that one path to switch models. The base compose uses `OPENPI_CHECKPOINT_DIR`, defaulting to the jialong44 14999 checkpoint. Explicit overlays take precedence over that base default. Prompt settings remain in compose/web controls; task text is not guessed from model folder names.

Offline verification (no robot commands):

```sh
/.venv/bin/python3 /app/scripts/serve_policy.py --model-dir CHECKPOINT --check-only
```

This verifies files, norm stats, input transforms, tokenizer and warmup shapes, without loading the model or starting a task. A separate real server startup/inference checks model execution.

Optional runtime timing: add `--latency-diagnostics`. The existing control-rate logs and video preview remain in the shared runtime. Old deployment-specific camera patches are replaced by the shared checkpoint-configured adapter.

## Legacy resolution override

`docker-compose.rinse-22000.yml` intentionally retains `scripts/serve_rinse_22000.py`.
That older checkpoint has no exported configuration snapshot and its registered
recipe now says 448 pixels, while the verified deployment uses 224 pixels. The
compatibility entrypoint preserves the original resolution. A successful
`--check-only` validates the selected transforms; without an exported snapshot it
cannot establish which recipe was originally used for training.

The removed step-6000 entrypoint is replaced by the generic `--model-dir` command:
its exported configuration loads 224-pixel, six-frame input directly. The old
step-14000 entrypoint depended on that separate checkpoint's metadata and is no
longer used by the model overlays.
