# Eight-task Sol inference

This branch wires the reviewed `sol62-twist34-merged-eight-task-v1-20261121` prompt into `GptMemoryBackend`. Its default model is `gpt-6.2-sol`, with low reasoning. It uses one current set of high/low/left-wrist/right-wrist images, lossless PNG at the supplied resolution, and no synthetic image history or inference dropout. The exact English/Chinese prompts are in the adjacent files.

Sol returns exactly `bottle_description`, `memory`, and integer `current_task_index` 1–8. The decoder rejects extra fields, booleans/fractional indices, adjustment-only memory, half a compound disposal, duplicate completion records, and noncanonical completion text. It converts the index into the exact eight-task catalog string.

The previous complete three-string state is the next request input; the returned complete state replaces it. The runtime does not append, reconstruct, or infer memory resets from task-index changes. The prompt instructs Sol to clear old bottle memory at the reviewed new-attempt boundary and to retain the final return task text when its memory is empty. Explicit episode reset invalidates all earlier pending results. There is no time-based result expiry: a valid result from the current episode is accepted regardless of elapsed time, and its complete state remains usable while the next request runs. Explicit reset/closure still isolates prior episodes; HTTP failures and the initial-result wait retain visible timeouts.

## What separate memory/current task means

These remain separate JSON string fields inside the existing `long_memory` observation field. The existing training tokenizer renders them as:

```text
Bottle description: green bottle
Memory: I picked up the bottle with my left gripper.
Current task: Unscrew the bottle cap.
```

They are labeled parts of the same language-token input, not independent encoders or model channels. The global task, tokenizer, low-level training transforms, task catalog, labels, dropout, optimizer and training configs are unchanged.

## Standard entrypoint, opt-in only

The standard `scripts/serve_policy.py` supports `--sol-high-level` or `OPENPI_SOL_HIGH_LEVEL=true`. It rejects checkpoints outside this eight-task completed-memory family. Root `docker-compose.yml` retains the existing checkpoint selection and adds this opt-in environment setting (default false), forwards the existing `OPENAI_API_KEY` environment variable and mounts `/home/eii/high-level-datasets` at `/high-level-data`. No alternate Compose file or robot deployment is created. Sol is installed only after dummy model warmup, which never triggers a paid request. Actual robot/ROS/GPU execution has not been tested by these offline tests.

Every paid dispatch requires an explicit writable `OPENPI_HIGH_LEVEL_DATASET_DIR`. The archive stores exact PNG bytes/SHA256, reconstructible request/schema, actual image-aligned ROS header timestamps and their domain, prior state, raw response/error, parsed output, usage, latency and applied/discarded decision. Missing/misaligned timestamps or archive failures block dispatch; persistence failures block application. Cost stays explicitly unknown unless a caller supplies a verified usage-cost calculator; no pricing is invented. Session and episode grouping are retained.

The robot recorder captures images and ROS timestamps under the same existing image lock. The standard WebSocket client sends and waits for an explicit reset acknowledgment only when server metadata advertises this protocol. Non-high-level clients retain their previous reset behavior. RTC arguments pass unchanged through the wrapper.

## Isolation and validation

The 15000-step robot deployment uses the actual training branch `codex/twist34-migrate3000-20261121`, with inference repairs merged after training finished. Training configuration, model, data transforms and task catalog remain identical to training commit `a46633110419b1aab3ff666a14b5de6529f98be9`. The web Inference conditioning drawer reads the exact active system prompt, reviewed Chinese translation and output schema from the running policy status. Changing conditions reconnects and acknowledges the high-level reset if the prior WebSocket has closed.
