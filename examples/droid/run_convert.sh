#!/bin/bash
conda run -n droid_convert python convert_droid_failures_to_lerobot_parallel.py \
    --repo_id michios/droid_failure_v2 \
    --num_workers 1 \
    --resume \
    --restart-every 200
