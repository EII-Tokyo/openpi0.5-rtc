# RL Token Decoder Experiment Report

Goal: compare three ways to reduce teacher-forcing shortcut and make `z_rl` carry sample-specific state.

Healthy target: `real_loss < shuffled_loss << zero_loss`.
External reference: `pravsels/pi05-build-block-tower-rlt-v1` reported at 10k shuffled about +40% and zero about +359% over real.

| experiment | step | real | shuffled | zero | shuffled/real | zero/real | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| masked_tf075 | 5000 | 0.533203 | 0.543457 | 0.529053 | 1.020 | 0.994 | failed_order |
| masked_tf075 | 10000 | 0.510986 | 0.519287 | 0.516846 | 1.014 | 1.010 | failed_order |
| masked_tf075 | 12000 | 0.522461 | 0.535156 | 0.530273 | 1.024 | 1.014 | failed_order |
| query | 5000 | 0.333984 | 0.537598 | 6.121094 | 1.614 | 18.523 | ideal |
| query | 10000 | 0.293701 | 0.552734 | 5.460938 | 1.892 | 18.844 | ideal |
| query | 12000 | 0.288208 | 0.580078 | 5.480469 | 2.022 | 19.273 | ideal |
| query_margin | 5000 | 0.341064 | 0.537842 | 3.310547 | 1.572 | 9.789 | ideal |
| query_margin | 10000 | 0.301514 | 0.594971 | 4.605469 | 1.977 | 15.438 | ideal |
| query_margin | 12000 | 0.294434 | 0.579834 | 4.703125 | 1.978 | 16.164 | ideal |

## Experiment Descriptions

- `masked_tf075`: teacher-forced decoder with 75% random history masking
- `query`: no-history z-conditioned learned-query decoder
- `query_margin`: query decoder plus zero/shuffle hinge margin losses

## Interpretation

- `masked_tf075` is conservative: if it improves gap, teacher-forcing history was the main shortcut but the paper-like decoder can be retained.
- `query` is stronger: if it improves gap, no-history reconstruction is a better stage-1 objective for this data.
- `query_margin` is aggressive: it directly optimizes the ablation ordering, but must be checked for stable real reconstruction loss.

## Current Best

Best by zero/real then shuffled/real is `query` at step 12000: real=0.288208, shuffled/real=2.022, zero/real=19.273, verdict=ideal.
