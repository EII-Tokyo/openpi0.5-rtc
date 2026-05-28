# RL Token Network Design

## Goal

Build the first RL Token network on top of the existing trained pi0.5 checkpoint:

```text
/app/checkpoints/eii_data_system_without_rinse_cam3_fullft_h200_return_home_29repo/no_rinse_cam3_fullft_return_home_29repo_bs256_nw64_fsdp4_20260520/19000
```

This first version trains only an RL Token autoencoder. It does not change action prediction and does not run online RL yet.

## Confirmed Backbone Dimension

The current pi0.5 config uses `paligemma_variant="gemma_2b"`.

`gemma_2b` has:

```text
width = 2048
```

`Pi0` builds the image encoder with:

```text
num_classes = paligemma_config.width
```

Therefore image tokens produced by `self.PaliGemma.img(...)` are:

```text
image_tokens: [B, S_img, 2048]
```

The full VLA prefix representation is:

```text
H_vla: [B, N, 2048]
```

## Network Version

```python
encoder_layers = 4
decoder_layers = 4
hidden_dim = 2048
rl_token_dim = 2048
```

The compression bottleneck is token count, not feature dimension:

```text
[B, N, 2048] -> [B, 1, 2048] -> [B, N, 2048]
```

## Reconstruction Quality Checks

After training, reconstruction loss alone is not enough. The RL token must contain sample-specific information, so evaluation should compare three decoder conditions:

```text
real:     decode(z_rl(real sample), H_vla previous tokens)
shuffled: decode(z_rl(from another batch sample), H_vla previous tokens)
zero:     decode(0, H_vla previous tokens)
```

Expected healthy ordering:

```text
real_loss < shuffled_loss << zero_loss
```

Meaning:

- `real_loss` checks normal autoencoding quality.
- `shuffled_loss` checks whether the compressed token is tied to the correct visual state.
- `zero_loss` checks whether the decoder is cheating by reconstructing mostly from teacher-forced previous tokens and positional information.

## H_vla Extraction

Use the frozen pi0.5 model to compute prefix hidden states:

```text
observation
 -> preprocess/model transforms
 -> Pi0.embed_prefix(observation)
 -> PaliGemma.llm([prefix_tokens, None], ...)
 -> prefix_out
```

Use:

```text
H_vla = prefix_out
prefix_mask = mask returned by embed_prefix
```

Do not use only raw image embeddings as the reconstruction target. The target should be VLA backbone output after image, language, and state/prompt fusion.

## Encoder

Input:

```text
H_vla: [B, N, 2048]
learned_rl_query: [1, 1, 2048]
```

Broadcast and concatenate:

```text
encoder_input = concat([H_vla, learned_rl_query], axis=1)
encoder_input: [B, N + 1, 2048]
```

Run a bidirectional Transformer encoder:

```text
encoder_layers = 4
num_heads = 8 or 16
mlp_dim = 8192
```

Output:

```text
encoder_out: [B, N + 1, 2048]
z_rl = encoder_out[:, -1, :]
z_rl: [B, 2048]
```

## Decoder

Decoder depth is intentionally matched with encoder depth:

```text
encoder_layers = decoder_layers = 4
```

Use teacher-forced autoregressive reconstruction, matching the RLT objective:

```text
decoder_input_i = [project(z_rl), stop_gradient(H_vla[:, :i-1])]
```

The decoder is causal. Prediction i can see z_rl and the real previous prefix embeddings, but not the current or future target embedding.

```text
decoder_layers = 4
num_heads = 8 or 16
mlp_dim = 8192
```

Output:

```text
H_hat: [B, N, 2048]
```

## Training Objective

Freeze the existing pi0.5 policy and train only RL Token parameters.

Target:

```text
H_hat ~= stop_gradient(H_vla)
```

Loss:

```text
masked_mse(H_hat, stop_gradient(H_vla), prefix_mask)
```

Only valid prefix tokens participate in the loss. Padding tokens must be ignored.

## Trainable Parameters

Trainable:

```text
RL query token
encoder transformer, 4 layers
decoder reconstruction queries
decoder transformer, 2 layers
LayerNorm/projection parameters inside the RL Token module
```

Frozen:

```text
PaliGemma image encoder
PaliGemma/Gemma language backbone
action expert
action projection heads
all existing pi0.5 policy parameters
```

## Implementation Constraints

- Keep the RL Token module separate from normal policy inference in the first version.
- Do not route `z_rl` into the action expert yet.
- Use `jax.lax.stop_gradient` on `H_vla`.
- Use `prefix_mask` for both encoder masking and reconstruction loss.
- Do not hard-code the runtime prefix length `N`; use `max_token_len` only for learned reconstruction parameter allocation and slice per batch.
- Start with bf16-compatible layers because the frozen VLA checkpoint uses bf16.

## First Verification Target

A minimal smoke test should instantiate the RL Token module and verify:

```text
H_vla: [B, N, 2048]
prefix_mask: [B, N]
z_rl: [B, 2048]
H_hat: [B, N, 2048]
loss: scalar finite
```

Only after this autoencoder training path is stable should the RL token be used by actor-critic or online RL code.
