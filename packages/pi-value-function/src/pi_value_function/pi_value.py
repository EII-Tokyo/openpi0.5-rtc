"""Pi Value model implementation based on RECAP/π*0.6 paper.

This module implements a distributional value function using:
- SigLIP for image encoding (since Gemma3_270M doesn't have built-in vision)
- Custom Gemma3 wrapper that accepts pre-computed embeddings
- Categorical value distribution with 201 bins over range (-1, 0)
"""

import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
import einops

from openpi.models.model import Observation, preprocess_observation
import openpi.shared.array_typing as at
from pi_value_function.value_model_base import BaseValueModel, BaseValueModelConfig
import openpi.models.siglip as _siglip
from pi_value_function.gemma3 import Gemma3Module, get_config as get_gemma3_config, GEMMA3_VOCAB_SIZE

class PiValue(BaseValueModel):
    """Value model using SigLIP + Gemma3 backbone.

    Architecture:
    1. SigLIP encodes images to embeddings (projected to Gemma3 width)
    2. Gemma3 embedder encodes text tokens
    3. Image and text embeddings are concatenated
    4. An EOS (end-of-sequence) token is appended to the sequence
    5. Gemma3 transformer processes the combined sequence
    6. The hidden state at the EOS token position is extracted
    7. Linear projection to categorical value distribution (201 bins)

    The value distribution follows RECAP/π*0.6:
    - 201 categorical bins over range (-1, 0)
    - Bin 0 = -1.0 (failure/start), Bin 200 = 0.0 (success/completion)
    - Cross-entropy loss between predicted distribution and discretized returns
    """

    def __init__(self, config: BaseValueModelConfig, rngs: nnx.Rngs):
        self.value_dims = config.value_dims
        self.value_min = config.value_min
        self.value_max = config.value_max
        self.rngs = rngs

        # EOS token ID for Gemma3 (standard EOS token)
        # Gemma uses token ID 1 as EOS
        self.eos_token_id = 1

        # Get Gemma3 config
        gemma3_config = get_gemma3_config("gemma3_270m")
        gemma_width = gemma3_config.width  # 640 for 270M

        # Initialize custom Gemma3 module (with separate embed/forward)
        self.gemma = nnx_bridge.ToNNX(
            Gemma3Module(config=gemma3_config, embed_dtype="bfloat16")
        )
        self.gemma.lazy_init(rngs=rngs, method="initialize_params")

        # siglip2-so400m-patch14-224
        # Use pool_type="none" to get unpooled tokens: (batch, num_patches, width)
        # Set num_classes=gemma_width to project tokens to Gemma's dimension
        self.siglip = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=gemma_width,  # Project to Gemma's embedding dimension
                variant="So400m/14",
                pool_type="none",  # Return all tokens unpooled
                scan=True,
                dtype_mm="float32",
                head_zeroinit=False,
            )
        )
        self.siglip.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)

        # Value projection head: Gemma output -> categorical distribution
        self.value_proj = nnx.Linear(gemma_width, self.value_dims, rngs=rngs)

    # TODO: Do I need masks?
    # should not be autoregressive for value function

    @at.typecheck
    def embed_prefix(
        self, obs: Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        """Embed observation (images + text) into a sequence of embeddings with EOS token.

        Args:
            obs: Observation containing images, image_masks, tokenized_prompt, etc.

        Returns:
            tokens: Concatenated embeddings [batch, seq_len + 1, embed_dim] (includes EOS)
            input_mask: Valid token mask [batch, seq_len + 1]
            ar_mask: Autoregressive mask pattern [seq_len + 1] (False = bidirectional)
        """
        input_mask = []
        ar_mask = []
        tokens = []

        # Embed images via SigLIP
        for name in obs.images:
            image_tokens, _ = self.siglip(obs.images[name], train=False)
            # SigLIP already projected to gemma_width via num_classes parameter
            tokens.append(image_tokens)
            input_mask.append(
                einops.repeat(
                    obs.image_masks[name],
                    "b -> b s",
                    s=image_tokens.shape[1],
                )
            )
            # Image tokens use bidirectional attention
            ar_mask += [False] * image_tokens.shape[1]

        # Embed language tokens via Gemma3 embedder
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.gemma(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # Language tokens also use bidirectional attention for value estimation
            ar_mask += [False] * tokenized_inputs.shape[1]

        # Concatenate all embeddings
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)

        # Append EOS token
        batch_size = tokens.shape[0]
        eos_token_ids = jnp.full((batch_size, 1), self.eos_token_id, dtype=jnp.int32)
        eos_embeddings = self.gemma(eos_token_ids, method="embed")  # (batch, 1, embed_dim)

        tokens = jnp.concatenate([tokens, eos_embeddings], axis=1)
        # EOS token is always valid
        eos_mask = jnp.ones((batch_size, 1), dtype=bool)
        input_mask = jnp.concatenate([input_mask, eos_mask], axis=1)
        # EOS token also uses bidirectional attention
        ar_mask += [False]

        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def discretize_returns(
        self,
        returns: at.Float[at.Array, "*b"],
    ) -> at.Int[at.Array, "*b"]:
        """Discretize continuous returns to bin indices.

        Linear mapping: bin_index = (value - value_min) / (value_max - value_min) * (num_bins - 1)

        Per the paper:
        - Bin 0 corresponds to value_min (-1.0 = failure/start)
        - Bin 200 corresponds to value_max (0.0 = success/completion)

        Args:
            returns: Continuous returns normalized to [value_min, value_max]

        Returns:
            Bin indices in range [0, value_dims - 1]
        """
        # Linear mapping from [value_min, value_max] to [0, num_bins - 1]
        normalized = (returns - self.value_min) / (self.value_max - self.value_min)
        bin_indices = normalized * (self.value_dims - 1)
        # Clamp to valid range and round to nearest integer
        bin_indices = jnp.clip(jnp.round(bin_indices), 0, self.value_dims - 1)
        return bin_indices.astype(jnp.int32)

    @at.typecheck
    def forward(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "b vd"]:
        """Forward pass to get value distribution logits.

        Args:
            rng: Random key for dropout and other stochastic operations
            observation: Input observation (images, state, language)
            train: Whether in training mode

        Returns:
            Logits over value bins, shape (batch, value_dims)
        """
        # Get observation embeddings (includes EOS token at the end)
        tokens, input_mask, ar_mask = self.embed_prefix(observation)

        # Build full attention mask (bidirectional for value estimation)
        # Shape: (batch, seq_len, seq_len)
        attn_mask = input_mask[:, None, :] & input_mask[:, :, None]

        # Compute positions (cumulative sum of valid tokens)
        positions = jnp.cumsum(input_mask.astype(jnp.int32), axis=1) - 1

        # Forward through Gemma3 transformer
        hidden_states = self.gemma(
            tokens,
            positions,
            attn_mask,
            deterministic=not train,
        )  # Shape: (batch, seq_len, embed_dim)

        # Extract hidden state at EOS token position (last position)
        # The EOS token is always at the last position since we appended it
        eos_hidden_state = hidden_states[:, -1, :]  # (batch, embed_dim)

        # Project to value distribution logits
        logits = self.value_proj(eos_hidden_state)  # Shape: (batch, value_dims)

        return logits

    @at.typecheck
    def compute_loss(
        self,
        rng: at.KeyArrayLike,
        observation: Observation,
        returns: at.Float[at.Array, "*b"],
        *,
        train: bool = False,
    ) -> at.Float[at.Array, "*b"]:
        """Compute cross-entropy loss between predicted value distribution and target.

        Implements Equation 1 from RECAP/π*0.6 paper:
        min_φ E_{τ∈D} [ Σ_{o_t∈τ} H(R^B_t(τ), p_φ(V|o_t, ℓ)) ]

        Where:
        - H(·,·) is cross-entropy
        - R^B_t is the discretized empirical return (one-hot over 201 bins)
        - p_φ(V|o_t, ℓ) is the predicted distribution (softmax over 201 logits)

        Args:
            rng: Random key for any stochastic operations
            observation: Input observation (images, state, language)
            returns: Monte Carlo returns normalized to [value_min, value_max]
            train: Whether in training mode

        Returns:
            Cross-entropy loss per sample in batch
        """
        # Preprocess observation (image augmentation, normalization, etc.)
        preprocess_rng, forward_rng = jax.random.split(rng)
        observation = preprocess_observation(preprocess_rng, observation, train=train)

        # Forward pass to get logits over value bins
        logits = self.forward(forward_rng, observation, train=train)  # Shape: (batch, value_dims)

        # Discretize target returns to bin indices
        target_bins = self.discretize_returns(returns)  # Shape: (batch,)

        # Create one-hot targets
        target_one_hot = jax.nn.one_hot(target_bins, self.value_dims)  # Shape: (batch, value_dims)

        # Compute softmax cross-entropy loss
        # H(target, pred) = -Σ target * log(softmax(logits))
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        loss = -jnp.sum(target_one_hot * log_probs, axis=-1)  # Shape: (batch,)

        return loss
