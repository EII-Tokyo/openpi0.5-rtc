# TOPReward Implementation Plan

## Overview

Implement TOPReward (Token Probabilities as Hidden Zero-Shot Rewards) using Qwen3-VL-8B-Instruct on a local RTX 5090. The system will:

1. Evaluate LeRobot episodes offline (progress estimation, success detection, AWR weights)
2. Serve real-time progress scores via an existing websocket server
3. Compute 3-tier advantage weights (good/neutral/bad) for π0.5 fine-tuning

---

## Project Structure

```
topreward/
├── topreward/
│   ├── __init__.py
│   ├── model.py              # Qwen3-VL-8B loading & logit extraction
│   ├── reward.py             # Core TOPReward: log-prob of "True" token
│   ├── progress.py           # Prefix sampling + min-max normalization
│   ├── success_detector.py   # Binary success/failure from final-frame probs
│   ├── advantage.py          # 3-tier advantage computation (good/neutral/bad)
│   ├── data/
│   │   ├── lerobot_loader.py # Load LeRobot episodes → video prefixes
│   │   └── camera.py         # Multi-camera handling (select or fuse views)
│   ├── serving/
│   │   ├── websocket_client.py  # Plug into existing websocket for real-time
│   │   └── real_time.py         # Streaming prefix accumulation + scoring
│   └── utils/
│       ├── metrics.py         # VOC (Spearman rank correlation)
│       └── visualization.py   # Progress curve plotting
├── scripts/
│   ├── evaluate_offline.py    # Batch evaluate LeRobot episodes
│   ├── compute_awr_weights.py # Generate advantage weights for π0.5 training
│   ├── benchmark_voc.py       # Reproduce VOC benchmarks
│   └── serve_realtime.py      # Launch real-time scoring service
├── configs/
│   └── default.yaml           # Hyperparameters (τ, δ_max, K, thresholds)
├── tests/
│   └── test_reward.py
└── requirements.txt
```

---

## Module-by-Module Specification

### 1. `model.py` — Model Loading & Logit Extraction

**Purpose:** Load Qwen3-VL-8B and extract logit/log-probability for a specific token given video + text input.

**Key details:**
- Load with `transformers` (AutoModelForCausalLM, AutoProcessor) in bfloat16
- Use `flash_attention_2` for speed on the 5090
- **CRITICAL: Do NOT use chat template** — the paper shows chat templates degrade VOC by ~47% on Qwen3-VL-8B (Table 5). Feed raw text directly.
- Return the log-probability of the "True" token (token ID for "True" in Qwen3-VL's vocabulary)

```python
class TOPRewardModel:
    def __init__(self, model_name="Qwen/Qwen3-VL-8B-Instruct", device="cuda"):
        # Load model + processor
        # Find token_id for "True" in tokenizer vocab
        # self.true_token_id = tokenizer.encode("True", add_special_tokens=False)[0]
    
    def get_log_prob(self, video_frames: list[PIL.Image], prompt: str) -> float:
        """
        Given video frames and a text prompt, return log P("True" | context).
        
        1. Build input: video tokens + prompt text (NO chat template)
        2. Forward pass, get logits at the last token position
        3. Apply log_softmax to logits
        4. Return logits[true_token_id]
        """
    
    def get_log_prob_batch(self, videos: list, prompts: list) -> list[float]:
        """Batched version for throughput on large evaluations."""
```

**Prompt template (from paper Section 3.1, no chat wrapping):**
```
<|video|> The above video shows a robot manipulation trajectory that completes the following task: {INSTRUCTION}. Decide whether the above statement is True or not. The answer is:
```

**Implementation notes:**
- The `<|video|>` token is a placeholder — check Qwen3-VL's actual video token format. Qwen3-VL uses `<|vision_start|>...<|vision_end|>` for video frames.
- For Qwen3-VL specifically, video frames are passed as a list of PIL images to the processor, which handles tokenization.
- Verify "True" is a single token in Qwen3-VL's tokenizer. If it's multi-token, sum log probs (but paper says it's single-token for their evaluated models).

### 2. `reward.py` — Core Reward Computation

**Purpose:** Implements Equation (1): `r_t = log p_θ(a | c(τ_{1:t}, u))`

```python
def compute_reward(
    model: TOPRewardModel,
    frames: list[PIL.Image],       # Full trajectory frames
    instruction: str,
    prefix_end: int                 # Use frames[0:prefix_end]
) -> float:
    """Return log-prob of 'True' for a trajectory prefix."""
    prefix_frames = frames[:prefix_end]
    prompt = build_prompt(instruction)  # No chat template!
    return model.get_log_prob(prefix_frames, prompt)
```

### 3. `progress.py` — Progress Estimation (Section 3.2)

**Purpose:** Generate the full progress curve for an episode.

```python
def estimate_progress(
    model: TOPRewardModel,
    frames: list[PIL.Image],
    instruction: str,
    K: int = 16                     # Number of prefix sample points
) -> dict:
    """
    Returns:
        raw_rewards: list[float]    # r_{t_k} for each prefix (log probs)
        normalized:  list[float]    # s_{t_k} after min-max norm (Eq. 2)
        timestamps:  list[int]      # Frame indices used
    """
    # 1. Uniformly sample K prefix lengths: t_1=1, ..., t_K=T
    T = len(frames)
    prefix_lengths = np.linspace(1, T, K, dtype=int)
    
    # 2. For each prefix, compute r_t = log P("True" | prefix, instruction)
    rewards = []
    for t in prefix_lengths:
        r_t = compute_reward(model, frames, instruction, prefix_end=t)
        rewards.append(r_t)
    
    # 3. Min-max normalize (Eq. 2)
    eps = 1e-8
    r_min, r_max = min(rewards), max(rewards)
    normalized = [(r - r_min) / (r_max - r_min + eps) for r in rewards]
    
    return {
        "raw_rewards": rewards,
        "normalized": normalized,
        "timestamps": prefix_lengths.tolist()
    }
```

**Performance considerations for K=16 on 5090:**
- Each prefix requires one forward pass of Qwen3-VL-8B
- With flash attention + bf16, expect ~0.5-1.5s per forward pass depending on frame count
- 16 prefixes per episode → ~8-24s per episode
- For 1000s of episodes, plan for batching and/or parallelism
- Consider KV-cache reuse: prefixes share earlier frames, but Qwen3-VL processes video frames as a set, so caching benefit depends on implementation

### 4. `success_detector.py` — Success Detection (Section 5.2)

**Purpose:** Binary success/failure classification using average log-prob of final frames.

```python
def detect_success(
    model: TOPRewardModel,
    frames: list[PIL.Image],
    instruction: str,
    n_final_frames: int = 3,
    threshold: float = None       # Calibrate from labeled data
) -> dict:
    """
    Use average log-prob over last n frames as success score.
    
    Returns:
        score: float              # Average log P("True") for last n frames
        is_success: bool          # score > threshold (if threshold provided)
    """
    T = len(frames)
    # Sample last 3 prefix points (e.g., T-2, T-1, T)
    final_prefixes = list(range(max(1, T - n_final_frames + 1), T + 1))
    
    scores = []
    for t in final_prefixes:
        r = compute_reward(model, frames, instruction, prefix_end=t)
        scores.append(r)
    
    avg_score = np.mean(scores)
    
    return {
        "score": avg_score,
        "is_success": avg_score > threshold if threshold is not None else None,
        "per_frame_scores": scores
    }
```

**Calibrating the threshold:**
- Use your labeled success/failure episodes
- Compute scores for all episodes
- Find threshold that maximizes ROC-AUC (or F1, depending on your use case)
- Save threshold to config

### 5. `advantage.py` — 3-Tier Advantage Weights for π0.5

**Purpose:** Compute per-step advantage weights with a 3-tier system: good, neutral, bad.

The paper uses Equation (3):
```
Δ_t = clip(τ · exp(s_t - s_{t-1}), min=0, max=δ_max)
```

We extend this to a 3-tier system:

```python
@dataclass
class AdvantageConfig:
    tau: float = 2.0              # Scaling factor (from paper)
    delta_max: float = 2.0        # Max reward cap (from paper)
    good_threshold: float = 0.6   # Above this → good action (upweight)
    bad_threshold: float = -0.2   # Below this → bad action (downweight)
    good_weight: float = 2.0      # Weight multiplier for good actions
    neutral_weight: float = 1.0   # Weight for neutral actions
    bad_weight: float = 0.1       # Weight for bad actions (near-zero, not zero)

def compute_advantages(
    normalized_progress: list[float],
    config: AdvantageConfig
) -> dict:
    """
    Compute per-step advantages from normalized progress curve.
    
    Returns:
        delta_t: list[float]       # Raw progress increments (Eq. 3)
        tiers: list[str]           # "good" / "neutral" / "bad" per step
        weights: list[float]       # Final per-step weights for AWR
    """
    deltas = []
    tiers = []
    weights = []
    
    for k in range(1, len(normalized_progress)):
        # Eq. 3: progress increment
        increment = normalized_progress[k] - normalized_progress[k-1]
        delta = np.clip(
            config.tau * np.exp(increment),
            a_min=0, a_max=config.delta_max
        )
        deltas.append(delta)
        
        # 3-tier classification
        if increment > config.good_threshold:
            tier = "good"
            w = config.good_weight
        elif increment < config.bad_threshold:
            tier = "bad"
            w = config.bad_weight
        else:
            tier = "neutral"
            w = config.neutral_weight
        
        tiers.append(tier)
        weights.append(w)
    
    return {"deltas": deltas, "tiers": tiers, "weights": weights}


def compute_episode_advantages(
    model: TOPRewardModel,
    frames: list[PIL.Image],
    instruction: str,
    action_timestamps: list[int],  # Frame index for each action
    config: AdvantageConfig
) -> list[float]:
    """
    Full pipeline: frames → progress → advantages aligned to action steps.
    
    For π0.5 training:
    - Compute progress at each action timestep
    - Derive increments and classify into tiers
    - Return weights to use in the AWR loss
    """
    # Get progress at each action timestamp
    rewards = []
    for t in action_timestamps:
        r = compute_reward(model, frames, instruction, prefix_end=t)
        rewards.append(r)
    
    # Normalize
    eps = 1e-8
    r_min, r_max = min(rewards), max(rewards)
    normalized = [(r - r_min) / (r_max - r_min + eps) for r in rewards]
    
    # Compute advantages
    return compute_advantages(normalized, config)
```

**Integration with π0.5 AWR training (Equation 5 from paper):**
```python
# In your π0.5 training loop:
# L_AWR = E[Δ_t · ||v_θ(a_t, t | o) - (a - ε)||²]
# 
# The `weights` from compute_advantages() are your Δ_t values.
# Multiply them into your flow-matching loss.
```

**Tuning the 3-tier thresholds:**
- Start with paper defaults (τ=2.0, δ_max=2.0)
- For the good/neutral/bad thresholds, visualize the distribution of progress increments across your labeled episodes
- Good actions: moments where the robot makes clear task progress
- Bad actions: moments where progress decreases (regression/mistakes)
- Neutral: small changes, approaching, repositioning

### 6. `data/lerobot_loader.py` — LeRobot Data Loading

```python
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

def load_episode_frames(
    dataset: LeRobotDataset,
    episode_idx: int,
    camera_key: str = "observation.images.top",  # or pick best view
    max_frames: int = None,
    subsample_fps: int = None
) -> tuple[list[PIL.Image], str, dict]:
    """
    Load frames + instruction from a LeRobot episode.
    
    Returns:
        frames: list of PIL Images
        instruction: task instruction string
        metadata: dict with episode info
    """

def select_camera(
    dataset: LeRobotDataset,
    episode_idx: int,
    strategy: str = "first"  # "first", "overhead", "wrist", "all"
) -> str:
    """
    Choose which camera view to use.
    
    For Franka: typically overhead or wrist cam
    For ALOHA: typically top camera
    
    Strategy:
    - "first": use first available camera
    - "overhead"/"wrist": match by key name
    - "all": concatenate views (experimental)
    """
```

**Multi-camera strategy:**
- The paper uses single camera views per evaluation
- Recommend: pick the view with the best overview of the workspace (usually overhead/top)
- For ALOHA: the `cam_high` or top camera typically works best
- Allow configuring per-robot-platform which camera key to use

### 7. `serving/real_time.py` — Real-Time WebSocket Integration

**Purpose:** Plug into existing websocket server to provide live progress scores.

```python
class RealTimeScorer:
    def __init__(self, model: TOPRewardModel, instruction: str):
        self.model = model
        self.instruction = instruction
        self.frame_buffer = []        # Accumulates frames
        self.score_interval = 5       # Score every N frames
        self.last_score = 0.0
        self.scores_history = []
    
    def on_frame(self, frame: PIL.Image) -> dict | None:
        """
        Called for each incoming frame from websocket.
        Returns progress score if interval reached, else None.
        """
        self.frame_buffer.append(frame)
        
        if len(self.frame_buffer) % self.score_interval == 0:
            score = self._compute_current_score()
            return {"progress": score, "frame_idx": len(self.frame_buffer)}
        return None
    
    def _compute_current_score(self) -> float:
        """Compute raw log-prob for current accumulated trajectory."""
        r = compute_reward(
            self.model, self.frame_buffer,
            self.instruction, prefix_end=len(self.frame_buffer)
        )
        self.scores_history.append(r)
        
        # Normalize against history so far
        if len(self.scores_history) > 1:
            r_min = min(self.scores_history)
            r_max = max(self.scores_history)
            normalized = (r - r_min) / (r_max - r_min + 1e-8)
        else:
            normalized = 0.0
        
        return normalized
    
    def reset(self, new_instruction: str = None):
        """Reset for new episode."""
        self.frame_buffer = []
        self.scores_history = []
        if new_instruction:
            self.instruction = new_instruction
```

**WebSocket integration sketch (adapts to your existing setup):**
```python
# In your existing websocket handler:
async def handle_message(ws, message):
    frame = decode_frame(message)  # Your existing frame decoding
    
    result = scorer.on_frame(frame)
    if result:
        await ws.send(json.dumps({
            "type": "progress_score",
            "progress": result["progress"],
            "frame_idx": result["frame_idx"]
        }))
```

**Real-time performance budget:**
- Target: score every 5 frames at ~10 FPS control → one inference every 0.5s
- Qwen3-VL-8B with ~50 accumulated frames + bf16 + flash attn → ~0.5-1s per forward
- This is tight. Optimizations:
  - Subsample frame buffer (e.g., keep every 3rd frame)
  - Cap max frames sent to model (e.g., last 64 frames)
  - Run inference in a separate thread/process to avoid blocking

### 8. `utils/metrics.py` — VOC Metric

```python
from scipy.stats import spearmanr

def compute_voc(predicted_values: list[float], timestamps: list[int]) -> float:
    """
    Value-Order Correlation (Eq. 4).
    Spearman rank correlation between chronological order and predicted values.
    """
    correlation, _ = spearmanr(timestamps, predicted_values)
    return correlation
```

---

## Scripts

### `scripts/evaluate_offline.py`
```
Usage: python scripts/evaluate_offline.py \
    --dataset_repo "your/lerobot-dataset" \
    --camera_key "observation.images.top" \
    --K 16 \
    --output results.json \
    --num_episodes 100 \
    --num_workers 1  # Model is on single GPU, parallelism via prefetching
```

Outputs per episode:
- Progress curve (normalized)
- VOC score
- Success detection score
- Raw log-probs

### `scripts/compute_awr_weights.py`
```
Usage: python scripts/compute_awr_weights.py \
    --dataset_repo "your/lerobot-dataset" \
    --output awr_weights.safetensors \
    --tau 2.0 --delta_max 2.0 \
    --good_threshold 0.6 --bad_threshold -0.2
```

Outputs a file mapping (episode_idx, step_idx) → advantage weight + tier label.
This can be loaded during π0.5 AWR fine-tuning.

---

## Config (`configs/default.yaml`)

```yaml
model:
  name: "Qwen/Qwen3-VL-8B-Instruct"
  dtype: "bfloat16"
  attn_implementation: "flash_attention_2"
  use_chat_template: false  # CRITICAL: must be false

progress:
  K: 16                      # Number of prefix sample points
  normalize: true            # Min-max normalization
  eps: 1e-8

success_detection:
  n_final_frames: 3
  threshold: null             # Calibrate from labeled data

advantage:
  tau: 2.0
  delta_max: 2.0
  # 3-tier thresholds (tune on your data)
  good_threshold: 0.6
  bad_threshold: -0.2
  good_weight: 2.0
  neutral_weight: 1.0
  bad_weight: 0.1

realtime:
  score_interval: 5           # Score every N frames
  max_buffer_frames: 64       # Cap frame buffer for latency
  subsample_factor: 3         # Keep every Nth frame in buffer

cameras:
  franka: "observation.images.top"
  aloha: "observation.images.cam_high"
  default: "observation.images.top"
```

---

## Implementation Order

1. **`model.py`** — Get Qwen3-VL-8B loaded, verify "True" token ID, confirm logit extraction works on a single image
2. **`reward.py`** — Verify log-prob computation on a toy example (one frame + instruction)
3. **`data/lerobot_loader.py`** — Load a few episodes, extract frames
4. **`progress.py`** — Run progress estimation on 5 episodes, plot curves, sanity check
5. **`utils/metrics.py`** — Compute VOC on those episodes, compare against expected ~0.94
6. **`success_detector.py`** — Calibrate threshold using labeled data
7. **`advantage.py`** — Compute weights, visualize distribution of good/neutral/bad
8. **`scripts/evaluate_offline.py`** — Scale to full dataset evaluation
9. **`scripts/compute_awr_weights.py`** — Generate weights for π0.5 training
10. **`serving/real_time.py`** — Integrate with existing websocket

---

## Critical Implementation Notes

### DO NOT use chat templates
The paper's ablation (Table 5) shows Qwen3-VL-8B VOC drops from **0.945 → 0.500** with chat templates. This is the single biggest implementation detail. Feed raw text without any system/user/assistant role wrapping.

### Token ID verification
Before anything else, verify:
```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
token_ids = tokenizer.encode("True", add_special_tokens=False)
print(token_ids)  # Should be a single token
# Also check: tokenizer.decode([token_ids[0]]) == "True"
```

### Video frame handling for Qwen3-VL
Qwen3-VL processes video as a sequence of image frames. The processor handles converting PIL images into the right format. Key: pass frames as a list, not as a video file. Check the exact input format Qwen3-VL expects (likely a list of PIL Images or a video tensor).

### Memory management for large evaluations
- With K=16 prefixes per episode and 1000+ episodes, that's 16,000+ forward passes
- Use `torch.no_grad()` and `torch.cuda.empty_cache()` between episodes
- Consider `torch.compile` for faster inference
- Monitor VRAM — the 5090's 32GB is comfortable for the 8B model (~16GB in bf16) but long video sequences add KV cache pressure

### Multi-camera episodes
- Evaluate on single best camera view (overhead recommended)
- Optionally: run on each camera separately and average/max scores
- Don't concatenate camera views into a single video — the paper doesn't do this

---

## Dependencies

```
torch>=2.4
transformers>=4.45
accelerate
flash-attn
qwen-vl-utils            # Qwen3-VL helper utilities
lerobot
scipy
numpy
pillow
pyyaml
tqdm
matplotlib                # For visualization
safetensors               # For saving AWR weights
```

---

## Testing Checklist

- [ ] "True" is single token in Qwen3-VL tokenizer
- [ ] Log-prob extraction gives negative values (log space) that increase toward 0 as task completes
- [ ] No chat template in prompt
- [ ] Progress curves are monotonically increasing on successful episodes
- [ ] VOC > 0.9 on a few test episodes (sanity check against paper's 0.947)
- [ ] Success detection ROC-AUC > 0.6 on labeled data
- [ ] 3-tier advantage weights: good steps cluster around task-progress moments
- [ ] Real-time scoring latency < 1s per score on 5090
