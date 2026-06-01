# Runtime image for real ALOHA collection with local RLT actor inference.
# Extends the robot ROS image with the JAX/Flax dependencies needed to load
# and run exported RLT actors inside rlt_warmup_runtime.

FROM lyl472324464/robot:aloha-ros1.0

ENV PIP_DEFAULT_TIMEOUT=1000

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install \
    "numpy==1.26.4" \
    "jax[cuda12]==0.5.3" \
    "flax==0.10.2" \
    "jaxtyping==0.2.36" \
    "beartype==0.19.0" \
    "nvidia-cublas-cu12" \
    "nvidia-cuda-cupti-cu12" \
    "nvidia-cuda-nvcc-cu12" \
    "nvidia-cuda-runtime-cu12" \
    "nvidia-cudnn-cu12" \
    "nvidia-cufft-cu12" \
    "nvidia-cusolver-cu12" \
    "nvidia-cusparse-cu12" \
    "nvidia-nccl-cu12"
