FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

RUN pip install \
    "accelerate>=1.11.0" \
    "huggingface_hub>=0.36.0" \
    "msgpack>=1.0.5" \
    "numpy>=2.0.0" \
    "peft>=0.18.0" \
    "pillow>=11.0.0" \
    "qwen-vl-utils>=0.0.14" \
    "safetensors>=0.6.0" \
    "transformers>=5.5.4" \
    "websockets>=15.0.0"

CMD ["python", "scripts/serve_qwen_high_level.py"]
