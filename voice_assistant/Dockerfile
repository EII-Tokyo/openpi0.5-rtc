# Voice Assistant Dockerfile with CUDA support for RTX 5090
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3 \
    python3-dev \
    portaudio19-dev \
    alsa-utils \
    libasound2-dev \
    build-essential \
    git \
    wget \
    curl \
    pulseaudio \
    pulseaudio-utils \
    libpulse-dev \
    mecab \
    mecab-ipadic-utf8 \
    libmecab-dev \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
COPY --from=ghcr.io/astral-sh/uv:0.5.6 /uv /uvx /bin/

ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV VIRTUAL_ENV=/.venv
ENV PATH="/.venv/bin:$PATH"

# 设置工作目录
WORKDIR /app

# 复制项目文件（先复制依赖文件以便缓存）
COPY pyproject.toml uv.lock /app/

# 安装Python依赖
RUN uv venv --python python3 $UV_PROJECT_ENVIRONMENT
RUN uv sync --python $UV_PROJECT_ENVIRONMENT/bin/python --no-install-project

# 复制项目代码
COPY . /app

# 初始化 unidic 字典（日语分词所需）
RUN uv run python3 -m unidic download || echo "unidic dictionary download failed, will download at runtime"

# 预下载Whisper模型 (CPU，运行时再切换到CUDA)
RUN uv run python3 -c "import whisper; whisper.load_model('large-v3', device='cpu')"

# 预下载TTS模型 (CPU，运行时再切换到CUDA)
ENV COQUI_TOS_AGREED=1
RUN uv run python3 -c "from TTS.api import TTS; tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False)" || echo "TTS model download failed, will download at runtime"

# 预下载Silero VAD模型
RUN uv run python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False)"

# 默认命令
CMD ["uv", "run", "python3", "/app/voice_assistant.py"]
