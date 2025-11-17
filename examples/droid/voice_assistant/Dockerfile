# Voice Assistant Dockerfile with CUDA support for RTX 5090
FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
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
    libmecab-dev \
    mecab-ipadic-utf8 \
    swig \
    && rm -rf /var/lib/apt/lists/*

# 安装 uv
RUN pip3 install uv

# 先安装PyTorch 2.9 (在复制文件前，以便缓存)
RUN pip3 install torch torchaudio

# 设置工作目录
WORKDIR /app

# 复制依赖文件
COPY pyproject.toml ./

# 安装其他Python依赖 (排除torch，因为已经安装) - 合并为一个RUN减少层数
RUN uv pip install --system --no-deps -e . && \
    uv pip install --system pyaudio numpy openai-whisper TTS==0.22.0 cutlet fugashi unidic transformers==4.34.1 pygame openai redis requests && \
    python3 -m unidic download

# 预下载所有模型 (在复制代码文件之前，这样代码改动不会触发重新下载)
ENV COQUI_TOS_AGREED=1
RUN python3 -c "import whisper; whisper.load_model('large-v3', device='cpu')" && \
    echo "Whisper model downloaded"

RUN python3 -c "from TTS.api import TTS; tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', progress_bar=False)" || \
    echo "TTS model download failed, will download at runtime"

RUN python3 -c "import torch; torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False, onnx=False)" && \
    echo "Silero VAD model downloaded"

# 最后才复制 Python 源文件 (这样改代码不会让前面的层失效)
COPY voice_assistant.py config.py system_prompt_template.txt ./

# 默认命令
CMD ["python3", "voice_assistant.py"]