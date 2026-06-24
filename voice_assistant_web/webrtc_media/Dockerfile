FROM lyl472324464/robot:aloha-ros1.0

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gstreamer1.0-tools \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      gstreamer1.0-libav && \
    rm -rf /var/lib/apt/lists/*

COPY voice_assistant_web/backend/requirements.txt /tmp/eii-pilot-backend-requirements.txt
RUN uv pip install --python 3.10 --system -r /tmp/eii-pilot-backend-requirements.txt

COPY voice_assistant_web /app/voice_assistant_web

CMD ["python3", "-m", "uvicorn", "voice_assistant_web.webrtc_media.media_service:app", "--host", "0.0.0.0", "--port", "8013"]
