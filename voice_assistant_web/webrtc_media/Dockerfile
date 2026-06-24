# syntax=docker/dockerfile:1.7

FROM lyl472324464/robot:aloha-ros1.0

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
      gstreamer1.0-tools \
      gstreamer1.0-plugins-base \
      gstreamer1.0-plugins-good \
      gstreamer1.0-plugins-bad \
      gstreamer1.0-plugins-ugly \
      gstreamer1.0-libav \
      python3-gi \
      gir1.2-gstreamer-1.0 \
      gir1.2-gst-plugins-base-1.0 \
      gir1.2-gst-plugins-bad-1.0 && \
    rm -rf /var/lib/apt/lists/*

COPY voice_assistant_web/webrtc_media/requirements.txt /tmp/eii-pilot-webrtc-media-requirements.txt
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /usr/bin/python3 --system -r /tmp/eii-pilot-webrtc-media-requirements.txt

ENV PYTHONPATH=/app

COPY voice_assistant_web /app/voice_assistant_web

CMD ["/usr/bin/python3", "-m", "uvicorn", "voice_assistant_web.webrtc_media.media_service:app", "--host", "0.0.0.0", "--port", "8013"]
