FROM lyl472324464/robot:aloha-ros1.0

WORKDIR /app

COPY voice_assistant_web/backend/requirements.txt /tmp/voice-assistant-web-requirements.txt
RUN uv pip install --python 3.10 --system -r /tmp/voice-assistant-web-requirements.txt

COPY voice_assistant_web /app/voice_assistant_web

CMD ["python3", "-m", "uvicorn", "voice_assistant_web.backend.app.main:app", "--host", "0.0.0.0", "--port", "8011"]
