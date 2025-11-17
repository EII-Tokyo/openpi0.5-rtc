# 语音助手

Please refer to the ALOHA voice assistant [README.md](examples/aloha_real/voice_assistant/README.md)

Additional changes for DROID:
- Running the whisper client + pi leads to potential GPU memory issues. When running serve_policy for openpi, use XLA_PYTHON_CLIENT_PREALLOCATE=false, or allocated amount to ~10GB when running inference if needed.