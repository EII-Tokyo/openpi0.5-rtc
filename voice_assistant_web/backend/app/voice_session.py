from __future__ import annotations

import base64
import json
import logging
import subprocess
import uuid
from pathlib import Path

from fastapi import HTTPException, UploadFile
from openai import BadRequestError, OpenAI

from .config import settings
from .redis_commands import TASK_MAPPING, publish_task
from .schemas import VoiceResponse

_SUPPORTED_AUDIO_SUFFIXES = {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}
_AUDIO_SUFFIX_BY_CONTENT_TYPE = {
    "audio/flac": ".flac",
    "audio/mp4": ".m4a",
    "audio/mpeg": ".mp3",
    "audio/mpga": ".mpga",
    "audio/ogg": ".ogg",
    "audio/wav": ".wav",
    "audio/webm": ".webm",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


class VoiceAssistantEngine:
    def __init__(self, redis_client) -> None:
        self._redis = redis_client
        self._openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self._base_prompt = (
            "The user is operating an aloha robot. Supported tasks are:\n"
            + "\n".join(f"{key}: {value}" for key, value in TASK_MAPPING.items())
            + "\nMap leader-follower demo or teleoperation experience requests to task 6."
            + "\nReturn JSON only with keys task_number and response_statement."
        )

    def _normalize_language(self, language: str | None, *, transcript: str = "") -> str:
        normalized = (language or "").strip().lower()
        if normalized in {"zh", "zh-cn", "zh-hans", "chinese", "cn"}:
            return "zh"
        if normalized in {"ja", "japanese", "jp"}:
            return "ja"
        if normalized in {"en", "english"}:
            return "en"
        if any("\u3040" <= ch <= "\u30ff" for ch in transcript):
            return "ja"
        if any("\u4e00" <= ch <= "\u9fff" for ch in transcript):
            return "zh"
        return "en"

    def _localized_text(self, language: str, *, en: str, ja: str, zh: str) -> str:
        if language == "ja":
            return ja
        if language == "zh":
            return zh
        return en

    def _rule_based_task(self, transcript: str) -> str | None:
        normalized = transcript.lower().strip()
        compact = normalized.replace(" ", "")

        task_1_keywords = [
            "twist off the bottle cap",
            "unscrew the cap",
            "open the bottle cap",
            "start twisting the bottle cap",
            "开始拧瓶盖",
            "拧瓶盖",
            "拧开瓶盖",
            "开瓶盖",
            "ボトルキャップを開け",
            "キャップを開け",
            "ねじって開け",
        ]
        if any(keyword in normalized or keyword in compact for keyword in task_1_keywords):
            return "1"

        task_2_keywords = [
            "process bottles",
            "rinse the bottle",
            "rinse bottle",
            "rinsing bottles",
            "tear off labels",
            "tear off the label",
            "remove labels",
            "处理瓶子",
            "清洗瓶子",
            "冲洗瓶子",
            "撕掉标签",
            "撕标签",
            "ラベルを剥がす",
            "ボトルをすすぐ",
            "ボトル処理",
        ]
        if any(keyword in normalized or keyword in compact for keyword in task_2_keywords):
            return "2"

        task_6_keywords = [
            "leader follower",
            "leader-follower",
            "leader demo",
            "follower demo",
            "teleoperation experience",
            "teleop experience",
            "teleoperation demo",
            "teleop demo",
            "customer demo",
            "遥操作体验",
            "遥操作演示",
            "客户演示",
            "主从演示",
            "leader跟随",
            "follower跟随",
            "リーダーフォロワー",
            "遠隔操作体験",
            "遠隔操作デモ",
        ]
        if any(keyword in normalized or keyword in compact for keyword in task_6_keywords):
            return "6"

        task_3_keywords = [
            "human control",
            "manual control",
            "teleop",
            "stop and human",
            "人工操作",
            "人工接管",
            "手动控制",
            "遥操作",
            "手動操作",
            "人が操作",
            "テレオペ",
        ]
        if any(keyword in normalized or keyword in compact for keyword in task_3_keywords):
            return "3"

        return None

    async def process_text(self, text: str, *, language: str = "en", debug: dict | None = None) -> VoiceResponse:
        transcript = text.strip()
        language = self._normalize_language(language, transcript=transcript)
        reply_language = {
            "ja": "Japanese",
            "zh": "Simplified Chinese",
        }.get(language, "English")
        if not transcript:
            return VoiceResponse(
                transcript="",
                reply_text=self._localized_text(
                    language,
                    en="No speech detected.",
                    ja="音声が検出されませんでした。",
                    zh="没有检测到语音。",
                ),
                task_number=None,
                task_name=None,
            )

        if self._openai is None:
            return VoiceResponse(
                transcript=transcript,
                reply_text=self._localized_text(
                    language,
                    en="OPENAI_API_KEY is not configured.",
                    ja="OPENAI_API_KEY が設定されていません。",
                    zh="OPENAI_API_KEY 尚未配置。",
                ),
                task_number=None,
                task_name=None,
            )

        forced_task = self._rule_based_task(transcript)
        logging.info(
            "voice_text start language=%s forced_task=%s transcript=%r",
            language,
            forced_task,
            transcript[:120],
        )

        completion = self._openai.chat.completions.create(
            model=settings.openai_chat_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"{self._base_prompt}\n"
                        f"The response_statement must be written in {reply_language}.\n"
                        "Map bottle cap opening / twisting requests to task 1.\n"
                        "Map bottle rinsing / washing requests to task 2.\n"
                        "Map manual takeover / teleoperation requests to task 3.\n"
                        "Do not choose task 3 unless the user explicitly asks for manual or human control."
                    ),
                },
                {"role": "user", "content": transcript},
            ],
        )
        content = completion.choices[0].message.content or "{}"
        parsed = json.loads(content)
        raw_task_number = parsed.get("task_number")
        task_number = forced_task or (str(raw_task_number) if raw_task_number is not None else None)
        reply_text = parsed.get("response_statement", "")
        task_name = TASK_MAPPING.get(task_number)
        if task_number in TASK_MAPPING:
            publish_task(self._redis, task_number)
        logging.info(
            "voice_text classified language=%s task=%s reply_len=%d",
            language,
            task_number,
            len(reply_text),
        )

        audio_base64 = None
        audio_mime_type = None
        tts_ok = False
        if reply_text:
            try:
                speech = self._openai.audio.speech.create(
                    model=settings.openai_tts_model,
                    voice=settings.openai_tts_voice,
                    input=reply_text,
                )
                audio_bytes = speech.read()
                audio_base64 = base64.b64encode(audio_bytes).decode("ascii")
                audio_mime_type = "audio/mpeg"
                tts_ok = True
                logging.info(
                    "voice_tts success language=%s bytes=%d",
                    language,
                    len(audio_bytes),
                )
            except Exception:
                logging.exception("voice_tts failed language=%s", language)

        return VoiceResponse(
            transcript=transcript,
            reply_text=reply_text,
            task_number=task_number,
            task_name=task_name,
            audio_base64=audio_base64,
            audio_mime_type=audio_mime_type,
            debug={
                **(debug or {}),
                "raw_response": parsed,
                "normalized_language": language,
                "tts_ok": tts_ok,
            },
        )

    async def process_audio(self, audio_file: UploadFile, *, language: str = "en") -> VoiceResponse:
        fallback_language = self._normalize_language(language)
        if self._openai is None:
            return VoiceResponse(
                transcript="",
                reply_text="OPENAI_API_KEY が設定されていません。" if fallback_language == "ja" else "OPENAI_API_KEY is not configured.",
                task_number=None,
                task_name=None,
            )

        content_type = (audio_file.content_type or "").split(";", 1)[0].strip().lower()
        suffix = Path(audio_file.filename or "").suffix.lower()
        if suffix not in _SUPPORTED_AUDIO_SUFFIXES:
            suffix = _AUDIO_SUFFIX_BY_CONTENT_TYPE.get(content_type, ".webm")

        audio_bytes = await audio_file.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio upload")

        temp_path = Path("/tmp") / f"voice_assistant_web_upload_{uuid.uuid4().hex}{suffix}"
        temp_path.write_bytes(audio_bytes)
        transcription_path = temp_path
        converted_path: Path | None = None
        try:
            logging.info(
                "voice_audio start filename=%s content_type=%s suffix=%s fallback_language=%s size=%d",
                audio_file.filename,
                audio_file.content_type,
                suffix,
                fallback_language,
                temp_path.stat().st_size,
            )
            if suffix != ".wav":
                converted_path = temp_path.with_suffix(".wav")
                try:
                    conversion = subprocess.run(
                        [
                            "ffmpeg",
                            "-hide_banner",
                            "-loglevel",
                            "error",
                            "-y",
                            "-i",
                            str(temp_path),
                            "-ac",
                            "1",
                            "-ar",
                            "16000",
                            "-f",
                            "wav",
                            str(converted_path),
                        ],
                        check=False,
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                except FileNotFoundError as exc:
                    logging.error("voice_audio ffmpeg is not installed")
                    raise HTTPException(status_code=500, detail="Audio conversion tool is not installed") from exc
                if conversion.returncode != 0 or not converted_path.exists() or converted_path.stat().st_size == 0:
                    logging.warning(
                        "voice_audio ffmpeg conversion failed filename=%s content_type=%s suffix=%s size=%d stderr=%s",
                        audio_file.filename,
                        audio_file.content_type,
                        suffix,
                        temp_path.stat().st_size,
                        conversion.stderr[-1000:],
                    )
                    raise HTTPException(status_code=400, detail="Invalid audio recording")
                transcription_path = converted_path
                logging.info(
                    "voice_audio converted to wav original_size=%d wav_size=%d",
                    temp_path.stat().st_size,
                    transcription_path.stat().st_size,
                )

            with transcription_path.open("rb") as handle:
                try:
                    transcription = self._openai.audio.transcriptions.create(
                        model=settings.openai_transcription_model,
                        file=handle,
                        response_format="verbose_json",
                    )
                except BadRequestError as exc:
                    logging.warning(
                        "voice_audio transcription rejected filename=%s content_type=%s suffix=%s size=%d error=%s",
                        audio_file.filename,
                        audio_file.content_type,
                        suffix,
                        transcription_path.stat().st_size,
                        exc,
                    )
                    raise HTTPException(status_code=400, detail="Unsupported or invalid audio format") from exc
            transcript = getattr(transcription, "text", "") or ""
            detected_language = self._normalize_language(getattr(transcription, "language", None), transcript=transcript)
            logging.info(
                "voice_audio transcribed detected_language=%s transcript=%r",
                detected_language,
                transcript[:120],
            )
            return await self.process_text(
                transcript,
                language=detected_language,
                debug={
                    "transcription_language": getattr(transcription, "language", None),
                    "detected_language": detected_language,
                    "transcript_preview": transcript[:200],
                },
            )
        finally:
            if converted_path is not None:
                converted_path.unlink(missing_ok=True)
            temp_path.unlink(missing_ok=True)
