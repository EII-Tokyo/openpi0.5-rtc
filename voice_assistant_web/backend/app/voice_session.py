from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

from fastapi import UploadFile
from openai import OpenAI

from .config import settings
from .redis_commands import TASK_MAPPING, publish_task
from .schemas import VoiceResponse


class VoiceAssistantEngine:
    def __init__(self, redis_client) -> None:
        self._redis = redis_client
        self._openai = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        self._base_prompt = (
            "The user is operating an aloha robot. Supported tasks are:\n"
            + "\n".join(f"{key}: {value}" for key, value in TASK_MAPPING.items())
            + "\nReturn JSON only with keys task_number and response_statement."
        )

    def translate_text(self, text: str, *, target_language: str) -> str:
        source = text.strip()
        if not source:
            return ""
        normalized_target = (target_language or "").strip().lower()
        if normalized_target not in {"zh", "ja"}:
            return source
        if self._openai is None:
            return source
        target_name = "Simplified Chinese" if normalized_target == "zh" else "Japanese"
        try:
            completion = self._openai.chat.completions.create(
                model=settings.openai_chat_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"Translate the given bottle description into {target_name} for robot speech output. "
                            "Keep product and brand names natural. Output only the translation."
                        ),
                    },
                    {"role": "user", "content": source},
                ],
            )
            translated = (completion.choices[0].message.content or "").strip()
            return translated or source
        except Exception:
            logging.exception("translate_text failed target_language=%s", normalized_target)
            return source

    def synthesize_announcement(self, text: str, *, target_language: str) -> tuple[str, str | None, str | None]:
        translated = self.translate_text(text, target_language=target_language)
        if not translated or self._openai is None:
            return self._localized_text(
                target_language,
                en=f"Now handling {translated}",
                ja=f"現在処理中: {translated}",
                zh=f"我要处理{translated}",
            ), None, None
        announcement_text = self._localized_text(
            target_language,
            en=f"Now handling {translated}",
            ja=f"現在処理中: {translated}",
            zh=f"我要处理{translated}",
        )
        try:
            speech = self._openai.audio.speech.create(
                model=settings.openai_tts_model,
                voice=settings.openai_tts_voice,
                input=announcement_text,
            )
            audio_bytes = speech.read()
            return announcement_text, base64.b64encode(audio_bytes).decode("ascii"), "audio/mpeg"
        except Exception:
            logging.exception("announcement_tts failed target_language=%s", target_language)
            return announcement_text, None, None

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

        task_2_keywords = [
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
        if any(keyword in normalized or keyword in compact for keyword in task_2_keywords):
            return "1"

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
            return "2"

        return None

    async def process_text(
        self,
        text: str,
        *,
        language: str = "en",
        dataset_dir: str | None = None,
        manual_dataset_dir: str | None = None,
        include_bottle_position: bool = False,
        forced_low_level_subtask: str | None = None,
        debug: dict | None = None,
    ) -> VoiceResponse:
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
                        "Map manual takeover / teleoperation requests to task 2.\n"
                        "Do not choose task 2 unless the user explicitly asks for manual or human control."
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
            publish_task(
                self._redis,
                task_number,
                dataset_dir=dataset_dir,
                manual_dataset_dir=manual_dataset_dir,
                include_bottle_position=include_bottle_position,
                forced_low_level_subtask=forced_low_level_subtask,
            )
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

    async def process_audio(
        self,
        audio_file: UploadFile,
        *,
        language: str = "en",
        dataset_dir: str | None = None,
        manual_dataset_dir: str | None = None,
        include_bottle_position: bool = False,
        forced_low_level_subtask: str | None = None,
    ) -> VoiceResponse:
        fallback_language = self._normalize_language(language)
        if self._openai is None:
            return VoiceResponse(
                transcript="",
                reply_text="OPENAI_API_KEY が設定されていません。" if fallback_language == "ja" else "OPENAI_API_KEY is not configured.",
                task_number=None,
                task_name=None,
            )

        suffix = Path(audio_file.filename or "recording.webm").suffix or ".webm"
        temp_path = Path("/tmp") / f"voice_assistant_web_upload{suffix}"
        temp_path.write_bytes(await audio_file.read())
        try:
            logging.info(
                "voice_audio start filename=%s suffix=%s fallback_language=%s size=%d",
                audio_file.filename,
                suffix,
                fallback_language,
                temp_path.stat().st_size,
            )
            with temp_path.open("rb") as handle:
                transcription = self._openai.audio.transcriptions.create(
                    model=settings.openai_transcription_model,
                    file=handle,
                    response_format="verbose_json",
                )
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
                dataset_dir=dataset_dir,
                manual_dataset_dir=manual_dataset_dir,
                include_bottle_position=include_bottle_position,
                forced_low_level_subtask=forced_low_level_subtask,
                debug={
                    "transcription_language": getattr(transcription, "language", None),
                    "detected_language": detected_language,
                    "transcript_preview": transcript[:200],
                },
            )
        finally:
            temp_path.unlink(missing_ok=True)
