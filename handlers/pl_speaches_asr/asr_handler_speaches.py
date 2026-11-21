"""ASR handler for OpenAvatarChat: transcription via a Speaches server
(OpenAI-compatible /v1/audio/transcriptions). Any faster-whisper model works;
defaults target English (set language:"" for auto-detect).
Loaded through OAC's handler_search_path -> custom_handlers.
"""

import io
import wave
from typing import Dict, Optional, cast

import httpx
import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from chat_engine.contexts.handler_context import HandlerContext
from chat_engine.contexts.session_context import SessionContext
from chat_engine.data_models.chat_data.chat_data_model import ChatData
from chat_engine.data_models.chat_data_type import ChatDataType
from chat_engine.data_models.chat_engine_config_data import ChatEngineConfigModel, HandlerBaseConfigModel
from chat_engine.common.handler_base import HandlerBase, HandlerBaseInfo, HandlerDataInfo, HandlerDetail
from chat_engine.data_models.runtime_data.data_bundle import DataBundle, DataBundleDefinition, DataBundleEntry


class ASRConfig(HandlerBaseConfigModel, BaseModel):
    base_url: str = Field(default="http://127.0.0.1:8000/v1")
    model_name: str = Field(default="deepdml/faster-whisper-large-v3-turbo-ct2")
    # Language code for the transcription request, e.g. "pl", "en", "de".
    # Set to "" to let faster-whisper auto-detect the language.
    language: str = Field(default="en")
    sample_rate: int = Field(default=16000)
    timeout_seconds: float = Field(default=15.0)


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.audio_chunks = []


def _float32_to_wav_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    """Mono float32 [-1, 1] -> 16-bit PCM WAV in memory."""
    audio = np.clip(audio.squeeze(), -1.0, 1.0)
    pcm = (audio * 32767.0).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, mode="wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())
    return buffer.getvalue()


class HandlerASR(HandlerBase):
    def __init__(self):
        super().__init__()
        self.config: Optional[ASRConfig] = None
        self.client: Optional[httpx.Client] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="SpeachesASR",
            config_model=ASRConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, 24000))
        inputs = {
            ChatDataType.HUMAN_AUDIO: HandlerDataInfo(
                type=ChatDataType.HUMAN_AUDIO,
            )
        }
        outputs = {
            ChatDataType.HUMAN_TEXT: HandlerDataInfo(
                type=ChatDataType.HUMAN_TEXT,
                definition=definition,
            )
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        self.config = cast(ASRConfig, handler_config)
        self.client = httpx.Client(timeout=self.config.timeout_seconds)
        logger.info(f"SpeachesASR: base_url={self.config.base_url} model={self.config.model_name}")

    def create_context(self, session_context, handler_config=None):
        context = ASRContext(session_context.session_info.session_id)
        return context

    def start_context(self, session_context, handler_context):
        pass

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.HUMAN_TEXT).definition
        context = cast(ASRContext, context)
        if inputs.type != ChatDataType.HUMAN_AUDIO:
            return
        audio = inputs.data.get_main_data()
        if audio is not None and audio.size > 0:
            context.audio_chunks.append(audio.squeeze())

        if not inputs.is_last_data:
            return
        if len(context.audio_chunks) == 0:
            return

        utterance = np.concatenate(context.audio_chunks)
        context.audio_chunks.clear()
        if utterance.shape[0] < self.config.sample_rate // 4:
            return  # below 250 ms — ignore

        wav_bytes = _float32_to_wav_bytes(utterance, self.config.sample_rate)
        form: dict = {"model": self.config.model_name}
        if self.config.language:  # empty -> faster-whisper auto-detect
            form["language"] = self.config.language
        try:
            response = self.client.post(
                f"{self.config.base_url}/audio/transcriptions",
                data=form,
                files={"file": ("speech.wav", wav_bytes, "audio/wav")},
            )
            response.raise_for_status()
        except Exception as error:
            logger.error(f"SpeachesASR request failed: {error}")
            return

        text = response.json().get("text", "").strip()
        logger.info(f"SpeachesASR transcript: {text}")
        if len(text) == 0:
            return
        output = DataBundle(output_definition)
        output.set_main_data(text)
        context.submit_data(output, finish_stream=True)

    def destroy_context(self, context: HandlerContext):
        context = cast(ASRContext, context)
        context.audio_chunks.clear()
