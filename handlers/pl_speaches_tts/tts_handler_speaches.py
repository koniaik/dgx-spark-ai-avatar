"""TTS handler for OpenAvatarChat: synthesis via a Speaches server
(OpenAI-compatible /v1/audio/speech). Default voice is English (Piper,
ONNX on CPU, low latency, MIT). Patterned after handlers/tts/edgetts
(per-sentence split + finish_stream). For other languages change model_name/voice.
"""

import io
import re
from typing import Dict, Optional, cast

import httpx
import librosa
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


class TTSConfig(HandlerBaseConfigModel, BaseModel):
    base_url: str = Field(default="http://127.0.0.1:8000/v1")
    # Voice id as documented by Speaches (confirm with curl {base_url}/models at install time)
    model_name: str = Field(default="rhasspy/piper-voices")
    voice: str = Field(default="pl_PL-darkman-medium")
    sample_rate: int = Field(default=22050)
    timeout_seconds: float = Field(default=20.0)
    min_sentence_length: int = Field(default=2)


# keep word characters of any script (Polish diacritics included) + speech punctuation;
# strip markdown/emoji/special symbols
_KEEP_PATTERN = re.compile(
    r"[^a-zA-ZąćęłńóśźżĄĆĘŁŃÓŚŹŻ0-9\s,.\-–—:;!?…()%\"']"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…:;])\s+|\n+")


class TTSContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.input_text = ""


class HandlerTTS(HandlerBase):
    def __init__(self):
        super().__init__()
        self.config: Optional[TTSConfig] = None
        self.client: Optional[httpx.Client] = None

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="SpeachesTTS",
            config_model=TTSConfig,
        )

    def get_handler_detail(self, session_context: SessionContext,
                           context: HandlerContext) -> HandlerDetail:
        definition = DataBundleDefinition()
        definition.add_entry(DataBundleEntry.create_audio_entry("avatar_audio", 1, self.config.sample_rate))
        inputs = {
            ChatDataType.AVATAR_TEXT: HandlerDataInfo(
                type=ChatDataType.AVATAR_TEXT,
            )
        }
        outputs = {
            ChatDataType.AVATAR_AUDIO: HandlerDataInfo(
                type=ChatDataType.AVATAR_AUDIO,
                definition=definition,
            )
        }
        return HandlerDetail(inputs=inputs, outputs=outputs)

    def load(self, engine_config: ChatEngineConfigModel, handler_config: Optional[BaseModel] = None):
        self.config = cast(TTSConfig, handler_config)
        self.client = httpx.Client(timeout=self.config.timeout_seconds)
        logger.info(f"SpeachesTTS: base_url={self.config.base_url} voice={self.config.voice}")

    def create_context(self, session_context, handler_config=None):
        return TTSContext(session_context.session_info.session_id)

    def start_context(self, session_context, context: HandlerContext):
        # warmup: pusta próba nie może wywalić sesji, tylko zalogować problem
        try:
            self._synthesize("Test.")
        except Exception as error:
            logger.warning(f"SpeachesTTS warmup failed (is Speaches running?): {error}")

    def _synthesize(self, sentence: str) -> np.ndarray:
        response = self.client.post(
            f"{self.config.base_url}/audio/speech",
            json={
                "model": self.config.model_name,
                "voice": self.config.voice,
                "input": sentence,
                "response_format": "wav",
            },
        )
        response.raise_for_status()
        # decode the WAV and resample to the declared sample_rate (DataBundle definition is fixed)
        audio, _ = librosa.load(io.BytesIO(response.content), sr=self.config.sample_rate, mono=True)
        return audio

    def _submit_audio(self, context: HandlerContext, output_definition, audio: np.ndarray, finish: bool = False):
        output = DataBundle(output_definition)
        if finish:
            output.set_main_data(np.zeros(shape=(1, 240), dtype=np.float32))
        else:
            output.set_main_data(audio[np.newaxis, ...])
        context.submit_data(output, finish_stream=finish)

    def handle(self, context: HandlerContext, inputs: ChatData,
               output_definitions: Dict[ChatDataType, HandlerDataInfo]):
        output_definition = output_definitions.get(ChatDataType.AVATAR_AUDIO).definition
        context = cast(TTSContext, context)
        if inputs.type != ChatDataType.AVATAR_TEXT:
            return
        text = inputs.data.get_main_data()
        if text is not None:
            text = re.sub(r"<\|.*?\|>", "", text)
            text = _KEEP_PATTERN.sub("", text)
            context.input_text += text

        # End-of-text: qwen_omni sets the "avatar_text_end" meta, but
        # openai_compatible (and others) only finish the stream — which the
        # engine reports as is_last_data. Accept BOTH signals; without this
        # the AVATAR_AUDIO stream never closes, LiteAvatar holds the audio
        # tail and duplex suppresses VAD after the first reply.
        text_end = inputs.data.get_meta("avatar_text_end", False) or inputs.is_last_data
        if not text_end:
            parts = _SENTENCE_SPLIT.split(context.input_text.strip())
            if len(parts) > 1:
                complete_sentences, context.input_text = parts[:-1], parts[-1]
                for sentence in complete_sentences:
                    sentence = sentence.strip()
                    if len(sentence) < self.config.min_sentence_length:
                        continue
                    try:
                        audio = self._synthesize(sentence)
                    except Exception as error:
                        logger.error(f"SpeachesTTS failed for sentence: {error}")
                        continue
                    self._submit_audio(context, output_definition, audio)
        else:
            remainder = context.input_text.strip()
            context.input_text = ""
            if len(remainder) >= self.config.min_sentence_length:
                try:
                    audio = self._synthesize(remainder)
                    self._submit_audio(context, output_definition, audio)
                except Exception as error:
                    logger.error(f"SpeachesTTS failed for last sentence: {error}")
            self._submit_audio(context, output_definition, None, finish=True)
            logger.info("speech end")

    def destroy_context(self, context: HandlerContext):
        context = cast(TTSContext, context)
        context.input_text = ""
