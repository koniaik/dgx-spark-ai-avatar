"""Local GPU ASR handler: whisper-large-v3-turbo via transformers, straight on the
CUDA device (torch cu13x nightly already used by the OAC venv on aarch64).

Faster than routing through a CPU-only ASR container: on a DGX Spark a short
utterance transcribes in well under a second vs ~8 s on CPU.
Input/output contract identical to pl_speaches_asr (HUMAN_AUDIO -> HUMAN_TEXT).
"""

import time
from typing import Dict, Optional, cast

import numpy as np
import torch
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
    model_name: str = Field(default="openai/whisper-large-v3-turbo")
    # ISO code ("pl"); "" lets whisper auto-detect the language
    language: str = Field(default="pl")
    sample_rate: int = Field(default=16000)


class ASRContext(HandlerContext):
    def __init__(self, session_id: str):
        super().__init__(session_id)
        self.audio_chunks = []


class HandlerASR(HandlerBase):
    def __init__(self):
        super().__init__()
        self.config: Optional[ASRConfig] = None
        self.model = None
        self.processor = None
        self.device = "cpu"

    def get_handler_info(self) -> HandlerBaseInfo:
        return HandlerBaseInfo(
            name="WhisperGPU",
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
        from transformers import AutoProcessor, WhisperForConditionalGeneration

        self.config = cast(ASRConfig, handler_config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        logger.info(f"WhisperGPU: loading {self.config.model_name} on {self.device} ({dtype})")
        self.processor = AutoProcessor.from_pretrained(self.config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.config.model_name, torch_dtype=dtype
        ).to(self.device)
        self.model.eval()
        logger.info(f"WhisperGPU: model ready on {self.device}")

    def create_context(self, session_context, handler_config=None):
        return ASRContext(session_context.session_info.session_id)

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

        utterance = np.concatenate(context.audio_chunks).astype(np.float32)
        context.audio_chunks.clear()
        if utterance.shape[0] < self.config.sample_rate // 4:
            return  # below 250 ms — ignore

        t0 = time.time()
        inputs_feat = self.processor(utterance, sampling_rate=self.config.sample_rate, return_tensors="pt")
        model_dtype = next(self.model.parameters()).dtype
        inputs_feat = {k: v.to(self.device, dtype=model_dtype if v.is_floating_point() else None) for k, v in inputs_feat.items()}
        gen_kwargs: dict = {}
        if self.config.language:
            gen_kwargs["forced_decoder_ids"] = self.processor.get_decoder_prompt_ids(
                language=self.config.language, task="transcribe")
        with torch.no_grad():
            generated = self.model.generate(**inputs_feat, **gen_kwargs)
        text = self.processor.batch_decode(generated, skip_special_tokens=True)[0].strip()
        logger.info(f"WhisperGPU: transcribed in {time.time() - t0:.2f}s: {text}")
        if len(text) == 0:
            return
        output = DataBundle(output_definition)
        output.set_main_data(text)
        context.submit_data(output, finish_stream=True)

    def destroy_context(self, context: HandlerContext):
        context = cast(ASRContext, context)
        context.audio_chunks.clear()
