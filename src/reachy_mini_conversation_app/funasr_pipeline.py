"""FunASR + Qwen/DeepSeek + CosyVoice local text-pipeline handler.

Pipeline
--------
    Mic (16 kHz PCM)
      │
      ▼  Energy-based VAD (silence detection)
    FunASR (SenseVoiceSmall, local, ~250 MB)
      │  Chinese / multilingual ASR
      ▼
    Qwen API  ─or─  DeepSeek API   (OpenAI-compatible HTTP, streaming)
      │  Configured via TEXT_LLM_PROVIDER env var
      ▼  Sentence-buffered streaming output
    CosyVoice2-0.5B (local)
      │  High-quality Chinese TTS, 22050 Hz PCM
      ▼
    Reachy Mini speaker

Installation
------------
    # 1. Install Python dependencies
    pip install 'reachy_mini_conversation_app[funasr_pipeline]'

    # 2. Install CosyVoice (requires cloning its repo)
    git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git
    cd CosyVoice && pip install -e .

    # 3. Download CosyVoice2-0.5B model
    python - <<'EOF'
    from modelscope import snapshot_download
    snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')
    EOF

    # 4. Set env vars (see .env.example)
    BACKEND_PROVIDER=funasr_pipeline
    TEXT_LLM_PROVIDER=qwen          # or "deepseek"
    QWEN_API_KEY=sk-...             # DashScope key from console.aliyun.com
    MODEL_NAME=qwen-turbo           # or qwen-plus / deepseek-chat / deepseek-reasoner
    COSYVOICE_MODEL_DIR=pretrained_models/CosyVoice2-0.5B

LLM Endpoints
-------------
    Qwen   : https://dashscope.aliyuncs.com/compatible-mode/v1
    DeepSeek: https://api.deepseek.com/v1

Both are fully OpenAI-API compatible, so the same AsyncOpenAI client is used.
"""
from __future__ import annotations

import re
import io
import asyncio
import logging
from typing import Any, Tuple, Optional

import httpx
import numpy as np
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample as scipy_resample

from reachy_mini_conversation_app.config import (
    FUNASR_PIPELINE_BACKEND,
    COSYVOICE_AVAILABLE_VOICES,
    COSYVOICE_DEFAULT_VOICE,
    QWEN_LLM_PROVIDER,
    DEEPSEEK_LLM_PROVIDER,
    config,
    get_available_voices_for_backend,
    set_custom_profile,
)


def _best_device() -> str:
    """Return the best available PyTorch device: cuda > mps > cpu."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


_DEVICE = _best_device()
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.conversation_handler import ConversationHandler


logger = logging.getLogger(__name__)

__all__ = ["FunASRPipelineHandler"]

# ── Audio constants ──────────────────────────────────────────────────────────

_INPUT_SAMPLE_RATE = 16_000    # FunASR / SenseVoice expects 16 kHz
_OUTPUT_SAMPLE_RATE = 22_050   # CosyVoice2-0.5B native output rate
_TTS_CHUNK_SAMPLES = 2_205     # ~100 ms chunks at 22050 Hz

# ── VAD constants ────────────────────────────────────────────────────────────

_RMS_THRESHOLD = 400    # int16 RMS; raise if background noise causes false triggers
_SILENCE_SECONDS = 1.2  # silence duration (s) that ends an utterance
_MIN_SPEECH_SECONDS = 0.4  # discard shorter utterances (noise bursts)

# ── LLM API endpoints ────────────────────────────────────────────────────────

_LLM_ENDPOINTS: dict[str, str] = {
    QWEN_LLM_PROVIDER: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    DEEPSEEK_LLM_PROVIDER: "https://api.deepseek.com/v1",
}
_DEFAULT_MODELS: dict[str, str] = {
    QWEN_LLM_PROVIDER: "qwen-turbo",
    DEEPSEEK_LLM_PROVIDER: "deepseek-chat",
}

# ── Misc ─────────────────────────────────────────────────────────────────────

_MAX_HISTORY_TURNS = 10
_SENTENCE_ENDS = re.compile(r"[.!?。！？\n]")

# SenseVoiceSmall emits emotion/language tags; strip them from transcription.
_TAG_PATTERN = re.compile(r"<\|[^|]+\|>")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_mono_int16(audio: NDArray[Any]) -> NDArray[np.int16]:
    """Reshape any audio array to 1-D mono int16."""
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = audio[:, 0]
    return audio_to_int16(audio)


def _build_llm_client(provider: str) -> tuple[AsyncOpenAI, str]:
    """Return (AsyncOpenAI client, resolved model name) for the given provider."""
    provider = provider.strip().lower()
    if provider not in _LLM_ENDPOINTS:
        logger.warning("Unknown TEXT_LLM_PROVIDER=%r; falling back to 'qwen'.", provider)
        provider = QWEN_LLM_PROVIDER

    base_url = _LLM_ENDPOINTS[provider]
    if provider == QWEN_LLM_PROVIDER:
        api_key = (config.QWEN_API_KEY or "").strip() or "DUMMY"
        if api_key == "DUMMY":
            logger.warning("QWEN_API_KEY / DASHSCOPE_API_KEY not set.")
    else:
        api_key = (config.DEEPSEEK_API_KEY or "").strip() or "DUMMY"
        if api_key == "DUMMY":
            logger.warning("DEEPSEEK_API_KEY not set.")

    model = (config.MODEL_NAME or "").strip() or _DEFAULT_MODELS[provider]
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    return client, model


# ── Handler ───────────────────────────────────────────────────────────────────

class FunASRPipelineHandler(ConversationHandler):
    """Turn-based FunASR ASR → Qwen/DeepSeek Chat → CosyVoice TTS handler.

    All heavy models are loaded lazily in ``start_up()`` and shared across
    ``copy()`` instances so multiple FastRTC connections avoid redundant loads.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        """Initialise lightweight state; models are loaded in start_up()."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_OUTPUT_SAMPLE_RATE,
            input_sample_rate=_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self._clear_queue = None  # injected by FastRTC

        # Voice state
        available = get_available_voices_for_backend(FUNASR_PIPELINE_BACKEND)
        self._voice: str = startup_voice if startup_voice in available else COSYVOICE_DEFAULT_VOICE

        # VAD state
        self._speech_buffer: list[NDArray[np.int16]] = []
        self._is_speaking: bool = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None

        # Pipeline state
        self._processing: bool = False
        self._conversation_history: list[dict[str, str]] = []

        # Shared model handles (populated in start_up, shared via copy)
        self._funasr_model: Any = None
        self._cosyvoice: Any = None
        self._llm_client: Optional[AsyncOpenAI] = None
        self._llm_model: str = "qwen-turbo"

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def copy(self) -> "FunASRPipelineHandler":
        """Return a new handler sharing the already-loaded heavy models."""
        clone = FunASRPipelineHandler(
            deps=self.deps,
            gradio_mode=self.gradio_mode,
            instance_path=self.instance_path,
            startup_voice=self._voice,
        )
        clone._funasr_model = self._funasr_model
        clone._cosyvoice = self._cosyvoice
        clone._llm_client = self._llm_client
        clone._llm_model = self._llm_model
        return clone

    async def start_up(self) -> None:
        """Load FunASR and CosyVoice models, build the LLM client."""
        await self._load_funasr()
        await self._load_cosyvoice()
        self._llm_client, self._llm_model = _build_llm_client(
            config.TEXT_LLM_PROVIDER or QWEN_LLM_PROVIDER
        )
        logger.info(
            "FunASR pipeline ready  ASR=%s  LLM=%s(%s)  TTS=CosyVoice2  voice=%s",
            config.FUNASR_MODEL,
            config.TEXT_LLM_PROVIDER,
            self._llm_model,
            self._voice,
        )

    async def _load_funasr(self) -> None:
        """Load FunASR SenseVoiceSmall (or configured model) on first call."""
        if self._funasr_model is not None:
            return
        try:
            from funasr import AutoModel  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "funasr is required for FunASR pipeline. "
                "Install with: pip install 'reachy_mini_conversation_app[funasr_pipeline]'"
            ) from exc

        model_id = config.FUNASR_MODEL or "iic/SenseVoiceSmall"
        logger.info("Loading FunASR model %r …", model_id)
        loop = asyncio.get_event_loop()
        logger.info("FunASR using device: %s", _DEVICE)
        self._funasr_model = await loop.run_in_executor(
            None,
            lambda: AutoModel(
                model=model_id,
                trust_remote_code=True,
                device=_DEVICE,
                disable_update=True,
            ),
        )
        logger.info("FunASR model ready.")

    async def _load_cosyvoice(self) -> None:
        """Load CosyVoice2 from the configured model directory on first call."""
        if self._cosyvoice is not None:
            return
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "CosyVoice is not installed. Clone and install it:\n"
                "  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git\n"
                "  cd CosyVoice && pip install -e .\n"
                "Then download the model:\n"
                "  python -c \"from modelscope import snapshot_download; "
                "snapshot_download('iic/CosyVoice2-0.5B', local_dir='pretrained_models/CosyVoice2-0.5B')\""
            ) from exc

        model_dir = config.COSYVOICE_MODEL_DIR or "pretrained_models/CosyVoice2-0.5B"
        logger.info("Loading CosyVoice2 from %r (device=%s) …", model_dir, _DEVICE)

        # CosyVoice2 has partial MPS support; try MPS then fall back to CPU.
        # fp16=False is required on MPS/CPU to avoid unsupported half-precision ops.
        def _load_cosyvoice() -> Any:
            try:
                import torch
                if _DEVICE == "mps":
                    # Enable MPS fallback for unsupported ops
                    import os
                    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
            except Exception:
                pass
            return CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

        loop = asyncio.get_event_loop()
        self._cosyvoice = await loop.run_in_executor(None, _load_cosyvoice)
        logger.info("CosyVoice2 ready.")

    async def shutdown(self) -> None:
        """Drain output queue and reset pipeline state."""
        self._processing = False
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── Audio I/O ─────────────────────────────────────────────────────────────

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive a microphone frame; run VAD to detect end of utterance."""
        input_sr, audio = frame
        audio = _to_mono_int16(audio)

        # Resample to 16 kHz for FunASR if needed
        if input_sr != _INPUT_SAMPLE_RATE:
            resampled = scipy_resample(audio, int(len(audio) * _INPUT_SAMPLE_RATE / input_sr))
            audio = resampled.astype(np.int16)

        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        now = asyncio.get_event_loop().time()
        is_speech = rms > _RMS_THRESHOLD

        if is_speech:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start = now
                self._speech_buffer = []
                # Interrupt any ongoing TTS playback
                if callable(self._clear_queue):
                    self._clear_queue()
            self._silence_start = None
            self._speech_buffer.append(audio.copy())
        else:
            if self._is_speaking:
                if self._silence_start is None:
                    self._silence_start = now
                elif now - self._silence_start >= _SILENCE_SECONDS:
                    speech_dur = now - (self._speech_start or now)
                    if speech_dur >= _MIN_SPEECH_SECONDS and self._speech_buffer and not self._processing:
                        asyncio.create_task(self._run_pipeline(list(self._speech_buffer)))
                    self._is_speaking = False
                    self._silence_start = None
                    self._speech_start = None
                    self._speech_buffer = []
                else:
                    # Short silence — keep buffering
                    self._speech_buffer.append(audio.copy())

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Return the next audio chunk or metadata from the output queue."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    # ── Pipeline ──────────────────────────────────────────────────────────────

    async def _run_pipeline(self, audio_frames: list[NDArray[np.int16]]) -> None:
        """ASR → LLM → TTS pipeline executed as a background task."""
        self._processing = True
        try:
            # ── ASR (local or remote) ─────────────────────────────────────────
            audio_concat = np.concatenate(audio_frames)
            user_text = await self._asr(audio_concat)

            if not user_text:
                logger.debug("FunASR returned empty transcription; skipping.")
                return

            logger.info("ASR → %r", user_text)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": user_text}))

            # ── LLM (streaming) ───────────────────────────────────────────────
            self._conversation_history.append({"role": "user", "content": user_text})
            if len(self._conversation_history) > _MAX_HISTORY_TURNS * 2:
                self._conversation_history = self._conversation_history[-_MAX_HISTORY_TURNS * 2 :]

            messages: list[dict[str, str]] = [
                {"role": "system", "content": get_session_instructions()},
                *self._conversation_history,
            ]

            if self._llm_client is None:
                logger.error("LLM client not initialised; call start_up() first.")
                return

            stream = await self._llm_client.chat.completions.create(
                model=self._llm_model,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            )

            # ── Sentence-buffered TTS ─────────────────────────────────────────
            full_response = ""
            sentence_buf = ""

            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                full_response += delta
                sentence_buf += delta

                # Flush TTS at sentence boundaries for low first-word latency
                if _SENTENCE_ENDS.search(sentence_buf) and sentence_buf.strip():
                    await self._tts_and_enqueue(sentence_buf.strip())
                    sentence_buf = ""

            if sentence_buf.strip():
                await self._tts_and_enqueue(sentence_buf.strip())

            logger.info("LLM → %r", full_response)
            self._conversation_history.append({"role": "assistant", "content": full_response})
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": full_response}))

        except Exception as exc:
            logger.error("FunASR pipeline error: %s", exc, exc_info=True)
        finally:
            self._processing = False

    # ── ASR / TTS dispatch (local vs remote) ──────────────────────────────────

    async def _asr(self, audio_int16: NDArray[np.int16]) -> str:
        """Transcribe audio, using the remote service if configured."""
        remote_url = (config.FUNASR_SERVICE_URL or "").strip()
        if remote_url:
            return await self._asr_remote(audio_int16, remote_url)
        return await self._asr_local(audio_int16)

    async def _asr_local(self, audio_int16: NDArray[np.int16]) -> str:
        """Run FunASR locally."""
        if self._funasr_model is None:
            logger.error("FunASR model not loaded.")
            return ""
        audio_float = audio_int16.astype(np.float32) / 32_768.0
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: self._funasr_model.generate(
                input=audio_float,
                language="auto",
                use_itn=True,
                batch_size_s=60,
            ),
        )
        raw = result[0]["text"] if result else ""
        return _TAG_PATTERN.sub("", raw).strip()

    async def _asr_remote(self, audio_int16: NDArray[np.int16], url: str) -> str:
        """Send raw PCM to the remote ASR service and return the transcription."""
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                url,
                content=audio_int16.tobytes(),
                headers={"Content-Type": "audio/pcm; rate=16000"},
            )
            resp.raise_for_status()
            return resp.json().get("text", "")

    async def _tts_and_enqueue(self, text: str) -> None:
        """Synthesise *text* and push PCM chunks to output_queue (local or remote)."""
        remote_url = (config.TTS_SERVICE_URL or "").strip()
        if remote_url:
            await self._tts_remote(text, remote_url)
        else:
            await self._tts_local(text)

    async def _tts_local(self, text: str) -> None:
        """Run CosyVoice2 locally."""
        if self._cosyvoice is None:
            logger.error("CosyVoice not loaded; skipping TTS.")
            return

        logger.debug("TTS (local): %r  speaker=%r", text, self._voice)
        loop = asyncio.get_event_loop()

        def _synth() -> list[NDArray[np.int16]]:
            chunks: list[NDArray[np.int16]] = []
            for output in self._cosyvoice.inference_sft(text, self._voice, stream=True):
                tensor = output["tts_speech"]
                pcm_float = tensor.squeeze().numpy()
                chunks.append((pcm_float * 32_768).clip(-32_768, 32_767).astype(np.int16))
            return chunks

        for chunk in await loop.run_in_executor(None, _synth):
            for start in range(0, len(chunk), _TTS_CHUNK_SAMPLES):
                sub = chunk[start : start + _TTS_CHUNK_SAMPLES]
                if len(sub) > 0:
                    await self.output_queue.put((_OUTPUT_SAMPLE_RATE, sub))

    async def _tts_remote(self, text: str, url: str) -> None:
        """Stream PCM audio from the remote TTS service into the output queue."""
        logger.debug("TTS (remote): %r  speaker=%r", text, self._voice)
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream(
                "POST",
                url,
                json={"text": text, "voice": self._voice},
            ) as resp:
                resp.raise_for_status()
                # Read the sample rate from the response header (default 22050)
                sample_rate = int(resp.headers.get("X-Sample-Rate", _OUTPUT_SAMPLE_RATE))
                buf = bytearray()
                async for raw_bytes in resp.aiter_bytes():
                    buf.extend(raw_bytes)
                    # Drain complete int16 samples (2 bytes each) from the buffer
                    n_samples = len(buf) // 2
                    if n_samples >= _TTS_CHUNK_SAMPLES:
                        n_bytes = (n_samples // _TTS_CHUNK_SAMPLES) * _TTS_CHUNK_SAMPLES * 2
                        chunk = np.frombuffer(buf[:n_bytes], dtype=np.int16).copy()
                        buf = buf[n_bytes:]
                        for start in range(0, len(chunk), _TTS_CHUNK_SAMPLES):
                            sub = chunk[start : start + _TTS_CHUNK_SAMPLES]
                            if len(sub) > 0:
                                await self.output_queue.put((sample_rate, sub))
                # Flush remaining bytes
                if len(buf) >= 2:
                    remainder = np.frombuffer(buf[: (len(buf) // 2) * 2], dtype=np.int16).copy()
                    if len(remainder) > 0:
                        await self.output_queue.put((sample_rate, remainder))

    # ── Voice management ──────────────────────────────────────────────────────

    async def get_available_voices(self) -> list[str]:
        """Return CosyVoice2 preset speaker list."""
        return list(COSYVOICE_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        """Return the active CosyVoice2 speaker ID."""
        return self._voice

    async def change_voice(self, voice: str) -> str:
        """Switch CosyVoice2 speaker; returns the effective speaker ID."""
        available = list(COSYVOICE_AVAILABLE_VOICES)
        if voice in available:
            self._voice = voice
            logger.info("Voice changed to %r.", voice)
        else:
            logger.warning(
                "Speaker %r not in preset list %s; keeping %r.", voice, available, self._voice
            )
        return self._voice

    # ── Personality ───────────────────────────────────────────────────────────

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Switch personality profile; clears conversation history."""
        set_custom_profile(profile)
        self._conversation_history = []
        logger.info("Applied personality profile %r; conversation history cleared.", profile)
        return profile or "default"
