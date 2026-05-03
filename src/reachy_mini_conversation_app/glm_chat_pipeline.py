"""GLM Chat text-pipeline handler: local Whisper ASR → GLM-4.x Chat → Edge-TTS.

Architecture
------------
Unlike the Realtime backends (which stream audio over WebSocket), this handler
implements a classic turn-based pipeline:

    Mic audio ──[VAD]──▶ [Whisper ASR] ──▶ [GLM-4.x Chat] ──▶ [Edge-TTS] ──▶ Speaker

Voice Activity Detection
------------------------
Simple energy-based VAD: speech is detected when the RMS of an audio frame
exceeds ``_RMS_THRESHOLD``.  After speech drops below the threshold for
``_SILENCE_SECONDS`` seconds the accumulated buffer is forwarded to ASR.

ASR
---
Uses ``faster-whisper`` with the ``small`` multilingual model running on CPU
with int8 quantisation.  The model is loaded once in ``start_up()`` and shared
across ``copy()`` instances to avoid repeated downloads.

LLM
---
Calls the GLM-4.x Chat Completions API (OpenAI-compatible HTTP, not WebSocket).
Responses are streamed and split at sentence boundaries so TTS starts as soon
as the first sentence is complete, reducing perceived latency.

TTS
---
Uses ``edge-tts`` (Microsoft Edge read-aloud, no API key required).  Audio is
returned as MP3 bytes, which are decoded to 24 kHz mono int16 PCM via the
``av`` library (already a project dependency).

Dependencies (install with ``pip install reachy_mini_conversation_app[glm_chat]``)
---
    faster-whisper
    edge-tts
"""
from __future__ import annotations

import io
import asyncio
import logging
from typing import Any, Tuple, Optional

import av
import numpy as np
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from numpy.typing import NDArray
from scipy.signal import resample

from reachy_mini_conversation_app.config import (
    GLM_CHAT_BACKEND,
    GLM_CHAT_AVAILABLE_VOICES,
    GLM_CHAT_DEFAULT_VOICE,
    config,
    get_default_voice_for_backend,
    get_available_voices_for_backend,
)
from reachy_mini_conversation_app.prompts import get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies
from reachy_mini_conversation_app.conversation_handler import ConversationHandler


logger = logging.getLogger(__name__)

__all__ = ["GlmChatPipelineHandler"]

# ── Audio / VAD constants ────────────────────────────────────────────────────

_INPUT_SAMPLE_RATE = 16_000   # Whisper expects 16 kHz
_OUTPUT_SAMPLE_RATE = 24_000  # Edge-TTS default output rate
_TTS_CHUNK_SAMPLES = 2_400    # 100 ms chunks at 24 kHz

# Energy threshold for speech detection (int16 RMS).
# Increase if background noise causes false triggers.
_RMS_THRESHOLD = 400

# How many seconds of continuous silence trigger ASR.
_SILENCE_SECONDS = 1.2

# Minimum speech duration (seconds) to bother running ASR.
_MIN_SPEECH_SECONDS = 0.4

# Maximum number of conversation turns kept in history.
_MAX_HISTORY_TURNS = 10

# GLM Chat endpoint (OpenAI-compatible HTTP)
_GLM_API_BASE = "https://open.bigmodel.cn/api/paas/v4/"

# Sentence-end characters used to flush TTS early.
_SENTENCE_ENDS = frozenset(".!?。！？\n")


# ── Helper ────────────────────────────────────────────────────────────────────

def _mp3_bytes_to_pcm(mp3_data: bytes, target_rate: int = _OUTPUT_SAMPLE_RATE) -> NDArray[np.int16]:
    """Decode MP3 bytes to mono int16 PCM at *target_rate* using av."""
    buf = io.BytesIO(mp3_data)
    container = av.open(buf, format="mp3")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
    frames: list[NDArray[np.int16]] = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            arr = resampled.to_ndarray()
            frames.append(arr.flatten().astype(np.int16))
    if not frames:
        return np.array([], dtype=np.int16)
    return np.concatenate(frames)


def _to_mono_int16(audio: NDArray[Any]) -> NDArray[np.int16]:
    """Reshape/convert an arbitrary audio array to 1-D int16."""
    if audio.ndim == 2:
        # scipy channels-last convention: (samples, channels) or (channels, samples)
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = audio[:, 0]
    return audio_to_int16(audio)


# ── Handler ───────────────────────────────────────────────────────────────────

class GlmChatPipelineHandler(ConversationHandler):
    """Turn-based ASR → GLM-4.x Chat → Edge-TTS pipeline handler.

    This handler does **not** inherit from ``BaseRealtimeHandler`` because it
    uses an ordinary HTTP Chat Completions API rather than a WebSocket Realtime
    API.  It implements the ``ConversationHandler`` interface directly.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        """Initialise pipeline state (Whisper / LLM clients loaded in start_up)."""
        super().__init__(
            expected_layout="mono",
            output_sample_rate=_OUTPUT_SAMPLE_RATE,
            input_sample_rate=_INPUT_SAMPLE_RATE,
        )

        self.deps = deps
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path

        # Output queue consumed by emit()
        self.output_queue: asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs] = asyncio.Queue()
        self._clear_queue = None  # set by FastRTC

        # Voice state
        available = get_available_voices_for_backend(GLM_CHAT_BACKEND)
        if startup_voice and startup_voice in available:
            self._voice: str = startup_voice
        else:
            self._voice = GLM_CHAT_DEFAULT_VOICE

        # VAD state
        self._speech_buffer: list[NDArray[np.int16]] = []
        self._is_speaking: bool = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None

        # Pipeline state
        self._processing: bool = False
        self._conversation_history: list[dict[str, str]] = []

        # Lazily shared across copy() instances
        self._whisper_model: Any = None
        self._glm_client: Optional[AsyncOpenAI] = None

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def copy(self) -> "GlmChatPipelineHandler":
        """Return a new handler sharing the (heavy) Whisper model."""
        clone = GlmChatPipelineHandler(
            deps=self.deps,
            gradio_mode=self.gradio_mode,
            instance_path=self.instance_path,
            startup_voice=self._voice,
        )
        # Share the already-loaded Whisper model to avoid re-downloading.
        clone._whisper_model = self._whisper_model
        return clone

    async def start_up(self) -> None:
        """Load Whisper model and initialise GLM client."""
        if self._whisper_model is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import]
            except ImportError as exc:
                raise RuntimeError(
                    "faster-whisper is required for GLM Chat pipeline. "
                    "Install it with: pip install 'reachy_mini_conversation_app[glm_chat]'"
                ) from exc

            logger.info("Loading Whisper 'small' model (first run may download ~500 MB)…")
            self._whisper_model = WhisperModel("small", device="cpu", compute_type="int8")
            logger.info("Whisper model ready.")

        api_key = (config.GLM_API_KEY or "").strip()
        if not api_key:
            logger.warning(
                "GLM_API_KEY / ZHIPUAI_API_KEY is not set. "
                "Set it in your .env file before connecting."
            )
            api_key = "DUMMY"

        self._glm_client = AsyncOpenAI(api_key=api_key, base_url=_GLM_API_BASE)
        logger.info("GLM Chat pipeline handler ready (model=%s, voice=%s).", config.MODEL_NAME, self._voice)

    async def shutdown(self) -> None:
        """Clear output queue and reset pipeline state."""
        self._processing = False
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    # ── Audio I/O ─────────────────────────────────────────────────────────────

    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive a microphone audio frame and run VAD logic.

        Speech is buffered until a period of silence triggers the ASR+LLM+TTS
        pipeline in a background task.
        """
        input_sr, audio = frame

        # Normalise to mono int16
        audio = _to_mono_int16(audio)

        # Resample to Whisper's expected 16 kHz if FastRTC provides a different rate
        if input_sr != _INPUT_SAMPLE_RATE:
            resampled = resample(audio, int(len(audio) * _INPUT_SAMPLE_RATE / input_sr))
            audio = resampled.astype(np.int16)

        # Energy-based VAD
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        now = asyncio.get_event_loop().time()
        is_speech = rms > _RMS_THRESHOLD

        if is_speech:
            if not self._is_speaking:
                # Transition: silence → speech
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
                # Ongoing silence during a speech segment
                if self._silence_start is None:
                    self._silence_start = now
                elif now - self._silence_start >= _SILENCE_SECONDS:
                    # Silence long enough → end of utterance
                    speech_duration = (self._speech_start or now) and (now - (self._speech_start or now))
                    if speech_duration >= _MIN_SPEECH_SECONDS and self._speech_buffer:
                        buffer_snapshot = list(self._speech_buffer)
                        if not self._processing:
                            asyncio.create_task(self._run_pipeline(buffer_snapshot))
                    # Reset VAD state
                    self._is_speaking = False
                    self._silence_start = None
                    self._speech_start = None
                    self._speech_buffer = []
                else:
                    # Silence, but not long enough yet — keep buffering
                    self._speech_buffer.append(audio.copy())

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Return the next queued audio chunk or metadata for playback."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    # ── Pipeline ──────────────────────────────────────────────────────────────

    async def _run_pipeline(self, audio_frames: list[NDArray[np.int16]]) -> None:
        """Run ASR → GLM Chat → TTS for a completed utterance (background task)."""
        self._processing = True
        try:
            # ── ASR ──────────────────────────────────────────────────────────
            audio_concat = np.concatenate(audio_frames)
            audio_float = audio_concat.astype(np.float32) / 32_768.0

            loop = asyncio.get_event_loop()
            segments_result = await loop.run_in_executor(
                None,
                lambda: list(self._whisper_model.transcribe(audio_float, beam_size=5)[0]),
            )
            user_text = "".join(seg.text for seg in segments_result).strip()

            if not user_text:
                logger.debug("Whisper returned empty transcription, skipping.")
                return

            logger.info("ASR → %r", user_text)
            await self.output_queue.put(AdditionalOutputs({"role": "user", "content": user_text}))

            # ── GLM Chat ──────────────────────────────────────────────────────
            self._conversation_history.append({"role": "user", "content": user_text})
            if len(self._conversation_history) > _MAX_HISTORY_TURNS * 2:
                self._conversation_history = self._conversation_history[-_MAX_HISTORY_TURNS * 2 :]

            messages: list[dict[str, str]] = [
                {"role": "system", "content": get_session_instructions()},
                *self._conversation_history,
            ]

            if self._glm_client is None:
                logger.error("GLM client not initialised; call start_up() first.")
                return

            model_name = config.MODEL_NAME or "glm-4.7"
            stream = await self._glm_client.chat.completions.create(
                model=model_name,
                messages=messages,  # type: ignore[arg-type]
                stream=True,
            )

            # ── Streaming LLM → sentence-buffered TTS ─────────────────────────
            full_response = ""
            sentence_buf = ""

            async for chunk in stream:
                delta = (chunk.choices[0].delta.content or "") if chunk.choices else ""
                full_response += delta
                sentence_buf += delta

                # Flush TTS when a sentence boundary is reached
                if any(c in sentence_buf for c in _SENTENCE_ENDS) and sentence_buf.strip():
                    await self._tts_and_enqueue(sentence_buf.strip())
                    sentence_buf = ""

            # Flush any remaining text
            if sentence_buf.strip():
                await self._tts_and_enqueue(sentence_buf.strip())

            logger.info("LLM → %r", full_response)
            self._conversation_history.append({"role": "assistant", "content": full_response})
            await self.output_queue.put(AdditionalOutputs({"role": "assistant", "content": full_response}))

        except Exception as exc:
            logger.error("Pipeline error: %s", exc, exc_info=True)
        finally:
            self._processing = False

    async def _tts_and_enqueue(self, text: str) -> None:
        """Convert *text* to speech via Edge-TTS and push PCM chunks to output_queue."""
        try:
            import edge_tts  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError(
                "edge-tts is required for GLM Chat pipeline. "
                "Install it with: pip install 'reachy_mini_conversation_app[glm_chat]'"
            ) from exc

        logger.debug("TTS: %r", text)
        communicate = edge_tts.Communicate(text, self._voice)
        mp3_chunks: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                mp3_chunks.append(chunk["data"])

        if not mp3_chunks:
            return

        # Decode MP3 → PCM in a thread (av is synchronous)
        mp3_data = b"".join(mp3_chunks)
        loop = asyncio.get_event_loop()
        pcm = await loop.run_in_executor(None, _mp3_bytes_to_pcm, mp3_data, _OUTPUT_SAMPLE_RATE)

        # Enqueue in small chunks so FastRTC streams smoothly
        for start in range(0, len(pcm), _TTS_CHUNK_SAMPLES):
            chunk_arr = pcm[start : start + _TTS_CHUNK_SAMPLES]
            if len(chunk_arr) > 0:
                await self.output_queue.put((_OUTPUT_SAMPLE_RATE, chunk_arr))

    # ── Voice management ──────────────────────────────────────────────────────

    async def get_available_voices(self) -> list[str]:
        """Return Edge-TTS voices available for GLM Chat backend."""
        return list(GLM_CHAT_AVAILABLE_VOICES)

    def get_current_voice(self) -> str:
        """Return the active Edge-TTS voice."""
        return self._voice

    async def change_voice(self, voice: str) -> str:
        """Switch to a different Edge-TTS voice; returns the effective voice."""
        available = list(GLM_CHAT_AVAILABLE_VOICES)
        if voice in available:
            self._voice = voice
            logger.info("Voice changed to %r.", voice)
        else:
            logger.warning("Voice %r not in available list %s; keeping %r.", voice, available, self._voice)
        return self._voice

    # ── Personality ───────────────────────────────────────────────────────────

    async def apply_personality(self, profile: Optional[str]) -> str:
        """Switch personality profile; clears conversation history."""
        from reachy_mini_conversation_app.config import set_custom_profile

        set_custom_profile(profile)
        self._conversation_history = []
        logger.info("Applied personality profile %r; conversation history cleared.", profile)
        return profile or "default"
