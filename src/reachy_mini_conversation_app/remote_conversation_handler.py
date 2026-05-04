"""Remote conversation handler for Reachy Mini (Pi 4).

Delegates the entire ASR → LLM → TTS pipeline to the Mac inference server.
By default the Pi 4 streams microphone frames over WebSocket so the Mac owns
VAD/turn detection, then plays back the returned PCM audio.  No local ML models
are required.

Architecture
------------
    Pi 4 (this handler)               Mac M4 Pro (inference_server.py)
    ┌────────────────────┐            ┌──────────────────────────────┐
    │  Mic  (16 kHz PCM) │            │                              │
    │  Audio streamer    │──WiFi──▶  │  WS /conversation/ws         │
    │                    │            │    ↓ Server-side VAD         │
    │                    │            │    ↓ FunASR (ASR)            │
    │                    │            │    ↓ Qwen/DeepSeek (LLM)     │
    │                    │◀──audio── │    ↓ CosyVoice2 (TTS)        │
    │  Speaker playback  │            └──────────────────────────────┘
    └────────────────────┘

Setup
-----
    # On Mac: start the inference server
    reachy-mini-inference-server

    # On Pi 4: add to .env
    BACKEND_PROVIDER=remote
    CONVERSATION_SERVICE_URL=http://<mac-ip>:8765/conversation
    # Optional: CONVERSATION_STREAM_URL=ws://<mac-ip>:8765/conversation/ws
    # Optional CosyVoice voice (default: 中文女)
    # VOICE=中文女
"""
from __future__ import annotations

import asyncio
import json
import logging
import socket
import time
from typing import Optional, Tuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import httpx
import numpy as np
import websockets
from fastrtc import AdditionalOutputs, wait_for_item, audio_to_int16
from numpy.typing import NDArray

from reachy_mini_conversation_app.config import (
    COSYVOICE_DEFAULT_VOICE,
    REMOTE_BACKEND,
    config,
    get_available_voices_for_backend,
    set_custom_profile,
)
from reachy_mini_conversation_app.conversation_handler import ConversationHandler
from reachy_mini_conversation_app.tools.core_tools import dispatch_tool_call
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies


logger = logging.getLogger(__name__)

__all__ = ["RemoteConversationHandler"]

# ── Audio constants ───────────────────────────────────────────────────────────

_INPUT_SAMPLE_RATE = 16_000   # microphone capture rate
_OUTPUT_SAMPLE_RATE = 22_050  # CosyVoice2 native rate on the Mac server

# ── VAD constants ─────────────────────────────────────────────────────────────

_RMS_THRESHOLD = 750      # int16 RMS; raise if noise causes false triggers
_SILENCE_SECONDS = 0.85   # silence duration (s) that ends an utterance
_LONG_SPEECH_SILENCE_SECONDS = 1.15
_LONG_SPEECH_SECONDS = 2.0
_MIN_SPEECH_SECONDS = 0.45  # discard utterances shorter than this
_MIN_POST_SAMPLES = int(_INPUT_SAMPLE_RATE * 0.8)
_MAX_SPEECH_SECONDS = 18.0
_BARGE_IN_RMS_THRESHOLD = int(config.REMOTE_BARGE_IN_RMS_THRESHOLD)
_BARGE_IN_SECONDS = float(config.REMOTE_BARGE_IN_SECONDS)
_BARGE_IN_PRE_ROLL_SECONDS = float(config.REMOTE_BARGE_IN_PRE_ROLL_SECONDS)
_PLAYBACK_GUARD_SECONDS = 0.35
_ALLOWED_REMOTE_ACTIONS = {
    "dance",
    "stop_dance",
    "play_emotion",
    "stop_emotion",
    "move_head",
    "head_tracking",
    "do_nothing",
}

# ── HTTP client settings ──────────────────────────────────────────────────────

_HTTP_TIMEOUT = httpx.Timeout(connect=5.0, read=60.0, write=30.0, pool=5.0)
_STREAM_CHUNK_SIZE = 4_096  # bytes per httpx read iteration
_STREAM_AUDIO_QUEUE_MAX = 64
_STREAM_RECONNECT_SECONDS = 1.5
_REMOTE_ACTION_TIMEOUT = 3.0
_REMOTE_CALL_TIMEOUT = 45.0
_IMMEDIATE_REMOTE_ACTIONS = {"stop_dance", "stop_emotion", "do_nothing"}


class _ClientConversationState:
    LISTENING = "LISTENING"
    THINKING = "THINKING"
    SPEAKING = "SPEAKING"
    INTERRUPTING = "INTERRUPTING"
    RECOVERING = "RECOVERING"
    DISCONNECTED = "DISCONNECTED"


def _to_mono_int16(audio: NDArray) -> NDArray[np.int16]:
    """Reshape any audio array to 1-D mono int16."""
    if audio.ndim == 2:
        if audio.shape[1] > audio.shape[0]:
            audio = audio.T
        audio = audio[:, 0]
    return audio_to_int16(audio)


def _derive_stream_url(conversation_url: str) -> str:
    """Derive ws://host/conversation/ws from http://host/conversation."""
    explicit = (getattr(config, "CONVERSATION_STREAM_URL", None) or "").strip()
    if explicit:
        return explicit

    parsed = urlsplit(conversation_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    path = parsed.path.rstrip("/")
    if path.endswith("/conversation"):
        path = f"{path}/ws"
    else:
        path = f"{path}/conversation/ws"
    return urlunsplit((scheme, parsed.netloc, path, "", ""))


def _initial_remote_session_id() -> str:
    configured = (getattr(config, "REMOTE_CONVERSATION_SESSION_ID", None) or "").strip()
    if configured:
        return configured
    return socket.gethostname().strip() or "reachy-mini"


class RemoteConversationHandler(ConversationHandler):
    """Minimal Pi 4 handler: stream audio → receive streamed audio back.

    All intelligence (ASR, LLM, TTS) runs on the Mac inference server.
    This handler has no local ML dependencies beyond ``httpx`` and ``numpy``.
    """

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ) -> None:
        """Initialise with lightweight audio streaming and HTTP fallback state."""
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

        available = get_available_voices_for_backend(REMOTE_BACKEND)
        self._voice: str = startup_voice if startup_voice in available else COSYVOICE_DEFAULT_VOICE

        # VAD state
        self._speech_buffer: list[NDArray[np.int16]] = []
        self._is_speaking: bool = False
        self._silence_start: Optional[float] = None
        self._speech_start: Optional[float] = None
        self._processing: bool = False
        self._ignore_until: float = 0.0
        self._playback_until: float = 0.0
        self._barge_in_start: Optional[float] = None
        self._barge_in_callback = None
        self._barge_in_pre_roll: list[NDArray[np.int16]] = []
        self._barge_in_pre_roll_samples = 0
        self._barge_in_max_pre_roll_samples = int(_INPUT_SAMPLE_RATE * _BARGE_IN_PRE_ROLL_SECONDS)

        # HTTP session tracking
        self._session_id: Optional[str] = _initial_remote_session_id()
        self._http_client: Optional[httpx.AsyncClient] = None
        self._remote_task: Optional[asyncio.Task[None]] = None

        # Streaming remote session. In this mode Reachy sends continuous mic
        # frames and the Mac server owns VAD/turn detection.
        self._streaming_enabled = bool(getattr(config, "REMOTE_AUDIO_STREAMING", True))
        self._stream_url: str = ""
        self._stream_task: Optional[asyncio.Task[None]] = None
        self._stream_audio_queue: asyncio.Queue[bytes | str] = asyncio.Queue(maxsize=_STREAM_AUDIO_QUEUE_MAX)
        self._stream_response_active = False
        self._stream_connected = False
        self._stream_output_sample_rate = _OUTPUT_SAMPLE_RATE
        self._cancelled_response_ids: set[str] = set()
        self._conversation_state = _ClientConversationState.DISCONNECTED

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def copy(self) -> "RemoteConversationHandler":
        """Return a new handler (no shared state needed — no local models)."""
        return RemoteConversationHandler(
            deps=self.deps,
            gradio_mode=self.gradio_mode,
            instance_path=self.instance_path,
            startup_voice=self._voice,
        )

    async def start_up(self) -> None:
        """Verify the remote server is reachable."""
        self._http_client = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        url = (config.CONVERSATION_SERVICE_URL or "").strip()
        if not url:
            logger.warning(
                "CONVERSATION_SERVICE_URL is not set. "
                "Set it to http://<mac-ip>:8765/conversation in .env"
            )
            return

        health_url = url.rsplit("/conversation", 1)[0] + "/health"
        try:
            resp = await self._http_client.get(health_url)
            data = resp.json()
            logger.info(
                "Remote inference server health: asr=%s tts=%s device=%s",
                data.get("asr"),
                data.get("tts"),
                data.get("device"),
            )
        except Exception as exc:
            logger.warning("Could not reach inference server at %s: %s", health_url, exc)

        if self._streaming_enabled:
            self._stream_url = _derive_stream_url(url)
            self._stream_task = asyncio.create_task(self._stream_loop(), name="remote-conversation-stream")
            logger.info("Remote audio streaming enabled: %s", self._stream_url)
        else:
            logger.info("Remote audio streaming disabled; using POST /conversation mode.")

    async def shutdown(self) -> None:
        """Close the remote HTTP client and clear queued audio."""
        if self._stream_task is not None and not self._stream_task.done():
            self._stream_task.cancel()
            try:
                await self._stream_task
            except asyncio.CancelledError:
                pass

        if self._remote_task is not None and not self._remote_task.done():
            self._remote_task.cancel()
            try:
                await self._remote_task
            except asyncio.CancelledError:
                pass

        if self._http_client is not None:
            try:
                await self._http_client.aclose()
            except Exception as exc:
                logger.debug("Error closing remote HTTP client: %s", exc)
            finally:
                self._http_client = None

        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def set_barge_in_callback(self, callback) -> None:
        """Register a local playback clear callback used when the user interrupts."""
        self._barge_in_callback = callback

    def note_playback_audio(self, sample_rate: int, sample_count: int) -> None:
        """Track approximate speaker activity for playback-aware VAD."""
        if sample_rate <= 0 or sample_count <= 0:
            return
        now = time.monotonic()
        duration = sample_count / sample_rate
        self._playback_until = max(self._playback_until, now) + duration + _PLAYBACK_GUARD_SECONDS

    def clear_output_queue(self) -> None:
        """Drop queued response audio without replacing the queue object."""
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def _is_playback_active(self) -> bool:
        return time.monotonic() < self._playback_until

    def _reset_vad_state(self) -> None:
        self._speech_buffer = []
        self._is_speaking = False
        self._silence_start = None
        self._speech_start = None
        self._barge_in_start = None
        self._barge_in_pre_roll = []
        self._barge_in_pre_roll_samples = 0

    def _append_barge_in_pre_roll(self, pcm: NDArray[np.int16]) -> None:
        chunk = pcm.copy()
        self._barge_in_pre_roll.append(chunk)
        self._barge_in_pre_roll_samples += chunk.size
        while self._barge_in_pre_roll and self._barge_in_pre_roll_samples > self._barge_in_max_pre_roll_samples:
            old = self._barge_in_pre_roll.pop(0)
            self._barge_in_pre_roll_samples -= old.size

    def _set_conversation_state(self, state: str, reason: str) -> None:
        if self._conversation_state == state:
            return
        previous = self._conversation_state
        self._conversation_state = state
        logger.info("Remote conversation state %s -> %s reason=%s", previous, state, reason)

    def _interrupt_response(self) -> None:
        """Cancel in-flight response generation and clear queued playback audio."""
        logger.info("Barge-in: interrupting current response")
        self._set_conversation_state(_ClientConversationState.INTERRUPTING, "local_barge_in")
        if self._remote_task is not None and not self._remote_task.done():
            self._remote_task.cancel()
        self.clear_output_queue()
        if callable(self._barge_in_callback):
            try:
                self._barge_in_callback()
            except Exception as exc:
                logger.debug("Barge-in callback failed: %s", exc)
        self._processing = False
        self._playback_until = 0.0

    def _decode_remote_actions(self, actions_header: str) -> list[dict[str, object]]:
        """Decode and validate Reachy-local actions from a response header."""
        if not actions_header:
            return []
        try:
            actions = json.loads(unquote(actions_header))
        except Exception as exc:
            logger.warning("Invalid X-Reachy-Actions header: %s", exc)
            return []
        if not isinstance(actions, list):
            return []
        if not actions:
            return []

        valid_actions: list[dict[str, object]] = []
        for action in actions[:4]:
            if not isinstance(action, dict):
                continue
            name = action.get("name")
            arguments = action.get("arguments", {})
            if name not in _ALLOWED_REMOTE_ACTIONS:
                logger.warning("Ignoring unsupported remote action: %r", name)
                continue
            if not isinstance(arguments, dict):
                arguments = {}
            timing = action.get("timing", "speech_start")
            if timing not in {"immediate", "speech_start", "after_speech"}:
                timing = "speech_start"
            valid_actions.append({"name": name, "arguments": arguments, "timing": timing})
        return valid_actions

    def _split_remote_actions(
        self,
        actions: list[dict[str, object]],
    ) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
        """Split actions by when they should run relative to speech playback."""
        immediate: list[dict[str, object]] = []
        speech_synced: list[dict[str, object]] = []
        after_speech: list[dict[str, object]] = []

        for action in actions:
            name = action.get("name")
            arguments = action.get("arguments", {})
            timing = action.get("timing", "speech_start")
            if timing == "immediate" or name in _IMMEDIATE_REMOTE_ACTIONS:
                immediate.append(action)
            elif name == "head_tracking" and isinstance(arguments, dict) and arguments.get("start") is False:
                immediate.append(action)
            elif timing == "after_speech":
                after_speech.append(action)
            else:
                speech_synced.append(action)
        return immediate, speech_synced, after_speech

    async def _execute_remote_action_items(self, actions: list[dict[str, object]], timing: str) -> None:
        """Execute Reachy-local actions requested by the Mac inference server."""
        if not actions:
            return

        for action in actions:
            name = action.get("name")
            arguments = action.get("arguments", {})
            if not isinstance(arguments, dict):
                arguments = {}

            logger.info("Executing %s remote action: %s %s", timing, name, arguments)
            try:
                result = await asyncio.wait_for(
                    dispatch_tool_call(str(name), json.dumps(arguments), self.deps),
                    timeout=_REMOTE_ACTION_TIMEOUT,
                )
                if result.get("error"):
                    logger.warning("Remote action %s failed: %s", name, result["error"])
                else:
                    logger.info("Remote action %s result: %s", name, result)
            except asyncio.TimeoutError:
                logger.warning("Remote action %s timed out after %.1fs", name, _REMOTE_ACTION_TIMEOUT)
            except Exception as exc:
                logger.exception("Remote action %s raised: %s", name, exc)

    async def _execute_remote_action_event(self, action: dict[str, object], timing: str) -> None:
        """Execute one event-protocol Reachy action and report the result upstream."""
        name = action.get("name")
        arguments = action.get("arguments", {})
        action_call_id = action.get("action_call_id")
        if not isinstance(arguments, dict):
            arguments = {}

        result: dict[str, object]
        logger.info("Executing %s action event: %s %s id=%s", timing, name, arguments, action_call_id)
        try:
            result = await asyncio.wait_for(
                dispatch_tool_call(str(name), json.dumps(arguments), self.deps),
                timeout=_REMOTE_ACTION_TIMEOUT,
            )
            if result.get("error"):
                logger.warning("Remote action event %s failed: %s", name, result["error"])
            else:
                logger.info("Remote action event %s result: %s", name, result)
        except asyncio.TimeoutError:
            result = {"error": f"timed out after {_REMOTE_ACTION_TIMEOUT:.1f}s"}
            logger.warning("Remote action event %s timed out after %.1fs", name, _REMOTE_ACTION_TIMEOUT)
        except Exception as exc:
            result = {"error": f"{type(exc).__name__}: {exc}"}
            logger.exception("Remote action event %s raised: %s", name, exc)

        await self._queue_stream_message(
            json.dumps(
                {
                    "type": "action.result",
                    "action_call_id": action_call_id,
                    "name": name,
                    "timing": timing,
                    "result": result,
                    "error": result.get("error"),
                },
                ensure_ascii=True,
            )
        )

    async def _execute_remote_actions(self, actions_header: str, timing: str = "immediate") -> None:
        """Decode and execute Reachy-local actions requested by the Mac inference server."""
        await self._execute_remote_action_items(self._decode_remote_actions(actions_header), timing)

    # ── FastRTC audio interface ────────────────────────────────────────────────

    async def receive(self, frame: tuple[int, NDArray]) -> None:
        """Accept incoming microphone frames and run energy-based VAD."""
        _, audio = frame
        pcm = _to_mono_int16(audio)

        if self._streaming_enabled:
            await self._receive_streaming(pcm)
            return

        rms = int(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        now = time.monotonic()
        playback_active = self._is_playback_active()
        response_active = playback_active or self._processing

        if now < self._ignore_until:
            return

        if response_active:
            if rms >= _BARGE_IN_RMS_THRESHOLD:
                if self._barge_in_start is None:
                    self._barge_in_start = now
                elif now - self._barge_in_start >= _BARGE_IN_SECONDS:
                    self._interrupt_response()
                    self._reset_vad_state()
                    self._is_speaking = True
                    self._speech_start = now
                    self._speech_buffer = [pcm]
                    logger.info("Remote VAD: barge-in speech start (rms=%d)", rms)
                    return
            else:
                self._barge_in_start = None
            return

        if rms >= _RMS_THRESHOLD:
            if not self._is_speaking:
                self._is_speaking = True
                self._speech_start = now
                self._speech_buffer = []
                logger.info("Remote VAD: speech start (rms=%d)", rms)
            self._speech_buffer.append(pcm)
            self._silence_start = None
        elif self._is_speaking:
            self._speech_buffer.append(pcm)
            if self._silence_start is None:
                self._silence_start = now
            speech_dur = (self._silence_start or now) - (self._speech_start or now)
            silence_seconds = (
                _LONG_SPEECH_SILENCE_SECONDS
                if speech_dur >= _LONG_SPEECH_SECONDS
                else _SILENCE_SECONDS
            )
            max_speech_reached = (
                self._speech_start is not None
                and now - self._speech_start >= _MAX_SPEECH_SECONDS
            )
            if now - self._silence_start >= silence_seconds or max_speech_reached:
                speech_dur = (self._silence_start or now) - (self._speech_start or now)
                buffered_samples = sum(chunk.size for chunk in self._speech_buffer)
                if (
                    speech_dur >= _MIN_SPEECH_SECONDS
                    and buffered_samples >= _MIN_POST_SAMPLES
                    and not self._processing
                ):
                    audio_data = np.concatenate(self._speech_buffer)
                    logger.info(
                        "Remote VAD: speech end after %.2fs silence=%.2fs, posting %d samples",
                        speech_dur,
                        now - self._silence_start,
                        audio_data.size,
                    )
                    self._remote_task = asyncio.create_task(self._call_remote(audio_data))
                else:
                    logger.info(
                        "Remote VAD: utterance too short (%.2fs, %d samples), discarding",
                        speech_dur,
                        buffered_samples,
                    )
                    self._ignore_until = time.monotonic() + 0.2
                self._reset_vad_state()

    async def emit(self) -> tuple[int, NDArray[np.int16]] | AdditionalOutputs:
        """Yield queued audio frames to FastRTC for playback."""
        return await wait_for_item(self.output_queue)

    # ── Remote streaming ──────────────────────────────────────────────────────

    async def _receive_streaming(self, pcm: NDArray[np.int16]) -> None:
        """Queue microphone PCM for the Mac-owned VAD stream."""
        now = time.monotonic()
        if now < self._ignore_until:
            return
        if not self._stream_connected:
            return

        if self._stream_response_active or self._is_playback_active():
            self._append_barge_in_pre_roll(pcm)
            rms = int(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
            if rms < _BARGE_IN_RMS_THRESHOLD:
                self._barge_in_start = None
                return
            if self._barge_in_start is None:
                self._barge_in_start = now
                return
            if now - self._barge_in_start < _BARGE_IN_SECONDS:
                return

            logger.info("Remote stream barge-in detected (rms=%d); interrupting response", rms)
            self._interrupt_response()
            self._stream_response_active = False
            self._processing = False
            await self._queue_stream_message(
                json.dumps({"type": "interrupt", "reason": "barge_in"}, ensure_ascii=True)
            )
            self._barge_in_start = None
            self._set_conversation_state(_ClientConversationState.RECOVERING, "barge_in_sent")
            pre_roll_chunks = self._barge_in_pre_roll
            pre_roll_samples = self._barge_in_pre_roll_samples
            self._barge_in_pre_roll = []
            self._barge_in_pre_roll_samples = 0
            logger.info(
                "Remote stream barge-in: forwarding %.2fs pre-roll audio",
                pre_roll_samples / _INPUT_SAMPLE_RATE,
            )
            for chunk in pre_roll_chunks:
                await self._queue_stream_message(chunk.tobytes())
            return
        else:
            self._barge_in_start = None
            self._barge_in_pre_roll = []
            self._barge_in_pre_roll_samples = 0

        await self._queue_stream_message(pcm.tobytes())

    async def _queue_stream_message(self, message: bytes | str) -> None:
        try:
            self._stream_audio_queue.put_nowait(message)
        except asyncio.QueueFull:
            try:
                self._stream_audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            try:
                self._stream_audio_queue.put_nowait(message)
            except asyncio.QueueFull:
                pass

    async def _stream_loop(self) -> None:
        """Maintain a long-lived websocket to the Mac inference server."""
        while True:
            if not self._stream_url:
                await asyncio.sleep(_STREAM_RECONNECT_SECONDS)
                continue
            try:
                async with websockets.connect(self._stream_url, max_size=None) as ws:
                    self._stream_connected = True
                    self._set_conversation_state(_ClientConversationState.LISTENING, "stream_connected")
                    while not self._stream_audio_queue.empty():
                        try:
                            self._stream_audio_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            break
                    logger.info("Remote conversation stream connected: %s", self._stream_url)
                    await ws.send(
                        json.dumps(
                            {
                                "type": "start",
                                "session_id": self._session_id,
                                "voice": self._voice,
                                "sample_rate": _INPUT_SAMPLE_RATE,
                                "capabilities": ["interrupt", "response.sentence", "action.call"],
                            },
                            ensure_ascii=True,
                        )
                    )
                    sender = asyncio.create_task(self._stream_sender(ws), name="remote-stream-sender")
                    receiver = asyncio.create_task(self._stream_receiver(ws), name="remote-stream-receiver")
                    done, pending = await asyncio.wait(
                        {sender, receiver},
                        return_when=asyncio.FIRST_EXCEPTION,
                    )
                    for task in pending:
                        task.cancel()
                    await asyncio.gather(*pending, return_exceptions=True)
                    for task in done:
                        if task.cancelled():
                            continue
                        exc = task.exception()
                        if exc is not None:
                            raise exc
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("Remote conversation stream disconnected: %s", exc)
                self._stream_connected = False
                self._stream_response_active = False
                self._processing = False
                self._barge_in_pre_roll = []
                self._barge_in_pre_roll_samples = 0
                self._set_conversation_state(_ClientConversationState.DISCONNECTED, "stream_disconnected")
                await asyncio.sleep(_STREAM_RECONNECT_SECONDS)

    async def _stream_sender(self, ws) -> None:
        while True:
            message = await self._stream_audio_queue.get()
            await ws.send(message)

    async def _stream_receiver(self, ws) -> None:
        deferred_actions: list[dict[str, object]] = []
        after_speech_actions: list[dict[str, object]] = []
        deferred_action_events: list[dict[str, object]] = []
        after_speech_action_events: list[dict[str, object]] = []
        deferred_actions_started = False

        async for message in ws:
            if isinstance(message, bytes):
                if (deferred_actions or deferred_action_events) and not deferred_actions_started:
                    deferred_actions_started = True
                    logger.info("Starting deferred remote actions with first streamed audio chunk.")
                    if deferred_actions:
                        asyncio.create_task(
                            self._execute_remote_action_items(deferred_actions, "speech-synced"),
                            name="remote-reachy-actions-speech-synced",
                        )
                    for action_event in deferred_action_events:
                        asyncio.create_task(
                            self._execute_remote_action_event(action_event, "speech-synced"),
                            name="remote-reachy-action-event-speech-synced",
                        )
                    deferred_action_events = []
                pcm_out = np.frombuffer(message, dtype=np.int16)
                await self.output_queue.put((self._stream_output_sample_rate, pcm_out))
                continue

            try:
                data = json.loads(message)
            except json.JSONDecodeError:
                logger.debug("Ignoring non-JSON stream message: %r", message[:120])
                continue

            msg_type = data.get("type")
            if msg_type == "ready":
                self._stream_output_sample_rate = int(data.get("sample_rate") or _OUTPUT_SAMPLE_RATE)
                logger.info("Remote conversation stream ready at %d Hz", self._stream_output_sample_rate)
            elif msg_type == "state.changed":
                state = data.get("state")
                reason = data.get("reason") or "server_state"
                if isinstance(state, str) and state:
                    self._set_conversation_state(state, str(reason))
            elif msg_type == "session.reset":
                self._session_id = None
                self._processing = False
                self._stream_response_active = False
                self._set_conversation_state(_ClientConversationState.LISTENING, "session_reset")
            elif msg_type == "user_text":
                self._session_id = data.get("session_id") or self._session_id
                user_text = data.get("text") or ""
                if user_text:
                    logger.info("User said: %s", user_text)
                actions = data.get("actions") or []
                if isinstance(actions, list):
                    valid_actions = self._decode_remote_actions(json.dumps(actions, ensure_ascii=True))
                    immediate_actions, deferred_actions, after_speech_actions = self._split_remote_actions(valid_actions)
                    deferred_action_events = []
                    after_speech_action_events = []
                    deferred_actions_started = False
                    if immediate_actions:
                        asyncio.create_task(
                            self._execute_remote_action_items(immediate_actions, "immediate"),
                            name="remote-reachy-actions-immediate",
                        )
            elif msg_type == "response.sentence":
                text = data.get("text") or ""
                if text:
                    logger.info("Assistant sentence: %s", text)
            elif msg_type == "action.call":
                logger.info(
                    "Received remote action call: %s %s timing=%s id=%s",
                    data.get("name"),
                    data.get("arguments", {}),
                    data.get("timing", "speech_start"),
                    data.get("action_call_id"),
                )
                action = {
                    "name": data.get("name"),
                    "arguments": data.get("arguments", {}),
                    "timing": data.get("timing", "speech_start"),
                }
                valid_actions = self._decode_remote_actions(json.dumps([action], ensure_ascii=True))
                immediate_actions, new_deferred_actions, new_after_speech_actions = self._split_remote_actions(
                    valid_actions
                )
                if immediate_actions:
                    for action_event in immediate_actions:
                        action_event["action_call_id"] = data.get("action_call_id")
                    for action_event in immediate_actions:
                        asyncio.create_task(
                            self._execute_remote_action_event(action_event, "immediate"),
                            name="remote-reachy-action-event-immediate",
                        )
                for action_event in new_deferred_actions:
                    action_event["action_call_id"] = data.get("action_call_id")
                    asyncio.create_task(
                        self._execute_remote_action_event(action_event, "speech-start"),
                        name="remote-reachy-action-event-speech-start",
                    )
                for action_event in new_after_speech_actions:
                    action_event["action_call_id"] = data.get("action_call_id")
                after_speech_action_events.extend(new_after_speech_actions)
            elif msg_type in {"audio_start", "response.audio.start"}:
                self._stream_response_active = True
                self._processing = True
                self._set_conversation_state(_ClientConversationState.SPEAKING, msg_type)
                self._stream_output_sample_rate = int(data.get("sample_rate") or self._stream_output_sample_rate)
                if (deferred_actions or deferred_action_events) and not deferred_actions_started:
                    deferred_actions_started = True
                    logger.info("Starting deferred remote actions at audio_start.")
                    if deferred_actions:
                        asyncio.create_task(
                            self._execute_remote_action_items(deferred_actions, "speech-synced"),
                            name="remote-reachy-actions-speech-synced",
                        )
                    for action_event in deferred_action_events:
                        asyncio.create_task(
                            self._execute_remote_action_event(action_event, "speech-synced"),
                            name="remote-reachy-action-event-speech-synced",
                        )
                    deferred_action_events = []
            elif msg_type == "response.cancelled":
                response_id = data.get("response_id")
                if isinstance(response_id, str):
                    self._cancelled_response_ids.add(response_id)
                self.clear_output_queue()
                self._stream_response_active = False
                self._processing = False
                self._playback_until = 0.0
                self._set_conversation_state(_ClientConversationState.LISTENING, "response_cancelled")
            elif msg_type in {"audio_end", "response.done"}:
                self._session_id = data.get("session_id") or self._session_id
                response_id = data.get("response_id")
                response_was_cancelled = (
                    isinstance(response_id, str)
                    and response_id in self._cancelled_response_ids
                )
                if response_was_cancelled:
                    self._cancelled_response_ids.discard(response_id)
                self._stream_response_active = False
                self._processing = False
                self._set_conversation_state(_ClientConversationState.LISTENING, msg_type)
                if not response_was_cancelled:
                    self._ignore_until = time.monotonic() + 0.2
                if (after_speech_actions or after_speech_action_events) and not response_was_cancelled:
                    logger.info("Starting after-speech remote actions.")
                    actions_to_run = after_speech_actions
                    after_speech_actions = []
                    if actions_to_run:
                        asyncio.create_task(
                            self._execute_remote_action_items(actions_to_run, "after-speech"),
                            name="remote-reachy-actions-after-speech",
                        )
                    action_events_to_run = after_speech_action_events
                    after_speech_action_events = []
                    for action_event in action_events_to_run:
                        asyncio.create_task(
                            self._execute_remote_action_event(action_event, "after-speech"),
                            name="remote-reachy-action-event-after-speech",
                        )
            elif msg_type == "ignored":
                text = data.get("text")
                logger.info("Remote stream ignored transcript: %r", text)
                self._processing = False
                self._set_conversation_state(_ClientConversationState.LISTENING, "ignored")
            elif msg_type == "error":
                self._stream_response_active = False
                self._processing = False
                self._set_conversation_state(_ClientConversationState.LISTENING, "error")
                logger.error("Remote stream error: %s", data.get("message"))

    # ── Remote call ───────────────────────────────────────────────────────────

    async def _call_remote(self, audio: NDArray[np.int16]) -> None:
        """POST audio to the Mac server and stream back synthesised PCM."""
        if self._processing:
            return
        self._processing = True

        url = (config.CONVERSATION_SERVICE_URL or "").strip()
        if not url:
            logger.error("CONVERSATION_SERVICE_URL is not configured.")
            self._processing = False
            return

        headers: dict[str, str] = {
            "Content-Type": "audio/pcm; rate=16000",
            "X-Voice": quote(self._voice),
        }
        if self._session_id:
            headers["X-Session-ID"] = self._session_id

        client = self._http_client or httpx.AsyncClient(timeout=_HTTP_TIMEOUT)
        pcm_bytes = audio.tobytes()
        stream_finished = False

        try:
            logger.info("Remote conversation: POST %s (%d bytes)", url, len(pcm_bytes))
            async with asyncio.timeout(_REMOTE_CALL_TIMEOUT):
                async with client.stream("POST", url, content=pcm_bytes, headers=headers) as resp:
                    logger.info("Remote conversation: response status=%d", resp.status_code)
                    if resp.status_code != 200:
                        body = await resp.aread()
                        logger.error("Server error %d: %s", resp.status_code, body[:200])
                        return

                    self._session_id = resp.headers.get("X-Session-ID", self._session_id)
                    output_sample_rate = int(resp.headers.get("X-Sample-Rate", _OUTPUT_SAMPLE_RATE))
                    actions_header = resp.headers.get("X-Reachy-Actions", "")
                    deferred_actions: list[dict[str, object]] = []
                    after_speech_actions: list[dict[str, object]] = []
                    deferred_actions_started = False
                    if actions_header:
                        logger.info("Received remote actions header: %s", unquote(actions_header))
                        actions = self._decode_remote_actions(actions_header)
                        immediate_actions, deferred_actions, after_speech_actions = self._split_remote_actions(actions)
                        if immediate_actions:
                            asyncio.create_task(
                                self._execute_remote_action_items(immediate_actions, "immediate"),
                                name="remote-reachy-actions-immediate",
                            )
                    user_text = unquote(resp.headers.get("X-User-Text", ""))
                    if user_text:
                        logger.info("User said: %s", user_text)

                    buf = b""
                    async for chunk in resp.aiter_bytes(_STREAM_CHUNK_SIZE):
                        buf += chunk
                        # Emit complete int16 samples (2 bytes each)
                        usable = len(buf) - (len(buf) % 2)
                        if usable > 0:
                            pcm_out = np.frombuffer(buf[:usable], dtype=np.int16)
                            if deferred_actions and not deferred_actions_started:
                                deferred_actions_started = True
                                logger.info("Starting deferred remote actions with first audio chunk.")
                                asyncio.create_task(
                                    self._execute_remote_action_items(deferred_actions, "speech-synced"),
                                    name="remote-reachy-actions-speech-synced",
                                )
                            await self.output_queue.put((output_sample_rate, pcm_out))
                            buf = buf[usable:]
                    stream_finished = True
                    logger.info("Remote conversation: audio stream finished")
                    self._ignore_until = time.monotonic() + 0.2

                    # Flush any remaining bytes
                    if len(buf) >= 2:
                        usable = len(buf) - (len(buf) % 2)
                        pcm_out = np.frombuffer(buf[:usable], dtype=np.int16)
                        if deferred_actions and not deferred_actions_started:
                            deferred_actions_started = True
                            logger.info("Starting deferred remote actions with final audio chunk.")
                            asyncio.create_task(
                                self._execute_remote_action_items(deferred_actions, "speech-synced"),
                                name="remote-reachy-actions-speech-synced",
                            )
                        await self.output_queue.put((output_sample_rate, pcm_out))
                    elif deferred_actions and not deferred_actions_started:
                        logger.info("Starting deferred remote actions after empty audio response.")
                        asyncio.create_task(
                            self._execute_remote_action_items(deferred_actions, "speech-synced"),
                            name="remote-reachy-actions-speech-synced",
                        )
                    if after_speech_actions:
                        logger.info("Starting after-speech remote actions.")
                        asyncio.create_task(
                            self._execute_remote_action_items(after_speech_actions, "after-speech"),
                            name="remote-reachy-actions-after-speech",
                        )

        except asyncio.CancelledError:
            logger.info("Remote conversation: cancelled")
            raise
        except TimeoutError:
            logger.error("Remote conversation timed out after %.1fs", _REMOTE_CALL_TIMEOUT)
        except Exception as exc:
            logger.error("Remote conversation error: %s", exc)
        finally:
            if not stream_finished:
                self.clear_output_queue()
                self._playback_until = 0.0
                self._ignore_until = time.monotonic() + 0.2
            self._processing = False

    # ── Gradio personality controls ────────────────────────────────────────────

    def set_voice(self, voice: str) -> None:
        """Update the CosyVoice speaker sent to the remote server."""
        available = get_available_voices_for_backend(REMOTE_BACKEND)
        if voice in available:
            self._voice = voice
            logger.info("Voice → %s", voice)

    async def get_available_voices(self) -> list[str]:
        """Return CosyVoice voices available through the remote server."""
        return get_available_voices_for_backend(REMOTE_BACKEND)

    def get_current_voice(self) -> str:
        """Return the active remote TTS voice."""
        return self._voice

    async def change_voice(self, voice: str) -> str:
        """Update the remote TTS voice and return the effective voice."""
        available = get_available_voices_for_backend(REMOTE_BACKEND)
        if voice in available:
            self._voice = voice
            logger.info("Voice changed to %r.", voice)
        else:
            logger.warning("Voice %r is not available for remote backend; keeping %r.", voice, self._voice)
        return self._voice

    def set_profile(self, profile: str | None) -> None:
        """Update the active personality profile."""
        set_custom_profile(profile)

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a personality profile and reset remote conversation state."""
        set_custom_profile(profile)
        self.reset_conversation()
        logger.info("Applied personality profile %r; remote session reset.", profile)
        return profile or "default"

    def reset_conversation(self) -> None:
        """Clear conversation session so next turn starts fresh."""
        self._session_id = None
        logger.info("Conversation session reset.")
