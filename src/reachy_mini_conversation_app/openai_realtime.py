import json
import time
import uuid
import wave
import base64
import random
import asyncio
import logging
from typing import Any, Final, Tuple, Literal, Optional
from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit, parse_qsl, urlunsplit

import cv2
import httpx
import numpy as np
import gradio as gr
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, AsyncStreamHandler, wait_for_item, audio_to_int16
from pydantic import Field, BaseModel
from numpy.typing import NDArray
from scipy.signal import resample
from openai.types.realtime import (
    AudioTranscriptionParam,
    RealtimeAudioConfigParam,
    RealtimeAudioConfigInputParam,
    RealtimeAudioConfigOutputParam,
    RealtimeResponseCreateParamsParam,
    RealtimeSessionCreateRequestParam,
)
from websockets.exceptions import ConnectionClosedError
from openai.resources.realtime.realtime import AsyncRealtimeConnection
from openai.types.realtime.realtime_audio_formats_param import AudioPCM
from openai.types.realtime.realtime_audio_input_turn_detection_param import ServerVad

from reachy_mini_conversation_app.config import (
    S2S_BACKEND,
    OPENAI_BACKEND,
    AVAILABLE_VOICES,
    config,
    get_s2s_session_url,
    get_s2s_direct_ws_url,
    get_default_voice_for_backend,
    get_available_voices_for_backend,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.tools.core_tools import ToolDependencies, get_tool_specs
from reachy_mini_conversation_app.tools.background_tool_manager import (
    ToolCallRoutine,
    ToolNotification,
    BackgroundToolManager,
)


logger = logging.getLogger(__name__)

OPENAI_REALTIME_SAMPLE_RATE: Final[int] = 24000
S2S_REALTIME_SAMPLE_RATE: Final[int] = 16000

# Cost tracking from usage data (pricing as of Feb 2026 https://openai.com/api/pricing/)
AUDIO_INPUT_COST_PER_1M = 32.0
AUDIO_OUTPUT_COST_PER_1M = 64.0
TEXT_INPUT_COST_PER_1M = 4.0
TEXT_OUTPUT_COST_PER_1M = 16.0
IMAGE_INPUT_COST_PER_1M = 5.0

_RESPONSE_DONE_TIMEOUT: Final[float] = 30.0


class InputTranscriptChunksByItem(BaseModel):
    """Current item_id and its accumulated deltas. Only one item at a time."""

    item_id: str | None = None
    deltas: list[str] = Field(default_factory=list)


def _compute_response_cost(usage: Any) -> float:
    """Compute dollar cost from a response usage object."""
    inp = getattr(usage, "input_token_details", None)
    out = getattr(usage, "output_token_details", None)
    cost = 0.0
    if inp:
        cost += (getattr(inp, "audio_tokens", 0) or 0) * AUDIO_INPUT_COST_PER_1M / 1e6
        cost += (getattr(inp, "text_tokens", 0) or 0) * TEXT_INPUT_COST_PER_1M / 1e6
        cost += (getattr(inp, "image_tokens", 0) or 0) * IMAGE_INPUT_COST_PER_1M / 1e6
    if out:
        cost += (getattr(out, "audio_tokens", 0) or 0) * AUDIO_OUTPUT_COST_PER_1M / 1e6
        cost += (getattr(out, "text_tokens", 0) or 0) * TEXT_OUTPUT_COST_PER_1M / 1e6
    return cost


def _normalize_startup_voice(voice: str | None) -> str | None:
    """Return a valid persisted OpenAI startup voice or None."""
    if voice in AVAILABLE_VOICES:
        return voice
    if voice:
        logger.warning("Ignoring persisted OpenAI startup voice %r; expected one of %s", voice, AVAILABLE_VOICES)
    return None


def _build_openai_compatible_client_from_realtime_url(
    realtime_url: str,
    api_key: str | None,
) -> tuple[AsyncOpenAI, dict[str, str]]:
    """Build an OpenAI-compatible realtime client from a direct websocket/base URL."""
    parsed = urlsplit(realtime_url)
    scheme = parsed.scheme.lower()
    if scheme not in {"ws", "wss", "http", "https"}:
        raise ValueError(
            "Expected speech-to-speech realtime URL to start with ws://, wss://, http://, or https://, "
            f"got: {realtime_url}"
        )

    path = parsed.path.rstrip("/")
    if path.endswith("/realtime"):
        base_path = path[: -len("/realtime")]
    else:
        base_path = path

    connect_query = {
        key: value
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if key != "model"
    }
    http_scheme = "https" if scheme in {"wss", "https"} else "http"
    websocket_scheme = "wss" if scheme in {"wss", "https"} else "ws"
    base_url = urlunsplit((http_scheme, parsed.netloc, base_path, "", ""))
    websocket_base_url = urlunsplit((websocket_scheme, parsed.netloc, base_path, "", ""))
    client = AsyncOpenAI(
        api_key=api_key or "DUMMY",
        base_url=base_url,
        websocket_base_url=websocket_base_url,
    )
    return client, connect_query


class OpenaiRealtimeHandler(AsyncStreamHandler):
    """An OpenAI realtime handler for fastrtc Stream."""

    def __init__(
        self,
        deps: ToolDependencies,
        gradio_mode: bool = False,
        instance_path: Optional[str] = None,
        startup_voice: Optional[str] = None,
    ):
        """Initialize the handler."""
        sample_rate = self._get_realtime_sample_rate()
        super().__init__(
            expected_layout="mono",
            output_sample_rate=sample_rate,
            input_sample_rate=sample_rate,
        )

        self.deps = deps

        self.output_sample_rate = sample_rate
        self.input_sample_rate = sample_rate

        self.client: AsyncOpenAI
        self.connection: AsyncRealtimeConnection | None = None
        self.output_queue: "asyncio.Queue[Tuple[int, NDArray[np.int16]] | AdditionalOutputs]" = asyncio.Queue()

        self.last_activity_time = asyncio.get_event_loop().time()
        self.start_time = asyncio.get_event_loop().time()
        self.is_idle_tool_call = False
        self.gradio_mode = gradio_mode
        self.instance_path = instance_path
        self._voice_override: str | None = _normalize_startup_voice(startup_voice)
        # Track how the API key was provided (env vs textbox) and its value
        self._key_source: Literal["env", "textbox"] = "env"
        self._provided_api_key: str | None = None
        self._realtime_connect_query: dict[str, str] = {}

        # Debouncing for partial transcripts
        self.partial_transcript_task: asyncio.Task[None] | None = None
        self.partial_debounce_delay = 0.5  # seconds
        self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

        # Internal lifecycle flags
        self._connected_event: asyncio.Event = asyncio.Event()

        # Background tool manager
        self.tool_manager = BackgroundToolManager()

        # Cost tracking
        self.cumulative_cost: float = 0.0

        # Response-in-progress guard: the Realtime API only allows one active
        # response per conversation at a time.  A dedicated worker task
        # (_response_sender_loop) dequeues and sends one request at a time
        self._pending_responses: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._response_done_event: asyncio.Event = asyncio.Event()
        self._response_done_event.set()
        self._response_started_or_rejected_event: asyncio.Event = asyncio.Event()
        self._last_response_rejected: bool = False
        self._assistant_audio_dump_enabled = logging.getLogger().isEnabledFor(logging.DEBUG)
        self._assistant_audio_dump_dir = self._resolve_assistant_audio_dump_dir(instance_path)
        self._assistant_audio_dump_stream_index = 0
        self._assistant_audio_dump_current_index: int | None = None
        self._assistant_audio_dump_chunks: list[bytes] = []
        self._turn_user_done_at: float | None = None
        self._turn_response_created_at: float | None = None
        self._turn_first_audio_at: float | None = None

    @staticmethod
    def _sanitize_tool_result_for_model(tool_name: str, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Remove bulky transport-only fields before echoing tool output back to the model."""
        if tool_name == "camera" and "b64_im" in tool_result:
            sanitized = dict(tool_result)
            sanitized.pop("b64_im", None)
            sanitized["image_attached"] = True
            return sanitized
        return tool_result

    @staticmethod
    def _get_realtime_sample_rate() -> int:
        """Use backend-native audio rates when possible."""
        if config.BACKEND_PROVIDER == S2S_BACKEND:
            return S2S_REALTIME_SAMPLE_RATE
        return OPENAI_REALTIME_SAMPLE_RATE

    def _get_openai_session_audio_rates(self) -> tuple[Literal[24000] | None, Literal[24000] | None]:
        """Return ``(input_rate, output_rate)`` for the openai realtime session config.

        The OpenAI realtime API mandates 24 kHz or None.  The S2S backend
        also accepts 24 kHz but performs best at 16 kHz; passing ``None``
        lets it default to that optimal rate while still satisfying the
        OpenAI SDK contract (which requires 24 kHz or None). The S2S backend
        will default to 16 kHz if None is passed.

        OpenAI SDK Server-side contract:
        https://github.com/openai/openai-python/blob/58184ad545ee2abd98e171ee09766f259d7f38cd/src/openai/types/realtime/realtime_audio_formats.py#L12

        """
        input_rate: int | None = self.input_sample_rate
        output_rate: int | None = self.output_sample_rate
        if config.BACKEND_PROVIDER == S2S_BACKEND:
            if self.input_sample_rate == 16000:
                input_rate = None
            if self.output_sample_rate == 16000:
                output_rate = None
        s2s_hint = (
            " To use the S2S backend optimal default rate (16 kHz), use an internal application sample rate of 16 kHz"
            " and pass None to the OpenAI realtime session config. None will be interpreted by the S2S backend as defaulting to 16 kHz."
            if config.BACKEND_PROVIDER == S2S_BACKEND
            else ""
        )
        assert input_rate in (24000, None) and output_rate in (24000, None), (
            f"OpenAI realtime session requires 24 kHz or None, "
            f"got input_rate={input_rate!r}, output_rate={output_rate!r}.{s2s_hint}"
        )
        return input_rate, output_rate  # type: ignore[return-value]

    def _mark_activity(self, reason: str) -> None:
        """Record non-idle conversation activity for the idle timer."""
        self.last_activity_time = asyncio.get_event_loop().time()
        logger.debug("last activity time updated to %s (%s)", self.last_activity_time, reason)

    @staticmethod
    def _resolve_assistant_audio_dump_dir(instance_path: Optional[str]) -> Path:
        """Store debug assistant PCM dumps in a predictable local directory."""
        base_dir = Path(instance_path) if instance_path is not None else Path.cwd()
        return base_dir / "debug_audio" / "assistant_audio"

    def _append_assistant_audio_dump_chunk(self, pcm_chunk: bytes) -> None:
        """Accumulate raw assistant PCM chunks for debug WAV export."""
        if not self._assistant_audio_dump_enabled:
            return
        if self._assistant_audio_dump_current_index is None:
            self._assistant_audio_dump_stream_index += 1
            self._assistant_audio_dump_current_index = self._assistant_audio_dump_stream_index
        self._assistant_audio_dump_chunks.append(pcm_chunk)

    def _flush_assistant_audio_dump(self, reason: str) -> None:
        """Write the accumulated assistant PCM stream to a mono 16-bit WAV file."""
        if not self._assistant_audio_dump_enabled:
            return

        stream_index = self._assistant_audio_dump_current_index
        chunks = self._assistant_audio_dump_chunks
        self._assistant_audio_dump_current_index = None
        self._assistant_audio_dump_chunks = []

        if stream_index is None or not chunks:
            return

        self._assistant_audio_dump_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%S_%fZ")
        provider = config.BACKEND_PROVIDER.replace("-", "_")
        file_path = (
            self._assistant_audio_dump_dir
            / f"{timestamp}_{provider}_stream{stream_index:03d}_{self.output_sample_rate}hz_{reason}.wav"
        )
        pcm_bytes = b"".join(chunks)

        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.output_sample_rate)
            wav_file.writeframes(pcm_bytes)

        logger.debug(
            "Saved assistant audio dump: path=%s samples=%d duration_ms=%.1f reason=%s",
            file_path,
            len(pcm_bytes) // 2,
            1000.0 * (len(pcm_bytes) // 2) / self.output_sample_rate,
            reason,
        )

    def copy(self) -> "OpenaiRealtimeHandler":
        """Create a copy of the handler."""
        return OpenaiRealtimeHandler(
            self.deps,
            self.gradio_mode,
            self.instance_path,
            startup_voice=self._voice_override,
        )

    async def change_voice(self, voice: str) -> str:
        """Change only the voice and restart the session."""
        self._voice_override = voice
        if getattr(self, "client", None) is not None:
            try:
                await self._restart_session()
                return f"Voice changed to {voice}."
            except Exception as e:
                logger.warning("Failed to restart session for voice change: %s", e)
                return "Voice change failed. Will take effect on next connection."
        return "Voice changed. Will take effect on next connection."

    def get_current_voice(self) -> str:
        """Return the voice currently selected for this handler."""
        return self._voice_override or get_session_voice()

    async def apply_personality(self, profile: str | None) -> str:
        """Apply a new personality (profile) at runtime if possible.

        - Updates the global config's selected profile for subsequent calls.
        - If a realtime connection is active, sends a session.update with the
          freshly resolved instructions so the change takes effect immediately.

        Returns a short status message for UI feedback.
        """
        try:
            # Update the in-process config value and env
            from reachy_mini_conversation_app.config import config as _config
            from reachy_mini_conversation_app.config import set_custom_profile

            set_custom_profile(profile)
            logger.info(
                "Set custom profile to %r (config=%r)", profile, getattr(_config, "REACHY_MINI_CUSTOM_PROFILE", None)
            )

            try:
                instructions = get_session_instructions()
                voice = self.get_current_voice()
            except BaseException as e:  # catch SystemExit from prompt loader without crashing
                logger.error("Failed to resolve personality content: %s", e)
                return f"Failed to apply personality: {e}"

            # Attempt a live update first, then force a full restart to ensure it sticks
            if self.connection is not None:
                try:
                    await self.connection.session.update(
                        session=RealtimeSessionCreateRequestParam(
                            type="realtime",
                            instructions=instructions,
                            audio=RealtimeAudioConfigParam(
                                output=RealtimeAudioConfigOutputParam(
                                    voice=voice,
                                ),
                            ),
                        ),
                    )
                    logger.info("Applied personality via live update: %s", profile or "built-in default")
                except Exception as e:
                    logger.warning("Live update failed; will restart session: %s", e)

                # Force a real restart to guarantee the new instructions/voice
                try:
                    await self._restart_session()
                    return "Applied personality and restarted realtime session."
                except Exception as e:
                    logger.warning("Failed to restart session after apply: %s", e)
                    return "Applied personality. Will take effect on next connection."
            else:
                logger.info(
                    "Applied personality recorded: %s (no live connection; will apply on next session)",
                    profile or "built-in default",
                )
                return "Applied personality. Will take effect on next connection."
        except Exception as e:
            logger.error("Error applying personality '%s': %s", profile, e)
            return f"Failed to apply personality: {e}"

    async def _emit_debounced_partial(self, transcript: str, item_id: str, sequence_counter: int) -> None:
        """Emit partial transcript after debounce delay."""
        try:
            await asyncio.sleep(self.partial_debounce_delay)

            input_transcript = self.input_transcript_chunks_by_item
            if input_transcript.item_id == item_id and len(input_transcript.deltas) - 1 == sequence_counter:
                await self.output_queue.put(AdditionalOutputs({"role": "user_partial", "content": transcript}))
                logger.debug(f"Debounced partial emitted: {transcript}")
        except asyncio.CancelledError:
            logger.debug("Debounced partial cancelled")
            raise

    async def start_up(self) -> None:
        """Start the handler with minimal retries on unexpected websocket closure."""
        openai_api_key = config.OPENAI_API_KEY
        if self.gradio_mode and config.BACKEND_PROVIDER == OPENAI_BACKEND and not openai_api_key:
            # api key was not found in .env or in the environment variables
            await self.wait_for_args()  # type: ignore[no-untyped-call]
            args = list(self.latest_args)
            textbox_api_key = args[3] if len(args[3]) > 0 else None
            if textbox_api_key is not None:
                openai_api_key = textbox_api_key
                self._key_source = "textbox"
                self._provided_api_key = textbox_api_key
            else:
                openai_api_key = config.OPENAI_API_KEY
        self.client = await self._build_realtime_client(api_key=openai_api_key)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            try:
                await self._run_realtime_session()
                # Normal exit from the session, stop retrying
                return
            except ConnectionClosedError as e:
                # Abrupt close (e.g., "no close frame received or sent") → retry
                logger.warning("Realtime websocket closed unexpectedly (attempt %d/%d): %s", attempt, max_attempts, e)
                if attempt < max_attempts:
                    if config.BACKEND_PROVIDER == S2S_BACKEND:
                        self.client = await self._build_realtime_client()
                    # exponential backoff with jitter
                    base_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s, 8s, etc.
                    jitter = random.uniform(0, 0.5)
                    delay = base_delay + jitter
                    logger.info("Retrying in %.1f seconds...", delay)
                    await asyncio.sleep(delay)
                    continue
                raise
            finally:
                # never keep a stale reference
                self.connection = None
                try:
                    self._connected_event.clear()
                except Exception:
                    pass

    async def _restart_session(self) -> None:
        """Force-close the current session and start a fresh one in background.

        Does not block the caller while the new session is establishing.
        """
        try:
            if self.connection is not None:
                try:
                    await self.connection.close()
                except Exception:
                    pass
                finally:
                    self.connection = None

            # Ensure we have a client (start_up must have run once)
            if getattr(self, "client", None) is None:
                logger.warning("Cannot restart: OpenAI client not initialized yet.")
                return

            # Fire-and-forget new session and wait briefly for connection
            try:
                self._connected_event.clear()
            except Exception:
                pass
            if config.BACKEND_PROVIDER == S2S_BACKEND:
                self.client = await self._build_realtime_client()
            asyncio.create_task(self._run_realtime_session(), name="openai-realtime-restart")
            try:
                await asyncio.wait_for(self._connected_event.wait(), timeout=5.0)
                logger.info("Realtime session restarted and connected.")
            except asyncio.TimeoutError:
                logger.warning("Realtime session restart timed out; continuing in background.")
        except Exception as e:
            logger.warning("_restart_session failed: %s", e)

    async def _safe_response_create(self, **kwargs: Any) -> None:
        """Enqueue a response.create() kwargs for the sender worker _response_sender_loop().

        This method never blocks the caller.
        """
        await self._pending_responses.put(kwargs)

    async def _response_sender_loop(self) -> None:
        """Dedicated worker that sends ``response.create()`` calls serially.

        This logic was designed to comply with the response.create() docstring specification for event ordering:
        https://github.com/openai/openai-python/blob/3e0c05b84a2056870abf3bd6a5e7849020209cc3/src/openai/resources/realtime/realtime.py#L649C1-L651C30

        For each queued request the worker:
        1. Waits until no response is active (_response_done_event).
        2. Sends response.create().
        3. Waits for the response cycle to complete (response.done).
        4. If the server rejected with active_response, retries from step 1.
        """
        while self.connection:
            try:
                kwargs = await self._pending_responses.get()
            except asyncio.CancelledError:
                return

            sent = False
            max_retries = 5
            attempts = 0
            while not sent and self.connection and attempts < max_retries:
                try:
                    await asyncio.wait_for(self._response_done_event.wait(), timeout=_RESPONSE_DONE_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for previous response to finish; forcing ahead")
                    self._response_done_event.set()

                if not self.connection:
                    break

                self._last_response_rejected = False
                try:
                    await self.connection.response.create(**kwargs)
                except Exception as e:
                    logger.debug("_response_sender_loop: send failed: %s", e)
                    self._response_done_event.set()
                    break

                # Check if we were rejected
                if self._last_response_rejected:
                    attempts += 1
                    if attempts >= max_retries:
                        logger.debug("response.create rejected %d times; giving up", attempts)
                        break
                    logger.debug("response.create was rejected; retrying (%d/%d)", attempts, max_retries)
                    continue

                try:
                    await asyncio.wait_for(self._response_done_event.wait(), timeout=_RESPONSE_DONE_TIMEOUT)
                except asyncio.TimeoutError:
                    logger.debug("Timed out waiting for response.done; assuming response completed")
                    self._response_done_event.set()
                    break

                sent = True

    async def _handle_tool_result(self, bg_tool: ToolNotification) -> None:
        """Process the result of a tool call."""
        if bg_tool.error is not None:
            logger.error("Tool '%s' (id=%s) failed with error: %s", bg_tool.tool_name, bg_tool.id, bg_tool.error)
            tool_result = {"error": bg_tool.error}
            tool_result_for_model = tool_result
        elif bg_tool.result is not None:
            tool_result = bg_tool.result
            tool_result_for_model = (
                self._sanitize_tool_result_for_model(bg_tool.tool_name, tool_result)
                if isinstance(tool_result, dict)
                else tool_result
            )
            logger.info(
                "Tool '%s' (id=%s) executed successfully.",
                bg_tool.tool_name,
                bg_tool.id,
            )
            logger.debug("Tool '%s' model-visible result: %s", bg_tool.tool_name, tool_result_for_model)
        else:
            logger.warning("Tool '%s' (id=%s) returned no result and no error", bg_tool.tool_name, bg_tool.id)
            tool_result = {"error": "No result returned from tool execution"}
            tool_result_for_model = tool_result

        # Connection may have closed while tool was running
        if not self.connection:
            logger.warning(
                "Connection closed during tool '%s' (id=%s) execution; cannot send result back",
                bg_tool.tool_name,
                bg_tool.id,
            )
            return

        try:
            self._mark_activity("tool_result_ready")
            # TODO: refactor this since it's repeated here, in the camera branch below, and in send_idle_signal
            if isinstance(bg_tool.id, str):
                await self.connection.conversation.item.create(
                    item={
                        "type": "function_call_output",
                        "call_id": bg_tool.id,
                        "output": json.dumps(tool_result_for_model),
                    },
                )

            await self.output_queue.put(
                AdditionalOutputs(
                    {
                        "role": "assistant",
                        "content": json.dumps(tool_result_for_model),
                        # Gradio UI metadata.status accept only "pending" and "done". Do not accept bg.tool.status values.
                        "metadata": {
                            "title": f"🛠️ Used tool {bg_tool.tool_name}",
                            "status": "done",
                        },
                    },
                ),
            )

            if bg_tool.tool_name == "camera" and "b64_im" in tool_result:
                # use raw base64, don't json.dumps (which adds quotes)
                b64_im = tool_result["b64_im"]
                if not isinstance(b64_im, str):
                    logger.warning("Unexpected type for b64_im: %s", type(b64_im))
                    b64_im = str(b64_im)
                image_width = tool_result.get("image_width")
                image_height = tool_result.get("image_height")
                jpeg_bytes_value = tool_result.get("jpeg_bytes")
                jpeg_bytes = jpeg_bytes_value if isinstance(jpeg_bytes_value, int) else (len(b64_im) * 3) // 4
                await self.connection.conversation.item.create(
                    item={
                        "type": "message",
                        "role": "user",
                        "content": [
                            {
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{b64_im}",
                            },
                        ],
                    },
                )
                if isinstance(image_width, int) and isinstance(image_height, int):
                    logger.info(
                        "Added camera image to conversation frame=%sx%s jpeg_bytes=%s",
                        image_width,
                        image_height,
                        jpeg_bytes,
                    )
                else:
                    logger.info(
                        "Added camera image to conversation jpeg_bytes=%s",
                        jpeg_bytes,
                    )

                if self.deps.camera_worker is not None:
                    np_img = self.deps.camera_worker.get_latest_frame()
                    if np_img is not None:
                        # Camera frames are BGR from OpenCV; convert so Gradio displays correct colors.
                        rgb_frame = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
                    else:
                        rgb_frame = None
                    img = gr.Image(value=rgb_frame)

                    await self.output_queue.put(
                        AdditionalOutputs(
                            {
                                "role": "assistant",
                                "content": img,
                            },
                        ),
                    )

            # If this tool call was triggered by an idle signal, don't make the robot speak.
            # For other tool calls, let the robot reply out loud.
            if not bg_tool.is_idle_tool_call:
                await self._safe_response_create(
                    response=RealtimeResponseCreateParamsParam(
                        instructions="Use the tool result just returned and answer concisely in speech.",
                    ),
                )

        except ConnectionClosedError:
            logger.warning("Connection closed while sending tool result")
            self.connection = None
            self._response_done_event.set()

    async def _run_realtime_session(self) -> None:
        """Establish and manage a single realtime session."""
        input_rate, output_rate = self._get_openai_session_audio_rates()
        async with self.client.realtime.connect(
            model=config.MODEL_NAME,
            extra_query=self._realtime_connect_query,
        ) as conn:
            try:
                session_config = RealtimeSessionCreateRequestParam(
                    type="realtime",
                    instructions=get_session_instructions(),
                    audio=RealtimeAudioConfigParam(
                        input=RealtimeAudioConfigInputParam(
                            format=AudioPCM(type="audio/pcm", rate=input_rate),  # type: ignore[typeddict-item]
                            transcription=AudioTranscriptionParam(model="gpt-4o-transcribe", language="en"),
                            turn_detection=ServerVad(type="server_vad", interrupt_response=True),
                        ),
                        output=RealtimeAudioConfigOutputParam(
                            format=AudioPCM(type="audio/pcm", rate=output_rate),  # type: ignore[typeddict-item]
                            voice=self.get_current_voice(),
                        ),
                    ),
                    tools=get_tool_specs(),  # type: ignore[typeddict-item]
                    tool_choice="auto",
                )
                await conn.session.update(session=session_config)
                logger.info(
                    "Realtime session initialized with profile=%r voice=%r",
                    getattr(config, "REACHY_MINI_CUSTOM_PROFILE", None),
                    self.get_current_voice(),
                )
                # If we reached here, the session update succeeded which implies the API key worked.
                # Persist the key to a newly created .env (copied from .env.example) if needed.
                self._persist_api_key_if_needed()
            except Exception:
                logger.exception("Realtime session.update failed; aborting startup")
                return

            logger.info("Realtime session updated successfully")

            # Reset the partial-transcript accumulator for each new session
            self.input_transcript_chunks_by_item = InputTranscriptChunksByItem()

            # Manage event received from the openai server
            self.connection = conn
            try:
                self._connected_event.set()
            except Exception:
                pass

            response_sender_task: asyncio.Task[None] | None = None
            try:
                # Start the background tool manager
                self.tool_manager.start_up(tool_callbacks=[self._handle_tool_result])

                # Start the response sender worker
                response_sender_task = asyncio.create_task(self._response_sender_loop(), name="response-sender")

                async for event in self.connection:
                    logger.debug(f"OpenAI event: {event.type}")
                    if event.type == "input_audio_buffer.speech_started":
                        self._flush_assistant_audio_dump("interrupted")
                        self._mark_activity("user_speech_started")
                        self._turn_user_done_at = None
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None
                        if hasattr(self, "_clear_queue") and callable(self._clear_queue):
                            self._clear_queue()
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.reset()
                        self.deps.movement_manager.set_listening(True)
                        logger.debug("User speech started")

                    if event.type == "input_audio_buffer.speech_stopped":
                        self._mark_activity("user_speech_stopped")
                        self.deps.movement_manager.set_listening(False)
                        logger.debug("User speech stopped - server will auto-commit with VAD")

                    if event.type == "response.output_audio.done":
                        if self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.request_reset_after_current_audio()
                        logger.debug("response completed")

                    if event.type == "response.created":
                        self._flush_assistant_audio_dump("orphaned")
                        self._mark_activity("response_created")
                        self._response_done_event.clear()
                        self._response_started_or_rejected_event.set()
                        if self._turn_user_done_at is not None and self._turn_response_created_at is None:
                            self._turn_response_created_at = time.perf_counter()
                            delta_ms = (self._turn_response_created_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: response.created %.0f ms after user transcript", delta_ms)
                        logger.debug("Response created (active)")

                    if event.type == "response.done":
                        # Doesn't mean the audio is done playing
                        self._response_done_event.set()
                        self.is_idle_tool_call = False
                        logger.debug("Response done")

                        response = getattr(event, "response", None)
                        usage = getattr(response, "usage", None) if response else None
                        if usage:
                            cost = _compute_response_cost(usage)
                            self.cumulative_cost += cost
                            logger.debug("Cost: $%.4f | Cumulative: $%.4f", cost, self.cumulative_cost)
                        else:
                            logger.warning("No usage data available for cost tracking")

                    if event.type == "conversation.item.input_audio_transcription.delta":
                        self._mark_activity("user_transcription_delta")
                        logger.debug(f"User partial transcript: {event.delta}")

                        item_id = event.item_id
                        delta = event.delta or ""

                        input_transcript = self.input_transcript_chunks_by_item
                        if input_transcript.item_id != item_id:
                            input_transcript.item_id = item_id
                            input_transcript.deltas = [delta]
                        else:
                            # speech-to-speech currently sends partial snapshots, not suffix-only deltas
                            input_transcript.deltas = [delta]

                        current_partial = "".join(input_transcript.deltas)
                        sequence_counter = len(input_transcript.deltas) - 1

                        # Cancel previous debounce task if it exists
                        if self.partial_transcript_task and not self.partial_transcript_task.done():
                            self.partial_transcript_task.cancel()
                            try:
                                await self.partial_transcript_task
                            except asyncio.CancelledError:
                                pass

                        # Start new debounce timer with the last delta
                        self.partial_transcript_task = asyncio.create_task(
                            self._emit_debounced_partial(current_partial, item_id, sequence_counter)
                        )

                    # Handle completed transcription (user finished speaking)
                    if event.type == "conversation.item.input_audio_transcription.completed":
                        self._mark_activity("user_transcription_completed")
                        logger.debug(f"User transcript: {event.transcript}")
                        self._turn_user_done_at = time.perf_counter()
                        self._turn_response_created_at = None
                        self._turn_first_audio_at = None

                        # Cancel any pending partial emission
                        if self.partial_transcript_task and not self.partial_transcript_task.done():
                            self.partial_transcript_task.cancel()
                            try:
                                await self.partial_transcript_task
                            except asyncio.CancelledError:
                                pass

                        await self.output_queue.put(AdditionalOutputs({"role": "user", "content": event.transcript}))

                    # Handle assistant transcription
                    if event.type == "response.output_audio_transcript.done":
                        self._mark_activity("assistant_transcript_done")
                        logger.debug(f"Assistant transcript: {event.transcript}")
                        await self.output_queue.put(
                            AdditionalOutputs({"role": "assistant", "content": event.transcript})
                        )

                    # Handle audio delta
                    if event.type == "response.output_audio.delta":
                        decoded_pcm_bytes = base64.b64decode(event.delta)
                        self._append_assistant_audio_dump_chunk(decoded_pcm_bytes)
                        decoded_pcm = np.frombuffer(decoded_pcm_bytes, dtype=np.int16).reshape(1, -1)
                        if self.gradio_mode and self.deps.head_wobbler is not None:
                            self.deps.head_wobbler.feed_pcm(decoded_pcm, self.output_sample_rate)
                        self._mark_activity("assistant_audio_delta")
                        if self._turn_user_done_at is not None and self._turn_first_audio_at is None:
                            self._turn_first_audio_at = time.perf_counter()
                            delta_ms = (self._turn_first_audio_at - self._turn_user_done_at) * 1000
                            logger.info("Turn latency: first audio delta %.0f ms after user transcript", delta_ms)
                        await self.output_queue.put(
                            (
                                self.output_sample_rate,
                                decoded_pcm,
                            ),
                        )
                    if event.type == "response.output_audio.done":
                        self._flush_assistant_audio_dump("complete")

                    # ---- tool-calling plumbing ----
                    if event.type == "response.function_call_arguments.done":
                        self._mark_activity("tool_call_received")
                        tool_name = getattr(event, "name", None)
                        args_json_str = getattr(event, "arguments", None)
                        call_id: str = str(getattr(event, "call_id", uuid.uuid4()))

                        logger.info(
                            "Tool call received — tool_name=%r, call_id=%s, is_idle=%s, args=%s",
                            tool_name,
                            call_id,
                            self.is_idle_tool_call,
                            args_json_str,
                        )

                        if not isinstance(tool_name, str) or not isinstance(args_json_str, str):
                            logger.error(
                                "Invalid tool call: tool_name=%s (type=%s), args=%s (type=%s), call_id=%s",
                                tool_name,
                                type(tool_name).__name__,
                                args_json_str,
                                type(args_json_str).__name__,
                                call_id,
                            )
                            continue

                        bg_tool = await self.tool_manager.start_tool(
                            call_id=call_id,
                            tool_call_routine=ToolCallRoutine(
                                tool_name=tool_name,
                                args_json_str=args_json_str,
                                deps=self.deps,
                            ),
                            is_idle_tool_call=self.is_idle_tool_call,
                        )

                        await self.output_queue.put(
                            AdditionalOutputs(
                                {
                                    "role": "assistant",
                                    "content": f"🛠️ Used tool {tool_name} with args {args_json_str}. The tool is now running. Tool ID: {bg_tool.tool_id}",
                                },
                            ),
                        )
                        logger.info(f"{tool_name} is_idle_tool_call: {self.is_idle_tool_call}")

                        logger.info(
                            "Started background tool: %s (id=%s, call_id=%s)", tool_name, bg_tool.tool_id, call_id
                        )

                    # server error
                    if event.type == "error":
                        err = getattr(event, "error", None)
                        msg = getattr(err, "message", str(err) if err else "unknown error")
                        code = getattr(err, "code", "") or getattr(err, "type", "")

                        if code == "conversation_already_has_active_response":
                            # response.create was rejected.  The sender worker
                            # is waiting on _response_done_event; when the active
                            # response finishes it will wake up and see this flag.
                            self._last_response_rejected = True
                            self._response_started_or_rejected_event.set()
                            logger.debug("response.create rejected; worker will retry after active response finishes")
                        else:
                            self._response_started_or_rejected_event.set()
                            logger.error("Realtime error [%s]: %s (raw=%s)", code, msg, err)

                        # Only show user-facing errors, not internal state errors
                        if code not in ("input_audio_buffer_commit_empty",):
                            await self.output_queue.put(
                                AdditionalOutputs({"role": "assistant", "content": f"[error] {msg}"})
                            )
            finally:
                # Stop the response sender worker.
                if response_sender_task is not None:
                    response_sender_task.cancel()
                    try:
                        await response_sender_task
                    except asyncio.CancelledError:
                        pass

                # Stop background tool manager tasks (listener + cleanup) in all patus.
                await self.tool_manager.shutdown()
                self._flush_assistant_audio_dump("session_end")

    # Microphone receive
    async def receive(self, frame: Tuple[int, NDArray[np.int16]]) -> None:
        """Receive audio frame from the microphone and send it to the OpenAI server.

        Handles both mono and stereo audio formats, converting to the expected
        mono format for OpenAI's API. Resamples if the input sample rate differs
        from the expected rate.

        Args:
            frame: A tuple containing (sample_rate, audio_data).

        """
        if not self.connection:
            return

        input_sample_rate, audio_frame = frame

        # Reshape if needed
        if audio_frame.ndim == 2:
            # Scipy channels last convention
            if audio_frame.shape[1] > audio_frame.shape[0]:
                audio_frame = audio_frame.T
            # Multiple channels -> Mono channel
            if audio_frame.shape[1] > 1:
                audio_frame = audio_frame[:, 0]

        # Resample if needed
        if self.input_sample_rate != input_sample_rate:
            audio_frame = resample(audio_frame, int(len(audio_frame) * self.input_sample_rate / input_sample_rate))

        # Cast if needed
        audio_frame = audio_to_int16(audio_frame)

        # Send to OpenAI (guard against races during reconnect)
        try:
            audio_message = base64.b64encode(audio_frame.tobytes()).decode("utf-8")
            await self.connection.input_audio_buffer.append(audio=audio_message)
        except Exception as e:
            logger.debug("Dropping audio frame: connection not ready (%s)", e)
            return

    async def emit(self) -> Tuple[int, NDArray[np.int16]] | AdditionalOutputs | None:
        """Emit audio frame to be played by the speaker."""
        # sends to the stream the stuff put in the output queue by the openai event handler
        # This is called periodically by the fastrtc Stream

        # Handle idle
        idle_duration = asyncio.get_event_loop().time() - self.last_activity_time
        if idle_duration > 180.0 and self._response_done_event.is_set() and self.deps.movement_manager.is_idle():
            try:
                await self.send_idle_signal(idle_duration)
            except Exception as e:
                logger.warning("Idle signal skipped (connection closed?): %s", e)
                return None

            self.last_activity_time = asyncio.get_event_loop().time()  # avoid repeated resets

        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def shutdown(self) -> None:
        """Shutdown the handler."""
        # Unblock the response sender worker so it can exit
        self._response_done_event.set()
        self._flush_assistant_audio_dump("shutdown")

        # Stop background tool manager tasks (listener + cleanup)
        await self.tool_manager.shutdown()

        # Cancel any pending debounce task
        if self.partial_transcript_task and not self.partial_transcript_task.done():
            self.partial_transcript_task.cancel()
            try:
                await self.partial_transcript_task
            except asyncio.CancelledError:
                pass

        if self.connection:
            try:
                await self.connection.close()
            except ConnectionClosedError as e:
                logger.debug(f"Connection already closed during shutdown: {e}")
            except Exception as e:
                logger.debug(f"connection.close() ignored: {e}")
            finally:
                self.connection = None

        # Clear any remaining items in the output queue
        while not self.output_queue.empty():
            try:
                self.output_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    def format_timestamp(self) -> str:
        """Format current timestamp with date, time, and elapsed seconds."""
        loop_time = asyncio.get_event_loop().time()  # monotonic
        elapsed_seconds = loop_time - self.start_time
        dt = datetime.now()  # wall-clock
        return f"[{dt.strftime('%Y-%m-%d %H:%M:%S')} | +{elapsed_seconds:.1f}s]"

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        fallback = get_available_voices_for_backend()
        if config.BACKEND_PROVIDER != OPENAI_BACKEND:
            return fallback
        try:
            # Best effort discovery; safe-guarded for unexpected shapes
            model = await self.client.models.retrieve(config.MODEL_NAME)
            # Try common serialization paths
            raw = None
            for attr in ("model_dump", "to_dict"):
                fn = getattr(model, attr, None)
                if callable(fn):
                    try:
                        raw = fn()
                        break
                    except Exception:
                        pass
            if raw is None:
                try:
                    raw = dict(model)
                except Exception:
                    raw = None
            # Scan for voice candidates
            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for k, v in obj.items():
                            kl = str(k).lower()
                            if "voice" in kl and isinstance(v, (list, tuple)):
                                for item in v:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and "name" in item and isinstance(item["name"], str):
                                        candidates.add(item["name"])
                            else:
                                _collect(v)
                    elif isinstance(obj, (list, tuple)):
                        for it in obj:
                            _collect(it)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)
            # Ensure default present and stable order
            voices = sorted(candidates) if candidates else fallback
            default_voice = get_default_voice_for_backend()
            if default_voice not in voices:
                voices = [default_voice, *[v for v in voices if v != default_voice]]
            return voices
        except Exception:
            return fallback

    async def _build_realtime_client(self, api_key: str | None = None) -> AsyncOpenAI:
        """Build the realtime SDK client, optionally via the s2s session allocator."""
        resolved_api_key = (api_key or self._provided_api_key or config.OPENAI_API_KEY or "").strip()
        if config.BACKEND_PROVIDER == OPENAI_BACKEND:
            self._realtime_connect_query = {}
            if not resolved_api_key:
                # In headless console mode, LocalStream now blocks startup until the key is provided.
                # However, unit tests may invoke this handler directly with a stubbed client.
                # To keep tests hermetic without requiring a real key, fall back to a placeholder.
                logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
                resolved_api_key = "DUMMY"
            return AsyncOpenAI(api_key=resolved_api_key)

        direct_realtime_url = get_s2s_direct_ws_url()
        if direct_realtime_url:
            if get_s2s_session_url():
                logger.info("S2S_REALTIME_WS_URL is set; bypassing S2S_REALTIME_SESSION_URL and using direct realtime.")
            client, connect_query = _build_openai_compatible_client_from_realtime_url(
                direct_realtime_url,
                resolved_api_key,
            )
            self._realtime_connect_query = connect_query
            logger.info("Using direct speech-to-speech realtime endpoint %s", direct_realtime_url)
            return client

        session_url = get_s2s_session_url()
        if not session_url:
            raise RuntimeError(
                f"Either S2S_REALTIME_SESSION_URL or S2S_REALTIME_WS_URL must be set when BACKEND_PROVIDER={S2S_BACKEND}"
            )

        async with httpx.AsyncClient(timeout=10.0) as http_client:
            response = await http_client.post(session_url)
            response.raise_for_status()
            payload = response.json()

        connect_url = payload.get("connect_url")
        if not isinstance(connect_url, str) or not connect_url:
            raise RuntimeError(f"Session allocator response did not contain a valid connect_url: {payload!r}")

        parsed = urlsplit(connect_url)
        path = parsed.path.rstrip("/")
        if not path.endswith("/realtime"):
            raise ValueError(f"Expected realtime connect URL ending with /realtime, got: {connect_url}")

        logger.info("Allocated realtime session %s", payload.get("session_id") or "<unknown>")
        client, connect_query = _build_openai_compatible_client_from_realtime_url(
            connect_url,
            resolved_api_key,
        )
        self._realtime_connect_query = connect_query
        return client

    async def send_idle_signal(self, idle_duration: float) -> None:
        """Send an idle signal to the openai server."""
        logger.debug("Sending idle signal")
        self.is_idle_tool_call = True
        timestamp_msg = f"[Idle time update: {self.format_timestamp()} - No activity for {idle_duration:.1f}s] You've been idle for a while. Feel free to get creative - dance, show an emotion, look around, do nothing, or just be yourself!"
        if not self.connection:
            logger.debug("No connection, cannot send idle signal")
            return
        await self.connection.conversation.item.create(
            item={
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": timestamp_msg}],
            },
        )
        await self._safe_response_create(
            response=RealtimeResponseCreateParamsParam(
                instructions="You MUST respond with function calls only - no speech or text. Choose appropriate actions for idle behavior.",
                tool_choice="required",
            ),
        )

    def _persist_api_key_if_needed(self) -> None:
        """Persist the API key into `.env` inside `instance_path/` when appropriate.

        - Only runs in Gradio mode when key came from the textbox and is non-empty.
        - Only saves if `self.instance_path` is not None.
        - Writes `.env` to `instance_path/.env` (does not overwrite if it already exists).
        - If `instance_path/.env.example` exists, copies its contents while overriding OPENAI_API_KEY.
        """
        try:
            if not self.gradio_mode:
                logger.warning("Not in Gradio mode; skipping API key persistence.")
                return

            if self._key_source != "textbox":
                logger.info("API key not provided via textbox; skipping persistence.")
                return

            key = (self._provided_api_key or "").strip()
            if not key:
                logger.warning("No API key provided via textbox; skipping persistence.")
                return
            if self.instance_path is None:
                logger.warning("Instance path is None; cannot persist API key.")
                return

            # Update the current process environment for downstream consumers
            try:
                import os

                os.environ["OPENAI_API_KEY"] = key
            except Exception:  # best-effort
                pass

            target_dir = Path(self.instance_path)
            env_path = target_dir / ".env"
            if env_path.exists():
                # Respect existing user configuration
                logger.info(".env already exists at %s; not overwriting.", env_path)
                return

            example_path = target_dir / ".env.example"
            content_lines: list[str] = []
            if example_path.exists():
                try:
                    content = example_path.read_text(encoding="utf-8")
                    content_lines = content.splitlines()
                except Exception as e:
                    logger.warning("Failed to read .env.example at %s: %s", example_path, e)

            # Replace or append the OPENAI_API_KEY line
            replaced = False
            for i, line in enumerate(content_lines):
                if line.strip().startswith("OPENAI_API_KEY="):
                    content_lines[i] = f"OPENAI_API_KEY={key}"
                    replaced = True
                    break
            if not replaced:
                content_lines.append(f"OPENAI_API_KEY={key}")

            # Ensure file ends with newline
            final_text = "\n".join(content_lines) + "\n"
            env_path.write_text(final_text, encoding="utf-8")
            logger.info("Created %s and stored OPENAI_API_KEY for future runs.", env_path)
        except Exception as e:
            # Never crash the app for QoL persistence; just log.
            logger.warning("Could not persist OPENAI_API_KEY to .env: %s", e)
