import logging
from typing import Any, Literal
from urllib.parse import urlsplit, parse_qsl, urlunsplit

import httpx
from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, wait_for_item
from numpy.typing import NDArray
from websockets.exceptions import ConnectionClosedError

from reachy_mini_conversation_app.config import (
    S2S_BACKEND,
    S2S_LOCAL_CONNECTION_MODE,
    config,
    get_s2s_session_url,
    get_s2s_direct_ws_url,
    get_s2s_selected_connection_mode,
)
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import (
    _RESPONSE_DONE_TIMEOUT,
    S2S_REALTIME_SAMPLE_RATE,
    OPENAI_REALTIME_SAMPLE_RATE,
    BaseRealtimeHandler,
    InputTranscriptChunksByItem,
)
from reachy_mini_conversation_app.tools.core_tools import get_active_tool_specs


logger = logging.getLogger(__name__)


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

    connect_query = {key: value for key, value in parse_qsl(parsed.query, keep_blank_values=True) if key != "model"}
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


class HuggingFaceRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for Hugging Face speech-to-speech endpoints."""

    backend_provider = S2S_BACKEND
    realtime_sample_rate = S2S_REALTIME_SAMPLE_RATE
    requires_api_key = False
    refresh_client_on_reconnect = True

    def _response_done_timeout(self) -> float:
        """Return the response completion timeout."""
        return _RESPONSE_DONE_TIMEOUT

    def _connection_closed_errors(self) -> tuple[type[BaseException], ...]:
        """Return websocket closure exceptions handled as reconnectable/ignorable."""
        return (ConnectionClosedError,)

    def _get_session_audio_rates(self) -> tuple[Literal[24000] | None, Literal[24000] | None]:
        """Return S2S audio rates for the OpenAI-compatible session config.

        The OpenAI SDK type accepts only 24 kHz or None. The Hugging Face
        speech-to-speech server interprets None as its native 16 kHz default.
        """
        input_rate: Literal[24000] | None
        output_rate: Literal[24000] | None

        if self.input_sample_rate == S2S_REALTIME_SAMPLE_RATE:
            input_rate = None
        elif self.input_sample_rate == OPENAI_REALTIME_SAMPLE_RATE:
            input_rate = 24000
        else:
            raise AssertionError(f"Unsupported S2S input sample rate: {self.input_sample_rate}")

        if self.output_sample_rate == S2S_REALTIME_SAMPLE_RATE:
            output_rate = None
        elif self.output_sample_rate == OPENAI_REALTIME_SAMPLE_RATE:
            output_rate = 24000
        else:
            raise AssertionError(f"Unsupported S2S output sample rate: {self.output_sample_rate}")

        return input_rate, output_rate

    def _get_session_instructions(self) -> str:
        """Return speech-to-speech session instructions."""
        return get_session_instructions()

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured speech-to-speech session voice."""
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    async def _wait_for_output_item(self) -> tuple[int, NDArray[Any]] | AdditionalOutputs | None:
        """Wait for the next output item."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    def _record_partial_transcript_delta(
        self,
        input_transcript: InputTranscriptChunksByItem,
        item_id: str,
        delta: str,
    ) -> None:
        """Record a speech-to-speech partial transcript snapshot."""
        input_transcript.item_id = item_id
        input_transcript.deltas = [delta]

    async def _build_realtime_client(self, api_key: str | None = None) -> AsyncOpenAI:
        """Build the speech-to-speech OpenAI-compatible realtime client."""
        resolved_api_key = (api_key or self._provided_api_key or config.OPENAI_API_KEY or "").strip()
        selected_connection_mode = get_s2s_selected_connection_mode()
        direct_realtime_url = get_s2s_direct_ws_url()
        if selected_connection_mode == S2S_LOCAL_CONNECTION_MODE:
            if not direct_realtime_url:
                raise RuntimeError("S2S_REALTIME_WS_URL must be set when S2S_REALTIME_CONNECTION_MODE=local")
            client, connect_query = _build_openai_compatible_client_from_realtime_url(
                direct_realtime_url,
                resolved_api_key,
            )
            self._realtime_connect_query = connect_query
            logger.info("Using direct speech-to-speech realtime endpoint %s", direct_realtime_url)
            return client

        session_url = get_s2s_session_url()
        if not session_url:
            raise RuntimeError("Built-in speech-to-speech session allocator URL is unavailable")
        if direct_realtime_url:
            logger.info("S2S_REALTIME_CONNECTION_MODE=deployed; ignoring S2S_REALTIME_WS_URL.")

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
