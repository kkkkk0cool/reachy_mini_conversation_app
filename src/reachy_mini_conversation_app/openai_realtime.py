import logging
from typing import Any, Literal

from openai import AsyncOpenAI
from fastrtc import AdditionalOutputs, wait_for_item
from numpy.typing import NDArray
from websockets.exceptions import ConnectionClosedError

from reachy_mini_conversation_app.config import OPENAI_BACKEND, config, get_default_voice_for_backend
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import (
    _RESPONSE_DONE_TIMEOUT,
    OPENAI_REALTIME_SAMPLE_RATE,
    BaseRealtimeHandler,
    _compute_response_cost,
    _normalize_startup_voice,
)
from reachy_mini_conversation_app.tools.core_tools import get_active_tool_specs


logger = logging.getLogger(__name__)

__all__ = ["OpenaiRealtimeHandler", "_compute_response_cost", "_normalize_startup_voice"]


class OpenaiRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for the direct OpenAI Realtime API."""

    backend_provider = OPENAI_BACKEND
    realtime_sample_rate = OPENAI_REALTIME_SAMPLE_RATE
    requires_api_key = True
    refresh_client_on_reconnect = False

    def _response_done_timeout(self) -> float:
        """Return the response completion timeout."""
        return _RESPONSE_DONE_TIMEOUT

    def _connection_closed_errors(self) -> tuple[type[BaseException], ...]:
        """Return websocket closure exceptions handled as reconnectable/ignorable."""
        return (ConnectionClosedError,)

    def _get_session_audio_rates(self) -> tuple[Literal[24000], Literal[24000]]:
        """OpenAI Realtime requires an explicit 24 kHz audio session config."""
        return 24000, 24000

    def _get_session_instructions(self) -> str:
        """Return OpenAI session instructions."""
        return get_session_instructions()

    def _get_session_voice(self, default: str | None = None) -> str:
        """Return the configured OpenAI session voice."""
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        """Return active tool specs for the current session dependencies."""
        return get_active_tool_specs(self.deps)

    async def _wait_for_output_item(self) -> tuple[int, NDArray[Any]] | AdditionalOutputs | None:
        """Wait for the next output item."""
        return await wait_for_item(self.output_queue)  # type: ignore[no-any-return]

    async def get_available_voices(self) -> list[str]:
        """Try to discover available voices for the configured OpenAI realtime model.

        Attempts to retrieve model metadata from the OpenAI Models API and look
        for any keys that might contain voice names. Falls back to a curated
        list known to work with realtime if discovery fails.
        """
        fallback = await super().get_available_voices()
        try:
            model = await self.client.models.retrieve(config.MODEL_NAME)
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

            candidates: set[str] = set()

            def _collect(obj: object) -> None:
                try:
                    if isinstance(obj, dict):
                        for key, value in obj.items():
                            key_lower = str(key).lower()
                            if "voice" in key_lower and isinstance(value, (list, tuple)):
                                for item in value:
                                    if isinstance(item, str):
                                        candidates.add(item)
                                    elif isinstance(item, dict) and isinstance(item.get("name"), str):
                                        candidates.add(item["name"])
                            else:
                                _collect(value)
                    elif isinstance(obj, (list, tuple)):
                        for item in obj:
                            _collect(item)
                except Exception:
                    pass

            if isinstance(raw, dict):
                _collect(raw)

            voices = sorted(candidates) if candidates else fallback
            default_voice = get_default_voice_for_backend(self.backend_provider)
            if default_voice not in voices:
                voices = [default_voice, *[voice for voice in voices if voice != default_voice]]
            return voices
        except Exception:
            return fallback

    async def _build_realtime_client(self, api_key: str | None = None) -> AsyncOpenAI:
        """Build the OpenAI realtime SDK client."""
        self._realtime_connect_query = {}
        resolved_api_key = (api_key or self._provided_api_key or config.OPENAI_API_KEY or "").strip()
        if not resolved_api_key:
            # In headless console mode, LocalStream blocks startup until the key is provided.
            # Unit tests may invoke this handler directly with a stubbed client.
            logger.warning("OPENAI_API_KEY missing. Proceeding with a placeholder (tests/offline).")
            resolved_api_key = "DUMMY"
        return AsyncOpenAI(api_key=resolved_api_key)
