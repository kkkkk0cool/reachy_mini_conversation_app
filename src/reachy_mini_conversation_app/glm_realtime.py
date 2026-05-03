"""GLM Realtime handler for Zhipu AI's GLM-Realtime WebSocket API.

GLM Realtime uses a WebSocket protocol compatible with OpenAI Realtime in terms of
event type names (session.update, response.audio.delta, etc.), but the session
configuration fields differ from OpenAI's nested audio config structure.

API docs: https://docs.bigmodel.cn/cn/guide/models/sound-and-video/glm-realtime
WebSocket endpoint: wss://open.bigmodel.cn/api/paas/v4/realtime
Authorization: Bearer {GLM_API_KEY}

Supported models:
  - glm-realtime       (default, balanced)
  - glm-realtime-flash (9B, faster/cheaper)
  - glm-realtime-air   (32B, highest quality)

Audio:
  - Input:  24kHz mono PCM-16 ("pcm24") — matches SAMPLE_RATE=24000
  - Output: 24kHz mono PCM-16 ("pcm")
"""
import logging
from typing import Any

from openai import AsyncOpenAI

from reachy_mini_conversation_app.config import GLM_BACKEND, config
from reachy_mini_conversation_app.prompts import get_session_voice, get_session_instructions
from reachy_mini_conversation_app.base_realtime import (
    BaseRealtimeHandler,
    InputTranscriptChunksByItem,
    to_realtime_tools_config,
)
from reachy_mini_conversation_app.tools.core_tools import get_active_tool_specs

logger = logging.getLogger(__name__)

__all__ = ["GlmRealtimeHandler"]

_GLM_BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
_GLM_WS_BASE_URL = "wss://open.bigmodel.cn/api/paas/v4/"


class GlmRealtimeHandler(BaseRealtimeHandler):
    """Realtime handler for Zhipu AI GLM Realtime API.

    Uses the OpenAI Python SDK's WebSocket transport with GLM's endpoint.
    The session config is sent as a plain dict matching GLM's protocol
    instead of OpenAI's nested audio config TypedDict.
    """

    BACKEND_PROVIDER = GLM_BACKEND
    # Both input and output use 24kHz PCM so a single SAMPLE_RATE covers both.
    SAMPLE_RATE = 24000
    REFRESH_CLIENT_ON_RECONNECT = False
    # Pricing is per-minute; set to 0.0 until per-token costs are published.
    AUDIO_INPUT_COST_PER_1M = 0.0
    AUDIO_OUTPUT_COST_PER_1M = 0.0
    TEXT_INPUT_COST_PER_1M = 0.0
    TEXT_OUTPUT_COST_PER_1M = 0.0
    IMAGE_INPUT_COST_PER_1M = 0.0

    def _get_session_instructions(self) -> str:
        return get_session_instructions()

    def _get_session_voice(self, default: str | None = None) -> str:
        return get_session_voice(default)

    def _get_active_tool_specs(self) -> list[dict[str, Any]]:
        return get_active_tool_specs(self.deps)

    def _get_session_config(self, tool_specs: list[dict[str, Any]]) -> Any:
        """Return GLM-compatible session config.

        GLM uses a flat session structure rather than OpenAI's nested audio config.
        The dict is passed directly to conn.session.update(); the OpenAI SDK
        serialises it to JSON without further validation at runtime.

        Key GLM-specific fields:
          input_audio_format  – "pcm24" = 24 kHz PCM (matches SAMPLE_RATE)
          output_audio_format – "pcm"   = 24 kHz PCM (GLM only supports this)
          beta_fields         – required; chat_mode must be set
        """
        return {
            "instructions": self._get_session_instructions(),
            "modalities": ["text", "audio"],
            "voice": self.get_current_voice(),
            "input_audio_format": "pcm24",
            "output_audio_format": "pcm",
            "turn_detection": {"type": "server_vad"},
            "tools": to_realtime_tools_config(tool_specs),
            "tool_choice": "auto",
            "beta_fields": {"chat_mode": "audio"},
        }

    def _record_partial_transcript_delta(
        self,
        input_transcript: InputTranscriptChunksByItem,
        item_id: str,
        delta: str,
    ) -> None:
        """Record a GLM partial transcript snapshot (same behaviour as HF)."""
        input_transcript.item_id = item_id
        input_transcript.deltas = [delta]

    async def _build_realtime_client(self) -> AsyncOpenAI:
        """Build the OpenAI-compatible client pointing at GLM's WebSocket endpoint.

        GLM accepts standard Bearer token authentication, so the OpenAI SDK's
        default header injection works without modification.
        """
        self._realtime_connect_query = {}
        api_key = (config.GLM_API_KEY or "").strip()
        if not api_key:
            logger.warning(
                "GLM_API_KEY / ZHIPUAI_API_KEY is not set. "
                "Set GLM_API_KEY in your .env file before connecting."
            )
            api_key = "DUMMY"
        return AsyncOpenAI(
            api_key=api_key,
            base_url=_GLM_BASE_URL,
            websocket_base_url=_GLM_WS_BASE_URL,
        )
