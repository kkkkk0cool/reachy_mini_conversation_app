"""Inference HTTP server: FunASR ASR + LLM API + CosyVoice2 TTS.

Run this on the Mac M4 Pro. Reachy Mini (Pi 4) only needs to capture/play
audio and call a single ``POST /conversation`` endpoint.

Architecture
------------
    Reachy Mini Pi 4  ──WiFi──▶  Mac M4 Pro (this server)
                                    POST /conversation
                                        ↓ FunASR SenseVoiceSmall  (ASR)
                                        ↓ Qwen / DeepSeek API     (LLM)
                                        ↓ CosyVoice2-0.5B         (TTS)
                                        ← streamed PCM audio

    Individual endpoints (for advanced / split deployments):
        POST /asr  → FunASR only
        POST /tts  → CosyVoice2 only

API
---
    POST /conversation
        Header X-Session-ID: <uuid>  (omit on first turn to start a new session)
        Content-Type: audio/pcm; rate=16000
        Body: raw little-endian int16 PCM bytes at 16 kHz
        Response 200: chunked audio/pcm (int16 LE, 22050 Hz)
                      Header X-Sample-Rate: 22050
                      Header X-Session-ID: <uuid>  (include in next request)
                      Header X-User-Text: <url-encoded user speech>

    POST /asr
        Content-Type: audio/pcm; rate=16000
        Body: raw little-endian int16 PCM bytes at 16 kHz
        Response 200: {"text": "...", "duration_ms": 123}
        Response 400: {"detail": "..."}

    POST /tts
        Content-Type: application/json
        Body: {"text": "...", "voice": "中文女"}
        Response 200: chunked audio/pcm (int16 LE, 22050 Hz)
                      Header X-Sample-Rate: 22050

    GET /voices
        Response 200: {"voices": ["中文女", "中文男", ...]}

    GET /health
        Response 200: {"status": "ok", "asr": "ready"|"loading"|"error",
                        "tts": "ready"|"loading"|"error"}

Usage
-----
    # Install deps
    pip install 'reachy_mini_conversation_app[inference_server]'
    # Install CosyVoice (see funasr_pipeline.py docstring)

    # Configure LLM API key and model (add to .env or environment):
    TEXT_LLM_PROVIDER=qwen           # or "deepseek"
    QWEN_API_KEY=sk-...
    MODEL_NAME=qwen-turbo

    # Start the server (on Mac)
    reachy-mini-inference-server

    # On Pi 4, add to .env:
    BACKEND_PROVIDER=remote
    CONVERSATION_SERVICE_URL=http://<mac-ip>:8765/conversation
"""
from __future__ import annotations

import re
import io
import json
import time
import uuid
import asyncio
import logging
import argparse
from pathlib import Path
from collections import deque
from typing import Any, AsyncIterator, Optional
from urllib.parse import quote, unquote

import numpy as np
import uvicorn
import av
from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, JSONResponse
from numpy.typing import NDArray
from openai import AsyncOpenAI
from pydantic import BaseModel

from reachy_mini_conversation_app.config import (
    COSYVOICE_AVAILABLE_VOICES,
    GLM_CHAT_AVAILABLE_VOICES,
    QWEN_LLM_PROVIDER,
    DEEPSEEK_LLM_PROVIDER,
    GLM_LLM_PROVIDER,
    OPENAI_COMPAT_LLM_PROVIDER,
    config,
)


logger = logging.getLogger(__name__)

# Strip SenseVoice emotion/language tags: <|HAPPY|>, <|zh|>, …
_TAG_PATTERN = re.compile(r"<\|[^|]+\|>")
_MEANINGFUL_TEXT_PATTERN = re.compile(r"[\u4e00-\u9fff]{2,}|[A-Za-z]{2,}|\d+")
_SPEAKABLE_TEXT_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]")

_OUTPUT_SAMPLE_RATE = 22_050  # CosyVoice2-0.5B native rate
_EDGE_OUTPUT_SAMPLE_RATE = 24_000
_ZERO_SHOT_PROMPT_TEXT = "希望你以后能够做的比我还好呦。"
_STREAM_INPUT_SAMPLE_RATE = 16_000
_STREAM_VAD_RMS_THRESHOLD = 650
_STREAM_VAD_MIN_SPEECH_SECONDS = 0.45
_STREAM_VAD_SILENCE_SECONDS = 0.95
_STREAM_VAD_LONG_SPEECH_SECONDS = 2.5
_STREAM_VAD_LONG_SILENCE_SECONDS = 1.25
_STREAM_VAD_PRE_ROLL_SECONDS = 0.55
_STREAM_VAD_MAX_SPEECH_SECONDS = 20.0


def _tts_provider() -> str:
    provider = (config.INFERENCE_TTS_PROVIDER or "cosyvoice").strip().lower()
    if provider not in {"cosyvoice", "edge"}:
        logger.warning("Unknown INFERENCE_TTS_PROVIDER=%r, falling back to cosyvoice", provider)
        return "cosyvoice"
    return provider


def _tts_sample_rate() -> int:
    return _EDGE_OUTPUT_SAMPLE_RATE if _tts_provider() == "edge" else _OUTPUT_SAMPLE_RATE


async def _empty_pcm_response(user_text: str = "") -> StreamingResponse:
    async def _empty():
        if False:
            yield b""

    return StreamingResponse(
        _empty(),
        media_type="audio/pcm",
        headers={
            "X-Sample-Rate": str(_tts_sample_rate()),
            "X-User-Text": quote(user_text),
        },
    )


def _is_meaningful_transcript(text: str) -> bool:
    """Return whether ASR text looks like real user speech, not punctuation/noise."""
    normalized = text.strip()
    if not normalized:
        return False
    return bool(_MEANINGFUL_TEXT_PATTERN.search(normalized))


def _is_speakable_text(text: str) -> bool:
    """Return whether a streamed segment contains content worth sending to TTS."""
    normalized = text.strip().strip("\"'“”‘’`")
    return bool(normalized and _SPEAKABLE_TEXT_PATTERN.search(normalized))


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


# ── LLM endpoints ─────────────────────────────────────────────────────────────

_LLM_CLOUD_ENDPOINTS: dict[str, str] = {
    QWEN_LLM_PROVIDER: "https://dashscope.aliyuncs.com/compatible-mode/v1",
    DEEPSEEK_LLM_PROVIDER: "https://api.deepseek.com/v1",
    GLM_LLM_PROVIDER: "https://open.bigmodel.cn/api/paas/v4/",
}
_LLM_DEFAULT_MODELS: dict[str, str] = {
    QWEN_LLM_PROVIDER: "qwen-turbo",
    DEEPSEEK_LLM_PROVIDER: "deepseek-chat",
    GLM_LLM_PROVIDER: "glm-4-flash",
    # OpenAI-compatible local endpoint: model name depends on the target server, e.g. Ollama:
    #   ollama pull nous-hermes-2-mistral
    OPENAI_COMPAT_LLM_PROVIDER: "nous-hermes-2-mistral",
}

# ── Global model state ────────────────────────────────────────────────────────

_SESSION_TTL = 1800  # seconds (30 min) of inactivity before session expires


class _Session:
    def __init__(self) -> None:
        self.history: list[dict[str, str]] = []
        self.last_seen: float = time.monotonic()

    def touch(self) -> None:
        self.last_seen = time.monotonic()


class _ModelState:
    funasr: Any = None
    cosyvoice: Any = None
    llm_client: Optional[AsyncOpenAI] = None
    llm_model: str = "qwen-turbo"
    funasr_status: str = "loading"
    tts_status: str = "loading"
    device: str = "cpu"
    sessions: dict[str, _Session] = {}
    cosyvoice_speakers: list[str] = []
    cosyvoice_prompt_wav: str = ""


_state = _ModelState()

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Reachy Mini Inference Server",
    description="FunASR ASR + CosyVoice2 TTS for Reachy Mini offload",
    version="1.0.0",
)


# ── Startup: load models ──────────────────────────────────────────────────────

@app.on_event("startup")
async def _startup() -> None:
    """Load FunASR and CosyVoice2 models in background threads; build LLM client."""
    _state.device = _best_device()
    logger.info("Inference server starting on device: %s", _state.device)
    _build_llm_client()
    loop = asyncio.get_event_loop()
    loop.create_task(_load_funasr())
    if _tts_provider() == "edge":
        _state.tts_status = "ready"
        logger.info("Edge TTS ready. voice=%s", config.EDGE_TTS_VOICE)
    else:
        loop.create_task(_load_cosyvoice())
    loop.create_task(_session_cleanup_loop())


def _build_llm_client() -> None:
    """Initialise the LLM client from environment configuration.

    Supported TEXT_LLM_PROVIDER values:
        "qwen"    – Alibaba DashScope (cloud)
        "deepseek"– DeepSeek API (cloud)
        "glm"     – Zhipu GLM API (cloud, e.g. glm-4.7 / glm-4-flash)
        "openai_compat" – Local OpenAI-compatible agent/server (no API key required).
                          Configure the endpoint with OPENAI_COMPAT_API_URL
                          (default: http://localhost:11434/v1).
    """
    _all_providers = list(_LLM_CLOUD_ENDPOINTS) + [OPENAI_COMPAT_LLM_PROVIDER]
    provider = (config.TEXT_LLM_PROVIDER or QWEN_LLM_PROVIDER).strip().lower()
    if provider not in _all_providers:
        logger.warning(
            "Unknown TEXT_LLM_PROVIDER=%r; expected one of %s. Falling back to 'qwen'.",
            provider,
            _all_providers,
        )
        provider = QWEN_LLM_PROVIDER

    api_key: str
    if provider == OPENAI_COMPAT_LLM_PROVIDER:
        # Some local servers ignore the key, while gateways like OpenClaw may require it.
        base_url = (config.OPENAI_COMPAT_API_URL or "http://localhost:11434/v1").rstrip("/")
        api_key = (config.OPENAI_COMPAT_API_KEY or "ollama").strip()
        logger.info("OpenAI-compatible local agent endpoint: %s", base_url)
    else:
        base_url = _LLM_CLOUD_ENDPOINTS[provider]
        if provider == QWEN_LLM_PROVIDER:
            api_key = (config.QWEN_API_KEY or "").strip() or "DUMMY"
            if api_key == "DUMMY":
                logger.warning("QWEN_API_KEY / DASHSCOPE_API_KEY not set — /conversation will fail.")
        elif provider == DEEPSEEK_LLM_PROVIDER:
            api_key = (config.DEEPSEEK_API_KEY or "").strip() or "DUMMY"
            if api_key == "DUMMY":
                logger.warning("DEEPSEEK_API_KEY not set — /conversation will fail.")
        else:  # GLM_LLM_PROVIDER
            api_key = (config.GLM_API_KEY or "").strip() or "DUMMY"
            if api_key == "DUMMY":
                logger.warning("GLM_API_KEY / ZHIPUAI_API_KEY not set — /conversation will fail.")

    _state.llm_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    raw_text_model = (config.TEXT_LLM_MODEL or "").strip()
    _state.llm_model = raw_text_model or (config.MODEL_NAME or "").strip() or _LLM_DEFAULT_MODELS[provider]
    logger.info("LLM client ready: provider=%s model=%s", provider, _state.llm_model)


async def _session_cleanup_loop() -> None:
    """Periodically evict idle sessions to avoid unbounded memory growth."""
    while True:
        await asyncio.sleep(300)  # check every 5 min
        cutoff = time.monotonic() - _SESSION_TTL
        expired = [sid for sid, s in _state.sessions.items() if s.last_seen < cutoff]
        for sid in expired:
            _state.sessions.pop(sid, None)
        if expired:
            logger.info("Evicted %d expired conversation session(s).", len(expired))


async def _load_funasr() -> None:
    try:
        from funasr import AutoModel  # type: ignore[import]
    except ImportError:
        logger.error("funasr not installed. Run: pip install funasr")
        _state.funasr_status = "error"
        return

    model_id = config.FUNASR_MODEL or "iic/SenseVoiceSmall"
    loop = asyncio.get_event_loop()

    devices = [_state.device]
    if _state.device != "cpu":
        devices.append("cpu")

    last_exc: Exception | None = None
    for device in devices:
        logger.info("Loading FunASR model %r on %s …", model_id, device)
        try:
            _state.funasr = await loop.run_in_executor(
                None,
                lambda device=device: AutoModel(
                    model=model_id,
                    trust_remote_code=True,
                    device=device,
                    disable_update=True,
                ),
            )
            _state.funasr_status = "ready"
            logger.info("FunASR ready on %s.", device)
            return
        except Exception as exc:
            last_exc = exc
            logger.warning("Failed to load FunASR on %s: %s", device, exc)

    logger.error("Failed to load FunASR on all devices: %s", last_exc)
    _state.funasr_status = "error"


async def _load_cosyvoice() -> None:
    try:
        from cosyvoice.cli.cosyvoice import CosyVoice2  # type: ignore[import]
    except ImportError:
        logger.error(
            "CosyVoice not installed. Clone https://github.com/FunAudioLLM/CosyVoice "
            "and run: pip install -e ."
        )
        _state.tts_status = "error"
        return

    model_dir = config.COSYVOICE_MODEL_DIR or "pretrained_models/CosyVoice2-0.5B"
    logger.info("Loading CosyVoice2 from %r …", model_dir)
    model_path = Path(model_dir).expanduser().resolve()
    prompt_wav = model_path.parent.parent / "asset" / "zero_shot_prompt.wav"

    def _load() -> Any:
        if _state.device == "mps":
            import os
            os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return CosyVoice2(model_dir, load_jit=False, load_trt=False, fp16=False)

    loop = asyncio.get_event_loop()
    try:
        _state.cosyvoice = await loop.run_in_executor(None, _load)
        try:
            _state.cosyvoice_speakers = list(_state.cosyvoice.list_available_spks())
        except Exception:
            _state.cosyvoice_speakers = []
        _state.cosyvoice_prompt_wav = str(prompt_wav)
        _state.tts_status = "ready"
        if _state.cosyvoice_speakers:
            logger.info("CosyVoice2 ready. SFT speakers: %s", _state.cosyvoice_speakers)
        else:
            logger.info("CosyVoice2 ready. No SFT speakers found; using zero-shot prompt %s", prompt_wav)
    except Exception as exc:
        logger.error("Failed to load CosyVoice2: %s", exc)
        _state.tts_status = "error"


def _cosyvoice_inference(text: str, voice: str, stream: bool = True):
    """Run CosyVoice in the mode supported by the loaded checkpoint."""
    if _state.cosyvoice_speakers:
        speaker = voice if voice in _state.cosyvoice_speakers else _state.cosyvoice_speakers[0]
        if speaker != voice:
            logger.warning("Unknown SFT speaker %r, falling back to %r", voice, speaker)
        return _state.cosyvoice.inference_sft(text, speaker, stream=stream)

    prompt_wav = _state.cosyvoice_prompt_wav
    if not prompt_wav or not Path(prompt_wav).exists():
        raise RuntimeError(f"CosyVoice zero-shot prompt wav not found: {prompt_wav}")
    return _state.cosyvoice.inference_zero_shot(text, _ZERO_SHOT_PROMPT_TEXT, prompt_wav, stream=stream)


def _mp3_bytes_to_pcm(mp3_data: bytes, target_rate: int = _EDGE_OUTPUT_SAMPLE_RATE) -> np.ndarray:
    """Decode Edge-TTS MP3 bytes to mono int16 PCM."""
    buf = io.BytesIO(mp3_data)
    container = av.open(buf, format="mp3")
    resampler = av.AudioResampler(format="s16", layout="mono", rate=target_rate)
    frames: list[np.ndarray] = []
    for frame in container.decode(audio=0):
        for resampled in resampler.resample(frame):
            frames.append(resampled.to_ndarray().flatten().astype(np.int16))
    if not frames:
        return np.array([], dtype=np.int16)
    return np.concatenate(frames)


async def _edge_tts_inference(text: str, voice: str) -> bytes:
    """Synthesize text with Edge TTS and return mono int16 PCM bytes."""
    if not _is_speakable_text(text):
        return b""

    try:
        import edge_tts  # type: ignore[import]
    except ImportError as exc:
        raise RuntimeError(
            "edge-tts is required for INFERENCE_TTS_PROVIDER=edge. "
            "Install it with: pip install 'reachy_mini_conversation_app[inference_server]'"
        ) from exc

    configured_voice = config.EDGE_TTS_VOICE or "zh-CN-XiaoxiaoNeural"
    if configured_voice not in GLM_CHAT_AVAILABLE_VOICES:
        configured_voice = "zh-CN-XiaoxiaoNeural"
    edge_voice = voice if voice in GLM_CHAT_AVAILABLE_VOICES else configured_voice
    communicate = edge_tts.Communicate(text, edge_voice)
    mp3_chunks: list[bytes] = []
    async for chunk in communicate.stream():
        if chunk["type"] == "audio":
            mp3_chunks.append(chunk["data"])
    if not mp3_chunks:
        return b""

    loop = asyncio.get_event_loop()
    pcm = await loop.run_in_executor(None, _mp3_bytes_to_pcm, b"".join(mp3_chunks), _EDGE_OUTPUT_SAMPLE_RATE)
    return pcm.tobytes()


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    """Return service readiness status."""
    return JSONResponse({
        "status": "ok",
        "asr": _state.funasr_status,
        "tts": _state.tts_status,
        "tts_provider": _tts_provider(),
        "device": _state.device,
    })


@app.get("/voices")
async def voices() -> JSONResponse:
    """Return available voices for the active TTS provider."""
    if _tts_provider() == "edge":
        return JSONResponse({"voices": list(GLM_CHAT_AVAILABLE_VOICES)})
    return JSONResponse({"voices": list(COSYVOICE_AVAILABLE_VOICES)})


@app.post("/asr")
async def asr(request: Request) -> JSONResponse:
    """Transcribe raw 16 kHz int16 PCM audio.

    Send ``Content-Type: audio/pcm; rate=16000`` with raw PCM bytes in the body.
    """
    if _state.funasr_status != "ready" or _state.funasr is None:
        raise HTTPException(503, detail=f"ASR model not ready: {_state.funasr_status}")

    body = await request.body()
    if not body:
        raise HTTPException(400, detail="Empty audio body")

    # Decode raw int16 LE PCM → float32 normalised
    audio_int16 = np.frombuffer(body, dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32_768.0

    t0 = time.perf_counter()
    loop = asyncio.get_event_loop()
    try:
        result = await loop.run_in_executor(
            None,
            lambda: _state.funasr.generate(
                input=audio_float,
                language="zh",
                use_itn=True,
                batch_size_s=60,
            ),
        )
    except Exception as exc:
        logger.error("ASR inference error: %s", exc)
        raise HTTPException(500, detail=str(exc))

    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    raw_text = result[0]["text"] if result else ""
    text = _TAG_PATTERN.sub("", raw_text).strip()
    logger.info("ASR (%d ms) → %r", elapsed_ms, text)

    return JSONResponse({"text": text, "duration_ms": elapsed_ms})


class _TTSRequest(BaseModel):
    text: str
    voice: str = "中文女"


@app.post("/tts")
async def tts(req: _TTSRequest) -> StreamingResponse:
    """Synthesise text with CosyVoice2 and stream raw PCM audio back.

    Response is ``audio/pcm`` (int16 LE, 22050 Hz) streamed in chunks.
    The response header ``X-Sample-Rate`` carries the sample rate.
    """
    if _state.tts_status != "ready" or (_tts_provider() == "cosyvoice" and _state.cosyvoice is None):
        raise HTTPException(503, detail=f"TTS model not ready: {_state.tts_status}")

    text = req.text.strip()
    if not text:
        raise HTTPException(400, detail="Empty text")

    voice = req.voice
    available_voices = GLM_CHAT_AVAILABLE_VOICES if _tts_provider() == "edge" else COSYVOICE_AVAILABLE_VOICES
    if voice not in available_voices:
        logger.warning("Unknown voice %r, falling back to 中文女", voice)
        voice = config.EDGE_TTS_VOICE if _tts_provider() == "edge" else "中文女"

    logger.debug("TTS text=%r voice=%r", text, voice)

    async def _stream_audio() -> AsyncIterator[bytes]:
        """Run CosyVoice inference in an executor and yield PCM byte chunks."""
        if _tts_provider() == "edge":
            pcm = await _edge_tts_inference(text, voice)
            if pcm:
                yield pcm
            return

        queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        loop = asyncio.get_event_loop()

        def _synth() -> None:
            try:
                for output in _cosyvoice_inference(text, voice, stream=True):
                    tensor = output["tts_speech"]
                    pcm_float = tensor.squeeze().numpy()
                    pcm_int16 = (pcm_float * 32_768).clip(-32_768, 32_767).astype(np.int16)
                    asyncio.run_coroutine_threadsafe(queue.put(pcm_int16.tobytes()), loop)
            except Exception as exc:
                logger.error("TTS synthesis error: %s", exc)
            finally:
                asyncio.run_coroutine_threadsafe(queue.put(None), loop)

        loop.run_in_executor(None, _synth)

        while True:
            chunk = await queue.get()
            if chunk is None:
                break
            yield chunk

    return StreamingResponse(
        _stream_audio(),
        media_type="audio/pcm",
        headers={"X-Sample-Rate": str(_tts_sample_rate())},
    )


_SENTENCE_ENDS = re.compile(r"[.!?。！？\n]")
_CLAUSE_ENDS = re.compile(r"[,，、;；:：]")
_MIN_CLAUSE_CHARS = 12
_MAX_TTS_CHARS = 36
_MAX_HISTORY_TURNS = 10
_TAIL_ACTION_MARKER = "REACHY_ACTIONS"
_ACTION_BLOCK_RE = re.compile(r"REACHY_ACTIONS\s*:\s*(\[.*?\])\s*$", re.DOTALL)
_ALLOWED_ACTION_NAMES = {
    "dance",
    "stop_dance",
    "play_emotion",
    "stop_emotion",
    "move_head",
    "head_tracking",
    "do_nothing",
}
_ALLOWED_ACTION_TIMINGS = {"immediate", "speech_start", "after_speech"}

_ACTION_PLANNER_PROMPT = (
    "你是 Reachy Mini 的动作规划器。"
    "根据用户一句话，判断是否需要机器人本体动作。"
    "只返回 JSON 数组，不要返回解释、Markdown 或其他文本。"
    "数组最多 4 项；没有动作时返回 []。"
    "每项格式：{\"name\":\"dance|stop_dance|play_emotion|stop_emotion|move_head|head_tracking|do_nothing\","
    "\"arguments\":{},\"timing\":\"immediate|speech_start|after_speech\"}。"
    "move_head.arguments.direction 只能是 left/right/up/down/front。"
    "head_tracking.arguments.start 是布尔值。"
    "stop_dance 和 stop_emotion 的 timing 用 immediate。"
    "普通动作默认 timing 用 speech_start；如果用户明确说“说完后/等会儿/最后再做”，用 after_speech。"
    "不要把聊天回复写进 JSON。"
)

_SYSTEM_PROMPT = (
    "你是 Reachy Mini，一个友好的中文机器人助手。"
    "无论用户说什么，默认用简体中文回答，除非用户明确要求其他语言。"
    "回答要口语化、适合语音播放，不要 Markdown。"
    "当用户要求你做动作时，用自然语言简短回应即可，不要输出 JSON、代码块或控制指令。"
)


def _extract_reachy_actions(text: str) -> tuple[str, list[dict[str, object]]]:
    """Remove optional REACHY_ACTIONS JSON block from assistant text."""
    match = _ACTION_BLOCK_RE.search(text.strip())
    if not match:
        return text, []

    spoken_text = text[: match.start()].strip()
    try:
        parsed = json.loads(match.group(1))
    except Exception as exc:
        logger.warning("Invalid REACHY_ACTIONS block: %s", exc)
        return spoken_text or text, []

    actions: list[dict[str, object]] = []
    if not isinstance(parsed, list):
        return spoken_text, []

    for item in parsed[:4]:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        arguments = item.get("arguments", {})
        if name not in _ALLOWED_ACTION_NAMES:
            continue
        if not isinstance(arguments, dict):
            arguments = {}
        actions.append({"name": name, "arguments": arguments})
    return spoken_text, actions


def _normalise_reachy_action(item: object) -> dict[str, object] | None:
    """Validate and normalise one remote Reachy action plan item."""
    if not isinstance(item, dict):
        return None
    name = item.get("name")
    arguments = item.get("arguments", {})
    timing = item.get("timing", "speech_start")
    if name not in _ALLOWED_ACTION_NAMES:
        return None
    if not isinstance(arguments, dict):
        arguments = {}
    if timing not in _ALLOWED_ACTION_TIMINGS:
        timing = "speech_start"

    if name == "move_head":
        direction = arguments.get("direction")
        if direction not in {"left", "right", "up", "down", "front"}:
            return None
        arguments = {"direction": direction}
    elif name == "head_tracking":
        arguments = {"start": bool(arguments.get("start"))}
    elif name in {"stop_dance", "stop_emotion"}:
        arguments = {"dummy": True}
        timing = "immediate"
    elif name in {"dance", "play_emotion"}:
        # Keep only optional tool-specific string arguments; random is fine when omitted.
        cleaned: dict[str, object] = {}
        for key in ("move", "emotion"):
            value = arguments.get(key)
            if isinstance(value, str) and value.strip():
                cleaned[key] = value.strip()
        if name == "dance":
            repeat = arguments.get("repeat")
            if isinstance(repeat, int) and 1 <= repeat <= 3:
                cleaned["repeat"] = repeat
        arguments = cleaned
    else:
        arguments = {}

    return {"name": name, "arguments": arguments, "timing": timing}


def _parse_action_plan_json(text: str) -> list[dict[str, object]]:
    """Parse a model-produced JSON action plan defensively."""
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    if not candidate.startswith("["):
        match = re.search(r"\[.*\]", candidate, re.DOTALL)
        candidate = match.group(0) if match else "[]"

    try:
        parsed = json.loads(candidate)
    except Exception as exc:
        logger.warning("Invalid action planner JSON: %s content=%r", exc, text[:240])
        return []
    if not isinstance(parsed, list):
        return []

    actions: list[dict[str, object]] = []
    for item in parsed[:4]:
        action = _normalise_reachy_action(item)
        if action:
            actions.append(action)
    return actions


async def _plan_reachy_actions_with_llm(user_text: str) -> list[dict[str, object]]:
    """Ask the local LLM for a structured action plan, independent of spoken reply."""
    if not config.REACHY_ACTION_PLANNER_ENABLED or _state.llm_client is None:
        return []

    timeout = max(0.1, float(config.REACHY_ACTION_PLANNER_TIMEOUT or 0.8))
    try:
        response = await asyncio.wait_for(
            _state.llm_client.chat.completions.create(
                model=_state.llm_model,
                messages=[
                    {"role": "system", "content": _ACTION_PLANNER_PROMPT},
                    {"role": "user", "content": user_text},
                ],
                temperature=0,
                max_tokens=220,
            ),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.info("Action planner timed out after %.2fs; using heuristic fallback.", timeout)
        return []
    except Exception as exc:
        logger.warning("Action planner failed: %s; using heuristic fallback.", exc)
        return []

    choices = getattr(response, "choices", None) or []
    if not choices:
        return []
    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) or ""
    actions = _parse_action_plan_json(content)
    if actions:
        logger.info("Reachy actions planned by LLM: %s", actions)
    return actions


def _sanitize_spoken_text(text: str) -> str:
    """Strip any leaked control marker before sending text to TTS."""
    spoken_text, _ = _extract_reachy_actions(text)
    if _TAIL_ACTION_MARKER in spoken_text:
        spoken_text = spoken_text.split(_TAIL_ACTION_MARKER, 1)[0]
    return spoken_text.strip()


def _infer_reachy_actions_from_user_text(text: str) -> list[dict[str, object]]:
    """Infer simple local robot actions from explicit user commands."""
    normalized = re.sub(r"\s+", "", text.lower())
    actions: list[dict[str, object]] = []
    default_timing = (
        "after_speech"
        if any(k in normalized for k in ("说完后", "说完之后", "等会儿", "等一下再", "最后再", "待会儿"))
        else "speech_start"
    )

    def add(name: str, arguments: dict[str, object] | None = None, timing: str = "speech_start") -> None:
        action = _normalise_reachy_action({"name": name, "arguments": arguments or {}, "timing": timing})
        if action:
            actions.append(action)

    if any(k in normalized for k in ("停止跳舞", "别跳舞", "不要跳舞", "停下跳舞", "别跳了")):
        add("stop_dance", {"dummy": True}, "immediate")
        return actions
    if any(k in normalized for k in ("停止动作", "停止一下", "停一下", "停下来", "别动了", "不要动", "停下动作")):
        add("stop_dance", {"dummy": True}, "immediate")
        add("stop_emotion", {"dummy": True}, "immediate")
        return actions

    if any(k in normalized for k in ("跳舞", "跳个舞", "跳一段", "跳一下", "舞蹈", "扭一扭", "动一动", "表演一下")):
        add("dance", {}, default_timing)

    direction: str | None = None
    if any(k in normalized for k in ("向左", "左边", "往左", "看左", "看看左", "看左边", "头往左", "左转", "转左")):
        direction = "left"
    elif any(k in normalized for k in ("向右", "右边", "往右", "看右", "看看右", "看右边", "头往右", "右转", "转右")):
        direction = "right"
    elif any(k in normalized for k in ("抬头", "抬一下头", "向上", "往上", "看上", "看上面")):
        direction = "up"
    elif any(k in normalized for k in ("低头", "低一下头", "向下", "往下", "看下", "看下面")):
        direction = "down"
    elif any(k in normalized for k in ("看前", "正前", "回正", "转回来", "看着我", "看我")):
        direction = "front"
    if direction:
        add("move_head", {"direction": direction}, default_timing)

    if any(k in normalized for k in ("开心一点", "笑一下", "高兴一点", "卖个萌", "表现开心", "开心一下")):
        add("play_emotion", {}, default_timing)
    if any(k in normalized for k in ("停止表情", "别做表情", "停止情绪")):
        add("stop_emotion", {"dummy": True})

    if any(k in normalized for k in ("开启头部追踪", "开始头部追踪", "开始跟踪", "看着我")):
        add("head_tracking", {"start": True})
    elif any(k in normalized for k in ("关闭头部追踪", "停止头部追踪", "停止跟踪")):
        add("head_tracking", {"start": False})

    if any(k in normalized for k in ("别动", "安静待着", "什么都别做")):
        add("do_nothing", {})

    return actions[:4]


def _pop_tts_segment(buffer: str) -> tuple[str | None, str]:
    """Pop one speech-friendly text segment from a streaming LLM buffer."""
    sentence_match = _SENTENCE_ENDS.search(buffer)
    if sentence_match:
        end = sentence_match.end()
        return buffer[:end].strip(), buffer[end:]

    if len(buffer) >= _MIN_CLAUSE_CHARS:
        clause_matches = list(_CLAUSE_ENDS.finditer(buffer))
        if clause_matches:
            for match in clause_matches:
                if match.end() >= _MIN_CLAUSE_CHARS:
                    end = match.end()
                    return buffer[:end].strip(), buffer[end:]

    if len(buffer) >= _MAX_TTS_CHARS:
        split_at = -1
        for pattern in (_CLAUSE_ENDS, re.compile(r"\\s+")):
            matches = [m for m in pattern.finditer(buffer[:_MAX_TTS_CHARS]) if m.end() >= _MIN_CLAUSE_CHARS]
            if matches:
                split_at = matches[-1].end()
                break
        if split_at < 0:
            split_at = _MAX_TTS_CHARS
        return buffer[:split_at].strip(), buffer[split_at:]

    return None, buffer


class _StreamingTurnDetector:
    """Server-side continuous audio VAD with pre-roll to avoid clipping speech starts."""

    def __init__(self) -> None:
        self._pre_roll: deque[NDArray[np.int16]] = deque()
        self._pre_roll_samples = 0
        self._speech_chunks: list[NDArray[np.int16]] = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speaking = False
        self._max_pre_roll_samples = int(_STREAM_INPUT_SAMPLE_RATE * _STREAM_VAD_PRE_ROLL_SECONDS)

    def _append_pre_roll(self, pcm: NDArray[np.int16]) -> None:
        chunk = pcm.copy()
        self._pre_roll.append(chunk)
        self._pre_roll_samples += chunk.size
        while self._pre_roll and self._pre_roll_samples > self._max_pre_roll_samples:
            old = self._pre_roll.popleft()
            self._pre_roll_samples -= old.size

    def _reset_speech(self) -> None:
        self._speech_chunks = []
        self._speech_samples = 0
        self._silence_samples = 0
        self._is_speaking = False

    def push(self, pcm: NDArray[np.int16]) -> NDArray[np.int16] | None:
        if pcm.size == 0:
            return None

        rms = int(np.sqrt(np.mean(pcm.astype(np.float32) ** 2)))
        is_speech = rms >= _STREAM_VAD_RMS_THRESHOLD

        if not self._is_speaking:
            if not is_speech:
                self._append_pre_roll(pcm)
                return None
            self._is_speaking = True
            self._speech_chunks = [chunk.copy() for chunk in self._pre_roll]
            self._speech_chunks.append(pcm.copy())
            self._speech_samples = sum(chunk.size for chunk in self._speech_chunks)
            self._silence_samples = 0
            self._pre_roll.clear()
            self._pre_roll_samples = 0
            logger.info("Streaming VAD: speech start (rms=%d)", rms)
            return None

        self._speech_chunks.append(pcm.copy())
        self._speech_samples += pcm.size
        if is_speech:
            self._silence_samples = 0
        else:
            self._silence_samples += pcm.size

        speech_seconds = self._speech_samples / _STREAM_INPUT_SAMPLE_RATE
        silence_seconds = self._silence_samples / _STREAM_INPUT_SAMPLE_RATE
        required_silence = (
            _STREAM_VAD_LONG_SILENCE_SECONDS
            if speech_seconds >= _STREAM_VAD_LONG_SPEECH_SECONDS
            else _STREAM_VAD_SILENCE_SECONDS
        )
        if silence_seconds < required_silence and speech_seconds < _STREAM_VAD_MAX_SPEECH_SECONDS:
            return None

        audio = np.concatenate(self._speech_chunks)
        self._reset_speech()
        if speech_seconds < _STREAM_VAD_MIN_SPEECH_SECONDS:
            logger.info("Streaming VAD: utterance too short (%.2fs), discarding", speech_seconds)
            return None
        logger.info(
            "Streaming VAD: speech end after %.2fs silence=%.2fs, %d samples",
            speech_seconds,
            silence_seconds,
            audio.size,
        )
        return audio


def _normalise_tts_voice(voice: str) -> str:
    available_voices = GLM_CHAT_AVAILABLE_VOICES if _tts_provider() == "edge" else COSYVOICE_AVAILABLE_VOICES
    if voice in available_voices:
        return voice
    return config.EDGE_TTS_VOICE if _tts_provider() == "edge" else "中文女"


async def _transcribe_audio_int16(audio_int16: NDArray[np.int16]) -> str:
    audio_float = audio_int16.astype(np.float32) / 32_768.0
    loop = asyncio.get_event_loop()
    try:
        asr_result = await loop.run_in_executor(
            None,
            lambda: _state.funasr.generate(
                input=audio_float,
                language="zh",
                use_itn=True,
                batch_size_s=60,
            ),
        )
    except Exception as exc:
        logger.error("ASR error: %s", exc)
        raise HTTPException(500, detail=f"ASR failed: {exc}")

    raw_text = asr_result[0]["text"] if asr_result else ""
    return _TAG_PATTERN.sub("", raw_text).strip()


async def _build_conversation_audio_stream(
    user_text: str,
    session_id: str | None,
    voice: str,
) -> tuple[str, list[dict[str, object]], AsyncIterator[bytes]]:
    planned_by_llm = False
    reachy_actions = await _plan_reachy_actions_with_llm(user_text)
    planned_by_llm = bool(reachy_actions)
    if not reachy_actions:
        reachy_actions = _infer_reachy_actions_from_user_text(user_text)
    if reachy_actions:
        source = "planner" if planned_by_llm else "heuristic"
        logger.info("Reachy actions selected by %s: %s", source, reachy_actions)

    active_session_id = session_id or str(uuid.uuid4())
    session = _state.sessions.setdefault(active_session_id, _Session())
    session.touch()
    session.history.append({"role": "user", "content": user_text})
    if len(session.history) > _MAX_HISTORY_TURNS * 2:
        session.history = session.history[-_MAX_HISTORY_TURNS * 2:]

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}, *session.history]
    try:
        llm_stream = await _state.llm_client.chat.completions.create(
            model=_state.llm_model,
            messages=messages,  # type: ignore[arg-type]
            stream=True,
        )
    except Exception as exc:
        logger.error("LLM error: %s", exc)
        raise HTTPException(502, detail=f"LLM failed: {exc}")

    async def _generate_audio() -> AsyncIterator[bytes]:
        """Consume LLM stream sentence-by-sentence, yield TTS PCM chunks."""
        sentence_queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        audio_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue()
        assistant_reply_parts: list[str] = []
        loop = asyncio.get_event_loop()

        def _synth_cosyvoice_sentence(text: str) -> None:
            try:
                for output in _cosyvoice_inference(text, voice, stream=True):
                    tensor = output["tts_speech"]
                    pcm_float = tensor.squeeze().numpy()
                    pcm_int16 = (pcm_float * 32_768).clip(-32_768, 32_767).astype(np.int16)
                    asyncio.run_coroutine_threadsafe(audio_queue.put(pcm_int16.tobytes()), loop)
            except Exception as exc:
                logger.error("TTS error: %s", exc)

        async def _produce_sentences() -> None:
            sentence_buf = ""
            try:
                async for delta_chunk in llm_stream:
                    choices = getattr(delta_chunk, "choices", None) or []
                    if not choices:
                        continue
                    delta_obj = getattr(choices[0], "delta", None)
                    delta = getattr(delta_obj, "content", None) or ""
                    if not delta:
                        continue
                    sentence_buf += delta
                    assistant_reply_parts.append(delta)

                    while True:
                        sentence, sentence_buf = _pop_tts_segment(sentence_buf)
                        if not sentence:
                            break
                        sentence = _sanitize_spoken_text(sentence)
                        if _is_speakable_text(sentence):
                            await sentence_queue.put(sentence)

                if sentence_buf.strip():
                    tail_text = _sanitize_spoken_text(sentence_buf.strip())
                    if _is_speakable_text(tail_text):
                        await sentence_queue.put(tail_text)
            except Exception as exc:
                logger.exception("LLM stream error: %s", exc)
            finally:
                await sentence_queue.put(None)

        async def _synth_sentences() -> None:
            try:
                while True:
                    sentence = await sentence_queue.get()
                    if sentence is None:
                        break
                    try:
                        if _tts_provider() == "edge":
                            audio = await _edge_tts_inference(sentence, voice)
                            if audio:
                                await audio_queue.put(audio)
                        else:
                            await loop.run_in_executor(None, _synth_cosyvoice_sentence, sentence)
                    except Exception as exc:
                        logger.exception("TTS sentence synthesis failed for %r: %s", sentence[:120], exc)
            finally:
                await audio_queue.put(None)

        producer_task = asyncio.create_task(_produce_sentences(), name="llm-sentence-producer")
        synth_task = asyncio.create_task(_synth_sentences(), name="tts-synth-worker")
        try:
            while True:
                audio_chunk = await audio_queue.get()
                if audio_chunk is None:
                    break
                yield audio_chunk
        finally:
            for task in (producer_task, synth_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(producer_task, synth_task, return_exceptions=True)

        assistant_text = _sanitize_spoken_text("".join(assistant_reply_parts))
        session.history.append({"role": "assistant", "content": assistant_text})

    return active_session_id, reachy_actions, _generate_audio()


@app.post("/conversation")
async def conversation(request: Request) -> StreamingResponse:
    """Full ASR → LLM → TTS pipeline in a single streaming request.

    Send raw 16 kHz int16 PCM audio; receive 22050 Hz int16 PCM audio back.

    Pass ``X-Session-ID`` from a previous response to continue a conversation.
    The reply carries a new ``X-Session-ID`` header to use in the next turn.
    """
    if _state.funasr_status != "ready" or _state.funasr is None:
        raise HTTPException(503, detail=f"ASR model not ready: {_state.funasr_status}")
    if _state.tts_status != "ready" or (_tts_provider() == "cosyvoice" and _state.cosyvoice is None):
        raise HTTPException(503, detail=f"TTS model not ready: {_state.tts_status}")
    if _state.llm_client is None:
        raise HTTPException(503, detail="LLM client not initialised")

    body = await request.body()
    if not body:
        raise HTTPException(400, detail="Empty audio body")

    voice = _normalise_tts_voice(unquote(request.headers.get("X-Voice", "中文女")))
    audio_int16 = np.frombuffer(body, dtype=np.int16)

    user_text = await _transcribe_audio_int16(audio_int16)
    if not _is_meaningful_transcript(user_text):
        logger.info("ASR ignored low-content transcript: %r", user_text)
        return await _empty_pcm_response(user_text)
    logger.info("ASR → %r", user_text)
    session_id, reachy_actions, audio_stream = await _build_conversation_audio_stream(
        user_text,
        request.headers.get("X-Session-ID"),
        voice,
    )

    response_headers = {
        "X-Sample-Rate": str(_tts_sample_rate()),
        "X-Session-ID": session_id,
        "X-User-Text": quote(user_text),
    }
    if reachy_actions:
        response_headers["X-Reachy-Actions"] = quote(json.dumps(reachy_actions, ensure_ascii=True))

    return StreamingResponse(
        audio_stream,
        media_type="audio/pcm",
        headers=response_headers,
    )


@app.websocket("/conversation/ws")
async def conversation_stream(websocket: WebSocket) -> None:
    """Continuous PCM stream endpoint.

    The Reachy client sends raw 16 kHz int16 PCM binary frames. The server owns
    turn detection, ASR, LLM, TTS, and streams response PCM back on the same socket.
    """
    await websocket.accept()
    turn_detector = _StreamingTurnDetector()
    session_id: str | None = None
    voice = _normalise_tts_voice("中文女")

    await websocket.send_json({"type": "ready", "sample_rate": _tts_sample_rate()})

    try:
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                break
            if "text" in message and message["text"] is not None:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    continue
                if data.get("type") in {"start", "config"}:
                    raw_session_id = data.get("session_id")
                    if isinstance(raw_session_id, str) and raw_session_id.strip():
                        session_id = raw_session_id.strip()
                    raw_voice = data.get("voice")
                    if isinstance(raw_voice, str) and raw_voice.strip():
                        voice = _normalise_tts_voice(raw_voice.strip())
                elif data.get("type") == "reset":
                    session_id = None
                    turn_detector = _StreamingTurnDetector()
                continue

            raw_bytes = message.get("bytes")
            if raw_bytes is None:
                continue
            if _state.funasr_status != "ready" or _state.funasr is None:
                await websocket.send_json({"type": "error", "message": f"ASR model not ready: {_state.funasr_status}"})
                continue
            if _state.tts_status != "ready" or (_tts_provider() == "cosyvoice" and _state.cosyvoice is None):
                await websocket.send_json({"type": "error", "message": f"TTS model not ready: {_state.tts_status}"})
                continue
            if _state.llm_client is None:
                await websocket.send_json({"type": "error", "message": "LLM client not initialised"})
                continue

            pcm = np.frombuffer(raw_bytes, dtype=np.int16)
            utterance = turn_detector.push(pcm)
            if utterance is None:
                continue

            try:
                user_text = await _transcribe_audio_int16(utterance)
                if not _is_meaningful_transcript(user_text):
                    logger.info("Streaming ASR ignored low-content transcript: %r", user_text)
                    await websocket.send_json({"type": "ignored", "text": user_text})
                    continue
                logger.info("Streaming ASR → %r", user_text)
                session_id, reachy_actions, audio_stream = await _build_conversation_audio_stream(
                    user_text,
                    session_id,
                    voice,
                )
                await websocket.send_json(
                    {
                        "type": "user_text",
                        "text": user_text,
                        "session_id": session_id,
                        "actions": reachy_actions,
                    }
                )
                await websocket.send_json({"type": "audio_start", "sample_rate": _tts_sample_rate()})
                async for audio_chunk in audio_stream:
                    if audio_chunk:
                        await websocket.send_bytes(audio_chunk)
                await websocket.send_json({"type": "audio_end", "session_id": session_id})
            except HTTPException as exc:
                await websocket.send_json({"type": "error", "message": str(exc.detail)})
            except WebSocketDisconnect:
                raise
            except Exception as exc:
                logger.exception("Streaming conversation turn failed: %s", exc)
                await websocket.send_json({"type": "error", "message": str(exc)})
    except WebSocketDisconnect:
        logger.info("Streaming conversation websocket disconnected")


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    """Start the inference server (used by reachy-mini-inference-server script)."""
    parser = argparse.ArgumentParser(description="Reachy Mini ASR/TTS Inference Server")
    parser.add_argument("--host", default=config.INFERENCE_SERVER_HOST, help="Listen host")
    parser.add_argument("--port", type=int, default=config.INFERENCE_SERVER_PORT, help="Listen port")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    print(f"\n  Reachy Mini Inference Server")
    print(f"   Listening on           http://{args.host}:{args.port}")
    print(f"   Full pipeline:         POST /conversation  (ASR + LLM + TTS)")
    print(f"   Streaming pipeline:    WS   /conversation/ws  (continuous audio + server VAD)")
    print(f"   ASR only:              POST /asr")
    print(f"   TTS only:              POST /tts")
    print(f"\n   Pi 4 .env (full pipeline — recommended):")
    print(f"   BACKEND_PROVIDER=remote")
    print(f"   CONVERSATION_SERVICE_URL=http://<this-mac-ip>:{args.port}/conversation")
    print(f"\n   Pi 4 .env (split ASR+TTS — advanced):")
    print(f"   FUNASR_SERVICE_URL=http://<this-mac-ip>:{args.port}/asr")
    print(f"   TTS_SERVICE_URL=http://<this-mac-ip>:{args.port}/tts\n")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
