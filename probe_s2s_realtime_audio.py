#!/usr/bin/env python3
"""Mic/speaker realtime probe for the speech-to-speech backend.

Uses the same resolved system prompt and voice as the conversation app, but
talks to the realtime endpoint directly from the local machine instead of the
robot.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import signal
import sys
import threading
import time
import wave
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

try:
    import websockets
except ImportError:  # pragma: no cover - runtime dependency
    websockets = None  # type: ignore[assignment]

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - runtime dependency
    sd = None


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from reachy_mini_conversation_app.config import DEFAULT_VOICE, config
from reachy_mini_conversation_app.prompts import get_session_instructions, get_session_voice
from reachy_mini_conversation_app.tools.core_tools import get_tool_specs


SAMPLE_WIDTH = 2


@dataclass
class ProbeArguments:
    session_url: Optional[str] = None
    ws_url: Optional[str] = None
    send_rate: int = 16000
    recv_rate: int = 16000
    chunk_size: int = 1024
    channels: int = 1
    input_device: Optional[str] = None
    output_device: Optional[str] = None
    authorization: Optional[str] = None
    save_output: Optional[str] = None
    allow_barge_in: bool = False
    instructions: Optional[str] = None
    print_json: bool = False
    show_prompt: bool = False
    show_session_config: bool = False
    list_devices: bool = False


@dataclass
class AllocatedSession:
    session_id: str
    websocket_url: str
    connect_url: str


class PlaybackBuffer:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self._lock = threading.Lock()

    def append(self, data: bytes) -> None:
        with self._lock:
            self._buffer.extend(data)

    def has_data(self) -> bool:
        with self._lock:
            return bool(self._buffer)

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()

    def read(self, size: int) -> bytes:
        with self._lock:
            if not self._buffer:
                return b"\x00" * size

            data = bytes(self._buffer[:size])
            del self._buffer[:size]

        if len(data) < size:
            data += b"\x00" * (size - len(data))
        return data


def parse_args() -> ProbeArguments:
    parser = argparse.ArgumentParser(
        description="Mic/speaker realtime probe for the speech-to-speech endpoint.",
    )
    parser.add_argument(
        "--session-url",
        default=os.getenv("S2S_REALTIME_SESSION_URL") or getattr(config, "S2S_REALTIME_SESSION_URL", None),
        help=(
            "Probe-only session allocation URL. Defaults to S2S_REALTIME_SESSION_URL if set for this probe, "
            "otherwise the app's built-in Hugging Face allocator URL."
        ),
    )
    parser.add_argument(
        "--ws-url",
        help="Direct websocket URL to connect to. If set, skips session allocation.",
    )
    parser.add_argument("--send-rate", type=int, default=16000)
    parser.add_argument("--recv-rate", type=int, default=16000)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--channels", type=int, default=1)
    parser.add_argument("--input-device")
    parser.add_argument("--output-device")
    parser.add_argument(
        "--authorization",
        default=os.getenv("S2S_AUTHORIZATION"),
        help="Optional Authorization header value for the session allocator.",
    )
    parser.add_argument("--save-output")
    parser.add_argument("--allow-barge-in", action="store_true")
    parser.add_argument(
        "--instructions",
        help="Override the resolved system prompt. By default the app prompt is used.",
    )
    parser.add_argument("--print-json", action="store_true")
    parser.add_argument("--show-prompt", action="store_true")
    parser.add_argument(
        "--show-session-config",
        action="store_true",
        help="Print the exact session.update payload the probe sends.",
    )
    parser.add_argument("--list-devices", action="store_true")
    namespace = parser.parse_args()
    return ProbeArguments(**vars(namespace))


def add_model_query_param(ws_url: str) -> str:
    """Mirror the conversation app's realtime connect query.

    The session allocator returns a `connect_url` with the session token, while
    the app also adds `model=<MODEL_NAME>` when opening the realtime
    websocket. Some deployed backends require that query parameter for
    `session.update` to be accepted.
    """
    parsed = urlsplit(ws_url)
    query_items = dict(parse_qsl(parsed.query, keep_blank_values=True))
    query_items.setdefault("model", config.MODEL_NAME)
    return urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            urlencode(query_items),
            parsed.fragment,
        )
    )


def require_runtime_dependencies() -> None:
    if websockets is None:
        raise SystemExit("websockets is required. Install it with `pip install websockets`.")
    if sd is None:
        raise SystemExit(
            "sounddevice is required for microphone/speaker probing. Install it with `pip install sounddevice`."
        )


async def allocate_session(session_url: str, authorization: str | None) -> AllocatedSession:
    headers = {}
    if authorization:
        headers["Authorization"] = authorization

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(session_url, headers=headers)
        response.raise_for_status()
        payload = response.json()

    connect_url = str(payload.get("connect_url") or "").strip()
    websocket_url = str(payload.get("websocket_url") or "").strip()
    session_id = str(payload.get("session_id") or "").strip()
    if not connect_url:
        raise RuntimeError(f"Session allocator response missing connect_url: {payload!r}")

    return AllocatedSession(
        session_id=session_id,
        websocket_url=websocket_url,
        connect_url=connect_url,
    )


def decode_output_audio_delta(event: dict[str, object]) -> bytes:
    delta = str(event.get("delta") or "").strip()
    if not delta:
        return b""
    return base64.b64decode(delta)


def _maybe_pcm_format(rate: int) -> dict[str, object]:
    if rate not in (16000, 24000):
        raise ValueError("Only 16000 and 24000 Hz are supported.")
    return {"type": "audio/pcm", "rate": rate}


def _session_audio_format_for_provider(rate: int) -> dict[str, object] | None:
    """Return a session.update format block, or None to omit it.

    The deployed speech-to-speech backend rejects session.update payloads that
    explicitly set audio/pcm at 16 kHz. Omitting the format lets the backend
    keep its default 16 kHz realtime session configuration.
    """
    if config.BACKEND_PROVIDER == "speech-to-speech" and rate == 16000:
        return None
    return _maybe_pcm_format(rate)


def build_session_update_payload(args: ProbeArguments, instructions: str, voice: str) -> dict[str, object]:
    input_audio: dict[str, object] = {
        "transcription": {"model": "gpt-4o-transcribe", "language": "en"},
        "turn_detection": {"type": "server_vad", "interrupt_response": True},
    }
    input_format = _session_audio_format_for_provider(args.send_rate)
    if input_format is not None:
        input_audio["format"] = input_format

    output_audio: dict[str, object] = {
        "voice": voice,
    }
    output_format = _session_audio_format_for_provider(args.recv_rate)
    if output_format is not None:
        output_audio["format"] = output_format

    return {
        "type": "session.update",
        "session": {
            "type": "realtime",
            "instructions": instructions,
            "audio": {
                "input": input_audio,
                "output": output_audio,
            },
            "tools": get_tool_specs(),
            "tool_choice": "auto",
        },
    }


def build_session_update_event(args: ProbeArguments, instructions: str, voice: str) -> str:
    return json.dumps(build_session_update_payload(args, instructions, voice))


def build_input_audio_append_event(chunk: bytes) -> str:
    return json.dumps(
        {
            "type": "input_audio_buffer.append",
            "audio": base64.b64encode(chunk).decode("ascii"),
        }
    )


def parse_realtime_event(message: str) -> dict[str, object]:
    payload = json.loads(message)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object event, got {type(payload).__name__}")
    return payload


def write_wav_pcm16(path: str, pcm_bytes: bytes, sample_rate: int, channels: int) -> None:
    output_path = Path(path)
    with wave.open(str(output_path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)


def _render_live_user_text(state: dict[str, object], text: str, *, final: bool = False) -> None:
    line = f"USER: {text}"
    width_value = state.get("width", 0)
    width = width_value if isinstance(width_value, int) else 0
    padded = line + (" " * max(0, width - len(line)))
    if final:
        print(f"\r{padded}", flush=True)
        state["width"] = 0
        return
    print(f"\r{padded}", end="", flush=True)
    state["width"] = len(line)


def _clear_live_user_text(state: dict[str, object]) -> None:
    width_value = state.get("width", 0)
    width = width_value if isinstance(width_value, int) else 0
    if width == 0:
        return
    print("\r" + (" " * width) + "\r", end="", flush=True)
    state["width"] = 0


def handle_realtime_event(
    message: str,
    playback: PlaybackBuffer,
    received_audio: bytearray,
    *,
    recv_rate: int,
    speaker_active_until: list[float],
    partial_user_text: dict[str, object],
    timing: dict[str, float | None],
    print_json: bool,
) -> None:
    try:
        event = parse_realtime_event(message)
    except Exception as exc:
        print(f"Invalid realtime event: {exc}: {message}", file=sys.stderr)
        return

    event_type = str(event.get("type") or "").strip()

    if print_json and event_type != "response.output_audio.delta":
        print(json.dumps(event, ensure_ascii=True))

    if event_type == "session.created":
        print("Realtime session created")
        return

    if event_type == "input_audio_buffer.speech_started":
        playback.clear()
        speaker_active_until[0] = 0.0
        partial_user_text["value"] = ""
        partial_user_text["saw_user_speech"] = True
        timing["turn_started_at"] = time.perf_counter()
        timing["user_done_at"] = None
        timing["response_created_at"] = None
        timing["first_audio_at"] = None
        return

    if event_type == "conversation.item.input_audio_transcription.delta":
        transcript = str(event.get("delta") or "").strip()
        partial_user_text["value"] = transcript
        if transcript:
            _render_live_user_text(partial_user_text, transcript)
        return

    if event_type == "conversation.item.input_audio_transcription.completed":
        transcript = str(event.get("transcript") or "").strip()
        partial_user_text["value"] = ""
        timing["user_done_at"] = time.perf_counter()
        if transcript:
            _render_live_user_text(partial_user_text, transcript, final=True)
        return

    if event_type == "response.created":
        _clear_live_user_text(partial_user_text)
        response_created_at = time.perf_counter()
        timing["response_created_at"] = response_created_at
        user_done_at = timing.get("user_done_at")
        if user_done_at is not None:
            delta_ms = (response_created_at - user_done_at) * 1000
            print(f"response.created: {delta_ms:.0f} ms", flush=True)
        return

    if event_type == "response.output_audio.delta":
        try:
            audio_chunk = decode_output_audio_delta(event)
        except Exception as exc:
            print(f"Failed to decode audio delta: {exc}", file=sys.stderr)
            return
        if audio_chunk:
            if timing.get("first_audio_at") is None:
                first_audio_at = time.perf_counter()
                timing["first_audio_at"] = first_audio_at
                user_done_at = timing.get("user_done_at")
                if user_done_at is not None:
                    delta_ms = (first_audio_at - user_done_at) * 1000
                    print(f"first audio delta: {delta_ms:.0f} ms", flush=True)
            playback.append(audio_chunk)
            received_audio.extend(audio_chunk)
            speaker_active_until[0] = time.monotonic() + max(0.15, len(audio_chunk) / (2 * recv_rate))
        return

    if event_type == "response.output_audio_transcript.done":
        transcript = str(event.get("transcript") or "").strip()
        if transcript:
            print(f"ASSISTANT: {transcript}", flush=True)
        return

    if event_type == "response.function_call_arguments.done":
        print(
            "TOOL: "
            f"{str(event.get('name') or '').strip()} "
            f"call_id={str(event.get('call_id') or '').strip()} "
            f"arguments={event.get('arguments')}",
            flush=True,
        )
        return

    if event_type == "error":
        _clear_live_user_text(partial_user_text)
        print("ERROR:", json.dumps(event, ensure_ascii=True), file=sys.stderr)
        return


async def listen_and_play_ws(args: ProbeArguments) -> None:
    require_runtime_dependencies()

    if args.list_devices:
        print(sd.query_devices())
        return

    instructions = args.instructions or get_session_instructions()
    voice = get_session_voice(default=DEFAULT_VOICE)

    if args.show_prompt:
        print("=== Prompt ===")
        print(instructions)
        print("=== End Prompt ===")
        print()

    if args.show_session_config:
        print("=== Session Update Payload ===")
        print(json.dumps(build_session_update_payload(args, instructions, voice), indent=2, ensure_ascii=True))
        print("=== End Session Update Payload ===")
        print()

    playback = PlaybackBuffer()
    received_audio = bytearray()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    mic_queue: asyncio.Queue[Optional[bytes]] = asyncio.Queue(maxsize=8)
    speaker_active_until = [0.0]
    partial_user_text: dict[str, object] = {"value": "", "width": 0, "saw_user_speech": False}
    timing: dict[str, float | None] = {
        "turn_started_at": None,
        "user_done_at": None,
        "response_created_at": None,
        "first_audio_at": None,
    }

    def queue_microphone_frame(frame: bytes) -> None:
        if stop_event.is_set():
            return
        if mic_queue.full():
            with suppress(asyncio.QueueEmpty):
                mic_queue.get_nowait()
        with suppress(asyncio.QueueFull):
            mic_queue.put_nowait(frame)

    def request_stop() -> None:
        if stop_event.is_set():
            return
        stop_event.set()
        with suppress(asyncio.QueueFull):
            mic_queue.put_nowait(None)

    def input_callback(indata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print(f"Input stream status: {status}", file=sys.stderr)
        if stop_event.is_set():
            return
        if not args.allow_barge_in and (playback.has_data() or time.monotonic() < speaker_active_until[0]):
            return
        loop.call_soon_threadsafe(queue_microphone_frame, bytes(indata))

    def output_callback(outdata, frames, time_info, status) -> None:  # type: ignore[no-untyped-def]
        if status:
            print(f"Output stream status: {status}", file=sys.stderr)
        outdata[:] = playback.read(len(outdata))

    def install_signal_handlers() -> None:
        for sig in (signal.SIGINT, signal.SIGTERM):
            with suppress(NotImplementedError):
                loop.add_signal_handler(sig, request_stop)

    async def send_audio(ws) -> None:  # type: ignore[no-untyped-def]
        try:
            while not stop_event.is_set():
                chunk = await mic_queue.get()
                if chunk is None:
                    break
                await ws.send(build_input_audio_append_event(chunk))
        except Exception:
            request_stop()

    async def receive_audio(ws) -> None:  # type: ignore[no-untyped-def]
        try:
            while not stop_event.is_set():
                msg = await ws.recv()
                if isinstance(msg, bytes):
                    playback.append(msg)
                    received_audio.extend(msg)
                else:
                    handle_realtime_event(
                        msg,
                        playback,
                        received_audio,
                        recv_rate=args.recv_rate,
                        speaker_active_until=speaker_active_until,
                        partial_user_text=partial_user_text,
                        timing=timing,
                        print_json=args.print_json,
                    )
        except Exception:
            request_stop()

    async def wait_for_user_stop() -> None:
        try:
            await asyncio.to_thread(input, "Streaming microphone audio. Press Enter to stop.\n")
        except EOFError:
            pass
        request_stop()

    ws_url = args.ws_url
    headers = {}
    if args.authorization:
        headers["Authorization"] = args.authorization

    allocated_session: Optional[AllocatedSession] = None
    if not ws_url:
        if not args.session_url:
            raise SystemExit("Pass --session-url or --ws-url.")
        alloc_start = time.perf_counter()
        allocated_session = await allocate_session(args.session_url, args.authorization)
        alloc_ms = (time.perf_counter() - alloc_start) * 1000
        ws_url = allocated_session.connect_url
        print(f"allocated session {allocated_session.session_id or '<unknown>'} in {alloc_ms:.0f} ms")
        if allocated_session.websocket_url:
            print(f"direct compute websocket: {allocated_session.websocket_url}")
        headers = {}

    ws_url = add_model_query_param(ws_url)

    install_signal_handlers()
    print(f"provider: {config.BACKEND_PROVIDER}")
    print(f"model: {config.MODEL_NAME}")
    print(f"voice: {voice}")
    print(f"prompt_chars: {len(instructions)}")
    print(f"tool_count: {len(get_tool_specs())}")

    try:
        async with websockets.connect(
            ws_url,
            additional_headers=headers or None,
            max_size=None,
            ping_interval=20,
            ping_timeout=20,
        ) as ws:
            print(f"Connected to {ws_url}")
            await ws.send(build_session_update_event(args, instructions, voice))

            with (
                sd.RawInputStream(
                    samplerate=args.send_rate,
                    channels=args.channels,
                    dtype="int16",
                    blocksize=args.chunk_size,
                    device=args.input_device,
                    callback=input_callback,
                ),
                sd.RawOutputStream(
                    samplerate=args.recv_rate,
                    channels=args.channels,
                    dtype="int16",
                    blocksize=args.chunk_size,
                    device=args.output_device,
                    callback=output_callback,
                ),
            ):
                tasks = [
                    asyncio.create_task(send_audio(ws)),
                    asyncio.create_task(receive_audio(ws)),
                    asyncio.create_task(wait_for_user_stop()),
                ]

                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                request_stop()

                for task in pending:
                    task.cancel()
                for task in pending:
                    with suppress(asyncio.CancelledError):
                        await task
                for task in done:
                    with suppress(asyncio.CancelledError):
                        task.result()
    except websockets.InvalidStatus as exc:
        raise SystemExit(f"Websocket rejected by {ws_url}: {exc}") from exc
    except websockets.ConnectionClosedError as exc:
        raise SystemExit(f"Websocket connection closed unexpectedly: {exc}") from exc

    if args.save_output:
        write_wav_pcm16(args.save_output, bytes(received_audio), args.recv_rate, args.channels)
        print(f"Wrote {args.save_output}")


def main() -> None:
    args = parse_args()
    asyncio.run(listen_and_play_ws(args))


if __name__ == "__main__":
    main()
