"""Tests for the headless console stream."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from reachy_mini.media.media_manager import MediaBackend
from reachy_mini_conversation_app.console import LocalStream


def test_clear_audio_queue_prefers_clear_player_when_available() -> None:
    """Local GStreamer audio should use the lower-level player flush when available."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    handler.output_queue.put_nowait("pending-audio")
    queue_ref = handler.output_queue
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.LOCAL))
    stream = LocalStream(handler, robot)
    stream._gstreamer_appsrc_pts_ns = 123

    stream.clear_audio_queue()

    audio.clear_player.assert_called_once()
    audio.clear_output_buffer.assert_not_called()
    assert handler.output_queue is queue_ref
    assert handler.output_queue.empty()
    assert stream._gstreamer_appsrc_pts_ns == 0


def test_clear_audio_queue_uses_output_buffer_for_webrtc() -> None:
    """WebRTC audio should flush queued playback via the output buffer API."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    handler.output_queue.put_nowait("pending-audio")
    queue_ref = handler.output_queue
    audio = SimpleNamespace(
        clear_player=MagicMock(),
        clear_output_buffer=MagicMock(),
    )
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)
    stream._gstreamer_appsrc_pts_ns = 123

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    audio.clear_player.assert_not_called()
    assert handler.output_queue is queue_ref
    assert handler.output_queue.empty()
    assert stream._gstreamer_appsrc_pts_ns == 0


def test_clear_audio_queue_falls_back_when_backend_is_unknown() -> None:
    """Unknown backends should still best-effort flush pending playback."""
    handler = MagicMock()
    handler.output_queue = asyncio.Queue()
    handler.output_queue.put_nowait("pending-audio")
    queue_ref = handler.output_queue
    audio = SimpleNamespace(clear_output_buffer=MagicMock())
    robot = SimpleNamespace(media=SimpleNamespace(audio=audio, backend=None))
    stream = LocalStream(handler, robot)
    stream._gstreamer_appsrc_pts_ns = 123

    stream.clear_audio_queue()

    audio.clear_output_buffer.assert_called_once()
    assert handler.output_queue is queue_ref
    assert handler.output_queue.empty()
    assert stream._gstreamer_appsrc_pts_ns == 0


def test_push_local_gstreamer_audio_sets_pts_and_duration(monkeypatch: pytest.MonkeyPatch) -> None:
    """Local GStreamer playback should timestamp buffers explicitly."""

    class FakeBuffer:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.pts = None
            self.duration = None

    class FakeGst:
        SECOND = 1_000_000_000

        class Buffer:
            @staticmethod
            def new_wrapped(payload: bytes) -> FakeBuffer:
                return FakeBuffer(payload)

    handler = MagicMock()
    appsrc = MagicMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_appsrc=appsrc), backend=MediaBackend.LOCAL))
    stream = LocalStream(handler, robot)
    monkeypatch.setattr("reachy_mini_conversation_app.console._load_gst", lambda: FakeGst)

    audio_frame = np.array([0.1, -0.1, 0.2, -0.2], dtype=np.float32)

    assert stream._push_local_gstreamer_audio(audio_frame, output_sample_rate=16000) is True

    pushed = appsrc.push_buffer.call_args.args[0]
    assert pushed.payload == audio_frame.tobytes()
    assert pushed.pts == 0
    assert pushed.duration == 250_000
    assert stream._gstreamer_appsrc_pts_ns == 250_000


def test_push_local_gstreamer_audio_accumulates_pts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Successive local playback buffers should advance the running playback clock."""

    class FakeBuffer:
        def __init__(self, payload: bytes) -> None:
            self.payload = payload
            self.pts = None
            self.duration = None

    class FakeGst:
        SECOND = 1_000_000_000

        class Buffer:
            @staticmethod
            def new_wrapped(payload: bytes) -> FakeBuffer:
                return FakeBuffer(payload)

    handler = MagicMock()
    appsrc = MagicMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_appsrc=appsrc), backend=MediaBackend.LOCAL))
    stream = LocalStream(handler, robot)
    monkeypatch.setattr("reachy_mini_conversation_app.console._load_gst", lambda: FakeGst)

    stream._push_local_gstreamer_audio(np.zeros(160, dtype=np.float32), output_sample_rate=16000)
    stream._push_local_gstreamer_audio(np.zeros(320, dtype=np.float32), output_sample_rate=16000)

    first = appsrc.push_buffer.call_args_list[0].args[0]
    second = appsrc.push_buffer.call_args_list[1].args[0]
    assert first.pts == 0
    assert first.duration == 10_000_000
    assert second.pts == 10_000_000
    assert second.duration == 20_000_000
    assert stream._gstreamer_appsrc_pts_ns == 30_000_000


def test_push_local_gstreamer_audio_returns_false_when_gst_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The workaround should fall back cleanly when GI bindings are unavailable."""
    handler = MagicMock()
    appsrc = MagicMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_appsrc=appsrc), backend=MediaBackend.LOCAL))
    stream = LocalStream(handler, robot)
    monkeypatch.setattr("reachy_mini_conversation_app.console._load_gst", lambda: None)

    assert stream._push_local_gstreamer_audio(np.zeros(160, dtype=np.float32), output_sample_rate=16000) is False
    appsrc.push_buffer.assert_not_called()


def test_push_local_gstreamer_audio_returns_false_for_non_local_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only local GStreamer playback should use explicit Gst timestamps."""
    handler = MagicMock()
    appsrc = MagicMock()
    robot = SimpleNamespace(media=SimpleNamespace(audio=SimpleNamespace(_appsrc=appsrc), backend=MediaBackend.WEBRTC))
    stream = LocalStream(handler, robot)
    monkeypatch.setattr("reachy_mini_conversation_app.console._load_gst", MagicMock())

    assert stream._push_local_gstreamer_audio(np.zeros(160, dtype=np.float32), output_sample_rate=16000) is False
    appsrc.push_buffer.assert_not_called()
