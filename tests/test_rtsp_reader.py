import queue
import time

import numpy as np
import pytest

from src.streaming.rtsp_reader import StreamReader


def make_reader(**kwargs) -> StreamReader:
    return StreamReader(stream_id="test", source="unused", **kwargs)


def test_latest_frame_wins_drops_stale_frame():
    reader = make_reader()
    frame_a = np.zeros((2, 2), dtype="uint8")
    frame_b = np.ones((2, 2), dtype="uint8")

    reader._push_frame(frame_a)
    reader._push_frame(frame_b)  # queue depth is 1: this should evict frame_a

    ts, frame = reader.read(timeout=0.1)
    np.testing.assert_array_equal(frame, frame_b)

    health = reader.health()
    assert health.frames_received == 2
    assert health.frames_dropped == 1


def test_read_returns_none_when_empty():
    reader = make_reader()
    assert reader.read(timeout=0.05) is None


def test_health_tracks_last_frame_timestamp():
    reader = make_reader()
    before = time.monotonic()
    reader._push_frame(np.zeros((1, 1), dtype="uint8"))
    health = reader.health()
    assert health.last_frame_ts is not None
    assert health.last_frame_ts >= before
    assert health.frames_received == 1


class _FakeCapture:
    """Minimal cv2.VideoCapture stand-in: N real frames, then EOF."""

    def __init__(self, n_frames: int):
        self._remaining = n_frames
        self._opened = True

    def isOpened(self):
        return self._opened

    def read(self):
        if self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((2, 2), dtype="uint8")

    def release(self):
        self._opened = False


def test_frame_skip_keeps_one_of_every_n(monkeypatch):
    reader = make_reader(frame_skip=1)  # keep 1 of every 2 frames
    monkeypatch.setattr(reader, "_open_capture", lambda: _FakeCapture(10))

    reader.start()
    time.sleep(0.3)
    reader.stop()

    # 10 source frames, frame_skip=1 -> keep frames 2,4,6,8,10 -> 5 pushed
    assert reader.health().frames_received == 5


class _FlakyThenGoodCapture:
    """Fails isOpened() the first `fail_calls_remaining` times, then opens once
    and delivers exactly 3 frames, then fails to open forever after - models a
    camera that reconnects after some trouble, streams briefly, then dies for
    good, so the test has a deterministic total frame count to assert on."""

    _fail_calls_remaining = 0
    _already_succeeded = False

    def __init__(self):
        cls = _FlakyThenGoodCapture
        if cls._already_succeeded:
            self._opened = False
            self._remaining = 0
            return
        if cls._fail_calls_remaining > 0:
            cls._fail_calls_remaining -= 1
            self._opened = False
            self._remaining = 0
        else:
            self._opened = True
            self._remaining = 3
            cls._already_succeeded = True

    def isOpened(self):
        return self._opened

    def read(self):
        if not self._opened or self._remaining <= 0:
            return False, None
        self._remaining -= 1
        return True, np.zeros((2, 2), dtype="uint8")

    def release(self):
        pass


def test_reconnect_after_open_failures(monkeypatch):
    _FlakyThenGoodCapture._fail_calls_remaining = 2
    _FlakyThenGoodCapture._already_succeeded = False
    reader = make_reader(reconnect_initial_backoff_s=0.01, reconnect_max_backoff_s=0.02)
    monkeypatch.setattr(reader, "_open_capture", lambda: _FlakyThenGoodCapture())

    reader.start()
    time.sleep(0.5)
    reader.stop()

    health = reader.health()
    # 2 initial open failures, one successful connection, then permanent
    # failure to reopen - reconnect_count keeps climbing on every failed
    # reopen attempt, so only the lower bound is deterministic.
    assert health.reconnect_count >= 2
    assert health.frames_received == 3  # exactly the one successful connection's frames
