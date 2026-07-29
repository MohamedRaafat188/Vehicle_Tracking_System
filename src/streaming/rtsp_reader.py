"""Background video capture with a latest-frame-wins buffer.

Speculative deployment-layer scaffold: written to run against RTSP cameras,
local video files, or webcams, but only exercised in this repo against local
files and the simulated RTSP setup in scripts/simulate_cameras.sh. Not run
against real IP cameras.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Optional, Union

import cv2

logger = logging.getLogger(__name__)


@dataclass
class StreamHealth:
    """Point-in-time health snapshot for a single stream."""
    stream_id: str
    connected: bool = False
    frames_received: int = 0
    frames_dropped: int = 0
    reconnect_count: int = 0
    measured_fps: float = 0.0
    last_frame_ts: Optional[float] = None


class StreamReader:
    """Reads one video source on a background thread.

    The main loop never blocks on cap.read(): frames land in a depth-1 queue,
    and a new frame overwrites (drops) whatever stale frame was waiting to be
    consumed, rather than queuing up and accumulating latency. For a live
    camera, a frame that's a second old is worse than no frame at all, so
    "freshest frame available" beats "every frame, eventually" here.
    """

    def __init__(
        self,
        stream_id: str,
        source: Union[str, int],
        frame_skip: int = 0,
        reconnect_initial_backoff_s: float = 1.0,
        reconnect_max_backoff_s: float = 30.0,
        reconnect_max_retries: int = 0,
    ):
        self.stream_id = stream_id
        self.source = source
        self.frame_skip = max(0, frame_skip)
        self.reconnect_initial_backoff_s = reconnect_initial_backoff_s
        self.reconnect_max_backoff_s = reconnect_max_backoff_s
        self.reconnect_max_retries = reconnect_max_retries

        self._queue: "queue.Queue[tuple[float, any]]" = queue.Queue(maxsize=1)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._health = StreamHealth(stream_id=stream_id)
        self._health_lock = threading.Lock()
        self._fps_window_start = time.monotonic()
        self._fps_window_count = 0

    def start(self) -> "StreamReader":
        self._thread = threading.Thread(
            target=self._run, name=f"StreamReader-{self.stream_id}", daemon=True
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    def read(self, timeout: Optional[float] = None) -> Optional["tuple[float, any]"]:
        """Return (timestamp, frame) for the freshest available frame, or None."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def health(self) -> StreamHealth:
        with self._health_lock:
            return StreamHealth(**self._health.__dict__)

    def _push_frame(self, frame) -> None:
        ts = time.monotonic()
        # Drop the stale frame instead of blocking: a full queue means the
        # consumer hasn't kept up, so the old frame is already outdated.
        if self._queue.full():
            try:
                self._queue.get_nowait()
                with self._health_lock:
                    self._health.frames_dropped += 1
            except queue.Empty:
                pass
        self._queue.put((ts, frame))

        with self._health_lock:
            self._health.frames_received += 1
            self._health.last_frame_ts = ts
            self._fps_window_count += 1
            elapsed = ts - self._fps_window_start
            if elapsed >= 1.0:
                self._health.measured_fps = self._fps_window_count / elapsed
                self._fps_window_count = 0
                self._fps_window_start = ts

    def _open_capture(self):
        source = self.source
        # Allow webcam indices to be passed as strings in config ("0" -> 0).
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        return cv2.VideoCapture(source)

    def _run(self) -> None:
        backoff = self.reconnect_initial_backoff_s
        retries = 0
        frame_counter = 0

        while not self._stop_event.is_set():
            cap = self._open_capture()
            if not cap.isOpened():
                cap.release()
                with self._health_lock:
                    self._health.connected = False
                    self._health.reconnect_count += 1
                logger.warning(
                    "[%s] failed to open source=%r, retrying in %.1fs",
                    self.stream_id, self.source, backoff,
                )
                if self._stop_event.wait(backoff):
                    break
                retries += 1
                if self.reconnect_max_retries and retries >= self.reconnect_max_retries:
                    logger.error("[%s] giving up after %d retries", self.stream_id, retries)
                    return
                backoff = min(backoff * 2, self.reconnect_max_backoff_s)
                continue

            with self._health_lock:
                self._health.connected = True
            logger.info("[%s] connected to %r", self.stream_id, self.source)
            frames_this_connection = 0

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    logger.warning("[%s] stream dropped, reconnecting", self.stream_id)
                    with self._health_lock:
                        self._health.connected = False
                    break

                frames_this_connection += 1
                frame_counter += 1
                if self.frame_skip and (frame_counter % (self.frame_skip + 1)):
                    continue

                self._push_frame(frame)

            cap.release()

            # A connection that delivered at least one frame was healthy, so the
            # next reconnect attempt starts fresh at the initial backoff. A
            # connection that opened but produced nothing (e.g. a camera that
            # accepts the connection then immediately drops it) is treated like
            # an open failure and escalates backoff instead of hammering it -
            # without this, a source that opens-then-instantly-EOFs (any finite
            # video file, or a flapping camera) reconnects in a tight loop with
            # no delay at all.
            with self._health_lock:
                self._health.reconnect_count += 1
            if frames_this_connection > 0:
                backoff = self.reconnect_initial_backoff_s
                retries = 0
            else:
                retries += 1
                if self.reconnect_max_retries and retries >= self.reconnect_max_retries:
                    logger.error("[%s] giving up after %d retries", self.stream_id, retries)
                    return
                backoff = min(backoff * 2, self.reconnect_max_backoff_s)

            if self._stop_event.wait(backoff):
                break

        with self._health_lock:
            self._health.connected = False


def make_reader(stream_id: str, source: Union[str, int], **kwargs) -> StreamReader:
    """Factory kept separate from StreamReader.__init__ so callers building
    many readers from a config list don't need to know the class name."""
    return StreamReader(stream_id=stream_id, source=source, **kwargs)
