"""Fan-in for multiple StreamReader instances.

Round-robins across streams rather than batching, since no engine in this
repo has actually been exported with batch > 1 (see src/export/export_tensorrt.py) -
batching support is left as a documented seam, not implemented speculatively.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple

from .rtsp_reader import StreamHealth, StreamReader

logger = logging.getLogger(__name__)


@dataclass
class TaggedFrame:
    stream_id: str
    timestamp: float
    frame: "any"


class StreamManager:
    """Owns a set of StreamReaders and fans out frames tagged with stream ID."""

    def __init__(self, readers: List[StreamReader]):
        if not readers:
            raise ValueError("StreamManager needs at least one reader")
        self._readers = readers
        self._started = False
        self._frames_emitted = 0
        self._start_time: Optional[float] = None

    @classmethod
    def from_config(cls, streams_cfg: List[dict], streaming_cfg: dict) -> "StreamManager":
        readers = [
            StreamReader(
                stream_id=s["id"],
                source=s["source"],
                frame_skip=streaming_cfg.get("frame_skip", 0),
                reconnect_initial_backoff_s=streaming_cfg.get("reconnect_initial_backoff_s", 1.0),
                reconnect_max_backoff_s=streaming_cfg.get("reconnect_max_backoff_s", 30.0),
                reconnect_max_retries=streaming_cfg.get("reconnect_max_retries", 0),
            )
            for s in streams_cfg
        ]
        return cls(readers)

    def start(self) -> None:
        for r in self._readers:
            r.start()
        self._started = True
        self._start_time = time.monotonic()
        logger.info("StreamManager started %d stream(s)", len(self._readers))

    def stop(self) -> None:
        for r in self._readers:
            r.stop()
        self._started = False

    def poll(self, timeout_per_stream: float = 0.01) -> Iterator[TaggedFrame]:
        """One pass round-robin over all streams, yielding whatever frames are
        currently available. Does not block waiting for a specific stream -
        a stalled camera should not stall the others."""
        for reader in self._readers:
            item = reader.read(timeout=timeout_per_stream)
            if item is not None:
                ts, frame = item
                self._frames_emitted += 1
                yield TaggedFrame(stream_id=reader.stream_id, timestamp=ts, frame=frame)

    def health(self) -> Dict[str, StreamHealth]:
        return {r.stream_id: r.health() for r in self._readers}

    def aggregate_fps(self) -> float:
        if not self._start_time:
            return 0.0
        elapsed = time.monotonic() - self._start_time
        return self._frames_emitted / elapsed if elapsed > 0 else 0.0
