"""SORT tracking + confidence-gated OCR.

OCR was the profiled bottleneck on the original Jetson deployment (per the
brief this scaffolds), not detection - so the optimization here targets OCR
*call count*, not detection speed. Once a track's OCR read clears
`gate_confidence`, the result is cached against that track ID and OCR is
skipped for it on every subsequent frame. A single trusted read persisting
for the life of a track is a deliberate tradeoff (see README limitations):
it's cheap, but a confident misread is never revisited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .sort import Sort

logger = logging.getLogger(__name__)

# (frame, bbox_xyxy) -> (plate_text, score), or None if nothing readable
OcrFn = Callable[[np.ndarray, np.ndarray], Optional[Tuple[str, float]]]


@dataclass
class TrackState:
    track_id: int
    cls: int
    first_seen_frame: int
    last_seen_frame: int
    bbox_history: List[np.ndarray] = field(default_factory=list)
    plate_text: Optional[str] = None
    plate_score: float = 0.0
    locked: bool = False


class GatedOcrTracker:
    """Wraps Sort with per-track OCR state and the confidence gate."""

    def __init__(self, ocr_fn: OcrFn, gate_confidence: float = 0.95,
                 max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        self._sort = Sort(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self._ocr_fn = ocr_fn
        self._gate_confidence = gate_confidence
        self._tracks: Dict[int, TrackState] = {}
        self._frame_idx = 0
        self._ocr_calls = 0
        self._ocr_calls_saved = 0
        self._known_ids: set = set()

    def update(self, boxes_xyxy: np.ndarray, classes: np.ndarray, frame: np.ndarray) -> List[TrackState]:
        self._frame_idx += 1
        confirmed = self._sort.update(boxes_xyxy, classes)

        current_ids = {t["id"] for t in confirmed}
        self._log_id_switches(current_ids)
        self._known_ids = current_ids

        out = []
        for t in confirmed:
            tid = t["id"]
            state = self._tracks.get(tid)
            if state is None:
                state = TrackState(
                    track_id=tid, cls=t["cls"],
                    first_seen_frame=self._frame_idx, last_seen_frame=self._frame_idx,
                )
                self._tracks[tid] = state

            state.last_seen_frame = self._frame_idx
            state.bbox_history.append(t["bbox"])

            if not state.locked:
                self._run_ocr_gated(state, frame, t["bbox"])
            else:
                self._ocr_calls_saved += 1

            out.append(state)

        return out

    def _run_ocr_gated(self, state: TrackState, frame: np.ndarray, bbox: np.ndarray) -> None:
        try:
            result = self._ocr_fn(frame, bbox)
        except Exception:
            logger.exception("OCR call failed for track %d; leaving track unlocked", state.track_id)
            return

        self._ocr_calls += 1
        if result is None:
            return

        text, score = result
        if score > state.plate_score:
            state.plate_text = text
            state.plate_score = score

        if score >= self._gate_confidence:
            state.locked = True
            logger.debug("track %d locked plate=%r score=%.3f after %d OCR call(s)",
                         state.track_id, text, score, self._frame_idx - state.first_seen_frame + 1)

    def _log_id_switches(self, current_ids: set) -> None:
        vanished = self._known_ids - current_ids
        for tid in vanished:
            state = self._tracks.get(tid)
            if state is not None and not state.locked:
                logger.info("track %d lost before OCR locked (last seen frame %d)",
                            tid, state.last_seen_frame)

    def ocr_savings_pct(self) -> float:
        total = self._ocr_calls + self._ocr_calls_saved
        return 100.0 * self._ocr_calls_saved / total if total else 0.0

    def stats(self) -> dict:
        return {
            "ocr_calls": self._ocr_calls,
            "ocr_calls_saved": self._ocr_calls_saved,
            "ocr_savings_pct": round(self.ocr_savings_pct(), 1),
            "active_tracks": len(self._tracks),
        }
