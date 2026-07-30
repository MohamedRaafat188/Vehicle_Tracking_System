"""Ultralytics tracker (ByteTrack / BoT-SORT) + confidence-gated OCR.

Tracking is delegated to Ultralytics' own trackers instead of a from-scratch
SORT: ByteTrack by default (motion-only association, no re-ID network, so OCR
stays the bottleneck rather than detection), with BoT-SORT selectable via the
tracker config for the occlusion-heavy case where appearance re-ID is worth
the extra compute.

The OCR gate is the point of this module and is unchanged: once a track's OCR
read clears `gate_confidence`, the result is cached against that track ID and
OCR is skipped for it on every subsequent frame. A single trusted read
persisting for the life of a track is a deliberate tradeoff (see README
limitations): it's cheap, but a confident misread is never revisited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from ultralytics.trackers import BOTSORT, BYTETracker
from ultralytics.utils import IterableSimpleNamespace, yaml_load
from ultralytics.utils.checks import check_yaml

logger = logging.getLogger(__name__)

# (frame, bbox_xyxy) -> (plate_text, score), or None if nothing readable
OcrFn = Callable[[np.ndarray, np.ndarray], Optional[Tuple[str, float]]]

_TRACKER_TYPES = {"bytetrack": BYTETracker, "botsort": BOTSORT}


class _Detections:
    """Minimal detections view in the shape Ultralytics' trackers expect.

    Their `update(results, img)` reads `results.conf`, `results.xywh` and
    `results.cls` as numpy arrays; this adapts our (xyxy, score, cls) arrays
    without pulling in a full ultralytics Results/Boxes object.
    """

    def __init__(self, boxes_xyxy: np.ndarray, scores: np.ndarray, classes: np.ndarray):
        boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(scores, dtype=np.float32).reshape(-1)
        self.cls = np.asarray(classes, dtype=np.float32).reshape(-1)
        w = boxes_xyxy[:, 2] - boxes_xyxy[:, 0]
        h = boxes_xyxy[:, 3] - boxes_xyxy[:, 1]
        cx = boxes_xyxy[:, 0] + w / 2.0
        cy = boxes_xyxy[:, 1] + h / 2.0
        self.xywh = np.stack([cx, cy, w, h], axis=1)

    def __len__(self) -> int:
        return len(self.conf)


def _build_tracker(tracker_type: str, frame_rate: int, overrides: Optional[dict]):
    """Load the shipped bytetrack.yaml / botsort.yaml, apply overrides, and
    construct the matching Ultralytics tracker."""
    cfg = yaml_load(check_yaml(tracker_type))
    if overrides:
        cfg.update(overrides)
    name = cfg.get("tracker_type")
    if name not in _TRACKER_TYPES:
        raise ValueError(
            f"unsupported tracker_type {name!r}; expected one of {sorted(_TRACKER_TYPES)}"
        )
    return _TRACKER_TYPES[name](IterableSimpleNamespace(**cfg), frame_rate=frame_rate)


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
    """Wraps an Ultralytics tracker with per-track OCR state and the gate."""

    def __init__(self, ocr_fn: OcrFn, gate_confidence: float = 0.95,
                 tracker_type: str = "bytetrack.yaml", frame_rate: int = 30,
                 overrides: Optional[dict] = None):
        self._tracker = _build_tracker(tracker_type, frame_rate, overrides)
        self._ocr_fn = ocr_fn
        self._gate_confidence = gate_confidence
        self._tracks: Dict[int, TrackState] = {}
        self._frame_idx = 0
        self._ocr_calls = 0
        self._ocr_calls_saved = 0
        self._known_ids: set = set()

    def update(self, boxes_xyxy: np.ndarray, scores: np.ndarray,
               classes: np.ndarray, frame: np.ndarray) -> List[TrackState]:
        """boxes_xyxy: Nx4 [x1,y1,x2,y2]. scores: N det confidences.
        classes: N class ids, same order. frame: the image (for GMC/re-ID)."""
        self._frame_idx += 1
        dets = _Detections(boxes_xyxy, scores, classes)
        # rows: [x1, y1, x2, y2, track_id, score, cls, det_idx]
        tracks = self._tracker.update(dets, frame)

        current_ids = {int(row[4]) for row in tracks}
        self._log_id_switches(current_ids)
        self._known_ids = current_ids

        out = []
        for row in tracks:
            tid = int(row[4])
            bbox = np.asarray(row[:4], dtype=np.float32)
            cls = int(row[6])

            state = self._tracks.get(tid)
            if state is None:
                state = TrackState(
                    track_id=tid, cls=cls,
                    first_seen_frame=self._frame_idx, last_seen_frame=self._frame_idx,
                )
                self._tracks[tid] = state

            state.last_seen_frame = self._frame_idx
            state.bbox_history.append(bbox)

            if not state.locked:
                self._run_ocr_gated(state, frame, bbox)
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
