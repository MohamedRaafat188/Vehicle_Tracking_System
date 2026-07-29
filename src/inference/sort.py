"""SORT: Simple Online and Realtime Tracking (Bewley et al., 2016).

A compact reimplementation - constant-velocity Kalman filter per track plus
Hungarian assignment on IoU. No appearance/re-ID model, by design: this
brief explicitly calls for SORT rather than DeepSORT/BoT-SORT, since the
target device's headroom goes to OCR, not a re-ID network.
"""
from __future__ import annotations

from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_batch(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Pairwise IoU between two arrays of [x1, y1, x2, y2] boxes."""
    if boxes_a.size == 0 or boxes_b.size == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    xx1 = np.maximum(a[..., 0], b[..., 0])
    yy1 = np.maximum(a[..., 1], b[..., 1])
    xx2 = np.minimum(a[..., 2], b[..., 2])
    yy2 = np.minimum(a[..., 3], b[..., 3])

    w = np.clip(xx2 - xx1, 0, None)
    h = np.clip(yy2 - yy1, 0, None)
    inter = w * h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter

    return np.where(union > 0, inter / union, 0.0)


class KalmanBoxTracker:
    """Tracks one object's bounding box as [cx, cy, s, r] with constant
    velocity on cx, cy, s (s = area, r = aspect ratio held fixed)."""

    count = 0

    def __init__(self, bbox: np.ndarray, cls: int):
        # State: [cx, cy, s, r, vcx, vcy, vs]
        self.x = np.zeros(7)
        self._update_pos(bbox)
        self.P = np.eye(7) * 10.0
        self.P[4:, 4:] *= 100.0  # high initial uncertainty on velocity

        self.F = np.eye(7)
        for i in range(3):
            self.F[i, i + 4] = 1.0
        self.H = np.zeros((4, 7))
        self.H[0, 0] = self.H[1, 1] = self.H[2, 2] = self.H[3, 3] = 1.0

        self.Q = np.eye(7) * 0.01
        self.R = np.eye(4) * 1.0

        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.cls = cls
        self.hits = 1
        self.time_since_update = 0
        self.age = 0
        self.history: List[np.ndarray] = [bbox.copy()]

    @staticmethod
    def _bbox_to_z(bbox: np.ndarray) -> np.ndarray:
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        cx = bbox[0] + w / 2.0
        cy = bbox[1] + h / 2.0
        s = max(w * h, 1e-6)
        r = w / max(h, 1e-6)
        return np.array([cx, cy, s, r])

    def _update_pos(self, bbox: np.ndarray) -> None:
        self.x[:4] = self._bbox_to_z(bbox)

    def to_bbox(self) -> np.ndarray:
        cx, cy, s, r = self.x[:4]
        s = max(s, 1e-6)
        w = np.sqrt(s * r)
        h = s / max(w, 1e-6)
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])

    def predict(self) -> np.ndarray:
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self.age += 1
        self.time_since_update += 1
        return self.to_bbox()

    def update(self, bbox: np.ndarray) -> None:
        z = self._bbox_to_z(bbox)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

        self.time_since_update = 0
        self.hits += 1
        self.history.append(bbox.copy())


class Sort:
    """Multi-object tracker. Feed it detections per frame, get back tracks."""

    def __init__(self, max_age: int = 15, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: np.ndarray, classes: np.ndarray) -> List[dict]:
        """detections: Nx4 [x1,y1,x2,y2]. classes: N class ids, same order.

        Returns confirmed tracks as dicts: id, bbox, cls, hits, time_since_update.
        """
        predicted = np.array([t.predict() for t in self.trackers]) if self.trackers else np.empty((0, 4))

        matches, unmatched_dets, unmatched_trks = self._associate(detections, predicted)

        for det_idx, trk_idx in matches:
            self.trackers[trk_idx].update(detections[det_idx])

        for det_idx in unmatched_dets:
            self.trackers.append(KalmanBoxTracker(detections[det_idx], int(classes[det_idx])))

        alive = []
        results = []
        for t in self.trackers:
            if t.time_since_update > self.max_age:
                continue  # drop: this is a silent track death, not an ID switch
            alive.append(t)
            if t.hits >= self.min_hits or t.age <= self.min_hits:
                results.append({
                    "id": t.id,
                    "bbox": t.to_bbox(),
                    "cls": t.cls,
                    "hits": t.hits,
                    "time_since_update": t.time_since_update,
                })
        self.trackers = alive
        return results

    def _associate(self, detections: np.ndarray, predicted: np.ndarray):
        if len(predicted) == 0 or len(detections) == 0:
            return [], list(range(len(detections))), list(range(len(predicted)))

        iou = iou_batch(detections, predicted)
        row_idx, col_idx = linear_sum_assignment(-iou)

        matches, unmatched_dets, unmatched_trks = [], [], []
        matched_dets, matched_trks = set(), set()

        for r, c in zip(row_idx, col_idx):
            if iou[r, c] < self.iou_threshold:
                continue
            matches.append((r, c))
            matched_dets.add(r)
            matched_trks.add(c)

        unmatched_dets = [i for i in range(len(detections)) if i not in matched_dets]
        unmatched_trks = [i for i in range(len(predicted)) if i not in matched_trks]
        return matches, unmatched_dets, unmatched_trks
