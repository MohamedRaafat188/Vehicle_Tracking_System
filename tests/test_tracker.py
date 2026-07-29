import numpy as np

from src.inference.tracker import GatedOcrTracker


def _box_frame():
    return np.zeros((100, 100, 3), dtype="uint8")


def test_ocr_gate_caches_after_lock():
    calls = []

    def ocr_fn(frame, bbox):
        calls.append(1)
        return "5545GZN", 0.97  # clears the default 0.95 gate immediately

    tracker = GatedOcrTracker(ocr_fn=ocr_fn, gate_confidence=0.95, min_hits=1)
    frame = _box_frame()
    boxes = np.array([[10, 10, 50, 50]])
    classes = np.array([3])

    tracks = tracker.update(boxes, classes, frame)
    assert tracks[0].locked is True
    assert len(calls) == 1

    # Same track (matching box) on subsequent frames should not call OCR again.
    for _ in range(5):
        tracks = tracker.update(boxes, classes, frame)
    assert len(calls) == 1
    assert tracker.stats()["ocr_calls_saved"] == 5


def test_ocr_gate_keeps_retrying_below_threshold():
    calls = []

    def ocr_fn(frame, bbox):
        calls.append(1)
        return "5545GZN", 0.5  # never clears the gate

    tracker = GatedOcrTracker(ocr_fn=ocr_fn, gate_confidence=0.95, min_hits=1)
    frame = _box_frame()
    boxes = np.array([[10, 10, 50, 50]])
    classes = np.array([3])

    for _ in range(4):
        tracks = tracker.update(boxes, classes, frame)

    assert tracks[0].locked is False
    assert len(calls) == 4  # retried every frame since it never locked


def test_ocr_fn_exception_does_not_crash_update():
    def ocr_fn(frame, bbox):
        raise RuntimeError("simulated OCR failure")

    tracker = GatedOcrTracker(ocr_fn=ocr_fn, min_hits=1)
    frame = _box_frame()
    boxes = np.array([[10, 10, 50, 50]])
    classes = np.array([2])

    tracks = tracker.update(boxes, classes, frame)  # must not raise
    assert tracks[0].locked is False


def test_track_loss_is_logged_not_crashed(caplog):
    def ocr_fn(frame, bbox):
        return None

    tracker = GatedOcrTracker(ocr_fn=ocr_fn, min_hits=1, max_age=0)
    frame = _box_frame()
    classes = np.array([2])

    tracker.update(np.array([[10, 10, 50, 50]]), classes, frame)
    # No matching detection next frame -> track should age out without crashing.
    tracks = tracker.update(np.empty((0, 4)), np.empty((0,)), frame)
    assert tracks == []
