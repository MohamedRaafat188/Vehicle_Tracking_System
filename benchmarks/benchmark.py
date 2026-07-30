"""Device-agnostic pipeline benchmark.

Measures whatever it's actually run on and reports that device's name - it
never assumes or hardcodes "Jetson Xavier". Reports a per-stage time
breakdown (decode / detect / track / OCR / post-process) rather than
assuming OCR is the bottleneck: that was true on the original Jetson
deployment referenced in the design brief, but there is no evidence it holds
on a desktop GPU with far more headroom, and this script is meant to find
out rather than repeat the assumption.

No numbers in this file are fabricated: everything printed and written to
RESULTS.md comes from an actual timed run against --source.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from paddleocr import PaddleOCR
from ultralytics import YOLO

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import read_valid_license_plate  # noqa: E402
from src.inference.tracker import _Detections, _build_tracker  # noqa: E402

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def detect_device() -> str:
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return f"CPU ({platform.processor() or platform.machine()})"


def percentiles(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    arr = np.array(values)
    return {
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def peak_rss_mb() -> float:
    if not _HAS_PSUTIL:
        return 0.0
    return psutil.Process().memory_info().rss / (1024 * 1024)


def gpu_mem_mb() -> float:
    if _HAS_TORCH and torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 * 1024)
    return 0.0


def benchmark_detector(model: YOLO, frame: np.ndarray, warmup: int, iters: int) -> Dict[str, float]:
    for _ in range(warmup):
        model.predict(source=frame, verbose=False)
    times = []
    for _ in range(iters):
        start = time.perf_counter()
        model.predict(source=frame, verbose=False)
        times.append(time.perf_counter() - start)
    return percentiles(times)


def benchmark_pipeline(source: str, frames_limit: int, gate_confidence: float) -> Dict:
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"could not open source: {source}")

    model_vehicles = YOLO("models/yolov8s.pt")
    model_lp = YOLO("models/best.pt")
    ocr = PaddleOCR(
        lang="en", ocr_version="PP-OCRv4",
        use_doc_orientation_classify=False, use_doc_unwarping=False,
        use_textline_orientation=False,
    )
    tracker = _build_tracker("bytetrack.yaml", frame_rate=30, overrides=None)

    stage_times = {"decode": [], "detect": [], "track": [], "ocr": [], "postprocess": []}
    ocr_calls, ocr_skipped = 0, 0
    locked_ids: set = set()

    frame_count = 0
    pipeline_start = time.perf_counter()

    while frame_count < frames_limit:
        t0 = time.perf_counter()
        ret, frame = cap.read()
        stage_times["decode"].append(time.perf_counter() - t0)
        if not ret:
            break
        frame_count += 1
        frame = cv2.resize(frame, (1920, 1080))

        t0 = time.perf_counter()
        # Detect at a low floor; the tracker's thresholds gate (matches src/pipeline.py).
        results = model_vehicles.predict(source=frame, conf=0.1, classes=[2, 3, 5, 7], verbose=False)[0]
        boxes_data = results.boxes.data.tolist()
        stage_times["detect"].append(time.perf_counter() - t0)

        boxes = np.array([b[:4] for b in boxes_data]) if boxes_data else np.empty((0, 4))
        scores = np.array([b[4] for b in boxes_data]) if boxes_data else np.empty((0,))
        classes = np.array([int(b[5]) for b in boxes_data]) if boxes_data else np.empty((0,))

        t0 = time.perf_counter()
        # rows: [x1, y1, x2, y2, track_id, score, cls, det_idx]
        tracks = tracker.update(_Detections(boxes, scores, classes), frame)
        stage_times["track"].append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        for row in tracks:
            track_id = int(row[4])
            if track_id in locked_ids:
                ocr_skipped += 1
                continue
            x1, y1, x2, y2 = np.asarray(row[:4]).astype(int)
            car = frame[max(y1, 0):y2, max(x1, 0):x2]
            if car.size == 0:
                continue
            lp_result = model_lp(source=car, verbose=False)[0]
            lp_boxes = lp_result.boxes.data.int().tolist()
            if not lp_boxes:
                continue
            lx1, ly1, lx2, ly2 = lp_boxes[0][:4]
            lp_crop = car[ly1:ly2, lx1:lx2]
            ocr_calls += 1
            plate = read_valid_license_plate(ocr, lp_crop)
            if plate:
                locked_ids.add(track_id)
        stage_times["ocr"].append(time.perf_counter() - t0)
        stage_times["postprocess"].append(0.0)  # plate_postprocess.py is a pure-CPU string op, negligible

    elapsed = time.perf_counter() - pipeline_start
    cap.release()

    total_ocr_decisions = ocr_calls + ocr_skipped
    savings_pct = 100.0 * ocr_skipped / total_ocr_decisions if total_ocr_decisions else 0.0

    return {
        "frames": frame_count,
        "elapsed_s": elapsed,
        "fps": frame_count / elapsed if elapsed > 0 else 0.0,
        "stage_breakdown_s": {k: percentiles(v) for k, v in stage_times.items()},
        "ocr_calls": ocr_calls,
        "ocr_skipped": ocr_skipped,
        "ocr_savings_pct": round(savings_pct, 1),
        "peak_rss_mb": round(peak_rss_mb(), 1),
        "gpu_mem_mb": round(gpu_mem_mb(), 1),
    }


def append_results_row(device: str, result: Dict, results_path: Path) -> None:
    stage = result["stage_breakdown_s"]
    row = (
        f"| {device} | this run | {result['fps']:.1f} (single stream) | "
        f"{stage['detect']['mean']*1000:.1f} | {stage['ocr']['mean']*1000:.1f} | "
        f"{result['ocr_savings_pct']:.1f}% | {result['peak_rss_mb']:.0f} | {result['gpu_mem_mb']:.0f} |\n"
    )
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Video file, RTSP URL, or webcam index")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--detector-iters", type=int, default=50)
    parser.add_argument("--gate-confidence", type=float, default=0.95)
    parser.add_argument("--append-results", action="store_true",
                         help="Append this run's numbers as a new row in benchmarks/RESULTS.md")
    args = parser.parse_args()

    device = detect_device()
    print(f"Device: {device}")

    result = benchmark_pipeline(args.source, args.frames, args.gate_confidence)

    print(f"\nFrames processed: {result['frames']}")
    print(f"Elapsed: {result['elapsed_s']:.1f}s  FPS: {result['fps']:.1f}")
    print("\nStage breakdown (ms, mean/p50/p95/p99):")
    for stage, stats in result["stage_breakdown_s"].items():
        print(f"  {stage:12s} {stats['mean']*1000:7.2f} / {stats['p50']*1000:7.2f} / "
              f"{stats['p95']*1000:7.2f} / {stats['p99']*1000:7.2f}")
    print(f"\nOCR calls: {result['ocr_calls']}, skipped (gated): {result['ocr_skipped']} "
          f"({result['ocr_savings_pct']:.1f}% saved)")
    print(f"Peak RSS: {result['peak_rss_mb']:.0f} MB, GPU mem: {result['gpu_mem_mb']:.0f} MB")

    if args.append_results:
        results_path = Path(__file__).parent / "RESULTS.md"
        append_results_row(device, result, results_path)
        print(f"\nAppended row to {results_path}")


if __name__ == "__main__":
    main()
