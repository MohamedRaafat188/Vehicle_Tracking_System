"""Deployment-layer pipeline entry point.

Wires: StreamManager (readers) -> YOLO vehicle detector (.pt or .engine) ->
per-stream GatedOcrTracker (SORT + confidence-gated OCR) -> plate
post-processing -> output sinks (annotated video / JSONL events / stdout).

Runs anywhere with --source pointing at a local video file, so it doesn't
require RTSP cameras or the Jetson/TensorRT engine to be tried out.

This module has been run locally against videos/tc.mp4 during scaffolding
(see the run instructions in the README) but never against real RTSP
cameras or a TensorRT engine - both remain speculative until exercised on
that hardware.
"""
from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import yaml
from paddleocr import PaddleOCR
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root, for utils.py
from utils import assemble_plate_text, check_license_plate_pattern  # noqa: E402

from src.inference.plate_postprocess import PlatePostProcessor
from src.inference.tracker import GatedOcrTracker
from src.streaming.stream_manager import StreamManager

logger = logging.getLogger(__name__)


def read_plate_with_score(ocr: PaddleOCR, lp_image: np.ndarray, min_score: float = 0.9):
    """Like utils.read_valid_license_plate, but also returns a confidence
    score so the OCR gate has something to compare against `gate_confidence`.
    Reuses utils' assembly/validation logic rather than re-implementing it.
    """
    if lp_image.size == 0:
        return None
    try:
        ocr_results = ocr.predict(lp_image)
    except Exception:
        return None

    for res in ocr_results:
        data = res.json["res"]
        texts, scores = data["rec_texts"], data["rec_scores"]
        lp_num = assemble_plate_text(texts, scores, data.get("rec_boxes"), min_score)
        if lp_num and check_license_plate_pattern(lp_num):
            overall_score = float(min(scores)) if scores else 0.0
            return lp_num, overall_score
    return None


class PipelineRunner:
    def __init__(self, config: dict, source_override: Optional[str] = None, max_frames: Optional[int] = None):
        self.config = config
        self.max_frames = max_frames

        streams_cfg = config["streams"]
        if source_override is not None:
            streams_cfg = [{"id": "cam1", "source": source_override}]

        self.manager = StreamManager.from_config(streams_cfg, config["streaming"])

        models_cfg = config["models"]
        weights = models_cfg["vehicle_engine"] if models_cfg.get("use_engine") else models_cfg["vehicle_weights"]
        if models_cfg.get("use_engine") and not Path(weights).exists():
            raise FileNotFoundError(
                f"use_engine=true but engine file not found: {weights}. "
                f"Build it on this device first with src/export/export_tensorrt.py, "
                f"or set use_engine: false to run the .pt weights instead."
            )
        self.model_vehicles = YOLO(weights)
        self.model_lp = YOLO(models_cfg["plate_weights"])
        self.ocr = PaddleOCR(
            lang="en", ocr_version="PP-OCRv4",
            use_doc_orientation_classify=False, use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        det_cfg = config["detection"]
        self.vehicle_conf = det_cfg["vehicle_conf"]
        self.vehicle_classes = det_cfg["vehicle_classes"]

        ocr_cfg = config["ocr"]
        self.min_line_score = ocr_cfg["min_line_score"]
        self.postproc = PlatePostProcessor(rules_path=Path(config["plate_rules_file"]))

        tracking_cfg = config["tracking"]
        self._trackers = {
            s["id"]: GatedOcrTracker(
                ocr_fn=self._ocr_fn,
                gate_confidence=ocr_cfg["gate_confidence"],
                max_age=tracking_cfg["max_age"],
                min_hits=tracking_cfg["min_hits"],
                iou_threshold=tracking_cfg["iou_threshold"],
            )
            for s in streams_cfg
        }

        out_cfg = config["output"]
        self.events_path = Path(out_cfg["events_path"])
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        self._events_file = open(self.events_path, "a", encoding="utf-8")
        self._locked_logged: set = set()

        self._stop = False

    def _ocr_fn(self, frame: np.ndarray, bbox: np.ndarray):
        x1, y1, x2, y2 = bbox.astype(int)
        x1, y1 = max(x1, 0), max(y1, 0)
        car = frame[y1:y2, x1:x2]
        if car.size == 0:
            return None
        lp_result = self.model_lp(source=car, verbose=False)[0]
        lp_boxes = lp_result.boxes.data.int().tolist()
        if not lp_boxes:
            return None
        lx1, ly1, lx2, ly2 = lp_boxes[0][:4]
        lp_crop = car[ly1:ly2, lx1:lx2]
        result = read_plate_with_score(self.ocr, lp_crop, self.min_line_score)
        return result

    def _emit_event(self, stream_id: str, state) -> None:
        key = (stream_id, state.track_id)
        if key in self._locked_logged:
            return
        self._locked_logged.add(key)

        corrected = self.postproc.process(state.plate_text)
        event = {
            "stream_id": stream_id,
            "track_id": state.track_id,
            "cls": state.cls,
            "plate_raw": state.plate_text,
            "plate_score": round(state.plate_score, 3),
            "plate_corrected": corrected.text,
            "plate_valid": corrected.valid,
            "first_seen_frame": state.first_seen_frame,
            "locked_frame": state.last_seen_frame,
            "ts": time.time(),
        }
        self._events_file.write(json.dumps(event) + "\n")
        self._events_file.flush()
        print(f"[{stream_id}] track {state.track_id}: {corrected.text} "
              f"(valid={corrected.valid}, score={state.plate_score:.3f})")

    def run(self) -> None:
        signal.signal(signal.SIGINT, self._handle_sigint)
        self.manager.start()
        logger.info("Pipeline started")

        frames_processed = 0
        try:
            while not self._stop:
                any_frame = False
                for tagged in self.manager.poll():
                    any_frame = True
                    self._process_frame(tagged.stream_id, tagged.frame)
                    frames_processed += 1
                    if self.max_frames and frames_processed >= self.max_frames:
                        self._stop = True
                        break
                if not any_frame:
                    time.sleep(0.005)
        finally:
            self.manager.stop()
            self._events_file.close()
            for stream_id, tracker in self._trackers.items():
                logger.info("[%s] stats: %s", stream_id, tracker.stats())

    def _process_frame(self, stream_id: str, frame: np.ndarray) -> None:
        frame = cv2.resize(frame, (1920, 1080))
        results = self.model_vehicles.predict(
            source=frame, conf=self.vehicle_conf, classes=self.vehicle_classes, verbose=False,
        )[0]
        boxes_data = results.boxes.data.tolist()
        if not boxes_data:
            return

        boxes = np.array([b[:4] for b in boxes_data])
        classes = np.array([int(b[5]) for b in boxes_data])

        tracker = self._trackers[stream_id]
        tracks = tracker.update(boxes, classes, frame)

        for state in tracks:
            if state.locked:
                self._emit_event(stream_id, state)

    def _handle_sigint(self, signum, frame) -> None:
        logger.info("SIGINT received, shutting down")
        self._stop = True


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--source", default=None,
                         help="Override configs/config.yaml streams with a single local video/webcam source")
    parser.add_argument("--max-frames", type=int, default=None, help="Stop after N frames (for testing)")
    args = parser.parse_args()

    config = load_config(args.config)
    runner = PipelineRunner(config, source_override=args.source, max_frames=args.max_frames)
    runner.run()


if __name__ == "__main__":
    main()
