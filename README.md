# Vehicle Tracking System

## Overview
This is a Vehicle Tracking System freelance project that uses computer vision to detect and track vehicles in real time. It leverages YOLOv8 model for detection and PaddleOCR for license plate recognition, along with data storage and visualization tools for tracking vehicle movement. These YOLO models will be deployed on the cloud and connected to the camera of the client using its url.

## Demo

<video src="assets/demo.mp4" controls width="640"></video>

Generated with `visualize.py` on a real traffic clip: vehicles are detected and tracked, license plates are located per-vehicle and read via OCR, and both car plates (single-line) and motorcycle plates (split across two lines) are correctly assembled and displayed.

## Features
- **Vehicle Detection:** Detects and identifies vehicles in each frame.
- **License Plate Recognition:** Recognizes and extracts license plate numbers from detected vehicles.
- **Data Storage:** Stores vehicle details, including images and detection timestamps.
- **Traffic Analysis:** Generates and plots traffic distribution based on time data.
- **Reporting:** Produces daily traffic reports in Excel format with hourly vehicle counts.

## Project Structure
- `models/`: Pre-trained model for vehicle detection, custom model for license plate detection and OCR.
- `License-Plate-Recognition-2/`: Dataset to train the license plate detection model. Download the data first from the code in file.ipynb
- `output/`: Saved results, such as CSV files, Excel reports, and graphs.
- `file.ipynb`: Download the dataset and train the license plate detection model. 
- `main.py`: Detecting vehicles and store data
- `visualize.py`: Visualize the results

## Installation

### Prerequisites
- **Python 3.10**
- **ultralytics**
- **pytorch torchvision pytorch-cuda=11.8** (optional to use gpu instead of cpu)
- **paddlepaddle** or **paddlepaddle-gpu**
- **paddleocr**

### Steps
1. **Clone the Repository:**
2. **install Prerequisites**
3. **run main.py** file

## Deployment Layer (scaffold, speculative)

Everything in this section — `src/`, `configs/`, `benchmarks/`, `scripts/`,
`tests/` — is a **speculative scaffold**, not a description of a real
deployment. `main.py` / `visualize.py` above are the only code in this repo
that has actually been run against real footage. Nothing here has been run
against RTSP cameras, a Jetson, or TensorRT; no historical benchmark numbers
exist for this project, and none are claimed below.

### Architecture (as designed, not yet deployed)

```
RTSP cameras (or local files / webcams)
        |
src/streaming/rtsp_reader.py   - one background thread per stream,
        |                          latest-frame-wins buffer, auto-reconnect
src/streaming/stream_manager.py - round-robins frames across streams
        |
YOLOv8 vehicle detector (.pt, or a TensorRT .engine once one is built)
        |
src/inference/tracker.py       - SORT + confidence-gated OCR
        |
PaddleOCR (gated: only called until a track's read clears the threshold)
        |
src/inference/plate_postprocess.py - confusion-char correction + format check
        |
src/pipeline.py output sinks: annotated video, JSONL event log, stdout
```

### Running it

Against a local file, no camera or GPU-specific setup required:

```
python -m src.pipeline --source videos/tc.mp4 --max-frames 200
```

Against `configs/config.yaml`'s stream list (edit it first — the shipped
config just points every entry at `videos/tc.mp4` as a placeholder):

```
python -m src.pipeline --config configs/config.yaml
```

Against simulated RTSP cameras (requires `mediamtx` and `ffmpeg` on PATH,
neither installed by requirements.txt):

```
scripts/simulate_cameras.sh videos/tc.mp4 4
scripts/kill_stream.sh 2   # test reconnect/backoff against a real drop
```

### TensorRT export

`src/export/export_tensorrt.py` exports `.pt` weights to a `.engine` via
`model.export(format="engine", half=True, ...)`. **Validated on a desktop
RTX 3060 Ti** (TensorRT 10.16.1.11, CUDA 12.9): `models/yolov8s.pt` exported
to FP16 in ~524s, and the script's own validation step confirmed the engine's
output matches the PyTorch model on a sample image. It has **not** been run
on a Jetson / JetPack target — that's what this brief's deployment path
actually calls for, and desktop CUDA and Jetson TensorRT builds are not
interchangeable, so Jetson support is still unvalidated. TensorRT engines are
tied to the exact GPU architecture, TensorRT version, and (on Jetson) JetPack
version they were built on; they must be built on the deployment device
itself and are `.gitignore`d, never committed.

### Benchmarks

`benchmarks/benchmark.py` is real and runnable — it measures whatever device
you run it on (CPU or GPU, detected automatically) and reports latency
percentiles, a decode/detect/track/OCR/postprocess breakdown, and OCR-gate
savings. `benchmarks/RESULTS.md` currently has **no rows**: no benchmark has
been run yet, and no number in this repo is invented. Run it yourself and
`--append-results` to fill it in.

### Engineering decisions

- **SORT, not DeepSORT/BoT-SORT:** no spare compute for a re-ID network on a
  device where OCR is meant to be the bottleneck, not detection.
- **OCR gated per track ID, not run every frame:** once a track's read clears
  `ocr.gate_confidence`, it's cached and skipped thereafter — this is the
  actual point of `tracker.py`, since OCR (not detection) is assumed to be
  the expensive stage.
- **Frame skipping is a config knob (`streaming.frame_skip`), default 0:**
  the earlier debugging of `main.py` in this same repo found that skipping
  frames can break tracker continuity for small/low-confidence objects (see
  git history around the motorcycle-plate fix) — so it's opt-in per
  deployment, not a hardcoded default.
- **640 inference size from a 1080p source:** matches `main.py`'s existing
  `cv2.resize(frame, (1920, 1080))` convention and is the input size
  `export_tensorrt.py` targets by default.
- **Latest-frame-wins buffering, not FIFO:** for a live camera, a stale
  frame is worse than no frame — queuing would just accumulate latency
  under load instead of shedding it.
- **Where the bottleneck actually is:** unknown for this repo's hardware.
  `benchmark.py`'s per-stage breakdown is designed to measure this rather
  than assume it, since the OCR-is-the-bottleneck framing came from the
  original design brief, not from anything measured here.

### Limitations

- SORT has no re-ID/appearance model, so it can lose a track through
  occlusion and never recover the same ID.
- A track's plate is whatever single OCR read first cleared the gate
  threshold — there's no voting across reads, so a confident misread
  persists for that track's whole lifetime.
- No multi-camera re-identification: a vehicle crossing between streams is
  tracked as unrelated objects.
- `plate_postprocess.py`'s confusion-character rules are specific to this
  plate format (`configs/plate_rules.yaml`) and are not derived from any
  observed error log — there isn't one yet.
- No INT8 export/validation path, only FP16/FP32.
- The TensorRT export path is validated on desktop CUDA but not on a Jetson
  target (see above) — those aren't interchangeable.