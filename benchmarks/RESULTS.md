# Benchmark Results

**No benchmarks have been run yet.** This file is a skeleton for
`benchmark.py` output; it does not contain any measured or estimated
figures. The deployment brief this scaffold is based on referenced Jetson
Xavier numbers, but there are no records of them available for this repo -
they are not reproduced here rather than invented.

Run `python benchmarks/benchmark.py --source <video-or-rtsp-url> --append-results`
on whatever device you want measured; it will detect the device name itself
and append a real row below.

| Device | Date | FPS | Detect (ms) | OCR (ms) | OCR calls saved | Peak RSS (MB) | GPU mem (MB) |
|---|---|---|---|---|---|---|---|

Notes for interpreting future rows:
- FPS above is for a single stream unless the row says otherwise. Aggregate
  multi-stream throughput has to be benchmarked separately once
  `stream_manager.py` batching is in real use.
- Detect/OCR columns are mean per-call latency for that stage, not
  end-to-end frame time.
- "OCR calls saved" is the confidence-gate savings from `tracker.py` -
  percentage of tracks where OCR was skipped because a prior read on that
  track already cleared `gate_confidence`.
- The TensorRT export path (`src/export/export_tensorrt.py`) is untested in
  this repo (see its module docstring), so no PyTorch-vs-TensorRT comparison
  row can exist until it has actually been run on target hardware.
| NVIDIA GeForce RTX 3060 Ti | this run | 3.4 (single stream) | 190.3 | 88.8 | 96.5% | 1491 | 87 |
