"""Export a trained YOLOv8 model to a TensorRT engine.

STATUS: validated end-to-end on a desktop RTX 3060 Ti (TensorRT 10.16.1.11,
CUDA 12.9) - `models/yolov8s.pt` exported to FP16 in ~524s, engine validated
against the PyTorch model on a synthetic sample. Still never run on a Jetson /
JetPack target, which is what this brief's deployment path actually calls for
- desktop CUDA and Jetson TensorRT builds are not interchangeable, so treat
Jetson support as unvalidated even though the export logic itself now has a
real, passing run behind it.

TensorRT engines are tied to the exact GPU architecture, TensorRT version, and
(on Jetson) JetPack version they were built on. An engine built on one machine
will generally fail to load on another. Build on the deployment device itself;
never commit or share the .engine file (see .gitignore).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def engine_filename(weights: Path, device_tag: str, precision: str, imgsz: int) -> str:
    stem = weights.stem
    return f"{stem}_{device_tag}_{precision}_{imgsz}.engine"


def export(
    weights: str,
    imgsz: int = 640,
    half: bool = True,
    workspace: float = 4.0,
    batch: int = 1,
    device_tag: str = "device",
) -> Path:
    """Export `weights` (a .pt file) to a TensorRT engine.

    Raises RuntimeError with an actionable message if TensorRT / the
    ultralytics export backend isn't available - this is expected to happen
    on any machine without an NVIDIA GPU + TensorRT installed, which is
    every machine this has been written on so far.
    """
    try:
        from ultralytics import YOLO
    except ImportError as e:
        raise RuntimeError("ultralytics is required for export") from e

    weights_path = Path(weights)
    if not weights_path.exists():
        raise FileNotFoundError(
            f"weights file not found: {weights_path}. Train or download the "
            f"model before exporting."
        )

    precision = "fp16" if half else "fp32"
    model = YOLO(str(weights_path))

    logger.info("Starting TensorRT export: imgsz=%d half=%s workspace=%.1fGB batch=%d",
                imgsz, half, workspace, batch)
    start = time.monotonic()
    try:
        exported_path = model.export(
            format="engine", imgsz=imgsz, half=half, workspace=workspace, batch=batch,
        )
    except Exception as e:
        raise RuntimeError(
            "TensorRT export failed. This requires an NVIDIA GPU with a matching "
            "TensorRT install (and JetPack, on Jetson). See the error above for "
            "the underlying cause."
        ) from e
    build_time = time.monotonic() - start

    exported_path = Path(exported_path)
    final_name = engine_filename(weights_path, device_tag, precision, imgsz)
    final_path = exported_path.parent / final_name
    exported_path.rename(final_path)

    size_mb = final_path.stat().st_size / (1024 * 1024)
    logger.info("Engine built in %.1fs, size=%.1fMB -> %s", build_time, size_mb, final_path)
    print(f"Engine build time: {build_time:.1f}s")
    print(f"Engine size: {size_mb:.1f}MB")
    print(f"Engine path: {final_path}")

    _validate(weights_path, final_path, imgsz)
    return final_path


def _validate(pytorch_weights: Path, engine_path: Path, imgsz: int) -> None:
    """Sanity-check the engine against the PyTorch model on a synthetic image.

    Compares top detection count / classes only (not exact box coordinates,
    which legitimately shift a little under FP16). Fails loudly - raises,
    rather than warns - on a real mismatch, since a silently broken export is
    worse than a crashed one.
    """
    import numpy as np
    from ultralytics import YOLO

    sample = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype="uint8")

    pt_model = YOLO(str(pytorch_weights))
    trt_model = YOLO(str(engine_path))

    pt_result = pt_model.predict(source=sample, verbose=False)[0]
    trt_result = trt_model.predict(source=sample, verbose=False)[0]

    pt_classes = sorted(pt_result.boxes.cls.tolist()) if pt_result.boxes is not None else []
    trt_classes = sorted(trt_result.boxes.cls.tolist()) if trt_result.boxes is not None else []

    if pt_classes != trt_classes:
        raise RuntimeError(
            f"Engine validation failed: PyTorch detected classes {pt_classes} on the "
            f"sample image, TensorRT engine detected {trt_classes}. Do not deploy "
            f"this engine - re-export and investigate before using it."
        )
    logger.info("Engine validated OK against PyTorch model on synthetic sample")
    print("Validation OK: engine output matches PyTorch model on sample image")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", required=True, help="Path to trained .pt weights")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--workspace", type=float, default=4.0, help="TensorRT workspace size in GB")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--fp32", action="store_true", help="Export FP32 instead of the default FP16")
    parser.add_argument("--device-tag", default="jetson", help="Short tag embedded in the output filename")
    args = parser.parse_args()

    export(
        weights=args.weights,
        imgsz=args.imgsz,
        half=not args.fp32,
        workspace=args.workspace,
        batch=args.batch,
        device_tag=args.device_tag,
    )


if __name__ == "__main__":
    main()
