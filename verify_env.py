"""
Environment check for the Vehicle Tracking System.

Run this straight after installing dependencies and BEFORE running main.py:

    python verify_env.py

It checks every import main.py needs, reports whether the GPU is visible to
torch, loads both YOLO weights, and - most importantly - runs PaddleOCR on a
synthetic plate image to confirm the result structure the code relies on
(res.json['res']['rec_texts']) actually exists in the installed version.
"""

import os
import sys
import traceback

os.environ.setdefault("YOLO_VERBOSE", "False")

failures = []
warnings = []


def check(label):
    """Decorator that runs a check and records the outcome."""
    def wrapper(fn):
        sys.stdout.write(f"  {label:<44}")
        sys.stdout.flush()
        try:
            note = fn()
        except Exception as exc:
            print("FAIL")
            failures.append((label, exc, traceback.format_exc()))
        else:
            print(f"ok{'  - ' + note if note else ''}")
        return fn
    return wrapper


print(f"\nPython {sys.version.split()[0]}")
print(f"Interpreter: {sys.executable}\n")

print("Imports")


@check("numpy / opencv")
def _cv():
    import cv2
    return f"opencv {cv2.__version__}"


@check("pandas / matplotlib")
def _data():
    import matplotlib
    matplotlib.use("Agg")          # headless: no window needed to save the graph
    import pandas
    return f"pandas {pandas.__version__}"


@check("torch")
def _torch():
    import torch
    return f"torch {torch.__version__}"


@check("ultralytics")
def _ultra():
    import ultralytics
    return f"ultralytics {ultralytics.__version__}"


@check("paddle")
def _paddle():
    import paddle
    return f"paddle {paddle.__version__}"


@check("paddleocr")
def _paddleocr():
    import paddleocr
    ver = getattr(paddleocr, "__version__", "unknown")
    if str(ver).startswith("2."):
        warnings.append(
            f"paddleocr {ver} is 2.x - the code needs 3.x (ocr.predict). "
            "Run: pip install -U 'paddleocr>=3.0,<4.0'")
    return f"paddleocr {ver}"


@check("project modules (utils)")
def _utils():
    from utils import read_valid_license_plate, save_cars  # noqa: F401
    return None


print("\nGPU")


@check("torch sees CUDA")
def _cuda():
    import torch
    if not torch.cuda.is_available():
        warnings.append(
            "torch cannot see the GPU - detection will run on CPU and be very slow. "
            "Reinstall torch from the cu126 index.")
        return "NOT available (CPU fallback)"
    return f"{torch.cuda.get_device_name(0)} / CUDA {torch.version.cuda}"


print("\nModels")


@check("models/yolov8s.pt loads")
def _m1():
    from ultralytics import YOLO
    YOLO("models/yolov8s.pt")
    return None


@check("models/best.pt loads")
def _m2():
    from ultralytics import YOLO
    YOLO("models/best.pt")
    return None


print("\nOCR pipeline (this downloads models on first run - be patient)")


@check("PaddleOCR init + predict + json shape")
def _ocr():
    import cv2
    import numpy as np
    from paddleocr import PaddleOCR

    ocr = PaddleOCR(
        lang="en",
        ocr_version="PP-OCRv4",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False)

    # Synthetic plate: white background, black text in a valid pattern
    plate = np.full((90, 340, 3), 255, dtype=np.uint8)
    cv2.putText(plate, "9079GCH", (18, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 0), 5)

    results = ocr.predict(plate)

    # This is the exact access pattern utils.read_valid_license_plate uses.
    for res in results:
        res.json["res"]["rec_texts"]
        res.json["res"]["rec_scores"]

    from utils import read_valid_license_plate
    got = read_valid_license_plate(ocr, plate)
    if got != "9079GCH":
        warnings.append(
            f"OCR read {got!r} instead of '9079GCH' on a clean synthetic image. "
            "The result structure is fine, but check recognition quality / min_score.")
        return f"structure ok, read {got!r}"
    return "read '9079GCH' correctly"


print()
if failures:
    print(f"{len(failures)} check(s) FAILED\n")
    for label, exc, tb in failures:
        print(f"--- {label} ---")
        print(f"{type(exc).__name__}: {exc}\n")
        print(tb)
else:
    print("All checks passed.")

if warnings:
    print(f"\n{len(warnings)} warning(s):")
    for w in warnings:
        print(f"  ! {w}")

if not failures and not warnings:
    print("\nEnvironment is ready. Run:  python main.py")

sys.exit(1 if failures else 0)
