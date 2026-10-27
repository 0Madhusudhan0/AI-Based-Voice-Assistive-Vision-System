
from pathlib import Path
import time

import numpy as np
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "yolo11n.pt"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"YOLO11n weights not found at {MODEL_PATH}. "
        "Check the path to your existing model."
    )

model = YOLO(str(MODEL_PATH))
frame = np.zeros((480, 640, 3), dtype=np.uint8)

def predict():
    model.predict(
        source=frame,
        imgsz=416,
        conf=0.35,
        device="cpu",
        verbose=False,
    )

for _ in range(3):
    predict()

times = []

for _ in range(20):
    start = time.perf_counter()
    predict()
    times.append((time.perf_counter() - start) * 1000)

avg_ms = sum(times) / len(times)

print(f"Model: {MODEL_PATH.name}")
print(f"Average inference pipeline time: {avg_ms:.2f} ms")
print(f"Approximate throughput: {1000 / avg_ms:.2f} FPS")
