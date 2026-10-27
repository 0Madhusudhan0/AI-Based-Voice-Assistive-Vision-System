
from pathlib import Path
import time

import cv2
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "doors_stairs_yolov8n.pt"

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = YOLO(str(MODEL_PATH))

# Use a representative 640x480 frame.
frame = torch.zeros((480, 640, 3), dtype=torch.uint8).numpy()

# Warm-up runs are excluded from timing.
for _ in range(3):
    model.predict(
        source=frame,
        imgsz=416,
        conf=0.35,
        device="cpu",
        verbose=False,
    )

iterations = 20
times = []

for _ in range(iterations):
    start = time.perf_counter()

    model.predict(
        source=frame,
        imgsz=416,
        conf=0.35,
        device="cpu",
        verbose=False,
    )

    times.append((time.perf_counter() - start) * 1000)

avg_ms = sum(times) / len(times)
fps = 1000 / avg_ms

print(f"Model: {MODEL_PATH.name}")
print(f"Device: CPU")
print(f"Iterations: {iterations}")
print(f"Average inference pipeline time: {avg_ms:.2f} ms")
print(f"Approximate throughput: {fps:.2f} FPS")
