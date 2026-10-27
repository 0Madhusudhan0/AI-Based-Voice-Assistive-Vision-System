
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from detection.custom_detector import CustomObjectDetector


def main():
    detector = CustomObjectDetector()

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    result, detections = detector.detect(frame)

    print("Custom model loaded successfully.")
    print("Classes:", detector.model.names)
    print("Inference completed.")
    print("Detections on blank frame:", len(detections))


if __name__ == "__main__":
    main()
