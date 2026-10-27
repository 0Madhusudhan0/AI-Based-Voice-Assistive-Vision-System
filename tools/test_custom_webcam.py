
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from detection.custom_detector import CustomObjectDetector


def main():
    detector = CustomObjectDetector()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Webcam test started. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Error: Could not read webcam frame.")
                break

            result, detections = detector.detect(frame)
            annotated_frame = result.plot()

            for detection in detections:
                label = (
                    f"{detection['class_name']} "
                    f"{detection['confidence']:.2f} | "
                    f"{detection['position']} | "
                    f"{detection['proximity']}"
                )

                x1, y1, _, _ = map(
                    int, detection["bbox"]
                )

                cv2.putText(
                    annotated_frame,
                    label,
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow(
                "SightSpeak AI - Custom Detection Test",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
