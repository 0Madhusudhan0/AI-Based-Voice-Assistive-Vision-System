
import time

import cv2

from camera.webcam import Webcam
from detection.detector import ObjectDetector
from safety.engine import SafetyEngine
from speech.speaker import Speaker

def main() -> None:
    camera = Webcam(camera_index=0)
    detector = ObjectDetector()
    safety_engine = SafetyEngine()
    speaker = Speaker()

    try:
        camera.open()
        previous_time = time.perf_counter()

        while True:
            frame = camera.read()
            result, detections = detector.detect(frame)
            alerts = safety_engine.evaluate(detections)
            for alert in alerts:
                print(
        f"[{alert['priority'].upper()}] "
        f"{alert['message']}"
    )
                speaker.speak(alert["message"], priority=alert["priority"],)
            annotated_frame = result.plot(labels=False)
            
            for detection in detections:
                 label = (
    f"{detection['class_name']} | "
    f"{detection['position']} | "
    f"{detection['proximity']} | "
    f"{detection['confidence']:.2f}"
)
                 x1, y1, _, _ = detection["bbox"]
                 text_x = max(5, int(x1))
                 text_y = max(25, int(y1) - 10)
                 cv2.putText(
                     annotated_frame,
        label,
        (text_x, text_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

            current_time = time.perf_counter()
            elapsed = current_time - previous_time
            previous_time = current_time

            fps = 1.0 / elapsed if elapsed > 0 else 0.0

            cv2.putText(
                annotated_frame,
                f"FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
            )

            cv2.imshow(
                "SightSpeak AI - Object Detection",
                annotated_frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.close()
        speaker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
