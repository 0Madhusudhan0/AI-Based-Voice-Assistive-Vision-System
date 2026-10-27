
from ultralytics import YOLO

from spatial import proximity
from spatial.position import get_position

from spatial.proximity import estimate_proximity


class ObjectDetector:
    """Run YOLO inference and estimate object positions."""

    def __init__(
        self,
        model_path: str = "yolo11n.pt",
        confidence: float = 0.50,
        image_size: int = 416,
    ) -> None:
        self.model = YOLO(model_path)
        self.confidence = confidence
        self.image_size = image_size

    def detect(self, frame):
        results = self.model.predict(
            source=frame,
            imgsz=self.image_size,
            conf=self.confidence,
            device="cpu",
            verbose=False,
        )

        result = results[0]
        frame_height, frame_width = frame.shape[:2]
        detections = []

        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence_score = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]

            position = get_position(
                x1=x1,
                x2=x2,
                frame_width=frame_width,
            )

            proximity = estimate_proximity(
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                frame_width=frame_width,
                frame_height=frame_height,
            )

            detections.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence_score,
                    "bbox": [x1, y1, x2, y2],
                    "position": position,
                    "proximity": proximity,
                }
            )

        return result, detections
