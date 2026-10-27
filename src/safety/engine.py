import time
from typing import Any


class SafetyEngine:
    """Generate prioritized alerts with cooldowns and duplicate suppression."""

    IMPORTANT_CLASSES = {
        "person",
        "car",
        "motorcycle",
        "bus",
        "truck",
        "bicycle",
        "chair",
        "bench",
        "stairs",
        "door",
    }

    def __init__(
        self,
        cooldown_seconds: float = 10.0,
        high_priority_cooldown_seconds: float = 6.0,
    ) -> None:
        if cooldown_seconds < 0 or high_priority_cooldown_seconds < 0:
            raise ValueError("Cooldowns cannot be negative.")

        self.cooldown_seconds = cooldown_seconds
        self.high_priority_cooldown_seconds = high_priority_cooldown_seconds
        self._last_alert_time: dict[str, float] = {}

    def evaluate(
        self,
        detections: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        now = time.monotonic()
        candidates = []

        for detection in detections:
            class_name = detection["class_name"]
            position = detection["position"]
            proximity = detection["proximity"]

            if class_name not in self.IMPORTANT_CLASSES:
                continue

            if proximity == "near" and position == "center":
                priority = "high"
                message = f"{class_name} very close in the center"

            elif proximity == "near":
                priority = "medium"
                message = f"{class_name} nearby on your {position}"

            elif proximity == "medium" and position == "center":
                priority = "medium"
                message = f"{class_name} ahead in the center"

            else:
                continue

            alert_key = f"{class_name}:{position}:{priority}"

            candidates.append(
                {
                    "priority": priority,
                    "class_name": class_name,
                    "position": position,
                    "message": message,
                    "_key": alert_key,
                }
            )

        # Handle the most urgent alerts first.
        priority_order = {"high": 0, "medium": 1}
        candidates.sort(key=lambda alert: priority_order[alert["priority"]])

        emitted = []
        emitted_keys = set()

        for alert in candidates:
            key = alert["_key"]

            # Avoid duplicate announcements for the same alert in one frame.
            if key in emitted_keys:
                continue
            emitted_keys.add(key)

            last_time = self._last_alert_time.get(key)
            cooldown = (
                self.high_priority_cooldown_seconds
                if alert["priority"] == "high"
                else self.cooldown_seconds
            )

            if last_time is not None and now - last_time < cooldown:
                continue

            self._last_alert_time[key] = now

            emitted.append(
                {
                    "priority": alert["priority"],
                    "class_name": alert["class_name"],
                    "position": alert["position"],
                    "message": alert["message"],
                }
            )

        return emitted