
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from safety.engine import SafetyEngine


class TestSafetyEngine(unittest.TestCase):

    def setUp(self):
        self.engine = SafetyEngine(cooldown_seconds=3.0)

    @staticmethod
    def make_detection(
        class_name="person",
        position="center",
        proximity="near",
    ):
        return {
            "class_name": class_name,
            "position": position,
            "proximity": proximity,
        }

    def test_near_object_in_center_generates_high_alert(self):
        alerts = self.engine.evaluate([
            self.make_detection()
        ])

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["priority"], "high")
        self.assertEqual(
            alerts[0]["message"],
            "person very close in the center",
        )

    def test_near_object_on_left_generates_medium_alert(self):
        alerts = self.engine.evaluate([
            self.make_detection(position="left")
        ])

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["priority"], "medium")
        self.assertEqual(
            alerts[0]["message"],
            "person nearby on your left",
        )

    def test_medium_object_in_center_generates_alert(self):
        alerts = self.engine.evaluate([
            self.make_detection(proximity="medium")
        ])

        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["priority"], "medium")
        self.assertEqual(
            alerts[0]["message"],
            "person ahead in the center",
        )

    def test_far_object_does_not_generate_alert(self):
        alerts = self.engine.evaluate([
            self.make_detection(proximity="far")
        ])

        self.assertEqual(alerts, [])

    def test_unimportant_class_is_ignored(self):
        alerts = self.engine.evaluate([
            self.make_detection(class_name="cup")
        ])

        self.assertEqual(alerts, [])

    def test_higher_priority_alert_comes_first(self):
        alerts = self.engine.evaluate([
            self.make_detection(
                class_name="person",
                position="left",
                proximity="near",
            ),
            self.make_detection(
                class_name="car",
                position="center",
                proximity="near",
            ),
        ])

        self.assertEqual(len(alerts), 2)
        self.assertEqual(alerts[0]["priority"], "high")
        self.assertEqual(alerts[0]["class_name"], "car")
        self.assertEqual(alerts[1]["priority"], "medium")

    def test_cooldown_suppresses_repeated_alert(self):
        with patch(
            "safety.engine.time.monotonic",
            side_effect=[100.0, 101.0, 104.0],
        ):
            first = self.engine.evaluate([
                self.make_detection()
            ])
            second = self.engine.evaluate([
                self.make_detection()
            ])
            third = self.engine.evaluate([
                self.make_detection()
            ])

        self.assertEqual(len(first), 1)
        self.assertEqual(second, [])
        self.assertEqual(len(third), 1)


if __name__ == "__main__":
    unittest.main()
