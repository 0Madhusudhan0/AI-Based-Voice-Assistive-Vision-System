
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from spatial.position import get_position


class TestObjectPosition(unittest.TestCase):

    def test_object_on_left(self):
        self.assertEqual(get_position(50, 150, 640), "left")

    def test_object_in_center(self):
        self.assertEqual(get_position(250, 350, 640), "center")

    def test_object_on_right(self):
        self.assertEqual(get_position(500, 600, 640), "right")

    def test_left_threshold_boundary(self):
        # Center x = 224; 224 / 640 = 0.35
        self.assertEqual(get_position(200, 248, 640), "center")

    def test_right_threshold_boundary(self):
        # Center x = 416; 416 / 640 = 0.65
        self.assertEqual(get_position(392, 440, 640), "center")

    def test_invalid_frame_width(self):
        with self.assertRaises(ValueError):
            get_position(10, 20, 0)

    def test_invalid_thresholds(self):
        with self.assertRaises(ValueError):
            get_position(
                10, 20, 640,
                left_threshold=0.7,
                right_threshold=0.3,
            )


if __name__ == "__main__":
    unittest.main()
