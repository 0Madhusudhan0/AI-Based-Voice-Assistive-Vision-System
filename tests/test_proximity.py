
import sys
import unittest
from pathlib import Path

sys.path.insert(
    0, str(Path(__file__).resolve().parents[1] / "src")
)

from spatial.proximity import estimate_proximity


class TestProximity(unittest.TestCase):

    def test_small_box_is_far(self):
        self.assertEqual(
            estimate_proximity(0, 0, 10, 10, 640, 480),
            "far",
        )

    def test_medium_sized_box(self):
        self.assertEqual(
            estimate_proximity(0, 0, 200, 100, 640, 480),
            "medium",
        )

    def test_large_box_is_near(self):
        self.assertEqual(
            estimate_proximity(0, 0, 300, 200, 640, 480),
            "near",
        )

    def test_five_percent_boundary(self):
        self.assertEqual(
            estimate_proximity(0, 0, 50, 10, 100, 100),
            "medium",
        )

    def test_fifteen_percent_boundary(self):
        self.assertEqual(
            estimate_proximity(0, 0, 50, 30, 100, 100),
            "near",
        )

    def test_invalid_frame_dimensions(self):
        with self.assertRaises(ValueError):
            estimate_proximity(0, 0, 10, 10, 0, 100)

    def test_box_outside_frame(self):
        with self.assertRaises(ValueError):
            estimate_proximity(0, 0, 120, 50, 100, 100)


if __name__ == "__main__":
    unittest.main()
