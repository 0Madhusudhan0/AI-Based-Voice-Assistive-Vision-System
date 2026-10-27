
from typing import Literal

Position = Literal["left", "center", "right"]


def get_position(
    x1: float,
    x2: float,
    frame_width: int,
    left_threshold: float = 0.35,
    right_threshold: float = 0.65,
) -> Position:
    """Estimate horizontal object position from its bounding box."""

    if frame_width <= 0:
        raise ValueError("frame_width must be positive.")

    if not 0 <= left_threshold < right_threshold <= 1:
        raise ValueError("Invalid position thresholds.")

    center_x = (x1 + x2) / 2
    relative_x = center_x / frame_width

    if relative_x < left_threshold:
        return "left"

    if relative_x > right_threshold:
        return "right"

    return "center"
