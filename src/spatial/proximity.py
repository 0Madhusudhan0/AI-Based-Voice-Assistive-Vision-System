from typing import Literal

Proximity = Literal["far", "medium", "near"]


def estimate_proximity(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    frame_width: int,
    frame_height: int,
) -> Proximity:
    """Estimate visual size in frame, not actual physical distance."""

    if frame_width <= 0 or frame_height <= 0:
        raise ValueError("Frame dimensions must be positive.")

    if not (
        0 <= x1 <= x2 <= frame_width
        and 0 <= y1 <= y2 <= frame_height
    ):
        raise ValueError("Bounding box must lie within the frame.")

    box_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_width * frame_height
    area_ratio = box_area / frame_area

    # Conservative visual-size thresholds.
    if area_ratio < 0.10:
        return "far"
    elif area_ratio < 0.30:
        return "medium"
    else:
        return "near"