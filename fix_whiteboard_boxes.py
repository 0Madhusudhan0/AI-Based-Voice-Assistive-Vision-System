from pathlib import Path
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "sightspeak-three-class"

SPLITS = ["train", "valid", "test"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

fixed = 0
checked = 0

for split in SPLITS:
    image_dir = DATASET / split / "images"
    label_dir = DATASET / split / "labels"

    for image_path in image_dir.iterdir():
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        label_path = label_dir / f"{image_path.stem}.txt"

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        lines = label_path.read_text(encoding="utf-8").splitlines()
        new_lines = []

        for line_number, line in enumerate(lines, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Malformed label: {label_path}:{line_number}")

            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            if class_id not in (0, 1, 2):
                raise ValueError(f"Unexpected class ID in {label_path}")

            if not all(map(lambda v: 0 <= v <= 1, (x, y, w, h))):
                raise ValueError(f"Out-of-range values in {label_path}:{line_number}")

            checked += 1

            # Convert normalized center/size to normalized corners.
            left = x - w / 2
            top = y - h / 2
            right = x + w / 2
            bottom = y + h / 2

            # Clip corners to the image boundaries.
            clipped_left = max(0.0, left)
            clipped_top = max(0.0, top)
            clipped_right = min(1.0, right)
            clipped_bottom = min(1.0, bottom)

            if (
                clipped_left != left
                or clipped_top != top
                or clipped_right != right
                or clipped_bottom != bottom
            ):
                if class_id != 2:
                    raise ValueError(
                        f"Unexpected out-of-bounds non-whiteboard box: "
                        f"{label_path}:{line_number}"
                    )

                fixed += 1

            new_w = clipped_right - clipped_left
            new_h = clipped_bottom - clipped_top

            if new_w <= 0 or new_h <= 0:
                raise ValueError(
                    f"Box collapsed during clipping: {label_path}:{line_number}"
                )

            new_x = (clipped_left + clipped_right) / 2
            new_y = (clipped_top + clipped_bottom) / 2

            new_lines.append(
                f"{class_id} {new_x:.8f} {new_y:.8f} "
                f"{new_w:.8f} {new_h:.8f}"
            )

        label_path.write_text(
            "\n".join(new_lines) + ("\n" if new_lines else ""),
            encoding="utf-8",
        )

print(f"Annotations checked: {checked}")
print(f"Bounding boxes clipped: {fixed}")
print("Repair complete. Run the validator again.")