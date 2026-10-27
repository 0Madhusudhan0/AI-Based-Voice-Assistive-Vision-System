
from pathlib import Path
import random

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "datasets" / "stairs-yolov8-detection"
SPLIT = "train"
NUM_IMAGES = 12

CLASS_NAMES = {
    0: "door",
    1: "stairs",
}

IMAGE_DIR = DATASET_DIR / SPLIT / "images"
LABEL_DIR = DATASET_DIR / SPLIT / "labels"

image_paths = sorted(
    path
    for path in IMAGE_DIR.iterdir()
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)

if not image_paths:
    raise RuntimeError(f"No images found in {IMAGE_DIR}")

random.seed(42)
selected = random.sample(image_paths, min(NUM_IMAGES, len(image_paths)))

for image_path in selected:
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path.name}")
        continue

    height, width = image.shape[:2]
    label_path = LABEL_DIR / f"{image_path.stem}.txt"

    print(f"\nImage: {image_path.name}")
    print(f"Label file: {label_path}")
    if label_path.exists():
        print(label_path.read_text())

    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()

            if len(parts) != 5:
                continue

            class_id = int(parts[0])
            xc, yc, box_w, box_h = map(float, parts[1:])

            x1 = round((xc - box_w / 2) * width)
            y1 = round((yc - box_h / 2) * height)
            x2 = round((xc + box_w / 2) * width)
            y2 = round((yc + box_h / 2) * height)

            label = CLASS_NAMES.get(class_id, str(class_id))

            cv2.rectangle(
                image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            cv2.putText(
                image,
                label,
                (max(0, x1), max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )

    cv2.imshow("SightSpeak AI - Annotation Preview", image)
    print(f"Showing: {image_path.name}")

    key = cv2.waitKey(0)

    if key == 27:  # Esc to stop
        break

cv2.destroyAllWindows()
