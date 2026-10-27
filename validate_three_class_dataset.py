from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent
DATASET = ROOT / "datasets" / "sightspeak-three-class"

SPLITS = ["train", "valid", "test"]
CLASS_NAMES = {
    0: "door",
    1: "stairs",
    2: "whiteboard",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

errors = []
overall_counts = Counter()
overall_images = Counter()
empty_labels = Counter()

for split in SPLITS:
    split_dir = DATASET / split
    image_dir = split_dir / "images"
    label_dir = split_dir / "labels"

    images = [
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    ]

    image_stems = {p.stem for p in images}
    label_files = list(label_dir.glob("*.txt"))
    label_stems = {p.stem for p in label_files}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    if missing_labels:
        errors.append(
            f"{split}: images missing labels: {sorted(missing_labels)[:10]}"
        )

    if orphan_labels:
        errors.append(
            f"{split}: labels without matching images: "
            f"{sorted(orphan_labels)[:10]}"
        )

    counts = Counter()
    empty_count = 0

    for label_file in label_files:
        lines = label_file.read_text(encoding="utf-8").splitlines()
        lines = [line.strip() for line in lines if line.strip()]

        if not lines:
            empty_count += 1

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                errors.append(
                    f"{label_file}, line {line_number}: expected 5 values"
                )
                continue

            try:
                class_id = int(parts[0])
                x, y, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(
                    f"{label_file}, line {line_number}: non-numeric value"
                )
                continue

            if class_id not in CLASS_NAMES:
                errors.append(
                    f"{label_file}, line {line_number}: "
                    f"unexpected class ID {class_id}"
                )
                continue

            if not (
                0 <= x <= 1
                and 0 <= y <= 1
                and 0 < w <= 1
                and 0 < h <= 1
                and x - w / 2 >= -1e-6
                and y - h / 2 >= -1e-6
                and x + w / 2 <= 1 + 1e-6
                and y + h / 2 <= 1 + 1e-6
            ):
                errors.append(
                    f"{label_file}, line {line_number}: "
                    f"invalid bounding box {line}"
            )
                continue

            counts[class_id] += 1

    overall_counts.update(counts)
    overall_images[split] = len(images)
    empty_labels[split] = empty_count

    print(f"\n--- {split.upper()} ---")
    print(f"Images: {len(images)}")
    print(f"Label files: {len(label_files)}")
    print(f"Empty label files: {empty_count}")

    for class_id, name in CLASS_NAMES.items():
        print(f"{name}: {counts[class_id]} instances")

print("\n--- OVERALL INSTANCE COUNTS ---")
for class_id, name in CLASS_NAMES.items():
    print(f"{name}: {overall_counts[class_id]} instances")

print("\n--- VALIDATION RESULT ---")
if errors:
    print(f"FAILED: {len(errors)} issue(s) found.")
    for error in errors[:30]:
        print("-", error)
    if len(errors) > 30:
        print(f"...and {len(errors) - 30} more.")
else:
    print("PASSED: no structural or bounding-box errors found.")
