
from pathlib import Path
import yaml

DATASET_DIR = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "stairs-yolov8-hardneg"
)

YAML_FILE = DATASET_DIR / "data.yaml"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
EXPECTED_CLASSES = {
    0: "door",
    1: "stairs",
}


def main():
    if not YAML_FILE.is_file():
        print(f"ERROR: YAML file not found: {YAML_FILE}")
        return

    with YAML_FILE.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    print(f"Dataset: {DATASET_DIR}")
    print(f"YAML: {YAML_FILE}")

    names = config.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))

    print("\nConfigured classes:")
    for class_id, name in names.items():
        print(f"  {class_id}: {name}")

    if names != EXPECTED_CLASSES:
        print("\nWARNING: Class names differ from expected mapping.")

    all_ok = True

    for split, key in [
        ("train", "train"),
        ("valid", "val"),
        ("test", "test"),
    ]:
        declared_path = config.get(key)

        if not declared_path:
            print(f"\n{split}: no '{key}' path configured")
            all_ok = False
            continue

        image_dir = (DATASET_DIR / declared_path).resolve()

        print(f"\n--- {split.upper()} ---")
        print(f"Configured path: {declared_path}")
        print(f"Resolved path:   {image_dir}")

        if not image_dir.is_dir():
            print("ERROR: Image directory does not exist.")
            all_ok = False
            continue

        images = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        labels_dir = image_dir.parent / "labels"
        print(f"Images found: {len(images)}")
        print(f"Labels folder: {labels_dir}")

        if not labels_dir.is_dir():
            print("ERROR: Matching labels folder not found.")
            all_ok = False
            continue

        missing_labels = []
        empty_labels = []
        invalid_lines = []
        class_counts = {i: 0 for i in EXPECTED_CLASSES}

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.is_file():
                missing_labels.append(image_path.name)
                continue

            content = label_path.read_text(
                encoding="utf-8"
            ).strip()

            # Empty label files can be valid for images with no objects.
            if not content:
                empty_labels.append(label_path.name)
                continue

            for line_number, line in enumerate(
                content.splitlines(), start=1
            ):
                parts = line.split()

                if len(parts) != 5:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number} "
                        f"(expected 5 values)"
                    )
                    continue

                try:
                    class_id = int(parts[0])
                    x, y, width, height = map(float, parts[1:])
                except ValueError:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number} "
                        f"(non-numeric value)"
                    )
                    continue

                if class_id not in class_counts:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number} "
                        f"(unknown class ID {class_id})"
                    )
                    continue

                if not all(
                    0.0 <= value <= 1.0
                    for value in (x, y, width, height)
                ) or width <= 0 or height <= 0:
                    invalid_lines.append(
                        f"{label_path.name}:{line_number} "
                        f"(invalid box coordinates)"
                    )
                    continue

                class_counts[class_id] += 1

        print(f"Missing label files: {len(missing_labels)}")
        print(f"Empty label files: {len(empty_labels)}")
        print(f"Invalid annotations: {len(invalid_lines)}")
        print("Valid annotation counts by class:")

        for class_id, class_name in EXPECTED_CLASSES.items():
            print(f"  {class_id} ({class_name}): "
                  f"{class_counts[class_id]}")

        if missing_labels:
            print("Examples of missing labels:", missing_labels[:5])
        if invalid_lines:
            print("Examples of invalid annotations:", invalid_lines[:5])

        if missing_labels or invalid_lines:
            all_ok = False

    print("\n" + "=" * 45)
    print("BASIC VALIDATION:", "PASSED" if all_ok else "NEEDS ATTENTION")
    print("Review path errors and annotation warnings before training.")


if __name__ == "__main__":
    main()
