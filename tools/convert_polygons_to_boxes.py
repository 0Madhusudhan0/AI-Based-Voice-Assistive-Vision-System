
from pathlib import Path
import shutil
import yaml

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "datasets"
    / "doors-stairs"
)
DEST = SOURCE.parent / "doors-stairs-detection"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def convert_polygon_line(line: str, label_path: Path, line_number: int):
    parts = line.split()

    if len(parts) < 7 or len(parts) % 2 == 0:
        raise ValueError(
            f"{label_path.name}:{line_number}: "
            "expected class ID followed by at least 3 (x, y) points"
        )

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"{label_path.name}:{line_number}: non-numeric value"
        ) from exc

    if class_id < 0:
        raise ValueError(
            f"{label_path.name}:{line_number}: negative class ID"
        )

    if not all(0.0 <= value <= 1.0 for value in coords):
        raise ValueError(
            f"{label_path.name}:{line_number}: "
            "polygon coordinates must be normalized to [0, 1]"
        )

    xs = coords[0::2]
    ys = coords[1::2]

    if len(xs) < 3 or len(xs) != len(ys):
        raise ValueError(
            f"{label_path.name}:{line_number}: invalid polygon points"
        )

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{label_path.name}:{line_number}: zero-area bounding box"
        )

    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2

    return (
        f"{class_id} {x_center:.8f} {y_center:.8f} "
        f"{width:.8f} {height:.8f}"
    )


def main():
    if not SOURCE.is_dir():
        raise FileNotFoundError(f"Source dataset not found: {SOURCE}")

    yaml_path = SOURCE / "data.yaml"
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {yaml_path}")

    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = config.get("names")

    if isinstance(names, list):
        class_names = names
    elif isinstance(names, dict):
        class_names = [
            names[key] for key in sorted(names, key=lambda k: int(k))
        ]
    else:
        raise ValueError("data.yaml must define class names.")

    if DEST.exists():
        raise FileExistsError(
            f"Destination already exists: {DEST}\n"
            "Choose a new destination or inspect/remove it manually. "
            "The script will not overwrite it."
        )

    # Validate all source annotations before writing the new dataset.
    planned = {}
    total_annotations = 0

    for split in SPLITS:
        image_dir = SOURCE / split / "images"
        labels_dir = SOURCE / split / "labels"

        if not image_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Expected image and label folders for split '{split}'."
            )

        images = sorted(
            p for p in image_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(f"No images found in {image_dir}")

        image_stems = [p.stem for p in images]
        if len(image_stems) != len(set(image_stems)):
            raise ValueError(
                f"Duplicate image stems in {image_dir}; "
                "cannot safely map labels by filename."
            )

        converted_labels = {}
        missing = []

        for image_path in images:
            label_path = labels_dir / f"{image_path.stem}.txt"

            if not label_path.is_file():
                missing.append(image_path.name)
                continue

            converted_lines = []
            content = label_path.read_text(encoding="utf-8").strip()

            for line_number, line in enumerate(content.splitlines(), 1):
                if not line.strip():
                    continue

                converted = convert_polygon_line(
                    line, label_path, line_number
                )
                class_id = int(converted.split()[0])

                if class_id >= len(class_names):
                    raise ValueError(
                        f"{label_path.name}:{line_number}: "
                        f"class ID {class_id} is not defined in data.yaml"
                    )

                converted_lines.append(converted)

            converted_labels[image_path.name] = (
                label_path.name,
                converted_lines,
            )
            total_annotations += len(converted_lines)

        if missing:
            raise ValueError(
                f"{split}: {len(missing)} images lack labels. "
                f"Examples: {missing[:5]}"
            )

        planned[split] = (images, converted_labels)

    # Only create the destination after all source labels pass validation.
    DEST.mkdir(parents=True)

    try:
        for split, (images, labels) in planned.items():
            out_images = DEST / split / "images"
            out_labels = DEST / split / "labels"
            out_images.mkdir(parents=True)
            out_labels.mkdir(parents=True)

            for image_path in images:
                shutil.copy2(image_path, out_images / image_path.name)

                _, converted_lines = labels[image_path.name]
                label_text = "\n".join(converted_lines)
                if label_text:
                    label_text += "\n"

                (out_labels / f"{image_path.stem}.txt").write_text(
                    label_text, encoding="utf-8"
                )

        new_config = {
            "path": str(DEST.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": len(class_names),
            "names": class_names,
        }

        (DEST / "data.yaml").write_text(
            yaml.safe_dump(new_config, sort_keys=False),
            encoding="utf-8",
        )

    except Exception:
        shutil.rmtree(DEST, ignore_errors=True)
        raise

    print(f"Conversion complete: {DEST}")
    print(f"Classes: {class_names}")
    print(f"Total polygon annotations converted: {total_annotations}")
    print("Original dataset and labels were not modified.")
    print("Next: run the validator against the new dataset.")


if __name__ == "__main__":
    main()
