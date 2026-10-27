
from pathlib import Path
import shutil
from turtle import left, width
from turtle import left
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets" / "stairs-yolov8"
DEST = ROOT / "datasets" / "stairs-yolov8-detection"

SPLITS = ("train", "valid", "test")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ["door", "stairs"]


def convert_line(line, filename, line_number):
    parts = line.split()

    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"{filename}:{line_number}: non-numeric annotation"
        ) from exc

    if class_id not in (0, 1):
        raise ValueError(
            f"{filename}:{line_number}: unknown class ID {class_id}"
        )

    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            f"{filename}:{line_number}: coordinates outside [0, 1]"
        )

    # Case 1: Standard YOLO detection bounding box.
    if len(values) == 4:
        xc, yc, width, height = values

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{filename}:{line_number}: zero-area box"
            )

        left = xc - width / 2
        right = xc + width / 2
        top = yc - height / 2
        bottom = yc + height / 2

        tolerance = 1e-4
        if (
    left < -tolerance
    or right > 1 + tolerance
    or top < -tolerance
    or bottom > 1 + tolerance
):
            raise ValueError(
        f"{filename}:{line_number}: box extends outside image "
        f"(left={left:.6f}, right={right:.6f}, "
        f"top={top:.6f}, bottom={bottom:.6f})"
    )
        # Correct tiny rounding errors at image boundaries.
        xc = min(max(xc, width / 2), 1 - width / 2)
        yc = min(max(yc, height / 2), 1 - height / 2)

        return (
            f"{class_id} {xc:.8f} {yc:.8f} "
            f"{width:.8f} {height:.8f}"
        )

    # Case 2: YOLO segmentation polygon.
    if len(values) < 6 or len(values) % 2 != 0:
        raise ValueError(
            f"{filename}:{line_number}: malformed polygon or box"
        )

    xs = values[0::2]
    ys = values[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{filename}:{line_number}: zero-area polygon"
        )

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2

    return (
        f"{class_id} {xc:.8f} {yc:.8f} "
        f"{width:.8f} {height:.8f}"
    )
    parts = line.split()

    try:
        class_id = int(parts[0])
        values = [float(value) for value in parts[1:]]
    except (ValueError, IndexError) as exc:
        raise ValueError(
            f"{filename}:{line_number}: non-numeric annotation"
        ) from exc

    if class_id not in (0, 1):
        raise ValueError(
            f"{filename}:{line_number}: unknown class ID {class_id}"
        )

    if not all(0.0 <= value <= 1.0 for value in values):
        raise ValueError(
            f"{filename}:{line_number}: coordinates outside [0, 1]"
        )

    # Standard YOLO detection: class + xc + yc + width + height.
    if len(values) == 4:
        xc, yc, width, height = values

        if width <= 0 or height <= 0:
            raise ValueError(
                f"{filename}:{line_number}: zero-area box"
            )

        left = xc - width / 2
        right = xc + width / 2
        top = yc - height / 2
        bottom = yc + height / 2
        
        if (
            left < -1e-6
            or right > 1 + 1e-6
            or top < -1e-6
            or bottom > 1 + 1e-6
):
            raise ValueError(
                f"{filename}:{line_number}: box extends outside image "
                f"(left={left:.6f}, right={right:.6f}, "
                f"top={top:.6f}, bottom={bottom:.6f})"
            )

    # Polygon: class + at least 3 (x, y) pairs.
    if len(values) < 6 or len(values) % 2 != 0:
        raise ValueError(
            f"{filename}:{line_number}: malformed polygon or box"
        )

    xs = values[0::2]
    ys = values[1::2]

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{filename}:{line_number}: zero-area polygon"
        )

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2

    return (
        f"{class_id} {xc:.8f} {yc:.8f} "
        f"{width:.8f} {height:.8f}"
    )

    parts = line.split()

    if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
        raise ValueError(
            f"{filename}:{line_number}: malformed polygon annotation"
        )

    try:
        class_id = int(parts[0])
        coords = [float(value) for value in parts[1:]]
    except ValueError as exc:
        raise ValueError(
            f"{filename}:{line_number}: non-numeric annotation"
        ) from exc

    if class_id not in (0, 1):
        raise ValueError(
            f"{filename}:{line_number}: unknown class ID {class_id}"
        )

    if not all(0.0 <= v <= 1.0 for v in coords):
        raise ValueError(
            f"{filename}:{line_number}: coordinate outside [0, 1]"
        )

    xs = coords[0::2]
    ys = coords[1::2]

    if len(xs) < 3 or len(xs) != len(ys):
        raise ValueError(
            f"{filename}:{line_number}: insufficient polygon points"
        )

    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)

    width = xmax - xmin
    height = ymax - ymin

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{filename}:{line_number}: zero-area bounding box"
        )

    xc = (xmin + xmax) / 2
    yc = (ymin + ymax) / 2

    return (
        f"{class_id} {xc:.8f} {yc:.8f} "
        f"{width:.8f} {height:.8f}"
    )


def main():
    if not (SOURCE / "data.yaml").is_file():
        raise FileNotFoundError(f"Missing dataset YAML in {SOURCE}")

    config = yaml.safe_load(
        (SOURCE / "data.yaml").read_text(encoding="utf-8")
    )

    names = config.get("names")
    if names != CLASS_NAMES:
        raise ValueError(
            f"Expected class order {CLASS_NAMES}, found {names}"
        )

    if DEST.exists():
        raise FileExistsError(
            f"{DEST} already exists. Inspect it before retrying."
        )

    planned = {}
    totals = {name: 0 for name in CLASS_NAMES}

    # Validate all annotations before creating the destination.
    for split in SPLITS:
        image_dir = SOURCE / split / "images"
        labels_dir = SOURCE / split / "labels"

        if not image_dir.is_dir() or not labels_dir.is_dir():
            raise FileNotFoundError(f"Missing folders for {split}")

        images = sorted(
            p for p in image_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        if not images:
            raise ValueError(f"No images found for {split}")

        converted = {}

        for image in images:
            label = labels_dir / f"{image.stem}.txt"

            if not label.is_file():
                raise FileNotFoundError(
                    f"Missing label for {image.name}"
                )

            lines = []
            content = label.read_text(encoding="utf-8").strip()

            for number, line in enumerate(content.splitlines(), 1):
                if not line.strip():
                    continue

                output_line = convert_line(line, label.name, number)
                class_id = int(output_line.split()[0])
                totals[CLASS_NAMES[class_id]] += 1
                lines.append(output_line)

            converted[image.name] = lines

        planned[split] = (images, converted)

    # Write to a new dataset directory.
    try:
        for split, (images, labels) in planned.items():
            out_images = DEST / split / "images"
            out_labels = DEST / split / "labels"
            out_images.mkdir(parents=True)
            out_labels.mkdir(parents=True)

            for image in images:
                shutil.copy2(image, out_images / image.name)
                output = "\n".join(labels[image.name])
                if output:
                    output += "\n"

                (out_labels / f"{image.stem}.txt").write_text(
                    output, encoding="utf-8"
                )

        new_yaml = {
            "path": str(DEST.resolve()),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 2,
            "names": CLASS_NAMES,
        }

        (DEST / "data.yaml").write_text(
            yaml.safe_dump(new_yaml, sort_keys=False),
            encoding="utf-8",
        )

    except Exception:
        shutil.rmtree(DEST, ignore_errors=True)
        raise

    print(f"Converted dataset: {DEST}")
    print(f"Images: {sum(len(v[0]) for v in planned.values())}")
    print(f"Converted boxes by class: {totals}")
    print("Original dataset remains unchanged.")


if __name__ == "__main__":
    main()
