from pathlib import Path
import shutil
import yaml

# Project and dataset paths
ROOT = Path(__file__).resolve().parent
DATASETS = ROOT / "datasets"

DOOR_STAIRS = DATASETS / "stairs-yolov8-detection"
WHITEBOARD = DATASETS / "whiteboard-source"
OUTPUT = DATASETS / "sightspeak-three-class"

SPLITS = {
    "train": "train",
    "valid": "valid",
    "test": "test",
}

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASSES = ["door", "stairs", "whiteboard"]


def get_images(folder):
    return sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_dataset_split(source, destination, dataset_type, split):
    source_images = source / split / "images"
    source_labels = source / split / "labels"

    if not source_images.is_dir():
        raise FileNotFoundError(f"Missing image directory: {source_images}")

    if not source_labels.is_dir():
        raise FileNotFoundError(f"Missing label directory: {source_labels}")

    images = get_images(source_images)

    for image_path in images:
        label_path = source_labels / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Missing label for {image_path}: {label_path}"
            )

        # Prefix filenames to avoid collisions between datasets.
        prefix = "wb" if dataset_type == "whiteboard" else "ds"
        new_stem = f"{prefix}_{image_path.stem}"

        destination_image = destination / "images" / (
            new_stem + image_path.suffix.lower()
        )
        destination_label = destination / "labels" / f"{new_stem}.txt"

        # Copy images without modifying the originals.
        shutil.copy2(image_path, destination_image)

        # Read and validate annotations.
        output_lines = []

        for line_number, line in enumerate(
            label_path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            line = line.strip()

            # Empty annotation files are allowed.
            if not line:
                continue

            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Invalid YOLO annotation: {label_path}, "
                    f"line {line_number}: {line}"
                )

            try:
                class_id = int(parts[0])
                coords = [float(value) for value in parts[1:]]
            except ValueError as exc:
                raise ValueError(
                    f"Non-numeric annotation: {label_path}, "
                    f"line {line_number}: {line}"
                ) from exc

            expected_id = 0 if dataset_type == "whiteboard" else None

            if dataset_type == "whiteboard":
                if class_id != expected_id:
                    raise ValueError(
                        f"Unexpected whiteboard class ID {class_id} "
                        f"in {label_path}. Expected 0."
                    )
                # Remap whiteboard class 0 to combined class 2.
                class_id = 2
            else:
                if class_id not in (0, 1):
                    raise ValueError(
                        f"Unexpected door/stairs class ID {class_id} "
                        f"in {label_path}."
                    )

            x_center, y_center, width, height = coords

            if not (
                0 <= x_center <= 1
                and 0 <= y_center <= 1
                and 0 < width <= 1
                and 0 < height <= 1
            ):
                raise ValueError(
                    f"Invalid bounding box coordinates in {label_path}: {line}"
                )

            output_lines.append(
                f"{class_id} "
                + " ".join(f"{value:.8f}" for value in coords)
            )

        destination_label.write_text(
            "\n".join(output_lines) + ("\n" if output_lines else ""),
            encoding="utf-8",
        )

    print(f"{dataset_type:10s} | {split:5s} | {len(images):4d} images")


def main():
    if not DOOR_STAIRS.is_dir():
        raise FileNotFoundError(DOOR_STAIRS)

    if not WHITEBOARD.is_dir():
        raise FileNotFoundError(WHITEBOARD)

    if OUTPUT.exists():
        raise FileExistsError(
            f"{OUTPUT} already exists. Nothing was overwritten. "
            "Inspect it or choose a new output directory."
        )

    # Create an entirely separate output dataset.
    for split in SPLITS:
        for subfolder in ("images", "labels"):
            (OUTPUT / split / subfolder).mkdir(
                parents=True, exist_ok=True
            )

    try:
        for split in SPLITS:
            copy_dataset_split(
                DOOR_STAIRS,
                OUTPUT / split,
                "door_stairs",
                SPLITS[split],
            )

            copy_dataset_split(
                WHITEBOARD,
                OUTPUT / split,
                "whiteboard",
                SPLITS[split],
            )

        data_yaml = {
            "path": OUTPUT.resolve().as_posix(),
            "train": "train/images",
            "val": "valid/images",
            "test": "test/images",
            "nc": 3,
            "names": CLASSES,
        }

        (OUTPUT / "data.yaml").write_text(
            yaml.safe_dump(data_yaml, sort_keys=False),
            encoding="utf-8",
        )

        print("\nDataset created successfully:")
        print(OUTPUT)
        print("\nDataset configuration:")
        print((OUTPUT / "data.yaml").read_text(encoding="utf-8"))

    except Exception:
        print(
            "\nPreparation failed. The incomplete output directory is being "
            "removed so that no partial dataset is mistaken for a valid one."
        )
        shutil.rmtree(OUTPUT, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
