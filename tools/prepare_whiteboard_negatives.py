
from pathlib import Path
import random
import shutil

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SOURCE_IMAGES = (
    PROJECT_ROOT
    / "datasets"
    / "whiteboard-source"
    / "train"
    / "images"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "datasets"
    / "whiteboard-negative-review"
)

OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_LABELS = OUTPUT_ROOT / "labels"

BATCH_SIZE = 20
SEED = 42

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    if not SOURCE_IMAGES.is_dir():
        raise FileNotFoundError(
            f"Source images folder not found: {SOURCE_IMAGES}"
        )

    # Avoid overwriting a previous review batch.
    if OUTPUT_ROOT.exists() and any(OUTPUT_ROOT.iterdir()):
        raise FileExistsError(
            f"Output folder is not empty: {OUTPUT_ROOT}\n"
            "Inspect or rename it before running this script again."
        )

    images = sorted(
        path
        for path in SOURCE_IMAGES.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if len(images) < BATCH_SIZE:
        raise ValueError(
            f"Expected at least {BATCH_SIZE} images, found {len(images)}."
        )

    random.Random(SEED).shuffle(images)
    selected = images[:BATCH_SIZE]

    OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
    OUTPUT_LABELS.mkdir(parents=True, exist_ok=True)

    for source in selected:
        shutil.copy2(source, OUTPUT_IMAGES / source.name)

        # REVIEW ONLY: these labels are empty placeholders.
        # Do not train on this batch until every image has been
        # checked for actual doors and stairs.
        (OUTPUT_LABELS / f"{source.stem}.txt").write_text(
            "", encoding="utf-8"
        )

    print(f"Prepared {len(selected)} images for manual review.")
    print(f"Images: {OUTPUT_IMAGES}")
    print(f"Placeholder labels: {OUTPUT_LABELS}")
    print("IMPORTANT: These are NOT approved training labels yet.")


if __name__ == "__main__":
    main()

