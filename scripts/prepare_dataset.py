"""
SecuroServ Dataset Preparation Script
Helps organize and prepare custom labeled data for YOLOv8 training.

Usage:
    python scripts/prepare_dataset.py --source /path/to/raw_images --output datasets/securoserv

Directory structure expected:
    raw_images/
        violence/          ← image files
        choking/
        camera_tampering/
        restricted_area/
        loitering/
        normal/            ← negative examples

Labels must be provided as YOLO-format .txt files alongside images,
OR use the --roboflow flag to download from a Roboflow project.
"""

import os
import sys
import shutil
import random
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

CLASSES = [
    "violence",
    "choking",
    "camera_tampering",
    "restricted_area",
    "loitering",
    "unattended_object",
]

SPLIT_RATIOS = {"train": 0.75, "val": 0.15, "test": 0.10}


def create_yaml(output_dir: str, class_names: list) -> str:
    """Create dataset YAML for YOLOv8."""
    yaml_content = f"""# SecuroServ Custom Dataset
path: {os.path.abspath(output_dir)}
train: images/train
val:   images/val
test:  images/test

nc: {len(class_names)}
names: {class_names}
"""
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    logger.info(f"Dataset YAML written to {yaml_path}")
    return yaml_path


def split_and_copy(source_dir: str, output_dir: str):
    """
    Walk source directory, collect image+label pairs, split, and copy.
    """
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    pairs = []

    for root, _, files in os.walk(source_dir):
        for fname in files:
            ext = Path(fname).suffix.lower()
            if ext in image_exts:
                img_path = os.path.join(root, fname)
                label_path = os.path.splitext(img_path)[0] + ".txt"
                if os.path.exists(label_path):
                    pairs.append((img_path, label_path))
                else:
                    logger.warning(f"No label found for {img_path}, skipping.")

    if not pairs:
        logger.error("No image+label pairs found. Exiting.")
        sys.exit(1)

    random.shuffle(pairs)
    n = len(pairs)
    n_train = int(n * SPLIT_RATIOS["train"])
    n_val   = int(n * SPLIT_RATIOS["val"])

    splits = {
        "train": pairs[:n_train],
        "val":   pairs[n_train:n_train + n_val],
        "test":  pairs[n_train + n_val:],
    }

    for split, items in splits.items():
        img_dir = os.path.join(output_dir, "images", split)
        lbl_dir = os.path.join(output_dir, "labels", split)
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(lbl_dir, exist_ok=True)
        for img_path, label_path in items:
            shutil.copy2(img_path, img_dir)
            shutil.copy2(label_path, lbl_dir)
        logger.info(f"  {split}: {len(items)} images")

    logger.info(f"\nTotal: {n} image-label pairs split across train/val/test.")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare SecuroServ training dataset."
    )
    parser.add_argument("--source",  required=True,  help="Source image directory")
    parser.add_argument("--output",  default="datasets/securoserv", help="Output directory")
    parser.add_argument("--classes", nargs="+", default=CLASSES, help="Class names")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    logger.info(f"Preparing dataset: {args.source} → {args.output}")
    logger.info(f"Classes: {args.classes}")

    split_and_copy(args.source, args.output)
    yaml_path = create_yaml(args.output, args.classes)

    logger.info(f"\nDataset ready! To train, run:")
    logger.info(f"  python scripts/train.py --data {yaml_path}")


if __name__ == "__main__":
    main()
