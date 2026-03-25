"""
SecuroServ Model Training Script
Fine-tunes YOLOv8 on your custom surveillance dataset.

Usage:
    python scripts/train.py --data datasets/securoserv/dataset.yaml

Requirements:
    pip install ultralytics torch torchvision
    GPU strongly recommended (CUDA or Apple Silicon MPS)
"""

import os
import argparse
import logging
from datetime import datetime

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def train(
    data_yaml: str,
    model_size: str = "n",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "auto",
    resume: bool = False,
    project: str = "runs/train",
):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed. Run: pip install ultralytics")
        return

    # Choose base weights
    base_model = f"yolov8{model_size}.pt"
    logger.info(f"Starting training with base model: {base_model}")
    logger.info(f"Dataset: {data_yaml}")
    logger.info(f"Epochs: {epochs}  |  ImgSz: {imgsz}  |  Batch: {batch}")

    model = YOLO(base_model)

    run_name = f"securoserv_{datetime.now():%Y%m%d_%H%M%S}"

    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=run_name,
        resume=resume,
        # Augmentation settings
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        shear=2.0,
        flipud=0.0,
        fliplr=0.5,
        mosaic=1.0,
        mixup=0.1,
        # Early stopping
        patience=20,
        # Save best weights
        save=True,
        save_period=10,
    )

    best_weights = os.path.join(project, run_name, "weights", "best.pt")
    logger.info(f"\nTraining complete!")
    logger.info(f"Best weights: {best_weights}")

    # Copy best weights to models/ for easy use
    os.makedirs("models", exist_ok=True)
    dest = "models/securoserv.pt"
    if os.path.exists(best_weights):
        import shutil
        shutil.copy2(best_weights, dest)
        logger.info(f"Best weights copied to: {dest}")
        logger.info("You can now run the application and it will use your custom model.")
    else:
        logger.warning("Best weights not found — check training output directory.")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train SecuroServ YOLOv8 model.")
    parser.add_argument("--data",    required=True, help="Path to dataset.yaml")
    parser.add_argument("--model",   default="n",  choices=["n","s","m","l","x"],
                        help="YOLOv8 model size (n=nano, s=small, m=medium, ...)")
    parser.add_argument("--epochs",  type=int, default=100)
    parser.add_argument("--imgsz",   type=int, default=640)
    parser.add_argument("--batch",   type=int, default=16)
    parser.add_argument("--device",  default="auto",
                        help="Device: 'cpu', '0' (GPU), 'mps' (Apple Silicon), or 'auto'")
    parser.add_argument("--resume",  action="store_true", help="Resume interrupted training")
    args = parser.parse_args()

    train(
        data_yaml=args.data,
        model_size=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
