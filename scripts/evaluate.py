"""
SecuroServ Model Evaluation Script
Evaluates a trained model on the test set and prints metrics.

Usage:
    python scripts/evaluate.py --model models/securoserv.pt --data datasets/securoserv/dataset.yaml
"""

import argparse
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def evaluate(model_path: str, data_yaml: str, imgsz: int = 640, split: str = "test"):
    try:
        from ultralytics import YOLO
    except ImportError:
        logger.error("ultralytics not installed.")
        return

    logger.info(f"Evaluating: {model_path}")
    logger.info(f"Dataset:    {data_yaml}  (split={split})")

    model = YOLO(model_path)
    metrics = model.val(
        data=data_yaml,
        split=split,
        imgsz=imgsz,
        verbose=True,
    )

    logger.info("\n===== EVALUATION RESULTS =====")
    logger.info(f"mAP@0.5:     {metrics.box.map50:.4f}")
    logger.info(f"mAP@0.5:0.95 {metrics.box.map:.4f}")
    logger.info(f"Precision:   {metrics.box.mp:.4f}")
    logger.info(f"Recall:      {metrics.box.mr:.4f}")
    logger.info("Per-class mAP@0.5:")
    for i, name in enumerate(model.names.values()):
        ap50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
        logger.info(f"  {name:<25} {ap50:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--data",  required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()
    evaluate(args.model, args.data, args.imgsz, args.split)


if __name__ == "__main__":
    main()
