"""Inference utilities for BeeVision."""
from pathlib import Path

from yolov10 import YOLO


def run_inference(
    weights: str,
    source: str,
    imgsz: int = 640,
    project: str = "runs",
) -> None:
    """Run inference using a YOLOv10 model.

    Parameters
    ----------
    weights: str
        Path to the model weights.
    source: str
        Path to an image, video, or directory for inference.
    imgsz: int, optional
        Inference image size (default 640).
    project: str, optional
        Directory to save inference results (default "runs").
    """
    model_path = Path(weights).resolve()
    yolo = YOLO(model_path)
    yolo.predict(source=source, imgsz=imgsz, project=project)
