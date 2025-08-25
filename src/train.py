"""Training utilities for BeeVision."""
from pathlib import Path

from yolov10 import YOLO


def train_yolov10(
    data_yaml: str,
    model: str = "yolov10n.pt",
    epochs: int = 100,
    imgsz: int = 640,
    project: str = "models",
) -> None:
    """Fine-tune a YOLOv10 model.

    Parameters
    ----------
    data_yaml: str
        Path to the dataset YAML file.
    model: str, optional
        Pretrained weights to start from (default "yolov10n.pt").
    epochs: int, optional
        Number of training epochs (default 100).
    imgsz: int, optional
        Training image size (default 640).
    project: str, optional
        Directory to save training results (default "models").
    """
    data_path = Path(data_yaml).resolve()
    yolo = YOLO(model)
    yolo.train(data=str(data_path), epochs=epochs, imgsz=imgsz, project=project)
