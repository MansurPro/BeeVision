# BeeVision

BeeVision fine-tunes YOLOv10 models for detecting birds and bees, leveraging Google Colab's GPU resources for training and real-time inference.

## Key Features
- **Custom Dataset Training**: Fine-tune YOLOv10 on bird and bee datasets from [Roboflow](https://roboflow.com).
- **Real-Time Inference**: Pipelines for single-image and streaming detection.
- **Cloud-Based Workflow**: Designed for Google Colab with GPU acceleration.

## Technologies Used
- Python
- YOLOv10
- Google Colab
- Roboflow
- OpenCV

## Project Structure
```
BeeVision/
├── data/            # Datasets
├── models/          # Trained weights
├── notebooks/       # Jupyter notebooks and Colab files
├── src/             # Source code for training and inference
└── README.md
```

## Getting Started
1. Open `notebooks/beevision_workflow.ipynb` in Google Colab.
2. Execute the cells sequentially to install dependencies, prepare the dataset, train a model, and run inference.

## Dataset Preparation
Use the provided utility to download datasets from Roboflow. Example usage:

```python
from src.dataset import download_roboflow_dataset

download_roboflow_dataset(
    api_key="YOUR_API_KEY",
    workspace="YOUR_WORKSPACE",
    project="birds-and-bees",
    version=1,
)
```

The dataset will be stored under the `data/` directory by default.

## Model Fine-Tuning
Use the training helper to fine-tune YOLOv10 once the dataset is prepared:

```python
from src.train import train_yolov10

train_yolov10(
    data_yaml="data/birds-and-bees/data.yaml",
    model="yolov10n.pt",
    epochs=10,
)
```

An end-to-end workflow demonstrating these steps is available in `notebooks/beevision_workflow.ipynb`.

## Running Inference
Use the inference helper to run predictions on images or video using trained weights:

```python
from src.inference import run_inference

run_inference(
    weights="models/train/weights/best.pt",
    source="path/to/image_or_video.jpg",
)
```

For an interactive demonstration, run the inference cells in `notebooks/beevision_workflow.ipynb`.
