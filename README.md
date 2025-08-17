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
1. Open `notebooks/beevision_setup.ipynb` in Google Colab.
2. Run the setup cells to verify Python and pip versions and install dependencies.
3. Follow forthcoming notebooks or scripts for training and inference.

