# Pothole Detection with YOLOv8

This project trains and evaluates a YOLOv8 model that detects potholes in road images.

## Requirements

- Python 3.10 or newer
- PyTorch
- Ultralytics YOLO

Install the Python dependencies:

```bash
pip install ultralytics torch
```

## Dataset layout

Place the dataset locally in this layout:

```text
data/
  data.yaml
  train/
    images/
    labels/
  valid/
    images/
    labels/
```

The dataset, trained checkpoints, and generated results are deliberately excluded from Git because they are large.

## Run

Run commands from the `scripts` directory:

```bash
python train_model.py
python test_model.py
python test_single_image.py
```

The trained model is expected at `results/pothole_detection_stable/weights/best.pt`.
