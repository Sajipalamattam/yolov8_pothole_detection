# Pothole Detection with YOLOv8

This project uses Ultralytics YOLOv8 to detect potholes in road images. It contains
the training, validation, and single-image prediction scripts used for the project.

## What is included

- `scripts/train_model.py` - trains a YOLOv8n model on the pothole dataset.
- `scripts/test_model.py` - runs the trained model on validation images and saves annotated results.
- `scripts/test_single_image.py` - runs the trained model on one image.
- `data/data.yaml` - YOLO dataset configuration.
- `ref/` - sample reference images.

The training dataset, YOLO weights, trained checkpoints, and generated result folders
are excluded from Git because they are large. Therefore, cloning this repository alone
does not provide the dataset or the trained `best.pt` checkpoint.

## How the project works

1. YOLO reads the image and label paths from `data/data.yaml`.
2. `train_model.py` starts with the YOLOv8n base model and trains it to identify one class: `pothole`.
3. The best checkpoint is written to `results/pothole_detection_stable/weights/best.pt`.
4. The test scripts load that checkpoint, predict bounding boxes, and save annotated images.

## Requirements

- Python 3.10 or newer
- PyTorch
- Ultralytics YOLO
- NVIDIA GPU with CUDA is optional; training also runs on CPU, but it will be slower.

Install the dependencies:

```bash
pip install ultralytics torch
```

## Setup after cloning

From the repository root, create this local layout:

```text
data/
  data.yaml
  yolov8n.pt
  train/
    images/
    labels/
  valid/
    images/
    labels/
results/
  pothole_detection_stable/
    weights/
      best.pt
```

You need to provide one of these before using the test scripts:

- Train the model yourself by following the training steps below; or
- Copy an existing trained checkpoint to `results/pothole_detection_stable/weights/best.pt`.

The dataset must use YOLO label files. Each label file contains rows in this format:

```text
class_id center_x center_y width height
```

The values are normalized from 0 to 1. For this project, the pothole class ID is `0`.

## Dataset configuration

Open `data/data.yaml` if the dataset location or class names need to change. The
important settings are:

```yaml
train: train/images
val: valid/images
nc: 1
names: [pothole]
```

If you use a different number of classes, update `nc`, `names`, and the label class
IDs, then adjust the training script's `single_cls` setting.

## Train the model

Place the dataset and `yolov8n.pt` in `data/`, then run the command from the
repository root:

```bash
cd scripts
python train_model.py
```

The current training configuration is:

- Base model: YOLOv8n
- Epochs: 100
- Image size: 416
- Batch size: 2
- Early stopping patience: 20 epochs
- Single class: pothole
- Output: `results/pothole_detection_stable/weights/best.pt`

To change the training behavior, edit the arguments inside `model.train(...)` in
`scripts/train_model.py`, especially `epochs`, `imgsz`, `batch`, `device`, and `name`.

Important: `scripts/train_model.py` currently contains a Windows-specific absolute
dataset path. Change this line to your own local project path before training:

```python
os.chdir('D:/pothole_detection_project/data')
```

## Use an already trained model

The model was trained during development, but the trained checkpoint is not stored in
this GitHub repository because `.pt` files are ignored. To use that model, obtain the
project's `best.pt` file separately and place it here:

```text
results/pothole_detection_stable/weights/best.pt
```

After that, run validation on up to five images:

```bash
cd scripts
python test_model.py
```

Annotated predictions are saved under `test_results/pothole_test/`.

## Test one image

Open `scripts/test_single_image.py` and change `image_path` to the image you want to
test:

```python
image_path = '../data/valid/images/your_image.jpg'
```

Then run:

```bash
cd scripts
python test_single_image.py
```

The annotated image is saved under `single_test_results/single_image_test/`. Change
`conf=0.5` in either test script to use a different confidence threshold.

## Project status

- The YOLOv8 training and prediction scripts are included in this repository.
- The model was trained locally during development.
- The trained checkpoint, complete dataset, and generated results are not included in
  GitHub because they are excluded by `.gitignore`.
- A fresh clone therefore requires the dataset for training or a separately supplied
  `best.pt` file for prediction.
