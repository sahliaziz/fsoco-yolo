# YOLO Cone Detection Project

A machine learning project for detecting traffic cones (blue, yellow, and orange) using YOLOv11 object detection models. This project utilizes the FSOCO (Formula Student Objects in Context) dataset and includes data preprocessing, model training, and evaluation capabilities.

## 📋 Project Overview

This project implements a complete pipeline for:
- **Dataset Processing**: Converting FSOCO dataset annotations to YOLO format
- **Model Training**: Training YOLOv11 models (nano and small variants) on cone detection
- **Object Detection**: Detecting and classifying traffic cones in images and video

### Dataset Information

The project uses two datasets:
1. **FSOCO Dataset** (`fsoco-dataset/`): 40+ formula student teams' annotated images with bounding boxes
2. **TraCon Dataset** (`TraCon_dataset/`): Additional training data for model validation

### Classes

The model detects three cone types:
- **Class 0**: Blue cone
- **Class 1**: Yellow cone  
- **Class 2**: Orange cone

## 📁 Project Structure

```
.
├── Model_training.ipynb          # Main training notebook
├── dataset.yaml                  # YOLO dataset configuration
├── pyproject.toml               # Project dependencies
├── yolo11n.pt                   # Pre-trained nano model
├── yolo11s.pt                   # Pre-trained small model
├── fsoco-dataset/               # Original FSOCO dataset (40+ teams)
│   ├── meta.json                # Dataset metadata
│   └── [team_name]/
│       ├── ann/                 # JSON annotations
│       └── img/                 # Images
├── fsoco-dataset-yolo/          # Converted YOLO format
│   ├── train/                   # Training split (70%)
│   ├── val/                     # Validation split (15%)
│   └── test/                    # Test split (15%)
├── TraCon_dataset/              # Alternative dataset
│   ├── train/
│   ├── val/
│   └── test/
└── runs/                        # Training outputs
    └── detect/
        └── train[N]/            # Model checkpoints and metrics
```

## 🚀 Getting Started

### Prerequisites

- Python 3.13+
- pip or conda package manager

### Installation

1. Clone or navigate to the project directory:
```bash
cd /home/ash/Documents/yolo
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install ultralytics>=8.3.228
```

Or use the project configuration:
```bash
pip install -e .
```

## 📚 Usage

### Dataset Preparation

The `Model_training.ipynb` notebook contains the complete pipeline:

1. **Download Dataset**:
   ```python
   !wget http://fsoco.cs.uni-freiburg.de/datasets/fsoco_bounding_boxes_train.zip
   ```

2. **Extract and Convert**:
   - Extracts the FSOCO dataset
   - Converts COCO-format JSON annotations to YOLO format
   - Normalizes bounding boxes to YOLO normalized XYWH format
   - Splits data into train/val/test sets (70/15/15)

### Model Training

Train a YOLOv11 model using the converted dataset:

```python
from ultralytics import YOLO

# Load a pretrained model
model = YOLO('yolo11n.pt')  # nano model
# or
model = YOLO('yolo11s.pt')  # small model

# Train the model
results = model.train(
    data='dataset.yaml',
    epochs=100,
    imgsz=640,
    device=0
)
```

### Inference

Perform object detection on images:

```python
from ultralytics import YOLO

model = YOLO('runs/detect/train[N]/weights/best.pt')
results = model.predict(source='image.jpg', conf=0.5)
```

## 📊 Dataset Details

### FSOCO Dataset
- **40+ Formula Student Teams** represented
- **Diverse Environments**: Different lighting, weather, and track conditions
- **Automatic Split**: Data is randomly split into 70% training, 15% validation, 15% test
- **Format**: Converted from COCO JSON annotations to YOLO format

### Annotation Format

Original COCO format (JSON):
```json
{
  "size": {"width": 1920, "height": 1080},
  "objects": [
    {
      "name": "blue_cone",
      "bndbox": {
        "xmin": 100, "ymin": 200,
        "xmax": 150, "ymax": 300
      }
    }
  ]
}
```

Converted to YOLO normalized format:
```
<class_id> <center_x> <center_y> <width> <height>
```

All coordinates are normalized to [0, 1].

## 🎯 Available Models

Two pretrained YOLOv11 models are included:

- **yolo11n.pt**: Nano model (faster, less accurate, ~2.6M parameters)
- **yolo11s.pt**: Small model (balanced, ~11.2M parameters)

## 📈 Training Results

Training outputs are saved to `runs/detect/train[N]/` containing:
- `weights/best.pt`: Best model checkpoint
- `weights/last.pt`: Latest model checkpoint
- `results.csv`: Training metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `results.png`: Training curves

## 🔧 Configuration

### Dataset Configuration (`dataset.yaml`)

```yaml
path: fsoco-dataset-yolo        # Dataset root path
train: train/images             # Training images path
val: val/images                 # Validation images path
test: test/images               # Test images path
nc: 3                           # Number of classes
names: ["blue_cone", "yellow_cone", "orange_cone"]
```

### Training Configuration

Modify hyperparameters in the training script:
- `epochs`: Number of training epochs
- `imgsz`: Input image size (default: 640)
- `batch`: Batch size
- `device`: GPU device ID (0 for first GPU, -1 for CPU)
- `patience`: Early stopping patience

## 📝 Notebook Structure

The `Model_training.ipynb` contains:

1. Dataset download and extraction
2. COCO to YOLO format conversion
3. Data validation and visualization
4. Model training with validation
5. Inference and evaluation
6. Results visualization

## 🐛 Troubleshooting

### Missing Images
If images are not found during conversion, check:
- Image extensions (.png vs .jpg)
- File paths match between annotations and images
- Dataset extraction was successful

### CUDA Issues
If GPU is not detected:
```bash
pip install ultralytics[export]
```

## 📚 Resources

- [Ultralytics YOLOv11 Documentation](https://github.com/ultralytics/ultralytics)
- [FSOCO Dataset](http://fsoco.cs.uni-freiburg.de/)
- [YOLO Format Documentation](https://docs.ultralytics.com/datasets/detect/)

## 📄 License

This project uses publicly available datasets and the Ultralytics YOLO framework.

## 🤝 Contributing

To extend this project:
- Add additional datasets in the same YOLO format
- Implement custom data augmentation
- Experiment with different model architectures
- Add real-time video inference capabilities

---

**Last Updated**: November 2025
