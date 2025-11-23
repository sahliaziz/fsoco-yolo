# YOLO11 Model Training for FSOCO Dataset

This project trains a YOLO11 model to detect cones using the [FSOCO Dataset](http://fsoco.cs.uni-freiburg.de/).

## 1. Setup & Environment

### Install Dependencies

**Option 1: using `uv` (Recommended)**

```bash
# Create the virtual environment
uv venv

# Activate the environment
source .venv/bin/activate

# Sync dependencies from the lock file
uv sync
```
**Option 2: using `pip`**
```bash
pip install ultralytics ipykernel
```

## 2. Create Jupyter kernel
```bash
python -m ipykernel install --user --name=fsoco-train --display-name "fsoco-train (python3)"
```
Select the `fsoco-train (python3)` kernel when using Jupyter.

## Training Results

Training outputs are saved to `runs/detect/fsoco_yolo11n/` containing:
- `weights/best.pt`: Best model checkpoint
- `results.png`: Training curves

## 📚 Resources

- [Ultralytics YOLO11](https://github.com/ultralytics/ultralytics)
- [FSOCO Dataset](http://https://github.com/fsoco/fsoco-dataset)