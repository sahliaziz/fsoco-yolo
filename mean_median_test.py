import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
from sklearn.metrics import average_precision_score


# ------------------------------------------------------
# 1. Load YOLOv11 model
# ------------------------------------------------------
model = YOLO("your_yolov11_cone_model.pt")

# ------------------------------------------------------
# 2. Define class names
# ------------------------------------------------------
CLASS_NAMES = ["blue", "yellow", "orange", "large_orange"]
NUM_CLASSES = len(CLASS_NAMES)

# ------------------------------------------------------
# 3. Settings
# ------------------------------------------------------
video_path = "input_video.mp4"
labels_folder = Path("labels")   # Ground-truth YOLO labels
segment_size_frames = 100        # Compute mean/median every N frames


# ------------------------------------------------------
# Helper function to load ground truth for one frame
# ------------------------------------------------------
def load_ground_truth(label_path):
    """
    Returns a list of GT objects:
    each item: (cls_id, x_center, y_center, w, h)
    """
    if not label_path.exists():
        return []
    gt = []
    with label_path.open() as f:
        for line in f:
            cls, xc, yc, w, h = map(float, line.strip().split())
            gt.append((int(cls), xc, yc, w, h))
    return gt


# ------------------------------------------------------
# 4. Process video frame-by-frame
# ------------------------------------------------------
cap = cv2.VideoCapture(video_path)

# Storage for per-frame counts
frame_counts = []

# Storage for mAP computation
all_gt_boxes = []
all_pred_boxes = []

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    height, width = frame.shape[:2]

    # YOLO detection
    results = model(frame, verbose=False)[0]

    # Count detected cones
    counts = {cls: 0 for cls in CLASS_NAMES}
    predictions_per_frame = []

    for box in results.boxes:
        cls_id = int(box.cls)
        cls_name = CLASS_NAMES[cls_id]
        counts[cls_name] += 1

        # Store prediction in absolute pixel format
        x1, y1, x2, y2 = box.xyxy[0]
        predictions_per_frame.append([cls_id, float(x1), float(y1), float(x2), float(y2)])

    frame_counts.append(counts)

    # -------- Load ground truth for mAP ----------
    label_file = labels_folder / f"{frame_idx:06d}.txt"
    gt = load_ground_truth(label_file)
    gt_abs = []

    for cls, xc, yc, w, h in gt:
        # Convert normalized YOLO format to ABS pixel coordinates
        x1 = (xc - w/2) * width
        y1 = (yc - h/2) * height
        x2 = (xc + w/2) * width
        y2 = (yc + h/2) * height
        gt_abs.append([cls, x1, y1, x2, y2])

    all_gt_boxes.append(gt_abs)
    all_pred_boxes.append(predictions_per_frame)

cap.release()


# ------------------------------------------------------
# 5. Compute MEAN / MEDIAN per segment
# ------------------------------------------------------
num_frames = len(frame_counts)
num_segments = (num_frames + segment_size_frames - 1) // segment_size_frames

print("\n================ Mean / Median Per Segment ================")

for seg in range(num_segments):
    start = seg * segment_size_frames
    end = min((seg + 1) * segment_size_frames, num_frames)

    segment_data = frame_counts[start:end]

    print(f"\n--- Segment {seg+1}: Frames {start+1} to {end} ---")

    for cls in CLASS_NAMES:
        values = np.array([fc[cls] for fc in segment_data])

        mean_val = np.mean(values)
        median_val = np.median(values)

        print(f"  {cls:12s} | mean: {mean_val:.2f} | median: {median_val:.2f}")


# ------------------------------------------------------
# 6. Compute mAP (very simplified IoU-matching version)
# ------------------------------------------------------
def compute_iou(box1, box2):
    """IoU between two boxes [x1,y1,x2,y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2]-box1[0]) * (box1[3]-box1[1])
    area2 = (box2[2]-box2[0]) * (box2[3]-box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


print("\n================ mAP (simplified) ================")
print("NOTE: This requires correct label files in YOLO format.\n")

map_per_class = []

for cls_id in range(NUM_CLASSES):
    y_true_all = []
    y_pred_all = []

    for gt_frame, pred_frame in zip(all_gt_boxes, all_pred_boxes):
        # Get all GT and predictions of this class only
        gt_cls = [g for g in gt_frame if g[0] == cls_id]
        pred_cls = [p for p in pred_frame if p[0] == cls_id]

        # For every prediction → compute IoU with the best GT match
        for p in pred_cls:
            best_iou = 0
            for g in gt_cls:
                iou = compute_iou(p[1:], g[1:])
                best_iou = max(best_iou, iou)

            # Prediction score (1 if IoU≥0.5 else 0)
            y_pred_all.append(1 if best_iou >= 0.5 else 0)
            y_true_all.append(1)

        # Add FN (GT with no prediction)
        if len(pred_cls) < len(gt_cls):
            missing = len(gt_cls) - len(pred_cls)
            y_true_all.extend([1] * missing)
            y_pred_all.extend([0] * missing)

    # Compute AP
    if len(y_true_all) > 0:
        ap = average_precision_score(y_true_all, y_pred_all)
    else:
        ap = 0.0

    map_per_class.append(ap)
    print(f"AP for class {CLASS_NAMES[cls_id]}: {ap:.4f}")

print(f"\nmAP: {np.mean(map_per_class):.4f}")
