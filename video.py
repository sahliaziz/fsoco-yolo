import cv2
import numpy as np
from ultralytics import YOLO

# Model and paths
model = YOLO("/home/ash/Downloads/fsoco_yolo11n_imgsz1216_fp16.engine")
video_out = "video_out.mp4"

# Classes and colors (BGR)
CLASS_NAMES = {0: "blue_cone", 1: "yellow_cone", 2: "orange_cone", 3: "large_orange_cone"}
COLORS = {0: (255, 0, 0), 1: (0, 255, 255), 2: (0, 165, 255), 3: (0, 140, 255)}

# Detection threshold
CONFIDENCE_THRESHOLD = 0.7

video_path = "/home/ash/Downloads/test_video.mp4"
cap = cv2.VideoCapture(video_path)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(video_out, fourcc, fps, (w, h))

frame_count = 0

def map_to_radar(cx, cy, img_w, img_h, radar_w, radar_h, margin=8):
    """Map image center (cx,cy) to radar coordinates.
    Mapping: image center -> radar center. X and Y are normalized by half-width/height.
    """
    img_cx = img_w / 2.0
    img_cy = img_h / 2.0

    # normalized offsets in [-1,1]
    nx = (cx - img_cx) / (img_w / 2.0)
    ny = (img_cy - cy) / (img_h / 2.0)  # invert y so forward/up maps up on radar

    # clamp
    nx = max(-1.0, min(1.0, nx))
    ny = max(-1.0, min(1.0, ny))

    rx = int(radar_w / 2.0 + nx * (radar_w / 2.0 - margin))
    ry = int(radar_h / 2.0 - ny * (radar_h / 2.0 - margin))
    return rx, ry

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    percent = (frame_count / max(1, total_frames)) * 100
    bar_length = 40
    filled = int(bar_length * frame_count / max(1, total_frames))
    bar = '█' * filled + '-' * (bar_length - filled)
    print(f'\rProcessing: |{bar}| {percent:.1f}%', end='', flush=True)

    results = model(frame, verbose=False)

    # Draw normal bboxes on the frame and collect detections for radar
    detections = []  # list of (cx, cy, area, cls)
    for box in results[0].boxes:
        conf = float(box.conf[0])
        if conf > CONFIDENCE_THRESHOLD:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            area = max(1, (x2 - x1) * (y2 - y1))
            detections.append((cx, cy, area, cls, conf, (x1, y1, x2, y2)))

            color = COLORS.get(cls, (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{CLASS_NAMES.get(cls, 'unknown')} {conf:.2f}",
                        (x1, max(0, y1-8)), cv2.FONT_HERSHEY_PLAIN, 0.8, color, 1)

    # Create radar image (square) sized relative to frame
    max_radar = min(360, w // 3, h // 3)
    radar_size = max(128, max_radar)
    radar_h = radar_w = radar_size
    radar = np.zeros((radar_h, radar_w, 3), dtype=np.uint8)  # black background

    # draw radar center crosshair
    cx_r, cy_r = radar_w // 2, radar_h // 2
    cv2.line(radar, (cx_r - 10, cy_r), (cx_r + 10, cy_r), (50, 50, 50), 1)
    cv2.line(radar, (cx_r, cy_r - 10), (cx_r, cy_r + 10), (50, 50, 50), 1)
    cv2.circle(radar, (cx_r, cy_r), 3, (80, 80, 80), -1)

    # draw each detection as a colored dot; size scaled by bbox area
    for (cx, cy, area, cls, conf, bbox) in detections:
        rx, ry = map_to_radar(cx, cy, w, h, radar_w, radar_h)

        # area -> radius mapping (area relative to image area)
        a_norm = area / float(max(1, w * h))
        # scale radius: make small dots for small boxes and larger for big boxes
        radius = int(max(2, min(40, 2 + a_norm * 400)))

        color = COLORS.get(cls, (255, 255, 255))
        # draw filled circle
        cv2.circle(radar, (rx, ry), radius, color, -1)
        # optional small outline for visibility
        cv2.circle(radar, (rx, ry), max(1, radius//3), (0, 0, 0), 1)

    # place radar on the top-right corner of the frame with a small margin
    margin = 10
    radar_h, radar_w = radar.shape[:2]
    if radar_w + margin < w and radar_h + margin < h:
        x_off = w - radar_w - margin
        y_off = margin
        frame[y_off:y_off+radar_h, x_off:x_off+radar_w] = radar
        # draw small border
        cv2.rectangle(frame, (x_off-1, y_off-1), (x_off+radar_w, y_off+radar_h), (100, 100, 100), 1)
    else:
        # if frame too small, skip overlay
        pass

    out.write(frame)

cap.release()
out.release()
print(f"\n✅ Video result: {video_out}")
