import cv2
import numpy as np
from ultralytics import YOLO

# camera
fx = 604.6896362304688
fy = 603.9588012695312
cx = 327.1522521972656
cy = 252.92861938476562

CLASS_HEIGHTS = {
    0: 0.325, # blue_cone
    1: 0.325, # yellow_cone
    2: 0.325, # orange_cone
    3: 0.505  # large_orange_cone
}

model = YOLO('/Users/haminhanh/Downloads/yolov11_nano.pt')
cap = cv2.VideoCapture("/Users/haminhanh/Downloads/test_video.mp4")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    results = model.predict(source=frame, conf=0.45, imgsz=640, verbose=False)
    

    estimation_data = {
        "left_lane": [],  
        "right_lane": [], 
        "others": []     
    }

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0].item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            h_pixel = y2 - y1
            u_center = (x1 + x2) / 2
            H_real = CLASS_HEIGHTS.get(cls_id, 0.325)

            if h_pixel > 0:
                dist_z = (fy * H_real) / h_pixel
                dist_x = ((u_center - cx) * dist_z) / fx

                cone_info = {
                    "id": cls_id,
                    "x": round(dist_x, 3),
                    "z": round(dist_z, 3)
                }

                if cls_id == 0: # Blue -> Left
                    estimation_data["left_lane"].append(cone_info)
                elif cls_id == 1: # Yellow -> Right
                    estimation_data["right_lane"].append(cone_info)
                else:
                    estimation_data["others"].append(cone_info)

    estimation_data["left_lane"].sort(key=lambda c: c['z'])
    estimation_data["right_lane"].sort(key=lambda c: c['z'])

    if estimation_data["left_lane"] or estimation_data["right_lane"]:
        print("--- Frame Estimation Data ---")
        print(f"Left (Blue): {len(estimation_data['left_lane'])} ")
        print(f"Right (Yellow): {len(estimation_data['right_lane'])} ")
        if estimation_data["left_lane"]:
            print(f"  closest left: {estimation_data['left_lane'][0]}")
        if estimation_data["right_lane"]:
            print(f"  closest right: {estimation_data['right_lane'][0]}")
        print("-" * 30)

    cv2.imshow("Camera View", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()