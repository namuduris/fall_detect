import cv2
import numpy as np
import pandas as pd
import cvzone
from ultralytics import YOLO
from mss import mss

# ── CONFIG ──
# Path to your YOLO model and class file
MODEL_PATH = "yolov10s.pt"
COCO_TXT  = "/Users/karthikeyanamuduri/Desktop/waste6/person_fall_detection/coco.txt"

# Region-of‑interest: change these to match the window you want to monitor
ROI_LEFT   = 100    # x‑coordinate of the top‑left corner
ROI_TOP    = 100    # y‑coordinate of the top‑left corner
ROI_WIDTH  = 800    # width of capture region
ROI_HEIGHT = 600    # height of capture region

# Detection cadence
FRAME_SKIP = 3      # process every 3rd frame

# ── SETUP ──
model = YOLO(MODEL_PATH)
with open(COCO_TXT, "r") as f:
    classes = f.read().splitlines()

sct = mss()
count = 0

# Create a named window (optional)
cv2.namedWindow("Fall Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Fall Detection", ROI_WIDTH, ROI_HEIGHT)

while True:
    count += 1
    # grab only the ROI you care about
    monitor = {
        "left":   ROI_LEFT,
        "top":    ROI_TOP,
        "width":  ROI_WIDTH,
        "height": ROI_HEIGHT
    }
    sct_img = sct.grab(monitor)
    frame   = cv2.cvtColor(np.array(sct_img), cv2.COLOR_BGRA2BGR)

    # skip detection some frames to save CPU
    if count % FRAME_SKIP == 0:
        results = model(frame)
        if results and hasattr(results[0], "boxes") and results[0].boxes.data is not None:
            det = results[0].boxes.data.cpu().numpy()
            df  = pd.DataFrame(det, columns=['x1','y1','x2','y2','score','cls'])
            for _, row in df.iterrows():
                x1, y1, x2, y2 = map(int, row[['x1','y1','x2','y2']])
                cls_id = int(row['cls'])
                if cls_id >= len(classes):
                    continue
                label = classes[cls_id]

                if label == "person":
                    w, h    = x2-x1, y2-y1
                    aspect  = w/float(h)
                    if aspect > 1.2:  # person is “wider than tall”
                        cvzone.putTextRect(frame, "🚨 FALL!", (x1, y1-10),
                                           scale=1, thickness=2, colorR=(0,0,255))
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
                    else:
                        cvzone.putTextRect(frame, "Standing", (x1, y1-10),
                                           scale=1, thickness=2)
                        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)
                else:
                    # draw all other classes in green
                    cvzone.putTextRect(frame, label, (x1, y1-10),
                                       scale=1, thickness=2)
                    cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

    # show your ROI only—no recursion since the preview is outside it
    cv2.imshow("Fall Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
