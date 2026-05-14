import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict

# Load model
model = YOLO("runs/detect/train/weights/best.pt")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Load video instead of RealSense
cap = cv2.VideoCapture("20260414_121029.mp4")  # <-- change this

class_counts = defaultdict(set)
frame_count = {}
min_frames = 30 

class_colours = {
    "Leaves": (0,255,0),
    "Strawberries": (0,0,255),
    "Flowers": (200,100,255)
}

defaultcolour = (200,200,200)

while cap.isOpened():
    ret, color_image = cap.read()
    if not ret:
        break

    results = model.track(color_image, persist=True)
    annotate = color_image.copy()

    h, w = color_image.shape[:2]
    boxes = results[0].boxes

    if boxes is not None and boxes.id is not None:
        ids = boxes.id.cpu().numpy().flatten()
        classes = boxes.cls.cpu().numpy().astype(int)

        for i in range(len(ids)):
            track_id = int(ids[i])
            cls = int(classes[i])
            name = model.names[cls]

            if track_id not in frame_count:
                frame_count[track_id] = 0

            frame_count[track_id] += 1

            if frame_count[track_id] == min_frames:
                class_counts[name].add(track_id)

    if boxes is not None and boxes.id is not None:
        for box, track_id in zip(boxes, boxes.id):
            track_id = int(track_id.item())
            coords = box.xyxy.cpu().numpy().flatten()
            x1, y1, x2, y2 = map(int, coords)

            cls = int(box.cls[0])
            name = model.names[cls]

            colour = class_colours.get(name, defaultcolour)

            # ❌ No depth available → remove or fake it
            label = f"{name} | ID {track_id}"

            cv2.rectangle(annotate, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(annotate, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 2)

    y_offset = 40

    for name, ids in sorted(class_counts.items()):
        count = len(ids)
        text = f"{name}: {count}"

        colour = class_colours.get(name, defaultcolour)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)

        cv2.putText(annotate, text, (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2)
        y_offset += th + 10

    cv2.imshow("Images", annotate)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()