import cv2
import numpy as np
from ultralytics import YOLO
import pyrealsense2 as rs
from collections import defaultdict

model = YOLO("runs/detect/train/weights/best.pt")

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

pipeline.start(config)

class_counts = defaultdict(set)
frame_count = {}
min_frames = 30

class_colours = {
    "Leaves":(0,255,0),
    "Strawberries":(0,0,255),
    "Flowers":(200,100,255)
}

defaultcolour = (200,200,200)

while True:
    frames = pipeline.wait_for_frames()

    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    # Convert to numpy arrays
    color_image = np.asanyarray(color_frame.get_data())

    results = model.track(color_image, persist=True)
    annotated_frame = results[0].plot()
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
            cx = int((x1 + x2) /2 )
            cy = int((y1 + y2) /2 )

            cx = max(0, min(cx, w -1))
            cy = max(0, min(cy, h -1))

            distance = depth_frame.get_distance(cx, cy)

            cls = int(box.cls[0])
            name = model.names[cls]
                
            colour = class_colours.get(name,defaultcolour)

            label = f"{name} | ID {track_id} | {distance:.2f}M"

            cv2.rectangle(annotate, (x1, y1), (x2, y2), colour,2)

            cv2.putText(annotate, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0),2)

    y_offset = 40

    for name, ids in sorted(class_counts.items()):
        count = len(ids)
        text = f"{name}: {count}"

        colour = class_colours.get(name,defaultcolour)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)


        cv2.putText(annotate, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour,2)
        y_offset += th + 10
    cv2.imshow("Images", annotate)

    if cv2.waitKey(1) == ord('q'):
        break


pipeline.stop()