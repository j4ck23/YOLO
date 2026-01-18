import cv2
import os
import supervision as sv
from ultralytics import YOLO

model = YOLO("runs/detect/train/weights/best.pt")

bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print ("Cannot open cap")

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

img_counter = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

    annotated_image = bounding_box_annotator.annotate(scene=frame, detections=detections)
    annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)


    cv2.imshow('Camera', annotated_image)

    k = cv2.waitKey(1)

    if k%256 == 27:
        print("escape pressed, Exiting...")
        break
    elif k%256 == ord('s'):
        img_name = os.path.join(output_dir, "opencv_frame_{}.png".format(img_counter))
        cv2.imwrite(img_name, frame)
        print("{} written".format(img_name))
        img_counter +=1

cap.release()
cv2.destroyAllWindows()