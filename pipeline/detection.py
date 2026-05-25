from ultralytics import YOLO
import cv2
import os

from database.db import SessionLocal
from database.models import Detection

# =========================
# LOAD MODEL
# =========================

model = YOLO("yolov8n.pt")

# =========================
# DATABASE SESSION
# =========================

db = SessionLocal()

# =========================
# FOLDERS
# =========================

INPUT_FOLDER = "inputs"

OUTPUT_FOLDER = "outputs/crops"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# =========================
# PROCESS IMAGES
# =========================

for image_file in os.listdir(INPUT_FOLDER):

    image_path = os.path.join(
        INPUT_FOLDER,
        image_file
    )

    print(f"Processing: {image_file}")

    image = cv2.imread(image_path)

    if image is None:

        print(f"Could not read: {image_file}")

        continue

    results = model(image)

    for result in results:

        boxes = result.boxes

        for i, box in enumerate(boxes):

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_id = int(
                box.cls[0]
            )

            crop = image[y1:y2, x1:x2]

            crop_filename = (
                f"{i}_{image_file}"
            )

            crop_path = os.path.join(
                OUTPUT_FOLDER,
                crop_filename
            )

            cv2.imwrite(
                crop_path,
                crop
            )

            detection = Detection(

                image_name=image_file,

                class_id=class_id,

                confidence=confidence,

                crop_path=crop_path
            )

            db.add(detection)

# =========================
# SAVE DATABASE
# =========================

db.commit()

print("Detection Pipeline Completed")