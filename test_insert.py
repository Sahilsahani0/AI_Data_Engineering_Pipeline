from database.db import SessionLocal
from database.models import Detection

db = SessionLocal()

data = Detection(
    image_name="bottle1.jpg",
    class_id=0,
    confidence=0.95,
    crop_path="outputs/crops/bottle1.jpg"
)

db.add(data)

db.commit()

print("Data Inserted Successfully")