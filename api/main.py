from fastapi import FastAPI
from database.db import SessionLocal
from database.models import Detection

app = FastAPI()

db = SessionLocal()


@app.get("/")
def home():

    return {
        "message": "AI Data Engineering Pipeline Running"
    }


@app.get("/detections")
def get_detections():

    detections = db.query(Detection).all()

    results = []

    for item in detections:

        results.append({
            "id": item.id,
            "image_name": item.image_name,
            "class_id": item.class_id,
            "confidence": item.confidence,
            "crop_path": item.crop_path
        })

    return results