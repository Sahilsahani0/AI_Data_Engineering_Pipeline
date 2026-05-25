from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime
)

from database.db import Base

from datetime import datetime


class Detection(Base):

    __tablename__ = "detections"

    id = Column(Integer, primary_key=True, index=True)

    image_name = Column(String)

    class_id = Column(Integer)

    class_name = Column(String)

    confidence = Column(Float)

    crop_path = Column(String)

    created_at = Column(
        DateTime,
        default=datetime.utcnow
    )