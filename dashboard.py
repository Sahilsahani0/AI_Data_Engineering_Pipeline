# =========================
# FIX TORCH CPU ISSUE
# =========================

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

# =========================
# IMPORTS
# =========================

import cv2
import sqlite3
import pandas as pd
import streamlit as st
import plotly.express as px

from ultralytics import YOLO

from database.db import SessionLocal
from database.models import Detection

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Detection Pipeline",
    layout="wide"
)

# =========================
# TITLE
# =========================

st.title("AI Detection Pipeline")

# =========================
# SIDEBAR
# =========================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go To",
    [
        "Upload Detection",
        "Detection History",
        "Analytics"
    ]
)

# =========================
# LOAD MODEL
# =========================

model = YOLO("yolov8n.pt")

# FORCE CPU
model.to("cpu")

# COCO LABELS
CLASS_NAMES = model.names

# =========================
# DATABASE SESSION
# =========================

db = SessionLocal()

# =========================
# FOLDERS
# =========================

INPUT_FOLDER = "inputs"

CROP_FOLDER = "outputs/crops"

os.makedirs(INPUT_FOLDER, exist_ok=True)

os.makedirs(CROP_FOLDER, exist_ok=True)

# =========================================================
# PAGE 1 — UPLOAD DETECTION
# =========================================================

if page == "Upload Detection":

    st.subheader("Upload Image")

    uploaded_file = st.file_uploader(
        "Choose Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        # =========================
        # SAVE IMAGE
        # =========================

        image_path = os.path.join(
            INPUT_FOLDER,
            uploaded_file.name
        )

        with open(image_path, "wb") as f:

            f.write(uploaded_file.getbuffer())

        # =========================
        # READ IMAGE
        # =========================

        image = cv2.imread(image_path)

        original_image = image.copy()

        # =========================
        # YOLO DETECTION
        # =========================

        results = model(image)

        # =========================
        # PROCESS DETECTIONS
        # =========================

        for result in results:

            boxes = result.boxes

            for i, box in enumerate(boxes):

                # Coordinates
                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                # Confidence
                confidence = float(
                    box.conf[0]
                )

                # Class ID
                class_id = int(
                    box.cls[0]
                )

                # Class Name
                class_name = CLASS_NAMES[class_id]

                # =========================
                # DRAW BOX
                # =========================

                cv2.rectangle(
                    image,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # =========================
                # DRAW LABEL
                # =========================

                cv2.putText(
                    image,
                    f"{class_name} {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # =========================
                # CROP OBJECT
                # =========================

                crop = original_image[
                    y1:y2,
                    x1:x2
                ]

                # =========================
                # SAVE CROP
                # =========================

                crop_filename = (
                    f"{i}_{uploaded_file.name}"
                )

                crop_path = os.path.join(
                    CROP_FOLDER,
                    crop_filename
                )

                cv2.imwrite(
                    crop_path,
                    crop
                )

                # =========================
                # SAVE DATABASE
                # =========================

                detection = Detection(
                    image_name=uploaded_file.name,
                    class_id=class_id,
                    class_name=class_name,
                    confidence=confidence,
                    crop_path=crop_path,
                )
                db.add(detection)

        # =========================
        # COMMIT DATABASE
        # =========================

        db.commit()

        # =========================
        # CONVERT COLOR
        # =========================

        original_rgb = cv2.cvtColor(
            original_image,
            cv2.COLOR_BGR2RGB
        )

        detected_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # =========================
        # DISPLAY IMAGES
        # =========================

        st.subheader("Detection Results")

        col1, col2 = st.columns(2)

        with col1:

            st.image(
                original_rgb,
                caption="Original Image",
                width=400
            )

        with col2:

            st.image(
                detected_rgb,
                caption="Detected Image",
                width=400
            )

        # =========================
        # SHOW CROPS
        # =========================

        st.subheader("Detected Crops")

        crop_columns = st.columns(4)

        crop_index = 0

        for result in results:

            boxes = result.boxes

            for i, box in enumerate(boxes):

                class_id = int(box.cls[0])

                confidence = float(box.conf[0])
                if confidence < 0.50:
                    continue

                class_name = CLASS_NAMES[class_id]

                crop_filename = (
                    f"{i}_{uploaded_file.name}"
                )

                crop_path = os.path.join(
                    CROP_FOLDER,
                    crop_filename
                )

                with crop_columns[crop_index % 4]:

                    st.image(
                        crop_path,
                        caption=f"{class_name} | {confidence:.2f}",
                        width=180
                    )

                crop_index += 1

        st.success("Detection Completed")

# =========================================================
# PAGE 2 — DETECTION HISTORY
# =========================================================

elif page == "Detection History":

    st.subheader("Detection Records")

    conn = sqlite3.connect(
        "ai_pipeline.db"
    )

    query = """
    SELECT * FROM detections
    """

    df = pd.read_sql(
        query,
        conn
    )

    st.dataframe(
        df,
        width="stretch"
    )

    st.success(
        f"Total Records: {len(df)}"
    )

# =========================================================
# PAGE 3 — ANALYTICS
# =========================================================

elif page == "Analytics":

    st.subheader("Detection Analytics")

    conn = sqlite3.connect(
        "ai_pipeline.db"
    )

    query = """
    SELECT * FROM detections
    """

    df = pd.read_sql(
        query,
        conn
    )

    if len(df) > 0:

        # =========================
        # TOTAL DETECTIONS
        # =========================

        st.metric(
            "Total Detections",
            len(df)
        )

        # =========================
        # CLASS COUNTS
        # =========================

        class_counts = (

            df["class_name"]

            .value_counts()

            .reset_index()
        )

        class_counts.columns = [
            "Class Name",
            "Count"
        ]

        # =========================
        # BAR CHART
        # =========================

        fig = px.bar(

            class_counts,

            x="Class Name",

            y="Count",

            title="Detections By Class"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

    else:

        st.warning(
            "No detections found"
        )