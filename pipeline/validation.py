import os
import shutil
from PIL import Image

RAW_FOLDER = "datasets/raw_images"
VALID_FOLDER = "datasets/processed_images"
INVALID_FOLDER = "datasets/invalid_images"

os.makedirs(VALID_FOLDER, exist_ok=True)
os.makedirs(INVALID_FOLDER, exist_ok=True)

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png"]


def validate_image(image_path):

    try:

        ext = os.path.splitext(image_path)[1].lower()

        if ext not in VALID_EXTENSIONS:
            return False, "Invalid Extension"

        with Image.open(image_path) as img:
            img.verify()

        return True, "Valid Image"

    except Exception as e:
        return False, str(e)


def process_images():

    files = os.listdir(RAW_FOLDER)

    for file in files:

        file_path = os.path.join(RAW_FOLDER, file)

        if not os.path.isfile(file_path):
            continue

        is_valid, message = validate_image(file_path)

        print(f"{file} -> {message}")

        if is_valid:
            destination = os.path.join(VALID_FOLDER, file)
        else:
            destination = os.path.join(INVALID_FOLDER, file)

        shutil.copy(file_path, destination)


if __name__ == "__main__":
    process_images()