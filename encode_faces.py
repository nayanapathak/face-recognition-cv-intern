import face_recognition
import os
import pickle
import cv2
import numpy as np

DATASET_DIR = "dataset"
ENCODINGS_FILE = "encodings.pickle"

known_encodings = []
known_names = []

print("[INFO] Starting face encoding...")

for person_name in os.listdir(DATASET_DIR):
    person_path = os.path.join(DATASET_DIR, person_name)

    if not os.path.isdir(person_path):
        continue

    print(f"[INFO] Processing {person_name}")

    person_encodings = []  # 🔹 NEW: store encodings per person

    for image_name in os.listdir(person_path):
        image_path = os.path.join(person_path, image_name)

        image = cv2.imread(image_path)
        if image is None:
            continue

        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb, model="hog")

        # 🔹 NEW: ensure exactly ONE face per image
        if len(boxes) != 1:
            print(f"[WARNING] Skipping {image_name} (faces detected: {len(boxes)})")
            continue

        encoding = face_recognition.face_encodings(rgb, boxes)[0]
        person_encodings.append(encoding)

    # 🔹 NEW: average encodings for this person
    if len(person_encodings) > 0:
        mean_encoding = np.mean(person_encodings, axis=0)
        known_encodings.append(mean_encoding)
        known_names.append(person_name)
    else:
        print(f"[WARNING] No valid images for {person_name}")

print(f"[INFO] Encoding complete. Total people encoded: {len(known_names)}")

data = {"encodings": known_encodings, "names": known_names}

with open(ENCODINGS_FILE, "wb") as f:
    pickle.dump(data, f)

print("[INFO] Encodings saved to encodings.pickle")

