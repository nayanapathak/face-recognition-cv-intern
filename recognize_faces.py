import face_recognition
import cv2
import pickle
import numpy as np

# ================= CONFIG =================
ENCODINGS_FILE = "encodings.pickle"
TOLERANCE = 0.48          # Tuned for stability
RESIZE_SCALE = 0.45       # Better multi-face detection
DETECTION_MODEL = "hog"   # Use "cnn" only if GPU is available
# ==========================================

print("[INFO] Loading face encodings...")
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

if len(known_encodings) == 0:
    raise ValueError("No face encodings found. Run encode_faces.py first.")

print("[INFO] Starting webcam...")
video = cv2.VideoCapture(0)

if not video.isOpened():
    raise RuntimeError("Could not open webcam.")

while True:
    ret, frame = video.read()
    if not ret:
        break
    # ✅ FIX: Flip frame horizontally (mirror correction)
    frame = cv2.flip(frame, 1)

    # Resize frame for speed & accuracy balance
    small_frame = cv2.resize(
        frame, (0, 0),
        fx=RESIZE_SCALE,
        fy=RESIZE_SCALE
    )

    # Convert BGR to RGB
    rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces (supports multiple faces)
    boxes = face_recognition.face_locations(
        rgb,
        model=DETECTION_MODEL
    )

    # Compute encodings for each detected face
    encodings = face_recognition.face_encodings(rgb, boxes)

    for encoding, box in zip(encodings, boxes):
        # Compare face against known encodings
        distances = face_recognition.face_distance(
            known_encodings,
            encoding
        )

        name = "Unknown"
        min_distance = np.min(distances)

        if min_distance < TOLERANCE:
            best_match_index = np.argmin(distances)
            name = known_names[best_match_index]

        # Scale face box back to original frame size
        top, right, bottom, left = box
        top = int(top / RESIZE_SCALE)
        right = int(right / RESIZE_SCALE)
        bottom = int(bottom / RESIZE_SCALE)
        left = int(left / RESIZE_SCALE)


        # Draw bounding box
        cv2.rectangle(
            frame,
            (left, top),
            (right, bottom),
            (0, 255, 0),
            2
        )

        # Display name
        cv2.putText(
            frame,
            name,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

    cv2.imshow("Face Recognition - Webcam", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video.release()
cv2.destroyAllWindows()








