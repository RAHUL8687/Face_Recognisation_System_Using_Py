import cv2
import os
import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

CASCADE_FILE = "haarcascade_frontalface_default.xml"

MODEL_FILE = "data/face_model.yml"

STUDENTS_FILE = "data/students.csv"


# ============================================================
# HAAR CASCADE
# ============================================================

def get_cascade_path():

    local_path = os.path.join(
        os.path.dirname(__file__),
        CASCADE_FILE
    )

    if os.path.exists(local_path):
        return local_path

    opencv_path = os.path.join(
        cv2.data.haarcascades,
        CASCADE_FILE
    )

    if os.path.exists(opencv_path):
        return opencv_path

    return None


# ============================================================
# FACE DETECTION
# ============================================================

def detect_face(image):

    cascade_path = get_cascade_path()

    if cascade_path is None:
        raise FileNotFoundError(
            "Haar Cascade file not found."
        )

    face_cascade = cv2.CascadeClassifier(
        cascade_path
    )

    if face_cascade.empty():
        raise RuntimeError(
            "Haar Cascade could not be loaded."
        )

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(100, 100)
    )

    if len(faces) == 0:
        return None, None

    largest_face = max(
        faces,
        key=lambda box: box[2] * box[3]
    )

    x, y, w, h = largest_face

    face_image = gray[
        y:y + h,
        x:x + w
    ]

    return face_image, largest_face


# ============================================================
# SAVE FACE
# ============================================================

def save_face(face_image, student_id):

    os.makedirs(
        "data/faces",
        exist_ok=True
    )

    file_path = (
        f"data/faces/student_{student_id}.jpg"
    )

    cv2.imwrite(
        file_path,
        face_image
    )

    return file_path


# ============================================================
# TRAIN LBPH MODEL
# ============================================================

def train_model():

    if not os.path.exists(STUDENTS_FILE):

        raise FileNotFoundError(
            "students.csv not found."
        )

    students = pd.read_csv(
        STUDENTS_FILE
    )

    if students.empty:

        raise ValueError(
            "No students registered yet."
        )

    faces = []
    labels = []

    for _, student in students.iterrows():

        student_id = int(
            student["student_id"]
        )

        face_file = str(
            student["face_file"]
        )

        if not os.path.exists(face_file):

            print(
                f"Skipping missing face: {face_file}"
            )

            continue

        face = cv2.imread(
            face_file,
            cv2.IMREAD_GRAYSCALE
        )

        if face is None:

            print(
                f"Could not read: {face_file}"
            )

            continue

        faces.append(face)

        labels.append(student_id)

    if len(faces) == 0:

        raise ValueError(
            "No valid face images found."
        )

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    labels = np.array(
        labels,
        dtype=np.int32
    )

    recognizer.train(
        faces,
        labels
    )

    os.makedirs(
        "data",
        exist_ok=True
    )

    recognizer.write(
        MODEL_FILE
    )

    return MODEL_FILE


# ============================================================
# LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_FILE):

        raise FileNotFoundError(
            "Face recognition model not found."
        )

    recognizer = (
        cv2.face.LBPHFaceRecognizer_create()
    )

    recognizer.read(
        MODEL_FILE
    )

    return recognizer


# ============================================================
# RECOGNIZE FACE
# ============================================================

def recognize_face(face_image):

    recognizer = load_model()

    student_id, confidence = (
        recognizer.predict(face_image)
    )

    return int(student_id), float(confidence)


# ============================================================
# GET STUDENT NAME
# ============================================================

def get_student_by_id(student_id):

    students = pd.read_csv(
        STUDENTS_FILE
    )

    student = students[
        students["student_id"] == student_id
    ]

    if student.empty:

        return None

    return student.iloc[0]