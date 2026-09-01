import pandas as pd
import os


# ============================================================
# FILE PATHS
# ============================================================

STUDENTS_FILE = "data/students.csv"
ATTENDANCE_FILE = "data/attendance.csv"


# ============================================================
# INITIALIZE FILES
# ============================================================

def initialize_files():

    os.makedirs("data", exist_ok=True)
    os.makedirs("data/faces", exist_ok=True)

    if not os.path.exists(STUDENTS_FILE):

        students_df = pd.DataFrame(
            columns=[
                "student_id",
                "name",
                "roll_no",
                "face_file"
            ]
        )

        students_df.to_csv(
            STUDENTS_FILE,
            index=False
        )

    if not os.path.exists(ATTENDANCE_FILE):

        attendance_df = pd.DataFrame(
            columns=[
                "student_id",
                "name",
                "roll_no",
                "date",
                "time",
                "status"
            ]
        )

        attendance_df.to_csv(
            ATTENDANCE_FILE,
            index=False
        )


# ============================================================
# STUDENT FUNCTIONS
# ============================================================

def get_students():

    initialize_files()

    try:

        return pd.read_csv(
            STUDENTS_FILE
        )

    except Exception:

        return pd.DataFrame(
            columns=[
                "student_id",
                "name",
                "roll_no",
                "face_file"
            ]
        )


def get_student(student_id):

    students_df = get_students()

    if students_df.empty:

        return None

    matches = students_df[
        students_df["student_id"].astype(str)
        == str(student_id)
    ]

    if matches.empty:

        return None

    return matches.iloc[0].to_dict()


def add_student(
    student_id,
    name,
    roll_no,
    face_file
):

    initialize_files()

    students_df = get_students()

    new_student = pd.DataFrame([
        {
            "student_id": student_id,
            "name": name,
            "roll_no": roll_no,
            "face_file": face_file
        }
    ])

    students_df = pd.concat(
        [
            students_df,
            new_student
        ],
        ignore_index=True
    )

    students_df.to_csv(
        STUDENTS_FILE,
        index=False
    )


# ============================================================
# GENERATE NEW STUDENT ID
# ============================================================

def get_next_student_id():

    students_df = get_students()

    if students_df.empty:

        return 1

    ids = pd.to_numeric(
        students_df["student_id"],
        errors="coerce"
    )

    ids = ids.dropna()

    if ids.empty:

        return 1

    return int(ids.max()) + 1


# ============================================================
# DELETE STUDENT
# ============================================================

def delete_student(student_id):

    initialize_files()

    students_df = get_students()

    if students_df.empty:

        return None

    # --------------------------------------------------------
    # FIND STUDENT
    # --------------------------------------------------------

    matches = students_df[
        students_df["student_id"].astype(str)
        == str(student_id)
    ]

    if matches.empty:

        return None

    student = matches.iloc[0]

    face_file = str(
        student["face_file"]
    )

    # --------------------------------------------------------
    # REMOVE STUDENT
    # --------------------------------------------------------

    students_df = students_df[
        students_df["student_id"].astype(str)
        != str(student_id)
    ]

    students_df.to_csv(
        STUDENTS_FILE,
        index=False
    )

    # --------------------------------------------------------
    # REMOVE ATTENDANCE RECORDS
    # --------------------------------------------------------

    if os.path.exists(
        ATTENDANCE_FILE
    ):

        try:

            attendance_df = pd.read_csv(
                ATTENDANCE_FILE
            )

            if not attendance_df.empty:

                attendance_df = attendance_df[
                    attendance_df["student_id"].astype(str)
                    != str(student_id)
                ]

                attendance_df.to_csv(
                    ATTENDANCE_FILE,
                    index=False
                )

        except Exception:

            pass

    # --------------------------------------------------------
    # RETURN FACE FILE
    # --------------------------------------------------------

    return face_file


# ============================================================
# ATTENDANCE FUNCTIONS
# ============================================================

def get_attendance():

    initialize_files()

    try:

        return pd.read_csv(
            ATTENDANCE_FILE
        )

    except Exception:

        return pd.DataFrame(
            columns=[
                "student_id",
                "name",
                "roll_no",
                "date",
                "time",
                "status"
            ]
        )


def mark_attendance(
    student_id,
    name,
    roll_no,
    date,
    time,
    status="Present"
):

    initialize_files()

    attendance_df = get_attendance()

    new_attendance = pd.DataFrame([
        {
            "student_id": student_id,
            "name": name,
            "roll_no": roll_no,
            "date": date,
            "time": time,
            "status": status
        }
    ])

    attendance_df = pd.concat(
        [
            attendance_df,
            new_attendance
        ],
        ignore_index=True
    )

    attendance_df.to_csv(
        ATTENDANCE_FILE,
        index=False
    )


# ============================================================
# CHECK TODAY ATTENDANCE
# ============================================================

def attendance_marked_today(
    student_id,
    date
):

    attendance_df = get_attendance()

    if attendance_df.empty:

        return False

    matches = attendance_df[
        (
            attendance_df["student_id"].astype(str)
            == str(student_id)
        )
        &
        (
            attendance_df["date"].astype(str)
            == str(date)
        )
        &
        (
            attendance_df["status"].astype(str)
            == "Present"
        )
    ]

    return not matches.empty