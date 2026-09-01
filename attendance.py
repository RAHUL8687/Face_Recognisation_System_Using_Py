import csv
import os
from datetime import datetime


# ============================================================
# FILE PATH
# ============================================================

ATTENDANCE_FILE = "data/attendance.csv"


# ============================================================
# INITIALIZE ATTENDANCE FILE
# ============================================================

def initialize_attendance_file():
    """
    Create attendance.csv with the correct columns.
    """

    os.makedirs(
        "data",
        exist_ok=True
    )

    if not os.path.exists(ATTENDANCE_FILE):

        with open(
            ATTENDANCE_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "student_id",
                "name",
                "roll_no",
                "date",
                "time",
                "status"
            ])


# ============================================================
# CHECK TODAY'S ATTENDANCE
# ============================================================

def already_marked_today(student_id):
    """
    Check whether a student is already marked
    present today.
    """

    initialize_attendance_file()

    today = datetime.now().strftime(
        "%Y-%m-%d"
    )

    with open(
        ATTENDANCE_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:

            if (
                str(row["student_id"]) == str(student_id)
                and row["date"] == today
                and row["status"] == "Present"
            ):

                return True

    return False


# ============================================================
# MARK ATTENDANCE
# ============================================================

def mark_attendance(
    student_id,
    name,
    roll_no
):
    """
    Mark a student present.
    """

    initialize_attendance_file()

    # Prevent duplicate attendance
    if already_marked_today(student_id):

        return False

    now = datetime.now()

    date = now.strftime(
        "%Y-%m-%d"
    )

    time = now.strftime(
        "%H:%M:%S"
    )

    with open(
        ATTENDANCE_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            student_id,
            name,
            roll_no,
            date,
            time,
            "Present"
        ])

    return True