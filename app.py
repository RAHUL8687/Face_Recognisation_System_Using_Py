import streamlit as st
import pandas as pd
import numpy as np
import cv2
import os

from datetime import date

from database import (
    initialize_files,
    get_students,
    add_student,
    delete_student,
    get_student,
    get_attendance
)

from face_recognition import (
    detect_face,
    save_face,
    recognize_face,
    get_student_by_id,
    train_model
)

from attendance import (
    initialize_attendance_file,
    mark_attendance,
    already_marked_today
)


# ============================================================
# INITIALIZE SYSTEM
# ============================================================

initialize_files()
initialize_attendance_file()


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Smart Face Attendance",
    page_icon="📸",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    section[data-testid="stSidebar"] {
        padding-top: 1rem;
    }

    section[data-testid="stSidebar"] h1 {
        font-size: 1.55rem;
    }

    h1 {
        font-weight: 700;
        letter-spacing: -0.5px;
    }

    h2 {
        font-weight: 650;
    }

    h3 {
        font-weight: 600;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 14px;
        padding: 18px;
        background: rgba(128, 128, 128, 0.06);
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 600;
    }

    .stButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    .stDownloadButton > button {
        border-radius: 10px;
        min-height: 44px;
        font-weight: 600;
    }

    div[data-baseweb="input"] {
        border-radius: 9px;
    }

    div[data-baseweb="select"] {
        border-radius: 9px;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    div[data-testid="stCameraInput"] {
        max-width: 620px;
    }

    div[data-testid="stAlert"] {
        border-radius: 10px;
    }

    .app-footer {
        text-align: center;
        padding-top: 25px;
        padding-bottom: 10px;
        opacity: 0.65;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "# 📸 Smart Attendance"
    )

    st.caption(
        "Face Recognition System"
    )

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Dashboard",
            "👥 Student Management",
            "👤 Register Student",
            "📷 Take Attendance",
            "📋 Attendance Records",
            "📊 Reports",
            "⚙️ Settings"
        ]
    )

    st.divider()

    st.success(
        "🟢 System Online"
    )

    st.caption(
        "Powered by Python + OpenCV"
    )


# ============================================================
# DASHBOARD
# ============================================================

if page == "🏠 Dashboard":

    students = get_students()
    attendance = get_attendance()

    total_students = len(students)

    today = pd.Timestamp.now().strftime(
        "%Y-%m-%d"
    )

    # --------------------------------------------------------
    # PREPARE TODAY'S DATA
    # --------------------------------------------------------

    if attendance.empty:

        today_attendance = pd.DataFrame()

    else:

        today_attendance = attendance[
            attendance["date"].astype(str) == today
        ].copy()

    if today_attendance.empty:

        present_today = 0

    else:

        present_today = len(
            today_attendance[
                today_attendance["status"].astype(str)
                == "Present"
            ]["student_id"].unique()
        )

    absent_today = max(
        total_students - present_today,
        0
    )

    if total_students > 0:

        attendance_rate = round(
            (present_today / total_students) * 100,
            1
        )

    else:

        attendance_rate = 0.0

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

    st.title(
        "📸 Smart Attendance"
    )

    st.caption(
        "Face Recognition Attendance Management System"
    )

    st.divider()

    # --------------------------------------------------------
    # TODAY HEADER
    # --------------------------------------------------------

    st.subheader(
        f"📅 Today's Overview — "
        f"{pd.Timestamp.now().strftime('%d %B %Y')}"
    )

    st.success(
        "🟢 Attendance system is ready"
    )

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "👥 Total Students",
            total_students
        )

    with col2:

        st.metric(
            "✅ Present Today",
            present_today
        )

    with col3:

        st.metric(
            "❌ Absent Today",
            absent_today
        )

    with col4:

        st.metric(
            "📊 Attendance Rate",
            f"{attendance_rate}%"
        )

    st.divider()

    # --------------------------------------------------------
    # TODAY'S ATTENDANCE + RECENT ACTIVITY
    # --------------------------------------------------------

    left_col, right_col = st.columns(
        [1.25, 1]
    )

    with left_col:

        st.subheader(
            "📋 Today's Attendance"
        )

        if students.empty:

            st.info(
                "No students registered yet."
            )

        else:

            today_rows = []

            present_ids = set()

            if not today_attendance.empty:

                present_ids = set(
                    today_attendance[
                        today_attendance["status"].astype(str)
                        == "Present"
                    ]["student_id"]
                    .astype(str)
                    .tolist()
                )

            for _, student in students.iterrows():

                student_id = str(
                    student["student_id"]
                )

                student_name = str(
                    student["name"]
                )

                roll_no = str(
                    student["roll_no"]
                )

                if student_id in present_ids:

                    status = "✅ Present"

                    student_records = (
                        today_attendance[
                            today_attendance[
                                "student_id"
                            ].astype(str)
                            == student_id
                        ]
                    )

                    if not student_records.empty:

                        latest_record = (
                            student_records
                            .sort_values("time")
                            .iloc[-1]
                        )

                        record_time = str(
                            latest_record["time"]
                        )

                    else:

                        record_time = "-"

                else:

                    status = "❌ Absent"

                    record_time = "-"

                today_rows.append({
                    "Student": student_name,
                    "Roll Number": roll_no,
                    "Time": record_time,
                    "Status": status
                })

            today_df = pd.DataFrame(
                today_rows
            )

            st.dataframe(
                today_df,
                use_container_width=True,
                hide_index=True
            )

    with right_col:

        st.subheader(
            "🕐 Recent Activity"
        )

        if attendance.empty:

            st.info(
                "No attendance activity yet."
            )

        else:

            recent = attendance.copy()

            recent["date_time"] = pd.to_datetime(
                recent["date"].astype(str)
                + " "
                + recent["time"].astype(str),
                errors="coerce"
            )

            recent = (
                recent
                .sort_values(
                    "date_time",
                    ascending=False
                )
                .head(8)
            )

            recent_rows = []

            for _, row in recent.iterrows():

                recent_rows.append({
                    "Student": str(
                        row["name"]
                    ),
                    "Date": str(
                        row["date"]
                    ),
                    "Time": str(
                        row["time"]
                    ),
                    "Status": str(
                        row["status"]
                    )
                })

            recent_df = pd.DataFrame(
                recent_rows
            )

            st.dataframe(
                recent_df,
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # --------------------------------------------------------
    # ATTENDANCE TREND
    # --------------------------------------------------------

    st.subheader(
        "📈 Attendance Trend"
    )

    if attendance.empty:

        st.info(
            "Attendance trend will appear "
            "after attendance is recorded."
        )

    else:

        trend = attendance.copy()

        trend["date_parsed"] = pd.to_datetime(
            trend["date"],
            errors="coerce"
        )

        trend = trend[
            trend["status"].astype(str)
            == "Present"
        ]

        if trend.empty:

            st.info(
                "No present attendance data available."
            )

        else:

            trend = (
                trend
                .groupby("date_parsed")
                ["student_id"]
                .nunique()
                .reset_index(
                    name="Present"
                )
                .sort_values(
                    "date_parsed"
                )
            )

            trend = trend.set_index(
                "date_parsed"
            )

            st.line_chart(
                trend["Present"],
                use_container_width=True
            )

    st.divider()

    # --------------------------------------------------------
    # QUICK ACTIONS
    # --------------------------------------------------------

    st.subheader(
        "🚀 Quick Actions"
    )

    action1, action2, action3 = st.columns(3)

    with action1:

        st.info(
            "👤 **Register Student**\n\n"
            "Add a new student and capture face data."
        )

    with action2:

        st.info(
            "📷 **Take Attendance**\n\n"
            "Recognize a face and mark attendance."
        )

    with action3:

        st.info(
            "📊 **Reports**\n\n"
            "Analyze attendance and download reports."
        )

    st.divider()

    # --------------------------------------------------------
    # DASHBOARD FOOT NOTE
    # --------------------------------------------------------

    st.caption(
        f"System currently manages "
        f"**{total_students} student(s)** and "
        f"has recorded "
        f"**{len(attendance)} attendance record(s)**."
    )


# ============================================================
# STUDENT MANAGEMENT
# ============================================================

elif page == "👥 Student Management":

    st.title("👥 Student Management")

    st.caption(
        "View, search and manage registered students."
    )

    st.divider()

    students = get_students()

    total_students = len(students)

    students_with_faces = 0

    if not students.empty:

        for _, student in students.iterrows():

            face_file = str(
                student["face_file"]
            )

            if os.path.exists(face_file):

                students_with_faces += 1

    students_without_faces = (
        total_students - students_with_faces
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "👥 Total Students",
            total_students
        )

    with col2:

        st.metric(
            "🟢 Face Data Available",
            students_with_faces
        )

    with col3:

        st.metric(
            "🔴 Face Data Missing",
            students_without_faces
        )

    st.divider()

    st.subheader(
        "🔎 Search Students"
    )

    search_text = st.text_input(
        "Search by name or roll number",
        placeholder="Type a name or roll number..."
    )

    if students.empty:

        st.info(
            "No students registered yet."
        )

    else:

        filtered_students = students.copy()

        if search_text.strip():

            search_value = (
                search_text
                .strip()
                .lower()
            )

            filtered_students = students[
                students["name"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_value,
                    na=False
                )
                |
                students["roll_no"]
                .astype(str)
                .str.lower()
                .str.contains(
                    search_value,
                    na=False
                )
            ]

        st.write(
            f"Showing **{len(filtered_students)}** student(s)"
        )

        display_data = []

        for _, student in filtered_students.iterrows():

            face_file = str(
                student["face_file"]
            )

            if os.path.exists(face_file):

                face_status = "🟢 Available"

            else:

                face_status = "🔴 Missing"

            display_data.append({
                "Student ID": student["student_id"],
                "Name": student["name"],
                "Roll Number": student["roll_no"],
                "Face Data": face_status
            })

        display_df = pd.DataFrame(
            display_data
        )

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        if len(filtered_students) == 0:

            st.warning(
                "⚠️ No student found."
            )

    st.divider()

    st.subheader(
        "🗑️ Delete Student"
    )

    st.warning(
        "Deleting a student removes their student record, "
        "face image and attendance records."
    )

    if students.empty:

        st.info(
            "There are no students to delete."
        )

    else:

        student_options = []

        for _, student in students.iterrows():

            student_id = int(
                student["student_id"]
            )

            student_name = str(
                student["name"]
            )

            roll_no = str(
                student["roll_no"]
            )

            student_options.append(
                f"ID {student_id} | "
                f"{student_name} | "
                f"{roll_no}"
            )

        selected_label = st.selectbox(
            "Select student to delete",
            options=[
                "-- Select a student --"
            ] + student_options,
            index=0
        )

        if selected_label == "-- Select a student --":

            st.info(
                "👆 Select a student to continue."
            )

        else:

            selected_id = int(
                selected_label.split("|")[0]
                .replace("ID", "")
                .strip()
            )

            selected_student = get_student(
                selected_id
            )

            if selected_student is not None:

                st.info(
                    f"Selected: **{selected_student['name']}** "
                    f"({selected_student['roll_no']})"
                )

                confirm_delete = st.checkbox(
                    "I understand that this action will "
                    "remove the student's face data and "
                    "attendance records."
                )

                if st.button(
                    "🗑️ Delete Selected Student",
                    type="secondary",
                    use_container_width=True
                ):

                    if not confirm_delete:

                        st.error(
                            "❌ Please confirm deletion first."
                        )

                    else:

                        face_file = str(
                            selected_student["face_file"]
                        )

                        student_name = str(
                            selected_student["name"]
                        )

                        deleted_face_file = delete_student(
                            selected_id
                        )

                        face_deleted = False

                        if (
                            deleted_face_file
                            and
                            os.path.exists(
                                deleted_face_file
                            )
                        ):

                            os.remove(
                                deleted_face_file
                            )

                            face_deleted = True

                        elif os.path.exists(
                            face_file
                        ):

                            os.remove(
                                face_file
                            )

                            face_deleted = True

                        model_updated = False

                        try:

                            train_model()

                            model_updated = True

                        except Exception as e:

                            st.warning(
                                "⚠️ Student deleted, "
                                "but model retraining failed."
                            )

                            st.error(
                                f"Training error: {e}"
                            )

                        st.success(
                            f"✅ {student_name} "
                            "deleted successfully!"
                        )

                        if face_deleted:

                            st.success(
                                "🖼️ Face image deleted."
                            )

                        if model_updated:

                            st.success(
                                "🧠 Recognition model "
                                "updated successfully!"
                            )

                        st.rerun()


# ============================================================
# REGISTER STUDENT
# ============================================================

elif page == "👤 Register Student":

    st.title("👤 Register Student")

    st.caption(
        "Create a student profile and capture face data."
    )

    st.divider()

    st.subheader(
        "📝 Student Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        name = st.text_input(
            "Student Name",
            placeholder="Enter full name"
        )

    with col2:

        roll_no = st.text_input(
            "Roll Number",
            placeholder="Enter roll number"
        )

    st.divider()

    st.subheader(
        "📸 Face Capture"
    )

    st.caption(
        "Keep your face clearly visible and look "
        "directly at the camera."
    )

    camera_col, empty_col = st.columns(
        [1, 1]
    )

    with camera_col:

        camera_photo = st.camera_input(
            "Take a photo"
        )

    if camera_photo is not None:

        image_bytes = camera_photo.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "❌ Could not read the captured image."
            )

        else:

            face_image, face_box = detect_face(
                image
            )

            if face_image is None:

                st.error(
                    "❌ No face detected. "
                    "Please take another photo."
                )

            else:

                st.success(
                    "✅ Face detected successfully!"
                )

                x, y, w, h = face_box

                image_with_box = image.copy()

                cv2.rectangle(
                    image_with_box,
                    (x, y),
                    (x + w, y + h),
                    (0, 255, 0),
                    3
                )

                image_rgb = cv2.cvtColor(
                    image_with_box,
                    cv2.COLOR_BGR2RGB
                )

                st.image(
                    image_rgb,
                    caption="Detected Face",
                    width=500
                )

                st.divider()

                if st.button(
                    "➕ Register Student",
                    type="primary",
                    use_container_width=True
                ):

                    if not name.strip():

                        st.error(
                            "❌ Please enter student name."
                        )

                    elif not roll_no.strip():

                        st.error(
                            "❌ Please enter roll number."
                        )

                    else:

                        students = get_students()

                        existing_roll_numbers = (
                            students["roll_no"]
                            .astype(str)
                            .str.strip()
                            .values
                        )

                        if (
                            roll_no.strip()
                            in existing_roll_numbers
                        ):

                            st.warning(
                                "⚠️ This roll number "
                                "already exists."
                            )

                        else:

                            if students.empty:

                                student_id = 1

                            else:

                                ids = pd.to_numeric(
                                    students["student_id"],
                                    errors="coerce"
                                )

                                ids = ids.dropna()

                                if ids.empty:

                                    student_id = 1

                                else:

                                    student_id = (
                                        int(ids.max()) + 1
                                    )

                            face_file = save_face(
                                face_image,
                                student_id
                            )

                            add_student(
                                student_id=student_id,
                                name=name.strip(),
                                roll_no=roll_no.strip(),
                                face_file=face_file
                            )

                            try:

                                train_model()

                                st.success(
                                    f"✅ {name.strip()} "
                                    "registered successfully!"
                                )

                                st.success(
                                    "🧠 Recognition model "
                                    "updated successfully!"
                                )

                            except Exception as e:

                                st.warning(
                                    "⚠️ Student was registered, "
                                    "but model training failed."
                                )

                                st.error(
                                    f"Training error: {e}"
                                )

                            st.balloons()

    st.divider()

    st.subheader(
        "📋 Registered Students"
    )

    students = get_students()

    if students.empty:

        st.info(
            "No students registered yet."
        )

    else:

        st.dataframe(
            students,
            use_container_width=True,
            hide_index=True
        )


# ============================================================
# TAKE ATTENDANCE
# ============================================================

elif page == "📷 Take Attendance":

    st.title("📷 Take Attendance")

    st.caption(
        "Recognize a registered student and "
        "mark today's attendance."
    )

    st.divider()

    st.info(
        "📌 Capture a clear face photo. "
        "The system will identify the student automatically."
    )

    st.subheader(
        "📸 Camera"
    )

    camera_col, empty_col = st.columns(
        [1, 1]
    )

    with camera_col:

        photo = st.camera_input(
            "Take a photo"
        )

    if photo is not None:

        image_bytes = photo.getvalue()

        image_array = np.frombuffer(
            image_bytes,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if image is None:

            st.error(
                "❌ Could not read the captured image."
            )

        else:

            face_image, face_box = detect_face(
                image
            )

            if face_image is None:

                st.error(
                    "❌ No face detected. "
                    "Please try again."
                )

            else:

                try:

                    student_id, confidence = (
                        recognize_face(face_image)
                    )

                    student = get_student_by_id(
                        student_id
                    )

                    x, y, w, h = face_box

                    result_image = image.copy()

                    if student is not None:

                        student_name = str(
                            student["name"]
                        )

                        roll_number = str(
                            student["roll_no"]
                        )

                        cv2.rectangle(
                            result_image,
                            (x, y),
                            (x + w, y + h),
                            (0, 255, 0),
                            3
                        )

                        cv2.putText(
                            result_image,
                            student_name,
                            (x, max(y - 10, 30)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                        result_rgb = cv2.cvtColor(
                            result_image,
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            result_rgb,
                            caption="Recognition Result",
                            width=500
                        )

                        st.success(
                            "✅ Student Recognized!"
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Student ID",
                                student_id
                            )

                        with col2:

                            st.metric(
                                "Name",
                                student_name
                            )

                        with col3:

                            st.metric(
                                "Roll Number",
                                roll_number
                            )

                        st.divider()

                        if already_marked_today(
                            student_id
                        ):

                            st.warning(
                                "⚠️ Attendance already "
                                "marked for today."
                            )

                        else:

                            marked = mark_attendance(
                                student_id,
                                student_name,
                                roll_number
                            )

                            if marked:

                                st.success(
                                    "✅ Attendance marked "
                                    "successfully!"
                                )

                            else:

                                st.warning(
                                    "⚠️ Attendance could "
                                    "not be marked."
                                )

                    else:

                        cv2.rectangle(
                            result_image,
                            (x, y),
                            (x + w, y + h),
                            (0, 0, 255),
                            3
                        )

                        result_rgb = cv2.cvtColor(
                            result_image,
                            cv2.COLOR_BGR2RGB
                        )

                        st.image(
                            result_rgb,
                            caption="Unknown Face",
                            width=500
                        )

                        st.warning(
                            "⚠️ Face detected, but "
                            "student is not registered."
                        )

                except FileNotFoundError:

                    st.error(
                        "❌ Face recognition model "
                        "not found. Please register "
                        "a student first."
                    )

                except Exception as e:

                    st.error(
                        f"❌ Recognition error: {e}"
                    )


# ============================================================
# ATTENDANCE RECORDS
# ============================================================

elif page == "📋 Attendance Records":

    st.title("📋 Attendance Records")

    st.caption(
        "View, filter and download attendance records."
    )

    st.divider()

    attendance = get_attendance()

    if attendance.empty:

        st.info(
            "No attendance records found."
        )

    else:

        st.subheader(
            "🔎 Filter Attendance"
        )

        filter_col1, filter_col2 = st.columns(2)

        with filter_col1:

            filter_mode = st.radio(
                "View",
                [
                    "All Records",
                    "Select Date"
                ],
                horizontal=True
            )

        with filter_col2:

            if filter_mode == "Select Date":

                selected_date = st.date_input(
                    "Select Date",
                    value=date.today()
                )

            else:

                selected_date = None

        if filter_mode == "Select Date":

            selected_date_string = (
                selected_date.strftime(
                    "%Y-%m-%d"
                )
            )

            filtered_attendance = attendance[
                attendance["date"].astype(str)
                == selected_date_string
            ]

            st.subheader(
                f"📅 Attendance for {selected_date_string}"
            )

        else:

            filtered_attendance = attendance.copy()

            st.subheader(
                "📋 All Attendance Records"
            )

        present_count = len(
            filtered_attendance[
                filtered_attendance["status"] == "Present"
            ]
        )

        total_records = len(
            filtered_attendance
        )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "📋 Records",
                total_records
            )

        with col2:

            st.metric(
                "✅ Present",
                present_count
            )

        st.divider()

        if filtered_attendance.empty:

            st.info(
                "No attendance records found "
                "for the selected date."
            )

        else:

            st.dataframe(
                filtered_attendance,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "📥 Export Attendance"
            )

            csv_data = filtered_attendance.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            if filter_mode == "Select Date":

                file_name = (
                    f"attendance_{selected_date_string}.csv"
                )

            else:

                file_name = (
                    "attendance_all_records.csv"
                )

            st.download_button(
                label="⬇️ Download CSV",
                data=csv_data,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# REPORTS
# ============================================================

elif page == "📊 Reports":

    st.title("📊 Attendance Reports")

    st.caption(
        "Analyze attendance by date range and student."
    )

    st.divider()

    students = get_students()
    attendance = get_attendance()

    if students.empty:

        st.info(
            "No students registered yet."
        )

    else:

        st.subheader(
            "📅 Select Report Period"
        )

        date_col1, date_col2 = st.columns(2)

        with date_col1:

            start_date = st.date_input(
                "From Date",
                value=date.today()
            )

        with date_col2:

            end_date = st.date_input(
                "To Date",
                value=date.today()
            )

        threshold = st.slider(
            "Low Attendance Threshold (%)",
            min_value=0,
            max_value=100,
            value=75,
            step=5
        )

        if start_date > end_date:

            st.error(
                "❌ From Date cannot be later than To Date."
            )

        else:

            if attendance.empty:

                range_attendance = pd.DataFrame()

            else:

                attendance_copy = attendance.copy()

                attendance_copy["date_parsed"] = (
                    pd.to_datetime(
                        attendance_copy["date"],
                        errors="coerce"
                    ).dt.date
                )

                range_attendance = attendance_copy[
                    (
                        attendance_copy["date_parsed"]
                        >= start_date
                    )
                    &
                    (
                        attendance_copy["date_parsed"]
                        <= end_date
                    )
                ].copy()

            if range_attendance.empty:

                total_classes = 0

            else:

                present_records = range_attendance[
                    range_attendance["status"].astype(str)
                    == "Present"
                ]

                total_classes = (
                    present_records["date_parsed"]
                    .nunique()
                )

            report_data = []

            for _, student in students.iterrows():

                student_id = student["student_id"]

                student_name = str(
                    student["name"]
                )

                student_roll = str(
                    student["roll_no"]
                )

                if range_attendance.empty:

                    present_days = 0

                else:

                    student_records = (
                        range_attendance[
                            (
                                range_attendance[
                                    "student_id"
                                ].astype(str)
                                == str(student_id)
                            )
                            &
                            (
                                range_attendance[
                                    "status"
                                ].astype(str)
                                == "Present"
                            )
                        ]
                    )

                    present_days = (
                        student_records[
                            "date_parsed"
                        ].nunique()
                    )

                absent_days = max(
                    total_classes - present_days,
                    0
                )

                if total_classes > 0:

                    attendance_percentage = round(
                        (
                            present_days
                            / total_classes
                        ) * 100,
                        1
                    )

                else:

                    attendance_percentage = 0.0

                if total_classes == 0:

                    attendance_status = "⚪ No Data"

                elif attendance_percentage < threshold:

                    attendance_status = "🔴 Low Attendance"

                else:

                    attendance_status = "🟢 Good"

                report_data.append({
                    "Student ID": student_id,
                    "Name": student_name,
                    "Roll Number": student_roll,
                    "Total Classes": total_classes,
                    "Present": present_days,
                    "Absent": absent_days,
                    "Attendance %": attendance_percentage,
                    "Status": attendance_status
                })

            report_df = pd.DataFrame(
                report_data
            )

            good_count = len(
                report_df[
                    report_df["Status"] == "🟢 Good"
                ]
            )

            low_count = len(
                report_df[
                    report_df["Status"] == "🔴 Low Attendance"
                ]
            )

            no_data_count = len(
                report_df[
                    report_df["Status"] == "⚪ No Data"
                ]
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "📅 Total Classes",
                    total_classes
                )

            with col2:

                st.metric(
                    "👥 Students",
                    len(report_df)
                )

            with col3:

                st.metric(
                    "🟢 Good Attendance",
                    good_count
                )

            with col4:

                st.metric(
                    "🔴 Low Attendance",
                    low_count
                )

            st.divider()

            st.info(
                f"📅 Report period: "
                f"**{start_date.strftime('%d-%m-%Y')}** "
                f"to "
                f"**{end_date.strftime('%d-%m-%Y')}**"
            )

            st.subheader(
                "📋 Student Attendance Report"
            )

            st.dataframe(
                report_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            st.subheader(
                "⚠️ Low Attendance Students"
            )

            low_attendance_df = report_df[
                report_df["Status"]
                == "🔴 Low Attendance"
            ]

            if low_attendance_df.empty:

                st.success(
                    "🎉 No students are below the "
                    "selected attendance threshold."
                )

            else:

                st.warning(
                    f"{len(low_attendance_df)} student(s) "
                    f"are below {threshold}% attendance."
                )

                st.dataframe(
                    low_attendance_df[
                        [
                            "Student ID",
                            "Name",
                            "Roll Number",
                            "Total Classes",
                            "Present",
                            "Absent",
                            "Attendance %",
                            "Status"
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True
                )

            st.divider()

            st.subheader(
                "📈 Daily Attendance Summary"
            )

            if range_attendance.empty:

                st.info(
                    "No attendance data available "
                    "for this date range."
                )

            else:

                daily_data = (
                    range_attendance[
                        range_attendance["status"].astype(str)
                        == "Present"
                    ]
                    .groupby("date_parsed")
                    .agg(
                        Present=("student_id", "nunique")
                    )
                    .reset_index()
                )

                daily_data = daily_data.sort_values(
                    "date_parsed"
                )

                daily_data["Date"] = (
                    daily_data["date_parsed"]
                    .apply(
                        lambda x: x.strftime("%d-%m-%Y")
                    )
                )

                daily_data = daily_data[
                    [
                        "Date",
                        "Present"
                    ]
                ]

                st.dataframe(
                    daily_data,
                    use_container_width=True,
                    hide_index=True
                )

                st.subheader(
                    "📊 Attendance Trend"
                )

                chart_data = daily_data.copy()

                chart_data["Date"] = pd.to_datetime(
                    chart_data["Date"],
                    format="%d-%m-%Y"
                )

                chart_data = chart_data.set_index(
                    "Date"
                )

                st.line_chart(
                    chart_data["Present"],
                    use_container_width=True
                )

            st.divider()

            st.subheader(
                "👤 Individual Student Details"
            )

            student_options = []

            for _, student in students.iterrows():

                student_options.append(
                    f"{student['student_id']} | "
                    f"{student['name']} | "
                    f"{student['roll_no']}"
                )

            selected_student_label = st.selectbox(
                "Select Student",
                student_options
            )

            selected_student_id = int(
                selected_student_label
                .split("|")[0]
                .strip()
            )

            selected_student = get_student_by_id(
                selected_student_id
            )

            if selected_student is not None:

                selected_name = str(
                    selected_student["name"]
                )

                selected_roll = str(
                    selected_student["roll_no"]
                )

                st.info(
                    f"👤 **{selected_name}**  |  "
                    f"Roll No: **{selected_roll}**"
                )

                if range_attendance.empty:

                    student_history = pd.DataFrame()

                else:

                    student_history = (
                        range_attendance[
                            (
                                range_attendance[
                                    "student_id"
                                ].astype(str)
                                == str(selected_student_id)
                            )
                        ]
                        .copy()
                    )

                if student_history.empty:

                    student_present = 0

                else:

                    student_present = (
                        student_history[
                            student_history[
                                "status"
                            ].astype(str)
                            == "Present"
                        ]["date_parsed"]
                        .nunique()
                    )

                student_absent = max(
                    total_classes - student_present,
                    0
                )

                if total_classes > 0:

                    student_percentage = round(
                        (
                            student_present
                            / total_classes
                        ) * 100,
                        1
                    )

                else:

                    student_percentage = 0.0

                student_col1, student_col2, student_col3, student_col4 = (
                    st.columns(4)
                )

                with student_col1:

                    st.metric(
                        "📅 Total Classes",
                        total_classes
                    )

                with student_col2:

                    st.metric(
                        "✅ Present",
                        student_present
                    )

                with student_col3:

                    st.metric(
                        "❌ Absent",
                        student_absent
                    )

                with student_col4:

                    st.metric(
                        "📊 Attendance",
                        f"{student_percentage}%"
                    )

                st.divider()

                st.subheader(
                    "📅 Date-wise Attendance History"
                )

                if student_history.empty:

                    st.info(
                        "No attendance records found "
                        "for this student in the selected period."
                    )

                else:

                    history_display = student_history[
                        [
                            "date_parsed",
                            "time",
                            "status"
                        ]
                    ].copy()

                    history_display["Date"] = (
                        history_display[
                            "date_parsed"
                        ].apply(
                            lambda x: x.strftime(
                                "%d-%m-%Y"
                            )
                        )
                    )

                    history_display = history_display[
                        [
                            "Date",
                            "time",
                            "status"
                        ]
                    ]

                    history_display = (
                        history_display
                        .rename(
                            columns={
                                "time": "Time",
                                "status": "Status"
                            }
                        )
                    )

                    st.dataframe(
                        history_display,
                        use_container_width=True,
                        hide_index=True
                    )

                    st.subheader(
                        "📈 Student Attendance Trend"
                    )

                    student_chart = student_history[
                        student_history["status"].astype(str)
                        == "Present"
                    ].copy()

                    if student_chart.empty:

                        st.info(
                            "No present records available "
                            "for chart."
                        )

                    else:

                        student_chart = (
                            student_chart
                            .groupby("date_parsed")
                            .agg(
                                Present=("student_id", "nunique")
                            )
                            .reset_index()
                        )

                        student_chart["Present"] = 1

                        student_chart = (
                            student_chart
                            .set_index("date_parsed")
                        )

                        st.line_chart(
                            student_chart["Present"],
                            use_container_width=True
                        )

                    st.subheader(
                        "📥 Export Student Attendance"
                    )

                    student_csv = history_display.to_csv(
                        index=False
                    ).encode(
                        "utf-8"
                    )

                    student_file_name = (
                        f"{selected_name.replace(' ', '_')}_"
                        f"attendance_"
                        f"{start_date.strftime('%Y-%m-%d')}_"
                        f"to_"
                        f"{end_date.strftime('%Y-%m-%d')}.csv"
                    )

                    st.download_button(
                        label="⬇️ Download Student Attendance",
                        data=student_csv,
                        file_name=student_file_name,
                        mime="text/csv",
                        use_container_width=True
                    )

            st.divider()

            st.subheader(
                "📥 Export Complete Report"
            )

            report_csv = report_df.to_csv(
                index=False
            ).encode(
                "utf-8"
            )

            file_name = (
                f"attendance_report_"
                f"{start_date.strftime('%Y-%m-%d')}_"
                f"to_"
                f"{end_date.strftime('%Y-%m-%d')}.csv"
            )

            st.download_button(
                label="⬇️ Download Complete Report",
                data=report_csv,
                file_name=file_name,
                mime="text/csv",
                use_container_width=True
            )


# ============================================================
# SETTINGS
# ============================================================

elif page == "⚙️ Settings":

    st.title("⚙️ Settings")

    st.caption(
        "System information and configuration."
    )

    st.divider()

    st.subheader(
        "ℹ️ System Information"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.info(
            "🐍 Python\n\n"
            "Application backend"
        )

        st.info(
            "👁️ OpenCV\n\n"
            "Face detection & recognition"
        )

    with col2:

        st.info(
            "📄 CSV Storage\n\n"
            "Student and attendance data"
        )

        st.info(
            "🌐 Streamlit\n\n"
            "Web application interface"
        )

    st.divider()

    st.subheader(
        "📁 Project Storage"
    )

    st.code(
        "data/students.csv\n"
        "data/attendance.csv\n"
        "data/faces/\n"
        "data/face_model.yml",
        language="text"
    )

    st.success(
        "🟢 All core modules are operational."
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="app-footer">
        Smart Face Attendance System •
        Python + OpenCV + Streamlit
    </div>
    """,
    unsafe_allow_html=True
)