# 📸 Smart Face Attendance System

A face-recognition based student attendance management system built using Python, OpenCV, Pandas and Streamlit.

The system allows students to be registered with face data, automatically recognizes registered faces, marks attendance, prevents duplicate attendance for the same day, and provides attendance records and reports.

---

## ✨ Features

### 👤 Student Management
- Register new students
- Store student ID, name and roll number
- Store face images for registered students
- Search students by name or roll number
- View face-data availability
- Delete students
- Remove associated face data
- Retrain the recognition model after student changes

### 📷 Face Recognition
- Capture face using camera
- Detect faces using OpenCV
- Recognize registered students
- Display recognized student's details
- Handle unknown/unregistered faces

### 📋 Attendance
- Automatically mark recognized students as Present
- Store date and time of attendance
- Prevent duplicate attendance on the same day
- View complete attendance history
- Filter attendance by date
- Export attendance records as CSV

### 📊 Reports
- Attendance statistics
- Student-wise attendance
- Present and absent days
- Attendance percentage
- Low-attendance detection
- Date-range reports
- Daily attendance summary
- Attendance trend visualization
- Individual student attendance history
- CSV report export

### 🏠 Dashboard
- Total students
- Present students today
- Absent students today
- Today's attendance percentage
- Today's attendance table
- Recent attendance activity
- Attendance trend
- Quick overview of the system

---

## 🛠️ Technologies Used

- Python 3
- Streamlit
- OpenCV
- NumPy
- Pandas
- CSV-based data storage

---

## 📁 Project Structure

```text
Face_Recognisation_System_Using_Py/
│
├── app.py
├── database.py
├── attendance.py
├── face_recognition.py
├── requirements.txt
├── README.md
├── haarcascade_frontalface_default.xml
│
├── assets/
│
├── data/
│   ├── students.csv
│   ├── attendance.csv
│   ├── attendance/
│   ├── faces/
│   └── face_model.yml
│
├── data_backup/
│
└── venv/