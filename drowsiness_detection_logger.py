import cv2
import dlib
import time
import os
import pandas as pd
from datetime import datetime
from scipy.spatial import distance
from imutils import face_utils
import winsound

# User input
user_name = input("Enter your full name: ").strip().title()
location = input("Enter your location: ").strip().title()

data_file = "drowsiness_log.xlsx"

# Assign driver ID based on name
def get_driver_id(name):
    if os.path.exists(data_file):
        try:
            df = pd.read_excel(data_file)
            if name in df["Name"].values:
                return df[df["Name"] == name]["Driver ID"].values[0]
            else:
                existing_ids = df["Driver ID"].dropna().unique()
                next_id = len(set(existing_ids)) + 1
                return f"Driver_{next_id:02}"
        except:
            return "Driver_01"
    else:
        return "Driver_01"

driver_id = get_driver_id(user_name)

# Alarm sound
def play_alarm():
    for _ in range(3):
        winsound.Beep(1000, 500)
        time.sleep(0.2)

# Save event to Excel
def log_event(eye_closure, reaction_time, confidence, status, alarm):
    new_row = pd.DataFrame([{
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Name": user_name,
        "Driver ID": driver_id,
        "Eye Closure (s)": round(eye_closure, 2),
        "Reaction Time (s)": round(reaction_time, 2),
        "Confidence (%)": round(confidence, 2),
        "Status": status,
        "Alarm": alarm,
        "Location": location
    }])

    try:
        if os.path.exists(data_file):
            df = pd.read_excel(data_file)
            df = pd.concat([df, new_row], ignore_index=True)
        else:
            df = new_row
        df.to_excel(data_file, index=False)
    except PermissionError:
        print("Please close the Excel file and try again.")

# EAR calculation
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# Thresholds
EAR_THRESHOLD = 0.22
SLEEP_DURATION = 2

# Initialize webcam
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Webcam not detected.")
    exit()

# Load Dlib face detector and predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

start_time = None
alarm_triggered = False
reaction_start = None
last_logged_status = None

print("System started. Press 'q' to quit.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Webcam read failed.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)

        status_text = ""
        color = (0, 255, 0)
        current_status = ""
        ear = 0.0
        eye_closure = 0.0
        reaction_time = 0.0
        confidence = 0.0

        if len(faces) == 0:
            current_status = "No Face"
            color = (0, 255, 255)
            status_text = "No Face Detected"
            start_time = None
            alarm_triggered = False
            reaction_start = None

        else:
            for face in faces:
                shape = predictor(gray, face)
                shape = face_utils.shape_to_np(shape)

                leftEye = shape[lStart:lEnd]
                rightEye = shape[rStart:rEnd]

                # Draw yellow dots on eye landmarks
                for (x, y) in leftEye + rightEye:
                    cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)

                # Draw enclosing blue circle around left eye
                leftEyeCenter = tuple(map(int, leftEye.mean(axis=0)))
                leftEyeRadius = int(cv2.norm(leftEye[0] - leftEye[3]) / 2)
                cv2.circle(frame, leftEyeCenter, leftEyeRadius + 5, (255, 0, 0), 2)

                # Draw enclosing blue circle around right eye
                rightEyeCenter = tuple(map(int, rightEye.mean(axis=0)))
                rightEyeRadius = int(cv2.norm(rightEye[0] - rightEye[3]) / 2)
                cv2.circle(frame, rightEyeCenter, rightEyeRadius + 5, (255, 0, 0), 2)

                # EAR calculation
                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)
                ear = (leftEAR + rightEAR) / 2.0

                if ear < EAR_THRESHOLD:
                    if start_time is None:
                        start_time = time.time()
                        reaction_start = time.time()
                    elif time.time() - start_time >= SLEEP_DURATION:
                        if not alarm_triggered:
                            print("Driver is sleeping.")
                            play_alarm()
                            eye_closure = time.time() - start_time
                            reaction_time = time.time() - reaction_start
                            confidence = (1 - ear / EAR_THRESHOLD) * 100
                            current_status = "Sleeping"
                            if last_logged_status != current_status:
                                log_event(eye_closure, reaction_time, confidence, current_status, "Yes")
                                last_logged_status = current_status
                            alarm_triggered = True
                        status_text = "Driver is Sleeping"
                        color = (0, 0, 255)
                    else:
                        current_status = "Getting Sleepy"
                        eye_closure = time.time() - start_time
                        reaction_time = time.time() - reaction_start
                        confidence = (1 - ear / EAR_THRESHOLD) * 100
                        status_text = "Driver is Getting Sleepy"
                        color = (0, 165, 255)
                        if last_logged_status != current_status:
                            log_event(eye_closure, reaction_time, confidence, current_status, "No")
                            last_logged_status = current_status
                else:
                    current_status = "Alert"
                    eye_closure = 0.0
                    reaction_time = 0.0
                    confidence = 0.0
                    status_text = "Driver is Alert"
                    color = (0, 255, 0)
                    start_time = None
                    alarm_triggered = False
                    reaction_start = None
                    if last_logged_status != current_status:
                        log_event(eye_closure, reaction_time, confidence, current_status, "No")
                        last_logged_status = current_status

        # Display status
        cv2.putText(frame, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Show webcam
        cv2.imshow("Driver Drowsiness Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped manually.")

cap.release()
cv2.destroyAllWindows()
