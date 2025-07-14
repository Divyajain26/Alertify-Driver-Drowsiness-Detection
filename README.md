# Alertify-Driver-Drowsiness-Detection
Real-time driver drowsiness detection using EAR and OpenCV with a webcam. Classifies driver status as Alert, Getting Sleepy, or Sleeping. Triggers an alarm if eyes remain closed too long for safety. Logs all events with timestamps into an Excel file for analysis.

# Driver Drowsiness Detection System

This project is a real-time Driver Drowsiness Detection System that uses a webcam and computer vision to monitor a driver's alertness. It calculates the Eye Aspect Ratio (EAR) using facial landmarks to determine whether the driver is Alert, Getting Sleepy, or Sleeping. When drowsiness is detected, the system triggers an audible alarm and logs the event details into an Excel file (`drowsiness_log.xlsx`) for record-keeping and analysis.

## Features

- Real-time eye tracking using a webcam
- EAR-based drowsiness detection
- Alarm alert when the driver is sleepy
- Event logging to Excel with the following details:
  - Timestamp
  - Driver Name and ID
  - Location
  - Eye Closure Duration
  - Reaction Time
  - Confidence Score
  - Status and Alarm status

## Technologies Used

- Python
- OpenCV
- dlib (facial landmark detection)
- Pandas
- Scipy
- Winsound (for alarm)

## Required File

This project uses the `shape_predictor_68_face_landmarks.dat` file, which is not included in the repository due to size limits.

Download the required model file from the link below:  
[Download shape_predictor_68_face_landmarks.dat.bz2](https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2)

Instructions:
1. Download and extract the `.bz2` file to get the `.dat` file.
2. Place the extracted `.dat` file in your project folder.
