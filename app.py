import cv2
import math
import mediapipe as mp
import numpy as np
import time
from playsound import playsound
import os

# Initialize MediaPipe Pose and webcam
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0)

def calculate_angle(a, b, c):
    """
    Calculate the angle between three points (a, b, c).
    Args:
        a (tuple): First point (x, y).
        b (tuple): Middle point (x, y).
        c (tuple): Last point (x, y).
    Returns:
        float: Angle in degrees.
    """
    # Calculate the vectors
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (b[0] - c[0], b[1] - c[1])

    # Calculate the dot product and magnitude of vectors
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

    # Avoid division by zero
    if mag_ab == 0 or mag_bc == 0:
        return 0

    # Calculate the angle in radians
    angle_rad = math.acos(dot_product / (mag_ab * mag_bc))

    # Convert to degrees
    angle_deg = math.degrees(angle_rad)

    return angle_deg

def draw_angle(frame, point1, point2, point3, angle, color):
    """
    Draw the angle on the video frame.

    Args:
        frame (numpy.ndarray): The video frame to draw on.
        point1 (tuple): The first point of the angle (x, y).
        point2 (tuple): The vertex of the angle (x, y).
        point3 (tuple): The third point of the angle (x, y).
        angle (float): The angle value to display.
        color (tuple): The color of the lines and text (B, G, R).
    """
    # Draw lines between points
    cv2.line(frame, point1, point2, color, 2)
    cv2.line(frame, point2, point3, color, 2)

    # Calculate the position for displaying the angle
    text_position = (point2[0] + 20, point2[1] - 20)

    # Put the angle text on the frame
    cv2.putText(frame, f"{int(angle)}°", text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

# Calibration-related variables
is_calibrated = False                     # Indicates if calibration is complete
calibration_frames = 0                    # Counter for calibration frames
calibration_shoulder_angles = []          # List to store shoulder angles during calibration
calibration_neck_angles = []              # List to store neck angles during calibration

# Thresholds for bad posture detection (these will be set after calibration)
shoulder_threshold = None
neck_threshold = None

# Alert-related variables
last_alert_time = 0                       # Last time an alert was triggered
alert_cooldown = 5                        # Minimum seconds between alerts
sound_file = "alert.mp3"                  # Path to alert sound (ensure the file exists)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # STEP 2: Pose Detection
        # Extract key body landmarks
        left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                          int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                    int(landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))
        right_ear = (int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].x * frame.shape[1]),
                     int(landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y * frame.shape[0]))

        # STEP 3: Angle Calculation
        shoulder_angle = calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

        # STEP 1: Calibration
        if not is_calibrated and calibration_frames < 30:
            calibration_shoulder_angles.append(shoulder_angle)
            calibration_neck_angles.append(neck_angle)
            calibration_frames += 1
            cv2.putText(frame, f"Calibrating... {calibration_frames}/30", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
        elif not is_calibrated:
            shoulder_threshold = np.mean(calibration_shoulder_angles) - 10
            neck_threshold = np.mean(calibration_neck_angles) - 10
            is_calibrated = True
            print(f"Calibration complete. Shoulder threshold: {shoulder_threshold:.1f}, Neck threshold: {neck_threshold:.1f}")

        # Draw skeleton and angles
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))

        # STEP 4: Feedback
        if is_calibrated:
            current_time = time.time()
            if shoulder_angle < shoulder_threshold or neck_angle < neck_threshold:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > alert_cooldown:
                    print("Poor posture detected! Please sit up straight.")
                    if os.path.exists(sound_file):
                        playsound(sound_file)
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{shoulder_threshold:.1f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{neck_threshold:.1f}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # Display the frame
    cv2.imshow('Posture Corrector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
