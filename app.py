import cv2
import time
from playsound import playsound
import os
from posture_detection import PostureDetector, PostureConfig

def main():
    # Initialize
    config = PostureConfig()
    detector = PostureDetector(config)
    cap = cv2.VideoCapture(0)
    last_alert_time = 0
    sound_file = "assets/alert.mp3"

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Process frame
        frame, landmarks_detected, shoulder_angle, neck_angle = detector.process_frame(frame)

        if landmarks_detected and detector.is_calibrated:
            current_time = time.time()
            good_posture = detector.check_posture(shoulder_angle, neck_angle)

            # Handle alerts and visualization
            if not good_posture:
                status = "Poor Posture"
                color = (0, 0, 255)  # Red
                if current_time - last_alert_time > config.alert_cooldown:
                    print("Poor posture detected! Please sit up straight.")
                    if os.path.exists(sound_file):
                        playsound(sound_file)
                    last_alert_time = current_time
            else:
                status = "Good Posture"
                color = (0, 255, 0)  # Green

            # Draw status and measurements
            cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Shoulder Angle: {shoulder_angle:.1f}/{config.shoulder_threshold:.1f}",
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, f"Neck Angle: {neck_angle:.1f}/{config.neck_threshold:.1f}",
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Posture Corrector', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
