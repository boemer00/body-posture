import cv2
import math
import mediapipe as mp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional

@dataclass
class PostureConfig:
    shoulder_threshold: Optional[float] = None
    neck_threshold: Optional[float] = None
    calibration_frames_needed: int = 30
    alert_cooldown: int = 5

class PostureDetector:
    def __init__(self, config: PostureConfig = PostureConfig()):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.config = config
        self.is_calibrated = False
        self.calibration_frames = 0
        self.calibration_shoulder_angles = []
        self.calibration_neck_angles = []

    def calculate_angle(self, a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        """Calculate angle between three points."""
        ab = (b[0] - a[0], b[1] - a[1])
        bc = (b[0] - c[0], b[1] - c[1])

        dot_product = ab[0] * bc[0] + ab[1] * bc[1]
        mag_ab = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2)

        if mag_ab == 0 or mag_bc == 0:
            return 0

        angle_rad = math.acos(dot_product / (mag_ab * mag_bc))
        return math.degrees(angle_rad)

    def draw_angle(self, frame, point1, point2, point3, angle, color):
        """Draw angle visualization on frame."""
        cv2.line(frame, point1, point2, color, 2)
        cv2.line(frame, point2, point3, color, 2)
        text_position = (point2[0] + 20, point2[1] - 20)
        cv2.putText(frame, f"{int(angle)}°", text_position,
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

    def process_frame(self, frame) -> Tuple[np.ndarray, bool, Optional[float], Optional[float]]:
        """Process a single frame and return the annotated frame and posture data."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)

        if not results.pose_landmarks:
            return frame, False, None, None

        landmarks = results.pose_landmarks.landmark

        # Extract key points
        left_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * frame.shape[1]),
                        int(landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * frame.shape[0]))
        right_shoulder = (int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * frame.shape[1]),
                         int(landmarks[self.mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * frame.shape[0]))
        left_ear = (int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].x * frame.shape[1]),
                   int(landmarks[self.mp_pose.PoseLandmark.LEFT_EAR.value].y * frame.shape[0]))

        # Calculate angles
        shoulder_angle = self.calculate_angle(left_shoulder, right_shoulder, (right_shoulder[0], 0))
        neck_angle = self.calculate_angle(left_ear, left_shoulder, (left_shoulder[0], 0))

        # Handle calibration
        if not self.is_calibrated:
            if self.calibration_frames < self.config.calibration_frames_needed:
                self.calibration_shoulder_angles.append(shoulder_angle)
                self.calibration_neck_angles.append(neck_angle)
                self.calibration_frames += 1
                cv2.putText(frame, f"Calibrating... {self.calibration_frames}/{self.config.calibration_frames_needed}",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)
            else:
                self.config.shoulder_threshold = np.mean(self.calibration_shoulder_angles) - 10
                self.config.neck_threshold = np.mean(self.calibration_neck_angles) - 10
                self.is_calibrated = True

        # Draw skeleton and angles
        self.mp_drawing.draw_landmarks(frame, results.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
        midpoint = ((left_shoulder[0] + right_shoulder[0]) // 2, (left_shoulder[1] + right_shoulder[1]) // 2)
        self.draw_angle(frame, left_shoulder, midpoint, (midpoint[0], 0), shoulder_angle, (255, 0, 0))
        self.draw_angle(frame, left_ear, left_shoulder, (left_shoulder[0], 0), neck_angle, (0, 255, 0))

        return frame, True, shoulder_angle, neck_angle

    def check_posture(self, shoulder_angle: float, neck_angle: float) -> bool:
        """Check if posture is good based on calibrated thresholds."""
        if not self.is_calibrated:
            return True
        return (shoulder_angle >= self.config.shoulder_threshold and
                neck_angle >= self.config.neck_threshold)
