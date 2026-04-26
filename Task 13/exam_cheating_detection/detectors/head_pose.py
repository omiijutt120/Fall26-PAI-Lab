import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, Tuple
import config


# Canonical 3-D face model points (mm scale, OpenCV convention)
_MODEL_POINTS = np.array([
    (  0.0,    0.0,   0.0),   # Nose tip
    (  0.0, -330.0, -65.0),   # Chin
    (-225.0,  170.0,-135.0),  # Left eye corner
    ( 225.0,  170.0,-135.0),  # Right eye corner
    (-150.0, -150.0,-125.0),  # Left mouth corner
    ( 150.0, -150.0,-125.0),  # Right mouth corner
], dtype=np.float64)

# Corresponding MediaPipe Face Mesh landmark indices
_LANDMARK_IDS = [1, 152, 263, 33, 287, 57]


@dataclass
class HeadPoseResult:
    yaw:              float = 0.0    # positive = looking right
    pitch:            float = 0.0    # positive = looking down
    roll:             float = 0.0    # positive = tilting right
    looking_left:     bool  = False
    looking_right:    bool  = False
    looking_down:     bool  = False
    looking_up:       bool  = False
    face_detected:    bool  = False
    nose_tip:         Optional[Tuple[int,int]] = None
    direction_end:    Optional[Tuple[int,int]] = None


class HeadPoseDetector:
    def __init__(self):
        self._mp_face = mp.solutions.face_mesh
        self._face_mesh = self._mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._camera_matrix: Optional[np.ndarray] = None
        self._dist_coeffs   = np.zeros((4, 1), dtype=np.float64)

    def _build_camera_matrix(self, h: int, w: int) -> np.ndarray:
        focal = w
        cx, cy = w / 2.0, h / 2.0
        return np.array([
            [focal,  0,    cx],
            [0,    focal,  cy],
            [0,      0,    1 ],
        ], dtype=np.float64)

    def process(self, frame_rgb: np.ndarray) -> HeadPoseResult:
        h, w = frame_rgb.shape[:2]
        if self._camera_matrix is None:
            self._camera_matrix = self._build_camera_matrix(h, w)

        result = HeadPoseResult()
        fm_result = self._face_mesh.process(frame_rgb)

        if not fm_result.multi_face_landmarks:
            return result

        result.face_detected = True
        landmarks = fm_result.multi_face_landmarks[0].landmark

        # Extract 2-D image points from selected landmarks
        image_points = np.array([
            (landmarks[i].x * w, landmarks[i].y * h)
            for i in _LANDMARK_IDS
        ], dtype=np.float64)

        success, rot_vec, trans_vec = cv2.solvePnP(
            _MODEL_POINTS, image_points,
            self._camera_matrix, self._dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success:
            return result

        # Rotation matrix → Euler angles
        rot_mat, _ = cv2.Rodrigues(rot_vec)
        proj_mat   = np.hstack((rot_mat, trans_vec))
        _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

        pitch = float(euler[0][0])
        yaw   = float(euler[1][0])
        roll  = float(euler[2][0])

        result.yaw   = yaw
        result.pitch = pitch
        result.roll  = roll

        result.looking_left  = yaw < -config.HEAD_YAW_THRESHOLD
        result.looking_right = yaw >  config.HEAD_YAW_THRESHOLD
        result.looking_down  = pitch >  config.HEAD_PITCH_THRESHOLD
        result.looking_up    = pitch < -config.HEAD_PITCH_THRESHOLD

        # Arrow start: nose tip
        nose_2d = (int(landmarks[1].x * w), int(landmarks[1].y * h))
        # Project a point 500 mm in front along the nose direction
        front_3d = np.array([[0.0, 0.0, 500.0]])
        front_2d, _ = cv2.projectPoints(
            front_3d, rot_vec, trans_vec,
            self._camera_matrix, self._dist_coeffs,
        )
        direction_end = (
            int(front_2d[0][0][0]),
            int(front_2d[0][0][1]),
        )

        result.nose_tip      = nose_2d
        result.direction_end = direction_end
        return result

    def draw_overlay(self, frame_bgr: np.ndarray, result: HeadPoseResult):
        if not result.face_detected:
            return
        if result.nose_tip and result.direction_end:
            # Direction arrow
            arrow_color = (config.COLOR_DANGER
                           if (result.looking_left or result.looking_right
                               or result.looking_down)
                           else config.COLOR_OK)
            cv2.arrowedLine(frame_bgr,
                            result.nose_tip, result.direction_end,
                            arrow_color, 2, tipLength=0.2)

    def release(self):
        self._face_mesh.close()
