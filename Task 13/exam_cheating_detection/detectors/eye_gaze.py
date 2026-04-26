import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Tuple
import config


# Eye outline landmark indices (MediaPipe Face Mesh)
_LEFT_EYE   = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466,
               388, 387, 386, 385, 384, 398]
_RIGHT_EYE  = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173,
               157, 158, 159, 160, 161, 246]

# Iris centre landmarks (from refined mesh)
_LEFT_IRIS  = [474, 475, 476, 477]
_RIGHT_IRIS = [469, 470, 471, 472]


@dataclass
class GazeResult:
    gaze_ratio_x: float = 0.0     # −1 (left) … +1 (right)
    gaze_ratio_y: float = 0.0     # −1 (up)   … +1 (down)
    looking_left:  bool = False
    looking_right: bool = False
    face_detected: bool = False
    left_iris_center:  Optional[Tuple[int,int]] = None
    right_iris_center: Optional[Tuple[int,int]] = None


class EyeGazeDetector:
    def __init__(self):
        self._mp_face  = mp.solutions.face_mesh
        self._face_mesh = self._mp_face.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )

    @staticmethod
    def _iris_ratio(iris_pts: np.ndarray,
                    eye_pts:  np.ndarray) -> Tuple[float, float]:
        """
        Returns normalised iris position within the eye bounding box.
        x: 0 = extreme left, 1 = extreme right
        y: 0 = extreme top,  1 = extreme bottom
        """
        eye_min = eye_pts.min(axis=0)
        eye_max = eye_pts.max(axis=0)
        eye_size = eye_max - eye_min + 1e-6

        iris_center = iris_pts.mean(axis=0)
        rx = (iris_center[0] - eye_min[0]) / eye_size[0]
        ry = (iris_center[1] - eye_min[1]) / eye_size[1]
        return float(rx), float(ry)

    def process(self, frame_rgb: np.ndarray) -> GazeResult:
        h, w   = frame_rgb.shape[:2]
        result = GazeResult()
        fm     = self._face_mesh.process(frame_rgb)

        if not fm.multi_face_landmarks:
            return result

        result.face_detected = True
        lm = fm.multi_face_landmarks[0].landmark

        def pts(indices):
            return np.array([(lm[i].x * w, lm[i].y * h) for i in indices])

        l_iris = pts(_LEFT_IRIS)
        r_iris = pts(_RIGHT_IRIS)
        l_eye  = pts(_LEFT_EYE)
        r_eye  = pts(_RIGHT_EYE)

        lrx, lry = self._iris_ratio(l_iris, l_eye)
        rrx, rry = self._iris_ratio(r_iris, r_eye)

        # Average both eyes; re-centre so 0.5 → 0
        gaze_x = ((lrx + rrx) / 2.0 - 0.5) * 2.0  # −1 … +1
        gaze_y = ((lry + rry) / 2.0 - 0.5) * 2.0

        result.gaze_ratio_x  = gaze_x
        result.gaze_ratio_y  = gaze_y
        result.looking_left  = gaze_x < -config.GAZE_HORIZONTAL_THRESHOLD
        result.looking_right = gaze_x >  config.GAZE_HORIZONTAL_THRESHOLD

        result.left_iris_center  = tuple(l_iris.mean(axis=0).astype(int))
        result.right_iris_center = tuple(r_iris.mean(axis=0).astype(int))
        return result

    def draw_overlay(self, frame_bgr: np.ndarray, result: GazeResult):
        if not result.face_detected:
            return
        color = (config.COLOR_DANGER
                 if (result.looking_left or result.looking_right)
                 else config.COLOR_OK)
        for center in [result.left_iris_center, result.right_iris_center]:
            if center:
                cv2.circle(frame_bgr, center, 4, color, -1)
                cv2.circle(frame_bgr, center, 7, color,  1)

    def release(self):
        self._face_mesh.close()
