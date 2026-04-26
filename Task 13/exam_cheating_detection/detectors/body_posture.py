import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass
from typing import Optional, Tuple
import config


@dataclass
class PostureResult:
    shoulder_angle:    float = 0.0
    torso_lean:        float = 0.0  
    body_rotation:     bool  = False
    body_lean:         bool  = False
    person_detected:   bool  = False
    left_shoulder:     Optional[Tuple[int,int]] = None
    right_shoulder:    Optional[Tuple[int,int]] = None
    left_hip:          Optional[Tuple[int,int]] = None
    right_hip:         Optional[Tuple[int,int]] = None


class BodyPostureDetector:
    def __init__(self):
        self._mp_pose = mp.solutions.pose
        self._pose    = self._mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._mp_draw = mp.solutions.drawing_utils

    @staticmethod
    def _landmark_px(lm, w: int, h: int) -> Tuple[int, int]:
        return int(lm.x * w), int(lm.y * h)

    @staticmethod
    def _shoulder_rotation(ls: Tuple, rs: Tuple) -> float:
        """
        Estimate body-plane rotation by measuring the Z-component ratio.
        When facing the camera directly the shoulders are at equal depth;
        rotation creates a depth disparity captured via x-spread vs expected.
        We approximate with the horizontal-to-distance ratio.
        """
        dx = abs(rs[0] - ls[0])
        return float(dx)     # raw pixel width — normalised by caller

    def process(self, frame_rgb: np.ndarray) -> PostureResult:
        h, w   = frame_rgb.shape[:2]
        result = PostureResult()
        pose   = self._pose.process(frame_rgb)

        if not pose.pose_landmarks:
            return result

        result.person_detected = True
        lm = pose.pose_landmarks.landmark
        P  = self._mp_pose.PoseLandmark

        def px(idx):
            return self._landmark_px(lm[idx], w, h)

        ls = px(P.LEFT_SHOULDER)
        rs = px(P.RIGHT_SHOULDER)
        lh = px(P.LEFT_HIP)
        rh = px(P.RIGHT_HIP)

        result.left_shoulder  = ls
        result.right_shoulder = rs
        result.left_hip       = lh
        result.right_hip      = rh

        # ── Shoulder rotation ─────────────────────────────────────────────
        # Use landmark visibility / z-values from MediaPipe
        lz = lm[P.LEFT_SHOULDER.value].z
        rz = lm[P.RIGHT_SHOULDER.value].z
        z_diff = (rz - lz) * 100   # scale to meaningful range

        result.shoulder_angle = float(z_diff)

        # ── Torso lateral lean ────────────────────────────────────────────
        # Midpoint of shoulders vs midpoint of hips
        mid_shoulder_x = (ls[0] + rs[0]) / 2.0
        mid_hip_x      = (lh[0] + rh[0]) / 2.0
        mid_shoulder_y = (ls[1] + rs[1]) / 2.0
        mid_hip_y      = (lh[1] + rh[1]) / 2.0

        torso_h = abs(mid_shoulder_y - mid_hip_y) + 1e-6
        lateral_offset = (mid_shoulder_x - mid_hip_x) / torso_h * 30
        result.torso_lean = float(lateral_offset)

        result.body_rotation = abs(z_diff) > config.SHOULDER_ROTATION_THRESHOLD
        result.body_lean     = abs(lateral_offset) > config.TORSO_LEAN_THRESHOLD

        return result

    def draw_overlay(self, frame_bgr: np.ndarray, result: PostureResult):
        if not result.person_detected:
            return

        ls = result.left_shoulder
        rs = result.right_shoulder
        lh = result.left_hip
        rh = result.right_hip

        line_color = (config.COLOR_DANGER if result.body_rotation
                      else config.COLOR_OK)

        if ls and rs:
            cv2.line(frame_bgr, ls, rs, line_color, 2)
        if lh and rh:
            cv2.line(frame_bgr, lh, rh, (100, 100, 200), 2)
        if ls and lh:
            cv2.line(frame_bgr, ls, lh, (120, 120, 180), 1)
        if rs and rh:
            cv2.line(frame_bgr, rs, rh, (120, 120, 180), 1)

        dot_color = (config.COLOR_DANGER if result.body_rotation
                     else config.COLOR_OK)
        for pt in [ls, rs, lh, rh]:
            if pt:
                cv2.circle(frame_bgr, pt, 5, dot_color, -1)

    def release(self):
        self._pose.close()
