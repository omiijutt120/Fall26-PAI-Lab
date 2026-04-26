# =============================================================================
#  Exam Cheating Detection System — Configuration
# =============================================================================

# ── Camera ────────────────────────────────────────────────────────────────────
CAMERA_INDEX      = 0          # 0 = default webcam
FRAME_WIDTH       = 1280
FRAME_HEIGHT      = 720
TARGET_FPS        = 30

# ── Head-Pose Thresholds (degrees) ────────────────────────────────────────────
HEAD_YAW_THRESHOLD   = 25      # Left / right turn
HEAD_PITCH_THRESHOLD = 20      # Looking down
HEAD_ROLL_THRESHOLD  = 20      # Tilting head sideways

# ── Body-Posture Thresholds ───────────────────────────────────────────────────
SHOULDER_ROTATION_THRESHOLD = 18   # degrees of shoulder twist
TORSO_LEAN_THRESHOLD        = 22   # excessive forward / sideways lean

# ── Gaze Thresholds ───────────────────────────────────────────────────────────
GAZE_HORIZONTAL_THRESHOLD = 0.38   # ratio — looking far left / right
GAZE_VERTICAL_THRESHOLD   = 0.30   # ratio — looking far up / down

# ── Phone Detection ───────────────────────────────────────────────────────────
PHONE_CONFIDENCE_THRESHOLD = 0.45  # YOLO confidence cutoff
YOLO_MODEL_NAME            = "yolov8n.pt"   # nano — fast & small

# ── Alert System ─────────────────────────────────────────────────────────────
# How many consecutive frames a violation must persist before it fires an alert
ALERT_MIN_FRAMES = 6

# Seconds between repeated alerts for the same violation type
ALERT_COOLDOWN = {
    "HEAD_LEFT"       : 4,
    "HEAD_RIGHT"      : 4,
    "HEAD_DOWN"       : 3,
    "HEAD_UP"         : 4,
    "BODY_ROTATION"   : 5,
    "BODY_LEAN"       : 5,
    "GAZE_LEFT"       : 3,
    "GAZE_RIGHT"      : 3,
    "PHONE_DETECTED"  : 3,
    "PERSON_ABSENT"   : 5,
}

# ── Colours  (BGR) ────────────────────────────────────────────────────────────
COLOR_OK      = (80,  200,  80)     # green
COLOR_WARNING = (0,   200, 255)     # amber
COLOR_DANGER  = (40,   40, 220)     # red
COLOR_INFO    = (220, 180,  50)     # blue
COLOR_WHITE   = (255, 255, 255)
COLOR_BLACK   = (0,   0,   0)
COLOR_PANEL   = (22,  22,  30)      # dark sidebar
COLOR_BORDER  = (55,  55,  70)

# ── Display ───────────────────────────────────────────────────────────────────
SIDEBAR_WIDTH   = 340
FONT            = 0   # cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SM   = 0.52
FONT_SCALE_MD   = 0.65
FONT_SCALE_LG   = 0.80
FONT_THICKNESS  = 1
MAX_LOG_ENTRIES = 12   # visible in the sidebar log
