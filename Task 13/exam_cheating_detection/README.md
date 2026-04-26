# Exam Cheating Detection System

A real-time AI-powered exam proctoring system that monitors students for suspicious behaviour using computer vision — no cloud services required, runs entirely on your local machine.

---

## What It Detects

| Detection | Method | Alert Type |
|---|---|---|
| Head turning left / right | MediaPipe Face Mesh + solvePnP | 🚨 Danger |
| Looking down (notes/phone) | Head pitch estimation | ⚠ Warning |
| Eye gaze shifting sideways | MediaPipe iris landmarks | ⚠ Warning |
| Body / shoulder rotation | MediaPipe Pose landmark Z-depth | 🚨 Danger |
| Torso lateral lean | Hip–shoulder midpoint alignment | ⚠ Warning |
| Mobile phone in frame | YOLOv8 object detection (COCO class 67) | 🚨 Danger |
| Student absent from seat | Pose landmark confidence | 🚨 Danger |

---

## Project Structure

```
exam_cheating_detection/
│
├── main.py                  ← Entry point — run this
├── config.py                ← All thresholds and settings
├── requirements.txt
│
├── detectors/
│   ├── head_pose.py         ← Yaw / pitch / roll via solvePnP
│   ├── eye_gaze.py          ← Iris position tracking
│   ├── body_posture.py      ← Shoulder rotation & torso lean
│   └── phone_detector.py    ← YOLOv8 phone / object detection
│
└── utils/
    ├── alert_manager.py     ← De-bounce, cooldowns, violation log
    ├── drawing.py           ← All OpenCV drawing helpers
    └── logger.py            ← CSV violation log writer
```

---

## Setup

### 1. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> On the first run, YOLOv8 will automatically download `yolov8n.pt` (~6 MB).  
> MediaPipe models download on first use as well (~15 MB).

### 3. Run

```bash
python main.py
```

---

## Controls

| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `P` | Pause / Resume |
| `R` | Reset all violation counters |
| `S` | Save a screenshot |

---

## Configuration (`config.py`)

Every sensitivity threshold is in one file so you can tune the system without touching detector code.

```python
# Example: make head-turn detection more lenient
HEAD_YAW_THRESHOLD = 30       # degrees (default: 25)

# Use a different camera
CAMERA_INDEX = 1

# Require more consecutive frames before firing an alert (reduces false positives)
ALERT_MIN_FRAMES = 10         # default: 6
```

### Key thresholds

| Setting | Default | Effect |
|---|---|---|
| `HEAD_YAW_THRESHOLD` | 25° | Horizontal head turn before alert |
| `HEAD_PITCH_THRESHOLD` | 20° | Looking down before alert |
| `SHOULDER_ROTATION_THRESHOLD` | 18 | Body twist sensitivity |
| `PHONE_CONFIDENCE_THRESHOLD` | 0.45 | YOLO detection confidence |
| `ALERT_MIN_FRAMES` | 6 | De-bounce: frames before alert fires |
| `ALERT_COOLDOWN` | varies | Seconds before the same alert fires again |

---

## Output

- **Live window** — annotated video feed with sidebar dashboard showing real-time metrics, violation counters and alert log.
- **`violation_log.csv`** — timestamped CSV written to the project folder each session.

---

## Requirements

- Python 3.9 +
- Webcam (USB or built-in)
- ~500 MB disk space (for model weights)

---

## How It Works

### Head Pose
Uses MediaPipe Face Mesh to detect 468 facial landmarks in 3-D.  Six specific landmarks (nose tip, chin, eye corners, mouth corners) are matched against a canonical 3-D face model and fed into OpenCV's `solvePnP` to recover the rotation vector, which is decomposed into Euler angles.

### Eye Gaze
MediaPipe's refined face mesh provides four iris landmarks per eye.  The iris centroid is normalised relative to the eye bounding box to produce a gaze ratio — how far the pupil is from the centre of the eye socket.

### Body Posture
MediaPipe Pose tracks 33 body landmarks including both shoulders and hips.  The Z-component depth difference between the two shoulders reveals body rotation (one shoulder moving closer to the camera).  The lateral displacement of the shoulder midpoint vs hip midpoint captures torso lean.

### Phone Detection
YOLOv8-nano (6 MB) runs inference on the full frame every tick.  COCO class 67 (`cell phone`) triggers the phone alert.  Books and notebooks are also flagged as potential hidden-notes objects.

---

## Troubleshooting

**Camera not opening** — Change `CAMERA_INDEX` in `config.py` (try `1` or `2`).

**Too many false positives** — Increase `ALERT_MIN_FRAMES` and `ALERT_COOLDOWN` values, or raise the angle thresholds.

**Slow FPS** — Switch to `yolov8n.pt` (already the default) or reduce `FRAME_WIDTH`/`FRAME_HEIGHT`.

**MediaPipe import error** — Run `pip install mediapipe --upgrade`.
