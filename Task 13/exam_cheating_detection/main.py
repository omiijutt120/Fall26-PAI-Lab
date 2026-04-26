import sys
import os
import time
import cv2
import numpy as np

import config
from detectors import (
    HeadPoseDetector,
    EyeGazeDetector,
    BodyPostureDetector,
    PhoneDetector,
)
from utils.alert_manager import AlertManager
from utils.logger        import ViolationLogger
from utils               import drawing as draw


# ─────────────────────────────────────────────────────────────────────────────
#  Dashboard renderer
# ─────────────────────────────────────────────────────────────────────────────

class Dashboard:
    """Builds and blends the full-frame display each tick."""

    def __init__(self, cam_w: int, cam_h: int):
        self.cam_w     = cam_w
        self.cam_h     = cam_h
        self.total_w   = cam_w + config.SIDEBAR_WIDTH
        self._frame_no = 0

    def render(self,
               camera_frame: np.ndarray,
               head_result,
               gaze_result,
               posture_result,
               phone_result,
               alert_mgr: AlertManager,
               fps: float,
               paused: bool) -> np.ndarray:

        self._frame_no += 1
        canvas = np.zeros((self.cam_h, self.total_w, 3), dtype=np.uint8)

        # ── Left: annotated camera feed ───────────────────────────────────
        cam = camera_frame.copy()
        self._overlay_detections(cam, head_result, gaze_result,
                                 posture_result, phone_result, alert_mgr)
        self._overlay_hud(cam, fps, paused, alert_mgr)
        canvas[:, :self.cam_w] = cam

        # ── Right: sidebar panel ──────────────────────────────────────────
        sidebar = draw.build_sidebar(self.cam_h)
        y = self._draw_sidebar_content(
            sidebar, head_result, gaze_result,
            posture_result, phone_result, alert_mgr,
        )
        canvas[:, self.cam_w:] = sidebar

        return canvas

    # ── Detection overlays on the camera feed ─────────────────────────────

    def _overlay_detections(self, frame, head, gaze, posture,
                            phone, alert_mgr):
        # Draw pose skeleton
        posture_det = BodyPostureDetector.__new__(BodyPostureDetector)
        posture_det.draw_overlay(frame, posture)

        # Head direction arrow
        head_det = HeadPoseDetector.__new__(HeadPoseDetector)
        head_det.draw_overlay(frame, head)

        # Iris markers
        gaze_det = EyeGazeDetector.__new__(EyeGazeDetector)
        gaze_det.draw_overlay(frame, gaze)

        # Phone bounding boxes
        phone_det = PhoneDetector.__new__(PhoneDetector)
        phone_det.draw_overlay(frame, phone)

        # ── Violation banners (centred at top of feed) ────────────────────
        self._draw_violation_banners(frame, head, gaze, posture, phone)

    def _draw_violation_banners(self, frame, head, gaze, posture, phone):
        h, w = frame.shape[:2]
        banners = []

        if not posture.person_detected:
            banners.append(("⚠  NO STUDENT DETECTED", "danger"))
        if head.looking_left:
            banners.append(("◀  LOOKING LEFT", "danger"))
        if head.looking_right:
            banners.append(("▶  LOOKING RIGHT", "danger"))
        if head.looking_down:
            banners.append(("▼  LOOKING DOWN", "warning"))
        if gaze.looking_left:
            banners.append(("◄  GAZE SHIFTED LEFT", "warning"))
        if gaze.looking_right:
            banners.append(("►  GAZE SHIFTED RIGHT", "warning"))
        if posture.body_rotation:
            banners.append(("↻  BODY ROTATION", "danger"))
        if posture.body_lean:
            banners.append(("↗  TORSO LEAN", "warning"))
        if phone.phone_found:
            banners.append(("📱  PHONE DETECTED", "danger"))

        bh, pad = 32, 8
        for i, (text, severity) in enumerate(banners[:4]):   # show max 4
            bw   = cv2.getTextSize(text, config.FONT, 0.58, 1)[0][0] + 24
            bx   = (w - bw) // 2
            by   = 14 + i * (bh + pad)
            color = config.COLOR_DANGER if severity == "danger" else config.COLOR_WARNING
            draw.draw_rounded_rect(frame, (bx, by), (bx+bw, by+bh),
                                   color, radius=6, alpha=0.85)
            draw.put_text(frame, text, (bx + 12, by + bh - 8),
                          scale=0.58, color=config.COLOR_WHITE, shadow=True)

    # ── HUD overlay (fps, title) ──────────────────────────────────────────

    def _overlay_hud(self, frame, fps: float, paused: bool,
                     alert_mgr: AlertManager):
        h, w = frame.shape[:2]

        # Title bar
        draw.draw_rounded_rect(frame, (8, 8), (280, 36),
                               (20, 20, 28), radius=6, alpha=0.80)
        draw.put_text(frame, "EXAM MONITOR  v1.0",
                      (16, 28), scale=0.58,
                      color=(200, 200, 200), shadow=True)

        # FPS
        fps_col = (config.COLOR_OK if fps > 20
                   else config.COLOR_WARNING if fps > 12
                   else config.COLOR_DANGER)
        draw.put_text(frame, f"FPS {fps:.0f}", (w - 80, 28),
                      scale=0.55, color=fps_col, shadow=True)

        # PAUSED banner
        if paused:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)
            draw.put_text(frame, "PAUSED  —  Press P to resume",
                          (w // 2 - 180, h // 2),
                          scale=0.90, color=config.COLOR_WHITE,
                          thickness=2, shadow=True)

        # Total violations indicator (bottom-left)
        viol = alert_mgr.total_violations
        vcol = (config.COLOR_OK if viol == 0
                else config.COLOR_WARNING if viol < 5
                else config.COLOR_DANGER)
        draw.draw_rounded_rect(frame, (8, h - 38), (210, h - 10),
                               (20, 20, 28), radius=6, alpha=0.80)
        draw.put_text(frame,
                      f"Total violations: {viol}",
                      (16, h - 18), scale=0.55, color=vcol, shadow=True)

    # ── Sidebar content ───────────────────────────────────────────────────

    def _draw_sidebar_content(self, panel, head, gaze,
                              posture, phone, alert_mgr) -> int:
        pw = panel.shape[1]
        x  = 12
        y  = draw.draw_sidebar_header(
            panel,
            title    = "EXAM CHEATING",
            subtitle = "Detection System",
        )

        # ── Status section ────────────────────────────────────────────────
        y = draw.draw_section_title(panel, "Status", x, y) + 6

        draw.draw_status_badge(panel, "Student",
                               "Present" if posture.person_detected else "Absent",
                               x, y, w=pw-24,
                               ok=posture.person_detected)
        y += 42
        draw.draw_status_badge(panel, "Face",
                               "Detected" if head.face_detected else "Not Found",
                               x, y, w=pw-24,
                               ok=head.face_detected)
        y += 42
        draw.draw_status_badge(panel, "Phone",
                               f"Found ({len(phone.phones)})" if phone.phone_found
                               else "Clear",
                               x, y, w=pw-24,
                               ok=not phone.phone_found)
        y += 52

        # ── Head angles ───────────────────────────────────────────────────
        y = draw.draw_section_title(panel, "Head Orientation", x, y) + 8

        draw.draw_meter(panel, "Yaw  (L/R)",
                        head.yaw, config.HEAD_YAW_THRESHOLD * 2,
                        x, y, w=pw-28)
        y += 22
        draw.draw_meter(panel, "Pitch (U/D)",
                        head.pitch, config.HEAD_PITCH_THRESHOLD * 2,
                        x, y, w=pw-28)
        y += 22
        draw.draw_meter(panel, "Roll  (tilt)",
                        head.roll, config.HEAD_ROLL_THRESHOLD * 2,
                        x, y, w=pw-28)
        y += 32

        # ── Gaze ──────────────────────────────────────────────────────────
        y = draw.draw_section_title(panel, "Eye Gaze", x, y) + 8

        draw.draw_meter(panel, "Gaze X",
                        gaze.gaze_ratio_x,
                        config.GAZE_HORIZONTAL_THRESHOLD * 2,
                        x, y, w=pw-28)
        y += 22
        draw.draw_meter(panel, "Gaze Y",
                        gaze.gaze_ratio_y,
                        config.GAZE_VERTICAL_THRESHOLD * 2,
                        x, y, w=pw-28)
        y += 32

        # ── Body ──────────────────────────────────────────────────────────
        y = draw.draw_section_title(panel, "Body Posture", x, y) + 8

        draw.draw_meter(panel, "Rotation",
                        posture.shoulder_angle,
                        config.SHOULDER_ROTATION_THRESHOLD * 2,
                        x, y, w=pw-28)
        y += 22
        draw.draw_meter(panel, "Lean",
                        posture.torso_lean,
                        config.TORSO_LEAN_THRESHOLD * 2,
                        x, y, w=pw-28)
        y += 32

        # ── Violation counts ──────────────────────────────────────────────
        y = draw.draw_section_title(panel, "Violation Counts", x, y) + 8

        counts_map = {
            "Head Left"    : alert_mgr.counts.get("HEAD_LEFT",     0),
            "Head Right"   : alert_mgr.counts.get("HEAD_RIGHT",    0),
            "Head Down"    : alert_mgr.counts.get("HEAD_DOWN",     0),
            "Gaze Shift"   : (alert_mgr.counts.get("GAZE_LEFT",  0) +
                              alert_mgr.counts.get("GAZE_RIGHT", 0)),
            "Body Rotation": alert_mgr.counts.get("BODY_ROTATION", 0),
            "Phone"        : alert_mgr.counts.get("PHONE_DETECTED",0),
        }
        for label, count in counts_map.items():
            col = (config.COLOR_DANGER  if count >= 5
                   else config.COLOR_WARNING if count >= 2
                   else (160, 160, 160))
            draw.put_text(panel, f"{label}:", (x, y),
                          scale=0.44, color=(160,160,160))
            draw.put_text(panel, str(count), (pw - 30, y),
                          scale=0.44, color=col)
            y += 18
        y += 10

        # ── Alert log ─────────────────────────────────────────────────────
        remaining = panel.shape[0] - y - 20
        if remaining > 30:
            y = draw.draw_section_title(panel, "Alert Log", x, y) + 6
            recent = alert_mgr.get_recent_log(
                n=min(config.MAX_LOG_ENTRIES, remaining // 16))
            for alert in reversed(recent):
                if y + 14 > panel.shape[0] - 8:
                    break
                draw.draw_alert_log_entry(
                    panel,
                    time_str=alert.time_str,
                    message=alert.message,
                    severity=alert.severity,
                    x=x, y=y, w=pw - 24,
                )
                y += 15

        return y


# ─────────────────────────────────────────────────────────────────────────────
#  Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  Exam Cheating Detection System")
    print("  Press Q/ESC to quit  |  P to pause  |  R to reset  |  S to save")
    print("="*60 + "\n")

    # ── Initialise camera ─────────────────────────────────────────────────
    cap = cv2.VideoCapture(config.CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open camera index {config.CAMERA_INDEX}")
        sys.exit(1)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          config.TARGET_FPS)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] Camera: {actual_w}×{actual_h} px")

    # ── Initialise detectors ──────────────────────────────────────────────
    print("[INFO] Loading detectors …")
    head_det    = HeadPoseDetector()
    gaze_det    = EyeGazeDetector()
    posture_det = BodyPostureDetector()

    try:
        phone_det = PhoneDetector()
        phone_available = True
        print("[INFO] YOLOv8 phone detector loaded.")
    except ImportError:
        print("[WARN] ultralytics not installed — phone detection disabled.")
        phone_available = False
        phone_det = None

    # ── Support objects ───────────────────────────────────────────────────
    alert_mgr = AlertManager()
    logger    = ViolationLogger()
    dashboard = Dashboard(actual_w, actual_h)

    # ── Runtime state ─────────────────────────────────────────────────────
    paused     = False
    fps        = 0.0
    t_prev     = time.time()
    frame_idx  = 0
    last_frame = np.zeros((actual_h, actual_w, 3), dtype=np.uint8)

    # Blank default results (shown when no frame has been processed yet)
    from detectors.head_pose    import HeadPoseResult
    from detectors.eye_gaze     import GazeResult
    from detectors.body_posture import PostureResult
    from detectors.phone_detector import PhoneDetectorResult

    head_result    = HeadPoseResult()
    gaze_result    = GazeResult()
    posture_result = PostureResult()
    phone_result   = PhoneDetectorResult()

    win_name = "Exam Cheating Detection"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, actual_w + config.SIDEBAR_WIDTH, actual_h)

    print("[INFO] System ready.  Monitoring …\n")

    while True:
        # ── Key handling ──────────────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):         # Q or ESC
            break
        elif key == ord("p"):
            paused = not paused
        elif key == ord("r"):
            alert_mgr.reset()
            print("[INFO] Counters reset.")
        elif key == ord("s"):
            fname = f"screenshot_{int(time.time())}.jpg"
            cv2.imwrite(fname, canvas)
            print(f"[INFO] Saved {fname}")

        if paused:
            canvas = dashboard.render(
                camera_frame   = last_frame,
                head_result    = head_result,
                gaze_result    = gaze_result,
                posture_result = posture_result,
                phone_result   = phone_result,
                alert_mgr      = alert_mgr,
                fps            = fps,
                paused         = True,
            )
            cv2.imshow(win_name, canvas)
            continue

        # ── Capture ───────────────────────────────────────────────────────
        ret, frame_bgr = cap.read()
        if not ret:
            print("[WARN] Frame capture failed — retrying …")
            continue

        frame_idx += 1
        last_frame = frame_bgr.copy()
        frame_rgb  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # ── Run detectors ─────────────────────────────────────────────────
        head_result    = head_det.process(frame_rgb)
        gaze_result    = gaze_det.process(frame_rgb)
        posture_result = posture_det.process(frame_rgb)

        if phone_available:
            phone_result = phone_det.process(frame_bgr)
        else:
            phone_result = PhoneDetectorResult()

        # ── Evaluate alerts ───────────────────────────────────────────────
        fired = []

        def check(code, triggered, msg, sev="warning"):
            a = alert_mgr.update(code, triggered, msg, sev)
            if a:
                fired.append(a)

        check("HEAD_LEFT",   head_result.looking_left,
              "Head turned LEFT",       "danger")
        check("HEAD_RIGHT",  head_result.looking_right,
              "Head turned RIGHT",      "danger")
        check("HEAD_DOWN",   head_result.looking_down,
              "Looking DOWN",           "warning")
        check("HEAD_UP",     head_result.looking_up,
              "Looking UP",             "warning")
        check("GAZE_LEFT",   gaze_result.looking_left,
              "Gaze shifted LEFT",      "warning")
        check("GAZE_RIGHT",  gaze_result.looking_right,
              "Gaze shifted RIGHT",     "warning")
        check("BODY_ROTATION", posture_result.body_rotation,
              "Body rotation detected", "danger")
        check("BODY_LEAN",  posture_result.body_lean,
              "Torso lean detected",    "warning")
        check("PHONE_DETECTED", phone_result.phone_found,
              "MOBILE PHONE detected",  "danger")
        check("PERSON_ABSENT",  not posture_result.person_detected,
              "Student absent / hidden","danger")

        for alert in fired:
            logger.log(alert.code, alert.message, alert.severity)
            sev_tag = "⚠" if alert.severity == "warning" else "🚨"
            print(f"  {sev_tag}  [{alert.time_str}]  {alert.message}")

        # ── Render dashboard ──────────────────────────────────────────────
        canvas = dashboard.render(
            camera_frame   = frame_bgr,
            head_result    = head_result,
            gaze_result    = gaze_result,
            posture_result = posture_result,
            phone_result   = phone_result,
            alert_mgr      = alert_mgr,
            fps            = fps,
            paused         = False,
        )

        cv2.imshow(win_name, canvas)

        # ── FPS ───────────────────────────────────────────────────────────
        t_now    = time.time()
        fps      = 1.0 / max(t_now - t_prev, 1e-6)
        t_prev   = t_now

    # ── Cleanup ───────────────────────────────────────────────────────────
    print("\n[INFO] Shutting down …")
    cap.release()
    head_det.release()
    gaze_det.release()
    posture_det.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Session ended.  Total violations: {alert_mgr.total_violations}")
    print( "[INFO] Log saved to violation_log.csv")


if __name__ == "__main__":
    main()
