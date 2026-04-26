import cv2
import numpy as np
import config
from typing import Tuple, Optional


# ── Primitives ────────────────────────────────────────────────────────────────

def draw_rounded_rect(img: np.ndarray, pt1: Tuple, pt2: Tuple,
                      color: Tuple, radius: int = 8,
                      thickness: int = -1, alpha: float = 1.0):
    """Filled or outlined rectangle with rounded corners."""
    if alpha < 1.0:
        overlay = img.copy()
        _draw_rounded_rect_solid(overlay, pt1, pt2, color, radius, thickness)
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    else:
        _draw_rounded_rect_solid(img, pt1, pt2, color, radius, thickness)


def _draw_rounded_rect_solid(img, pt1, pt2, color, radius, thickness):
    x1, y1 = pt1
    x2, y2 = pt2
    r = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)
    if thickness == -1:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1)
        cv2.circle(img, (x1 + r, y1 + r), r, color, -1)
        cv2.circle(img, (x2 - r, y1 + r), r, color, -1)
        cv2.circle(img, (x1 + r, y2 - r), r, color, -1)
        cv2.circle(img, (x2 - r, y2 - r), r, color, -1)
    else:
        cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, thickness)
        cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, thickness)
        for cx, cy in [(x1+r,y1+r),(x2-r,y1+r),(x1+r,y2-r),(x2-r,y2-r)]:
            cv2.circle(img, (cx, cy), r, color, thickness)


def put_text(img: np.ndarray, text: str, pos: Tuple[int,int],
             scale: float = config.FONT_SCALE_MD,
             color: Tuple = config.COLOR_WHITE,
             thickness: int = config.FONT_THICKNESS,
             shadow: bool = False):
    if shadow:
        cv2.putText(img, text, (pos[0]+1, pos[1]+1),
                    config.FONT, scale, (0,0,0), thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos, config.FONT, scale, color,
                thickness, cv2.LINE_AA)


# ── Status Badge ──────────────────────────────────────────────────────────────

def draw_status_badge(img: np.ndarray, label: str, value: str,
                      x: int, y: int, w: int = 300,
                      ok: Optional[bool] = None):
    """
    Compact badge used in the sidebar.
      ok=True  → green indicator
      ok=False → red indicator
      ok=None  → blue/neutral indicator
    """
    h = 34
    draw_rounded_rect(img, (x, y), (x + w, y + h),
                      config.COLOR_BORDER, radius=6, thickness=1)

    dot_color = (config.COLOR_OK if ok is True
                 else config.COLOR_DANGER if ok is False
                 else config.COLOR_INFO)
    cv2.circle(img, (x + 12, y + h // 2), 5, dot_color, -1)

    put_text(img, label, (x + 24, y + h // 2 + 5),
             scale=config.FONT_SCALE_SM, color=(200, 200, 200))
    put_text(img, value, (x + w - 8 - len(value)*8, y + h // 2 + 5),
             scale=config.FONT_SCALE_SM, color=config.COLOR_WHITE)


# ── Violation Box ─────────────────────────────────────────────────────────────

def draw_violation_box(img: np.ndarray, label: str,
                       pt1: Tuple, pt2: Tuple,
                       severity: str = "warning"):
    color = config.COLOR_DANGER if severity == "danger" else config.COLOR_WARNING
    cv2.rectangle(img, pt1, pt2, color, 2)

    tw, th = cv2.getTextSize(label, config.FONT,
                             config.FONT_SCALE_SM, 1)[0]
    lx, ly = pt1[0], pt1[1] - 4
    draw_rounded_rect(img, (lx, ly - th - 4), (lx + tw + 8, ly + 2),
                      color, radius=4)
    put_text(img, label, (lx + 4, ly - 2),
             scale=config.FONT_SCALE_SM, color=config.COLOR_BLACK,
             thickness=1)


# ── Sidebar Panel ─────────────────────────────────────────────────────────────

def build_sidebar(height: int, width: int = config.SIDEBAR_WIDTH) -> np.ndarray:
    panel = np.full((height, width, 3), config.COLOR_PANEL, dtype=np.uint8)
    return panel


def draw_sidebar_header(panel: np.ndarray, title: str, subtitle: str,
                        x: int = 12, y_start: int = 16):
    put_text(panel, title, (x, y_start + 20),
             scale=config.FONT_SCALE_LG, color=config.COLOR_WHITE,
             thickness=2, shadow=True)
    put_text(panel, subtitle, (x, y_start + 42),
             scale=config.FONT_SCALE_SM, color=(160, 160, 160))
    cv2.line(panel, (x, y_start + 52),
             (panel.shape[1] - x, y_start + 52),
             config.COLOR_BORDER, 1)
    return y_start + 62


def draw_section_title(panel: np.ndarray, title: str,
                       x: int, y: int) -> int:
    put_text(panel, title.upper(), (x, y),
             scale=0.44, color=(130, 180, 255), thickness=1)
    cv2.line(panel, (x, y + 4), (panel.shape[1] - x, y + 4),
             (50, 50, 70), 1)
    return y + 14


def draw_alert_log_entry(panel: np.ndarray, time_str: str,
                         message: str, severity: str,
                         x: int, y: int, w: int):
    color = config.COLOR_DANGER if severity == "danger" else config.COLOR_WARNING
    put_text(panel, time_str, (x, y),
             scale=0.40, color=(140, 140, 140))
    put_text(panel, message, (x + 56, y),
             scale=0.42, color=color)


# ── Meter / Progress Bar ──────────────────────────────────────────────────────

def draw_meter(img: np.ndarray, label: str, value: float,
               max_val: float, x: int, y: int, w: int = 290):
    """Horizontal bar-graph meter."""
    ratio     = min(abs(value) / max_val, 1.0)
    bar_w     = int((w - 80) * ratio)
    bar_color = (config.COLOR_OK if ratio < 0.5
                 else config.COLOR_WARNING if ratio < 0.85
                 else config.COLOR_DANGER)

    put_text(img, label, (x, y), scale=0.42, color=(180, 180, 180))
    # track
    cv2.rectangle(img, (x + 82, y - 10), (x + w - 4, y + 2),
                  (50, 50, 60), -1)
    # fill
    if bar_w > 0:
        cv2.rectangle(img, (x + 82, y - 10),
                      (x + 82 + bar_w, y + 2), bar_color, -1)
    put_text(img, f"{value:+.1f}", (x + w + 2, y),
             scale=0.40, color=(200, 200, 200))
