import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple
import config


@dataclass
class PhoneDetection:
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2
    confidence: float
    label:      str


@dataclass
class PhoneDetectorResult:
    phones:       List[PhoneDetection] = field(default_factory=list)
    phone_found:  bool = False


class PhoneDetector:
    # COCO IDs we want to flag
    _TARGET_CLASSES = {
        67: "Mobile Phone",
        73: "Book / Notes",     # book
        76: "Clipboard",        # scissors — removed, replaced with clipboard proxy
        84: "Notebook",         # book variant
    }

    def __init__(self):
        from ultralytics import YOLO
        self._model = YOLO(config.YOLO_MODEL_NAME)
        self._model.overrides["verbose"] = False

    def process(self, frame_bgr: np.ndarray) -> PhoneDetectorResult:
        result = PhoneDetectorResult()

        # Run inference (returns a list, one item per image)
        predictions = self._model.predict(
            frame_bgr,
            conf=config.PHONE_CONFIDENCE_THRESHOLD,
            classes=list(self._TARGET_CLASSES.keys()),
            verbose=False,
        )

        if not predictions:
            return result

        for box in predictions[0].boxes:
            cls_id = int(box.cls[0].item())
            if cls_id not in self._TARGET_CLASSES:
                continue

            conf   = float(box.conf[0].item())
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detection = PhoneDetection(
                bbox=(x1, y1, x2, y2),
                confidence=conf,
                label=self._TARGET_CLASSES[cls_id],
            )
            result.phones.append(detection)

        result.phone_found = len(result.phones) > 0
        return result

    def draw_overlay(self, frame_bgr: np.ndarray,
                     result: PhoneDetectorResult):
        for det in result.phones:
            x1, y1, x2, y2 = det.bbox
            label = f"{det.label}  {det.confidence:.0%}"

            # Flashing red box
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2),
                          config.COLOR_DANGER, 3)

            # Label background
            tw, th = cv2.getTextSize(
                label, config.FONT, config.FONT_SCALE_SM, 1)[0]
            cv2.rectangle(frame_bgr,
                          (x1, y1 - th - 10), (x1 + tw + 8, y1),
                          config.COLOR_DANGER, -1)
            cv2.putText(frame_bgr, label, (x1 + 4, y1 - 4),
                        config.FONT, config.FONT_SCALE_SM,
                        config.COLOR_WHITE, 1, cv2.LINE_AA)
