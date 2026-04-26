import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import config


@dataclass
class Alert:
    code:        str
    message:     str
    severity:    str          # "warning" | "danger"
    timestamp:   float = field(default_factory=time.time)
    frame_count: int   = 0

    @property
    def time_str(self) -> str:
        t = time.localtime(self.timestamp)
        return f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}"


class AlertManager:
    """
    Central hub for all cheating alerts.

    Responsibilities
    ----------------
    - De-bounce noisy detections (requires N consecutive positive frames).
    - Enforce per-type cooldown windows so the log stays readable.
    - Keep a running violation counter used on the dashboard.
    """

    def __init__(self):
        self._consecutive: Dict[str, int]   = defaultdict(int)
        self._last_fired:  Dict[str, float] = defaultdict(float)
        self._active:      Dict[str, bool]  = defaultdict(bool)

        self.log:     deque[Alert] = deque(maxlen=200)
        self.counts:  Dict[str, int] = defaultdict(int)
        self.current_alerts: List[Alert] = []   # fired this frame

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, code: str, triggered: bool, message: str,
               severity: str = "warning") -> Optional[Alert]:
        """
        Call every frame for each violation type.

        Returns an Alert object the first time a violation fires (after the
        de-bounce window), otherwise None.
        """
        if triggered:
            self._consecutive[code] += 1
        else:
            self._consecutive[code] = 0
            self._active[code] = False
            return None

        # Not yet persistent enough
        if self._consecutive[code] < config.ALERT_MIN_FRAMES:
            return None

        now      = time.time()
        cooldown = config.ALERT_COOLDOWN.get(code, 4)

        if now - self._last_fired[code] < cooldown:
            return None     # still in cooldown

        # Fire the alert
        self._last_fired[code] = now
        self._active[code]     = True
        self.counts[code]     += 1

        alert = Alert(
            code=code,
            message=message,
            severity=severity,
            frame_count=self._consecutive[code],
        )
        self.log.append(alert)
        return alert

    def is_active(self, code: str) -> bool:
        """True while a violation is currently ongoing (past de-bounce)."""
        return self._consecutive[code] >= config.ALERT_MIN_FRAMES

    def get_recent_log(self, n: int = config.MAX_LOG_ENTRIES) -> List[Alert]:
        """Return the N most recent logged alerts."""
        items = list(self.log)
        return items[-n:]

    @property
    def total_violations(self) -> int:
        return sum(self.counts.values())

    def reset(self):
        self._consecutive.clear()
        self._last_fired.clear()
        self._active.clear()
        self.log.clear()
        self.counts.clear()
