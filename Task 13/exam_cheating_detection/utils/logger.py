import csv
import os
import time


class ViolationLogger:
    def __init__(self, path: str = "violation_log.csv"):
        self.path = path
        self._init_file()

    def _init_file(self):
        exists = os.path.isfile(self.path)
        with open(self.path, "a", newline="") as f:
            if not exists:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "time", "code", "message", "severity"])

    def log(self, code: str, message: str, severity: str):
        now = time.time()
        t   = time.strftime("%H:%M:%S", time.localtime(now))
        with open(self.path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([now, t, code, message, severity])
