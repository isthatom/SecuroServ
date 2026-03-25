"""
SecuroServ Alert Manager
Handles alert logging, sounds, and popup notifications.
"""

import os
import time
import logging
import threading
from datetime import datetime
from collections import deque

import cv2

logger = logging.getLogger("SecuroServ.Alerts")

SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
}


class AlertRecord:
    def __init__(self, detection, snapshot=None):
        self.detection = detection
        self.snapshot = snapshot
        self.timestamp = datetime.now()
        self.id = int(time.time() * 1000)

    @property
    def severity(self):
        return self.detection.severity

    @property
    def label(self):
        return self.detection.label

    @property
    def confidence(self):
        return self.detection.confidence

    def to_log_line(self):
        emoji = SEVERITY_EMOJI.get(self.severity, "⚪")
        ts = self.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        label = self.label.replace("_", " ").upper()
        return f"[{ts}] {emoji} [{self.severity}] {label} — conf: {self.confidence:.1%}"


class AlertManager:
    """
    Manages the alert lifecycle:
    - Logs to file
    - Saves incident snapshots
    - Plays sound cues
    - Maintains in-memory alert history
    - Calls registered UI callbacks
    """

    MAX_HISTORY = 200

    def __init__(self, config: dict):
        self.config = config
        self._history: deque = deque(maxlen=self.MAX_HISTORY)
        self._callbacks: list = []
        self._lock = threading.Lock()

        # Prepare directories
        log_dir = os.path.dirname(config.get("log_path", "logs/alerts.log"))
        incident_dir = config.get("save_path", "logs/incidents/")
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(incident_dir, exist_ok=True)

        self._log_path = config.get("log_path", "logs/alerts.log")
        self._incident_dir = incident_dir
        self._enable_log = config.get("enable_log", True)
        self._auto_save = config.get("auto_save_incidents", True)

    # ------------------------------------------------------------------
    def register_callback(self, fn):
        """Register a function to be called with each new AlertRecord."""
        self._callbacks.append(fn)

    # ------------------------------------------------------------------
    def handle(self, detection, frame=None):
        """Process a Detection — log, save, notify."""
        snapshot = None
        if self._auto_save and frame is not None:
            snapshot = self._save_snapshot(detection, frame)

        record = AlertRecord(detection=detection, snapshot=snapshot)

        with self._lock:
            self._history.appendleft(record)

        if self._enable_log:
            self._write_log(record)

        # Fire UI callbacks in background thread to avoid blocking
        threading.Thread(
            target=self._fire_callbacks, args=(record,), daemon=True
        ).start()

        logger.info(record.to_log_line())
        return record

    # ------------------------------------------------------------------
    def _save_snapshot(self, detection, frame):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        label = detection.label.replace(" ", "_")
        filename = f"{ts}_{label}_{detection.confidence:.0%}.jpg"
        path = os.path.join(self._incident_dir, filename)
        try:
            cv2.imwrite(path, frame)
            return path
        except Exception as exc:
            logger.warning(f"Could not save snapshot: {exc}")
            return None

    def _write_log(self, record: AlertRecord):
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(record.to_log_line() + "\n")
        except Exception as exc:
            logger.warning(f"Could not write log: {exc}")

    def _fire_callbacks(self, record: AlertRecord):
        for fn in self._callbacks:
            try:
                fn(record)
            except Exception as exc:
                logger.warning(f"Alert callback error: {exc}")

    # ------------------------------------------------------------------
    def get_history(self, limit: int = 50) -> list:
        with self._lock:
            return list(self._history)[:limit]

    def clear_history(self):
        with self._lock:
            self._history.clear()

    def get_stats(self) -> dict:
        with self._lock:
            history = list(self._history)

        stats = {
            "total": len(history),
            "by_severity": {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0},
            "by_label": {},
        }
        for r in history:
            sev = r.severity or "LOW"
            stats["by_severity"][sev] = stats["by_severity"].get(sev, 0) + 1
            stats["by_label"][r.label] = stats["by_label"].get(r.label, 0) + 1

        return stats
