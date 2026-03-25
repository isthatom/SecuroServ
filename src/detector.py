"""
SecuroServ Detection Engine
Handles YOLOv8 inference for abnormal behavior detection.
"""

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import defaultdict, deque

import cv2
import numpy as np

logger = logging.getLogger("SecuroServ.Detector")


@dataclass
class Detection:
    """Represents a single detection result."""
    label: str
    confidence: float
    bbox: tuple          # (x1, y1, x2, y2) in pixel coords
    severity: str = "MEDIUM"
    timestamp: float = field(default_factory=time.time)
    frame_id: int = 0

    @property
    def center(self):
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self):
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


@dataclass
class FrameResult:
    """Result object for a processed frame."""
    frame: np.ndarray
    detections: list
    fps: float
    frame_id: int
    timestamp: float = field(default_factory=time.time)
    tamper_detected: bool = False
    tamper_score: float = 0.0


# ---------------------------------------------------------------------------
# Behavior class names mapped to severity
# ---------------------------------------------------------------------------
BEHAVIOR_SEVERITY = {
    "violence":           "HIGH",
    "choking":            "CRITICAL",
    "camera_tampering":   "HIGH",
    "restricted_area":    "MEDIUM",
    "loitering":          "LOW",
    "unattended_object":  "MEDIUM",
    # COCO classes repurposed for demo when no custom model is available
    "person":             None,   # tracked but not alerted by default
    "knife":              "HIGH",
    "scissors":           "MEDIUM",
    "cell phone":         "LOW",
    "backpack":           "MEDIUM",
    "handbag":            "MEDIUM",
    "suitcase":           "MEDIUM",
}

ALERT_CLASSES = {k for k, v in BEHAVIOR_SEVERITY.items() if v is not None}


class CameraTamperDetector:
    """
    Lightweight tamper detection using frame differencing and
    scene-change analysis — works independently of YOLO.
    """

    def __init__(self, sensitivity: float = 0.35, history: int = 30):
        self.sensitivity = sensitivity
        self.reference_frame: Optional[np.ndarray] = None
        self.frame_history: deque = deque(maxlen=history)
        self.tamper_score = 0.0

    def update(self, frame: np.ndarray) -> tuple[bool, float]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        # Initialise reference
        if self.reference_frame is None:
            self.reference_frame = gray.copy()
            return False, 0.0

        # Diff vs reference
        diff = cv2.absdiff(self.reference_frame, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        changed_ratio = np.sum(thresh > 0) / thresh.size

        # Track history to detect sudden scene change (covering)
        self.frame_history.append(changed_ratio)

        if len(self.frame_history) < 10:
            return False, 0.0

        recent_avg = np.mean(list(self.frame_history)[-5:])
        historical_avg = np.mean(list(self.frame_history)[:-5])

        # Spike detection: recent suddenly much higher than history
        if historical_avg > 0:
            spike = recent_avg / (historical_avg + 1e-6)
        else:
            spike = 0.0

        # Check for near-total black-out (camera covered)
        mean_brightness = np.mean(gray)
        blackout = 1.0 - (mean_brightness / 128.0)
        blackout = max(0.0, min(1.0, blackout))

        self.tamper_score = min(1.0, (spike / 10.0) * 0.5 + blackout * 0.5)

        if self.tamper_score > self.sensitivity:
            # Slowly update reference so scene changes don't permanently trigger
            self.reference_frame = cv2.addWeighted(
                self.reference_frame, 0.95, gray, 0.05, 0
            )
            return True, self.tamper_score

        # Slowly update reference to adapt to normal lighting changes
        self.reference_frame = cv2.addWeighted(
            self.reference_frame, 0.99, gray, 0.01, 0
        )
        return False, self.tamper_score


class LoiterTracker:
    """Tracks person centroids across frames to detect loitering."""

    def __init__(self, threshold_seconds: float = 8.0, distance_px: int = 80):
        self.threshold = threshold_seconds
        self.distance_px = distance_px
        self._tracks: dict = {}   # track_id → {"center", "first_seen", "last_seen"}
        self._next_id = 0

    def update(self, person_detections: list) -> list:
        """
        Returns list of Detection objects upgraded to 'loitering' severity.
        """
        now = time.time()
        matched_ids = set()
        loitering = []

        for det in person_detections:
            cx, cy = det.center
            best_id, best_dist = None, float("inf")

            for tid, track in self._tracks.items():
                tx, ty = track["center"]
                dist = ((cx - tx) ** 2 + (cy - ty) ** 2) ** 0.5
                if dist < self.distance_px and dist < best_dist:
                    best_id, best_dist = tid, dist

            if best_id is not None:
                self._tracks[best_id]["center"] = (cx, cy)
                self._tracks[best_id]["last_seen"] = now
                matched_ids.add(best_id)
                duration = now - self._tracks[best_id]["first_seen"]
                if duration >= self.threshold:
                    loitering.append(Detection(
                        label="loitering",
                        confidence=min(0.99, 0.5 + duration / 60.0),
                        bbox=det.bbox,
                        severity="LOW",
                        timestamp=now,
                    ))
            else:
                new_id = self._next_id
                self._next_id += 1
                self._tracks[new_id] = {
                    "center": (cx, cy),
                    "first_seen": now,
                    "last_seen": now,
                }
                matched_ids.add(new_id)

        # Remove stale tracks
        stale = [tid for tid, t in self._tracks.items()
                 if now - t["last_seen"] > 3.0 and tid not in matched_ids]
        for tid in stale:
            del self._tracks[tid]

        return loitering


class DetectionEngine:
    """
    Core detection engine.

    Uses YOLOv8 for object/behavior detection plus supplementary
    rules for camera tampering and loitering.
    """

    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.tamper_detector = CameraTamperDetector()
        self.loiter_tracker = LoiterTracker()
        self._frame_id = 0
        self._fps_history = deque(maxlen=30)
        self._last_frame_time = time.time()
        self._alert_cooldowns: dict = defaultdict(float)
        self._load_lock = threading.Lock()
        self._model_loaded = False

        # Restricted area polygon (can be configured via UI)
        self.restricted_zones: list = []   # list of np.array polygons

        self._load_model()

    # ------------------------------------------------------------------
    def _load_model(self):
        """Load YOLOv8 model (custom or pretrained fallback)."""
        try:
            from ultralytics import YOLO
            custom_path = self.config.get("model_path", "models/securoserv.pt")
            fallback = self.config.get("fallback_model", "yolov8n.pt")

            if os.path.exists(custom_path):
                logger.info(f"Loading custom model: {custom_path}")
                self.model = YOLO(custom_path)
            else:
                logger.warning(
                    f"Custom model not found at '{custom_path}'. "
                    f"Loading pretrained fallback: {fallback}"
                )
                self.model = YOLO(fallback)

            self._model_loaded = True
            logger.info("Model loaded successfully.")
        except ImportError:
            logger.error(
                "ultralytics not installed. Run: pip install ultralytics"
            )
        except Exception as exc:
            logger.error(f"Failed to load model: {exc}")

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> FrameResult:
        """Run full pipeline on a single frame and return FrameResult."""
        self._frame_id += 1
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        fps = 1.0 / dt if dt > 0 else 0.0
        self._fps_history.append(fps)
        smooth_fps = np.mean(self._fps_history)

        detections = []
        tamper_detected = False
        tamper_score = 0.0

        # --- Camera tamper check (always runs) --------------------------
        if self.config.get("behaviors", {}).get("camera_tampering", {}).get("enabled", True):
            tamper_detected, tamper_score = self.tamper_detector.update(frame)
            if tamper_detected and self._cooldown_ok("camera_tampering"):
                detections.append(Detection(
                    label="camera_tampering",
                    confidence=tamper_score,
                    bbox=(0, 0, frame.shape[1], frame.shape[0]),
                    severity="HIGH",
                    timestamp=now,
                    frame_id=self._frame_id,
                ))

        # --- YOLO inference ---------------------------------------------
        if self._model_loaded and self.model is not None:
            conf = self.config.get("confidence_threshold", 0.45)
            iou  = self.config.get("iou_threshold", 0.45)
            size = self.config.get("inference_size", 640)

            results = self.model.predict(
                frame,
                conf=conf,
                iou=iou,
                imgsz=size,
                verbose=False,
            )

            person_dets = []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.model.names.get(cls_id, str(cls_id))
                    confidence = float(box.conf[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    severity = BEHAVIOR_SEVERITY.get(label)

                    det = Detection(
                        label=label,
                        confidence=confidence,
                        bbox=(x1, y1, x2, y2),
                        severity=severity or "LOW",
                        timestamp=now,
                        frame_id=self._frame_id,
                    )

                    if label == "person":
                        person_dets.append(det)

                    if label in ALERT_CLASSES and self._cooldown_ok(label):
                        detections.append(det)
                    elif label == "person":
                        detections.append(det)  # always add persons for tracking

            # --- Loitering detection ------------------------------------
            if self.config.get("behaviors", {}).get("loitering", {}).get("enabled", True):
                loitering = self.loiter_tracker.update(person_dets)
                for ld in loitering:
                    if self._cooldown_ok("loitering"):
                        detections.append(ld)

            # --- Restricted zone intrusion ------------------------------
            if self.restricted_zones:
                for det in person_dets:
                    cx, cy = det.center
                    for zone in self.restricted_zones:
                        if cv2.pointPolygonTest(zone, (cx, cy), False) >= 0:
                            if self._cooldown_ok("restricted_area"):
                                detections.append(Detection(
                                    label="restricted_area",
                                    confidence=det.confidence,
                                    bbox=det.bbox,
                                    severity="MEDIUM",
                                    timestamp=now,
                                    frame_id=self._frame_id,
                                ))

        # --- Draw overlays ----------------------------------------------
        annotated = self._draw_overlays(frame.copy(), detections, smooth_fps)

        return FrameResult(
            frame=annotated,
            detections=detections,
            fps=smooth_fps,
            frame_id=self._frame_id,
            timestamp=now,
            tamper_detected=tamper_detected,
            tamper_score=tamper_score,
        )

    # ------------------------------------------------------------------
    def _cooldown_ok(self, label: str) -> bool:
        """Return True if enough time has passed since last alert for label."""
        cooldown = self.config.get("cooldown_seconds", 5)
        now = time.time()
        if now - self._alert_cooldowns[label] >= cooldown:
            self._alert_cooldowns[label] = now
            return True
        return False

    # ------------------------------------------------------------------
    def _draw_overlays(
        self, frame: np.ndarray, detections: list, fps: float
    ) -> np.ndarray:
        """Draw bounding boxes, labels, and HUD onto frame."""
        SEVERITY_COLORS = {
            "CRITICAL": (0, 0, 255),
            "HIGH":     (0, 60, 255),
            "MEDIUM":   (0, 165, 255),
            "LOW":      (0, 255, 180),
            None:       (200, 200, 200),
        }

        show_boxes = self.config.get("show_bounding_boxes", True)
        show_conf  = self.config.get("show_confidence", True)
        show_fps   = self.config.get("show_fps", True)

        for det in detections:
            if not show_boxes:
                break
            color = SEVERITY_COLORS.get(det.severity, (200, 200, 200))
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label_text = det.label.replace("_", " ").upper()
            if show_conf:
                label_text += f"  {det.confidence:.0%}"

            (tw, th), _ = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
            )
            cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(
                frame, label_text,
                (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                cv2.LINE_AA,
            )

        # Draw restricted zones
        for zone in self.restricted_zones:
            cv2.polylines(frame, [zone], True, (0, 0, 220), 2)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [zone], (0, 0, 180))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

        # HUD: FPS counter
        if show_fps:
            h, w = frame.shape[:2]
            cv2.putText(
                frame, f"FPS: {fps:.1f}",
                (w - 120, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 100), 2, cv2.LINE_AA,
            )

        return frame

    # ------------------------------------------------------------------
    def set_restricted_zone(self, polygon: np.ndarray):
        """Add a restricted zone polygon."""
        self.restricted_zones.append(polygon)

    def clear_restricted_zones(self):
        self.restricted_zones.clear()

    def reload_model(self, model_path: str):
        """Hot-reload the model (e.g. after training)."""
        with self._load_lock:
            self._model_loaded = False
            self.config["model_path"] = model_path
            self._load_model()
