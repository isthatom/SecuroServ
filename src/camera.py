"""
SecuroServ Camera Module
Handles camera capture in a background thread with frame buffering.
"""

import time
import logging
import threading
from collections import deque
from typing import Optional, Callable

import cv2
import numpy as np

logger = logging.getLogger("SecuroServ.Camera")


class CameraCapture:
    """
    Thread-safe camera capture with frame buffering and health monitoring.
    """

    def __init__(self, config: dict):
        self.device_index = config.get("device_index", 0)
        self.target_width  = config.get("frame_width", 1280)
        self.target_height = config.get("frame_height", 720)
        self.target_fps    = config.get("fps_target", 30)

        self._cap: Optional[cv2.VideoCapture] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._frame_buffer: deque = deque(maxlen=2)
        self._lock = threading.Lock()
        self._frame_count = 0
        self._drop_count = 0
        self._last_frame_time = 0.0
        self._on_error: Optional[Callable] = None

    # ------------------------------------------------------------------
    def start(self) -> bool:
        """Open the camera and start capture thread. Returns True on success."""
        self._cap = cv2.VideoCapture(self.device_index)
        if not self._cap.isOpened():
            logger.error(f"Could not open camera at index {self.device_index}")
            return False

        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.target_width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)
        self._cap.set(cv2.CAP_PROP_FPS,          self.target_fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        logger.info(f"Camera opened: {actual_w}x{actual_h} @ {actual_fps:.0f}fps")

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        return True

    def stop(self):
        """Stop capture and release resources."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        if self._cap:
            self._cap.release()
            self._cap = None
        logger.info("Camera released.")

    # ------------------------------------------------------------------
    def read(self) -> Optional[np.ndarray]:
        """Return the latest frame (or None if none available)."""
        with self._lock:
            if self._frame_buffer:
                return self._frame_buffer[-1].copy()
        return None

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def resolution(self) -> tuple:
        if self._cap:
            w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            return (w, h)
        return (self.target_width, self.target_height)

    def set_error_callback(self, fn: Callable):
        self._on_error = fn

    # ------------------------------------------------------------------
    def _capture_loop(self):
        """Background thread: continuously grab frames from camera."""
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                logger.error("Camera disconnected.")
                if self._on_error:
                    self._on_error("Camera disconnected")
                break

            ret, frame = self._cap.read()
            if not ret:
                self._drop_count += 1
                if self._drop_count > 10:
                    logger.warning("Excessive frame drops — camera may be disconnected.")
                time.sleep(0.01)
                continue

            self._drop_count = 0
            self._frame_count += 1
            self._last_frame_time = time.time()

            with self._lock:
                self._frame_buffer.append(frame)

        self._running = False

    # ------------------------------------------------------------------
    @staticmethod
    def list_cameras(max_index: int = 5) -> list:
        """Return list of available camera indices."""
        available = []
        for i in range(max_index):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available.append(i)
                cap.release()
        return available
