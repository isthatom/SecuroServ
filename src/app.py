"""
SecuroServ Application
Wires together all components and launches the UI.
"""

import os
import sys
import logging
import yaml

from src.camera import CameraCapture
from src.detector import DetectionEngine
from src.alerts import AlertManager
from src.ui import SecuroServUI

logger = logging.getLogger("SecuroServ")


def setup_logging():
    os.makedirs("logs", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/securoserv.log", encoding="utf-8"),
        ],
    )


def load_config(path: str = "config/settings.yaml") -> dict:
    if not os.path.exists(path):
        logger.warning(f"Config not found at {path}, using defaults.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class SecuroServApp:
    def __init__(self):
        setup_logging()
        logger.info("SecuroServ starting up…")

        cfg = load_config()

        # Flatten relevant sub-configs for component constructors
        det_cfg  = {**cfg.get("detection", {}), **cfg.get("ui", {})}
        det_cfg["behaviors"] = cfg.get("behaviors", {})
        det_cfg["cooldown_seconds"] = cfg.get("alerts", {}).get("cooldown_seconds", 5)

        cam_cfg    = cfg.get("camera", {})
        alert_cfg  = {
            **cfg.get("alerts", {}),
            **cfg.get("recording", {}),
        }

        self.camera  = CameraCapture(cam_cfg)
        self.engine  = DetectionEngine(det_cfg)
        self.alerts  = AlertManager(alert_cfg)

    def run(self):
        app = SecuroServUI(
            alert_manager=self.alerts,
            detection_engine=self.engine,
            camera=self.camera,
            config={},
        )
        app.protocol("WM_DELETE_WINDOW", app.on_close)
        logger.info("UI ready.")
        app.mainloop()
        logger.info("SecuroServ shut down.")
