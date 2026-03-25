#  SecuroServ

**Real-Time Security Surveillance System**

SecuroServ is a YOLOv8-based video surveillance application that uses your laptop (or any connected) camera to detect abnormal and potentially dangerous behaviors in real time — with a sleek, dark-themed desktop UI.

---

##  Features

| Capability | Description |
|---|---|
|  **Violence Detection** | Detects physical altercations and aggressive contact |
|  **Choking / Distress** | Identifies persons in respiratory or physical distress |
|  **Camera Tampering** | Detects attempts to block, cover, or move the camera |
|  **Restricted Area Intrusion** | Alerts when a person enters a defined restricted zone |
|  **Loitering Detection** | Flags individuals lingering suspiciously in an area |
|  **Unattended Objects** | Spots bags or objects left unattended |
|  **Real-time Dashboard** | Live feed, FPS counter, alert feed, and session stats |
|  **Incident Recording** | Auto-saves frame snapshots when events are detected |
|  **Alert Log Export** | Export timestamped logs for reporting |

---

##  Architecture

```
SecuroServ/
├── main.py                    # Entry point
├── requirements.txt
├── config/
│   └── settings.yaml          # All tunable parameters
├── src/
│   ├── app.py                 # App orchestrator
│   ├── camera.py              # Thread-safe camera capture
│   ├── detector.py            # YOLOv8 inference + tamper/loiter logic
│   ├── alerts.py              # Alert management & logging
│   └── ui.py                  # Tkinter desktop UI
├── scripts/
│   ├── prepare_dataset.py     # Dataset preparation for training
│   ├── train.py               # YOLOv8 fine-tuning
│   └── evaluate.py            # Model evaluation & metrics
├── models/                    # Place your trained weights here
│   └── securoserv.pt          # (add after training)
└── logs/
    ├── alerts.log
    └── incidents/             # Auto-saved incident snapshots
```

---

### 1. Clone and Install

```bash
git clone https://github.com/YOUR_USERNAME/SecuroServ.git
cd SecuroServ

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

pip install -r requirements.txt
```

### 2. Run the Application

```bash
python main.py
```

Click **▶ START SURVEILLANCE** in the sidebar to begin.

> **Note:** On first run without a custom model, SecuroServ automatically downloads the pretrained `yolov8n.pt` weights and falls back to COCO-class detection (persons, bags, etc.) until you train a custom model.

---

##  Training Your Own Model

### Prepare your dataset

Collect and label images for each behavior class using a tool like [Roboflow](https://roboflow.com) or [LabelImg](https://github.com/HumanSignal/labelImg). Labels must be in **YOLO format** (`.txt` files alongside each image).

```bash
python scripts/prepare_dataset.py \
    --source /path/to/raw_labeled_images \
    --output datasets/securoserv
```

Expected class names (configurable):
- `violence`
- `choking`
- `camera_tampering`
- `restricted_area`
- `loitering`
- `unattended_object`

###  Train

```bash
python scripts/train.py \
    --data datasets/securoserv/dataset.yaml \
    --model s \        # n / s / m / l / x (larger = more accurate, slower)
    --epochs 100 \
    --batch 16
```

Best weights are automatically copied to `models/securoserv.pt`.

###  Evaluate

```bash
python scripts/evaluate.py \
    --model models/securoserv.pt \
    --data datasets/securoserv/dataset.yaml
```

### Use in the App

Restart SecuroServ — it will automatically load your custom model from `models/securoserv.pt`.

---

##  Configuration

Edit `config/settings.yaml` to tune detection behavior:

```yaml
detection:
  confidence_threshold: 0.45    # Lower = more detections, more false positives
  model_path: "models/securoserv.pt"

camera:
  device_index: 0               # Change for external cameras

alerts:
  cooldown_seconds: 5           # Minimum time between repeat alerts
  auto_save_incidents: true

behaviors:
  violence:
    enabled: true
    severity: "HIGH"
  # ... per-behavior toggles
```

---

##  Technical Details

### Detection Pipeline

1. **Camera Capture** — Background thread grabs frames at target FPS, buffered to avoid blocking.
2. **Camera Tamper Detection** — Frame-differencing + brightness analysis runs on every frame, independent of YOLO.
3. **YOLOv8 Inference** — Each frame is run through the model at configurable confidence/IOU thresholds.
4. **Loitering Tracker** — Person centroids are tracked across frames; extended dwell time triggers a loitering alert.
5. **Restricted Zone Check** — Point-in-polygon test for each detected person against user-defined zones.
6. **Alert** — Deduplicates alerts via per-class cooldowns, logs to file, saves snapshots.

### Camera Tamper Detection

Tamper detection uses two signals combined:
- **Spike detection**: sudden large change in frame difference vs. rolling history
- **Blackout detection**: dramatic drop in mean frame brightness

No YOLO inference is needed.

### Technologies Used

| Library | Role |
|---|---|
| **YOLOv8 / Ultralytics** | Object detection & behavior recognition |
| **OpenCV** | Camera capture, frame processing, drawing |
| **PyTorch** | Deep learning backend |
| **Tkinter** | Cross-platform desktop UI |
| **PyYAML** | Configuration management |
| **NumPy / Pillow** | Image processing utilities |

---

##  Notes

- **GPU strongly recommended** for real-time inference at 30+ FPS
- On CPU (laptop): expect 5–15 FPS depending on hardware; use `yolov8n` (nano) for best speed and heavier models for best accuracy
- Tamper detection and loitering tracking add near-zero overhead

---


