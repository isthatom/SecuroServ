"""
SecuroServ UI
Main application window built with CustomTkinter.
"""

import os
import time
import logging
import threading
import tkinter as tk
from tkinter import messagebox, filedialog
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

try:
    import customtkinter as ctk
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")
    HAS_CTK = True
except ImportError:
    HAS_CTK = False
    import tkinter.ttk as ttk

logger = logging.getLogger("SecuroServ.UI")

SEVERITY_COLORS_HEX = {
    "CRITICAL": "#FF2D2D",
    "HIGH":     "#FF6B00",
    "MEDIUM":   "#FFD600",
    "LOW":      "#00E676",
}

PALETTE = {
    "bg_dark":    "#0A0D14",
    "bg_card":    "#111827",
    "bg_sidebar": "#0D1117",
    "accent":     "#00D4FF",
    "accent2":    "#7C3AED",
    "text":       "#E2E8F0",
    "text_dim":   "#64748B",
    "border":     "#1E293B",
    "green":      "#10B981",
    "red":        "#EF4444",
    "orange":     "#F59E0B",
}


class FeedCanvas(tk.Canvas):
    """Canvas widget that displays the live camera feed."""

    def __init__(self, parent, width=854, height=480, **kwargs):
        super().__init__(
            parent,
            width=width,
            height=height,
            bg=PALETTE["bg_dark"],
            highlightthickness=2,
            highlightbackground=PALETTE["border"],
            **kwargs,
        )
        self._img_ref = None
        self._canvas_w = width
        self._canvas_h = height
        self._show_placeholder()

    def _show_placeholder(self):
        self.delete("all")
        cx, cy = self._canvas_w // 2, self._canvas_h // 2
        self.create_rectangle(0, 0, self._canvas_w, self._canvas_h, fill=PALETTE["bg_dark"])
        self.create_text(
            cx, cy - 20,
            text="📷",
            fill=PALETTE["text_dim"],
            font=("Courier", 48),
        )
        self.create_text(
            cx, cy + 40,
            text="Camera feed will appear here",
            fill=PALETTE["text_dim"],
            font=("Courier", 13),
        )

    def update_frame(self, frame_bgr: np.ndarray):
        """Convert BGR frame and display on canvas."""
        h, w = frame_bgr.shape[:2]
        # Scale to fit canvas
        scale = min(self._canvas_w / w, self._canvas_h / h)
        nw, nh = int(w * scale), int(h * scale)
        resized = cv2.resize(frame_bgr, (nw, nh))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(rgb)
        photo = ImageTk.PhotoImage(img)
        self._img_ref = photo  # prevent GC
        self.delete("all")
        # Center image
        x_off = (self._canvas_w - nw) // 2
        y_off = (self._canvas_h - nh) // 2
        self.create_image(x_off, y_off, anchor="nw", image=photo)

    def show_tamper_warning(self):
        self.create_rectangle(0, 0, self._canvas_w, self._canvas_h,
                              outline=PALETTE["red"], width=6)
        self.create_text(
            self._canvas_w // 2, 30,
            text="⚠ CAMERA TAMPERING DETECTED",
            fill=PALETTE["red"],
            font=("Courier", 14, "bold"),
        )


class AlertRow(tk.Frame):
    """Single row in the alert feed panel."""

    def __init__(self, parent, record, **kwargs):
        super().__init__(parent, bg=PALETTE["bg_card"], **kwargs)

        sev = record.severity or "LOW"
        color = SEVERITY_COLORS_HEX.get(sev, "#999")

        # Severity badge
        badge = tk.Label(
            self,
            text=f" {sev} ",
            bg=color,
            fg="#000" if sev == "LOW" else "#fff",
            font=("Courier", 9, "bold"),
            padx=4, pady=2,
        )
        badge.pack(side="left", padx=(8, 6), pady=6)

        # Label
        label_txt = record.label.replace("_", " ").upper()
        lbl = tk.Label(
            self,
            text=label_txt,
            bg=PALETTE["bg_card"],
            fg=PALETTE["text"],
            font=("Courier", 11, "bold"),
            anchor="w",
            width=22,
        )
        lbl.pack(side="left")

        # Confidence
        conf_lbl = tk.Label(
            self,
            text=f"{record.confidence:.0%}",
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_dim"],
            font=("Courier", 10),
        )
        conf_lbl.pack(side="left", padx=8)

        # Timestamp
        ts = record.timestamp.strftime("%H:%M:%S")
        ts_lbl = tk.Label(
            self,
            text=ts,
            bg=PALETTE["bg_card"],
            fg=PALETTE["text_dim"],
            font=("Courier", 10),
        )
        ts_lbl.pack(side="right", padx=10)

        # Separator line
        sep = tk.Frame(self, bg=PALETTE["border"], height=1)
        sep.pack(side="bottom", fill="x")


class SecuroServUI(tk.Tk):
    """Main application window."""

    def __init__(self, alert_manager, detection_engine, camera, config):
        super().__init__()

        self.alert_manager = alert_manager
        self.engine = detection_engine
        self.camera = camera
        self.config = config

        self._running = False
        self._pipeline_thread: Optional[threading.Thread] = None
        self._status = "IDLE"
        self._alert_rows: list = []
        self._stats = {"total": 0, "critical": 0, "high": 0, "medium": 0, "low": 0}

        self._setup_window()
        self._build_ui()
        self._start_ui_refresh()

        # Register alert callback
        alert_manager.register_callback(self._on_new_alert)

    # ------------------------------------------------------------------
    def _setup_window(self):
        self.title("SecuroServ — AI Surveillance System")
        self.configure(bg=PALETTE["bg_dark"])
        self.geometry("1320x820")
        self.minsize(1100, 700)
        self.resizable(True, True)

        # App icon (create simple colored icon)
        try:
            icon_img = tk.PhotoImage(width=32, height=32)
            self.iconphoto(True, icon_img)
        except Exception:
            pass

    # ------------------------------------------------------------------
    def _build_ui(self):
        """Build all UI sections."""
        # ---- Title Bar ----
        title_bar = tk.Frame(self, bg=PALETTE["bg_sidebar"], height=56)
        title_bar.pack(fill="x", side="top")
        title_bar.pack_propagate(False)

        tk.Label(
            title_bar,
            text="🛡  SECURO",
            bg=PALETTE["bg_sidebar"],
            fg=PALETTE["accent"],
            font=("Courier", 18, "bold"),
        ).pack(side="left", padx=18, pady=12)

        tk.Label(
            title_bar,
            text="SERV",
            bg=PALETTE["bg_sidebar"],
            fg=PALETTE["text"],
            font=("Courier", 18, "bold"),
        ).pack(side="left", pady=12)

        tk.Label(
            title_bar,
            text="AI-POWERED SECURITY SURVEILLANCE",
            bg=PALETTE["bg_sidebar"],
            fg=PALETTE["text_dim"],
            font=("Courier", 10),
        ).pack(side="left", padx=24, pady=12)

        # Status indicator
        self._status_dot = tk.Label(
            title_bar,
            text="● OFFLINE",
            bg=PALETTE["bg_sidebar"],
            fg=PALETTE["text_dim"],
            font=("Courier", 10, "bold"),
        )
        self._status_dot.pack(side="right", padx=20)

        # ---- Main Content ----
        content = tk.Frame(self, bg=PALETTE["bg_dark"])
        content.pack(fill="both", expand=True)

        # Left sidebar
        sidebar = tk.Frame(content, bg=PALETTE["bg_sidebar"], width=240)
        sidebar.pack(side="left", fill="y")
        sidebar.pack_propagate(False)
        self._build_sidebar(sidebar)

        # Center feed
        center = tk.Frame(content, bg=PALETTE["bg_dark"])
        center.pack(side="left", fill="both", expand=True, padx=12, pady=12)
        self._build_feed_panel(center)

        # Right panel
        right = tk.Frame(content, bg=PALETTE["bg_sidebar"], width=340)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)
        self._build_alert_panel(right)

    # ------------------------------------------------------------------
    def _build_sidebar(self, parent):
        # Camera Controls
        section_label(parent, "CAMERA CONTROLS")

        self._btn_start = sidebar_btn(parent, "▶  START SURVEILLANCE", self._start_surveillance, PALETTE["green"])
        self._btn_stop  = sidebar_btn(parent, "■  STOP", self._stop_surveillance, PALETTE["red"])
        self._btn_stop.configure(state="disabled")

        section_label(parent, "CAMERA")
        self._cam_var = tk.StringVar(value="0")
        cam_frame = tk.Frame(parent, bg=PALETTE["bg_sidebar"])
        cam_frame.pack(fill="x", padx=12, pady=4)
        tk.Label(cam_frame, text="Device Index:", bg=PALETTE["bg_sidebar"],
                 fg=PALETTE["text_dim"], font=("Courier", 10)).pack(side="left")
        tk.Entry(cam_frame, textvariable=self._cam_var, width=4,
                 bg=PALETTE["bg_card"], fg=PALETTE["text"],
                 insertbackground=PALETTE["text"],
                 font=("Courier", 10), relief="flat").pack(side="left", padx=6)

        section_label(parent, "DETECTION SETTINGS")

        # Confidence slider
        tk.Label(parent, text="Confidence Threshold",
                 bg=PALETTE["bg_sidebar"], fg=PALETTE["text_dim"],
                 font=("Courier", 9)).pack(anchor="w", padx=14)
        self._conf_var = tk.DoubleVar(value=0.45)
        self._conf_slider = tk.Scale(
            parent, from_=0.1, to=0.95, resolution=0.05,
            orient="horizontal", variable=self._conf_var,
            bg=PALETTE["bg_sidebar"], fg=PALETTE["text"],
            troughcolor=PALETTE["bg_card"], highlightthickness=0,
            font=("Courier", 9),
            command=self._on_conf_change,
        )
        self._conf_slider.pack(fill="x", padx=12)

        section_label(parent, "BEHAVIOR TOGGLES")
        self._behavior_vars = {}
        behaviors = [
            ("violence",         "Violence"),
            ("choking",          "Choking / Distress"),
            ("camera_tampering", "Camera Tampering"),
            ("restricted_area",  "Restricted Area"),
            ("loitering",        "Loitering"),
            ("unattended_object","Unattended Object"),
        ]
        for key, label in behaviors:
            var = tk.BooleanVar(value=True)
            self._behavior_vars[key] = var
            cb = tk.Checkbutton(
                parent, text=label, variable=var,
                bg=PALETTE["bg_sidebar"], fg=PALETTE["text"],
                selectcolor=PALETTE["bg_card"],
                activebackground=PALETTE["bg_sidebar"],
                activeforeground=PALETTE["accent"],
                font=("Courier", 10),
                anchor="w",
            )
            cb.pack(fill="x", padx=14, pady=1)

        section_label(parent, "TOOLS")
        sidebar_btn(parent, "📁  Export Log", self._export_log, PALETTE["accent2"])
        sidebar_btn(parent, "🗑  Clear Alerts", self._clear_alerts, PALETTE["text_dim"])
        sidebar_btn(parent, "📸  Save Snapshot", self._save_snapshot, PALETTE["accent"])

        # Stats at bottom
        section_label(parent, "SESSION STATS")
        self._stats_frame = tk.Frame(parent, bg=PALETTE["bg_sidebar"])
        self._stats_frame.pack(fill="x", padx=12)
        self._stat_labels = {}
        for sev, color in SEVERITY_COLORS_HEX.items():
            row = tk.Frame(self._stats_frame, bg=PALETTE["bg_sidebar"])
            row.pack(fill="x", pady=1)
            tk.Label(row, text=f"  {sev}", bg=PALETTE["bg_sidebar"],
                     fg=color, font=("Courier", 10), width=12, anchor="w").pack(side="left")
            lbl = tk.Label(row, text="0", bg=PALETTE["bg_sidebar"],
                           fg=PALETTE["text"], font=("Courier", 10, "bold"))
            lbl.pack(side="right")
            self._stat_labels[sev] = lbl

    # ------------------------------------------------------------------
    def _build_feed_panel(self, parent):
        # Feed label row
        label_row = tk.Frame(parent, bg=PALETTE["bg_dark"])
        label_row.pack(fill="x", pady=(0, 6))
        tk.Label(
            label_row, text="LIVE FEED",
            bg=PALETTE["bg_dark"], fg=PALETTE["accent"],
            font=("Courier", 12, "bold"),
        ).pack(side="left")

        self._rec_indicator = tk.Label(
            label_row, text="",
            bg=PALETTE["bg_dark"], fg=PALETTE["red"],
            font=("Courier", 11, "bold"),
        )
        self._rec_indicator.pack(side="right")

        # Canvas
        self.feed_canvas = FeedCanvas(parent, width=900, height=506)
        self.feed_canvas.pack(fill="both", expand=True)

        # Bottom info bar
        info_bar = tk.Frame(parent, bg=PALETTE["bg_card"], height=36)
        info_bar.pack(fill="x", pady=(6, 0))
        info_bar.pack_propagate(False)

        self._fps_label = tk.Label(
            info_bar, text="FPS: --",
            bg=PALETTE["bg_card"], fg=PALETTE["text_dim"],
            font=("Courier", 10),
        )
        self._fps_label.pack(side="left", padx=12)

        self._res_label = tk.Label(
            info_bar, text="RES: --",
            bg=PALETTE["bg_card"], fg=PALETTE["text_dim"],
            font=("Courier", 10),
        )
        self._res_label.pack(side="left", padx=12)

        self._det_label = tk.Label(
            info_bar, text="DETECTIONS: --",
            bg=PALETTE["bg_card"], fg=PALETTE["text_dim"],
            font=("Courier", 10),
        )
        self._det_label.pack(side="left", padx=12)

        self._time_label = tk.Label(
            info_bar, text="",
            bg=PALETTE["bg_card"], fg=PALETTE["text_dim"],
            font=("Courier", 10),
        )
        self._time_label.pack(side="right", padx=12)

    # ------------------------------------------------------------------
    def _build_alert_panel(self, parent):
        tk.Label(
            parent, text="ALERT FEED",
            bg=PALETTE["bg_sidebar"], fg=PALETTE["accent"],
            font=("Courier", 12, "bold"),
        ).pack(anchor="w", padx=14, pady=(12, 6))

        # Scrollable alert list
        container = tk.Frame(parent, bg=PALETTE["bg_sidebar"])
        container.pack(fill="both", expand=True, padx=4)

        self._alert_canvas = tk.Canvas(
            container, bg=PALETTE["bg_sidebar"],
            highlightthickness=0,
        )
        scrollbar = tk.Scrollbar(container, orient="vertical",
                                  command=self._alert_canvas.yview)
        self._alert_canvas.configure(yscrollcommand=scrollbar.set)

        scrollbar.pack(side="right", fill="y")
        self._alert_canvas.pack(side="left", fill="both", expand=True)

        self._alert_inner = tk.Frame(self._alert_canvas, bg=PALETTE["bg_sidebar"])
        self._alert_canvas_window = self._alert_canvas.create_window(
            (0, 0), window=self._alert_inner, anchor="nw"
        )
        self._alert_inner.bind("<Configure>", self._on_alert_frame_configure)
        self._alert_canvas.bind("<Configure>", self._on_alert_canvas_configure)

    # ------------------------------------------------------------------
    def _on_alert_frame_configure(self, event):
        self._alert_canvas.configure(scrollregion=self._alert_canvas.bbox("all"))

    def _on_alert_canvas_configure(self, event):
        self._alert_canvas.itemconfig(self._alert_canvas_window, width=event.width)

    # ------------------------------------------------------------------
    # Control Methods
    # ------------------------------------------------------------------
    def _start_surveillance(self):
        if self._running:
            return

        try:
            cam_idx = int(self._cam_var.get())
        except ValueError:
            cam_idx = 0

        self.camera.device_index = cam_idx

        if not self.camera.start():
            messagebox.showerror(
                "Camera Error",
                f"Could not open camera at index {cam_idx}.\n"
                "Check your camera connection and device index."
            )
            return

        self._running = True
        self._status = "ACTIVE"
        self._pipeline_thread = threading.Thread(
            target=self._pipeline_loop, daemon=True
        )
        self._pipeline_thread.start()
        self._update_status_indicator()
        self._btn_start.configure(state="disabled")
        self._btn_stop.configure(state="normal")

    def _stop_surveillance(self):
        self._running = False
        self.camera.stop()
        self._status = "IDLE"
        self._update_status_indicator()
        self._btn_start.configure(state="normal")
        self._btn_stop.configure(state="disabled")
        self.feed_canvas._show_placeholder()

    def _update_status_indicator(self):
        if self._status == "ACTIVE":
            self._status_dot.configure(text="● LIVE", fg=PALETTE["green"])
            self._rec_indicator.configure(text="⬤ REC")
        else:
            self._status_dot.configure(text="● OFFLINE", fg=PALETTE["text_dim"])
            self._rec_indicator.configure(text="")

    # ------------------------------------------------------------------
    # Pipeline loop (runs in background thread)
    # ------------------------------------------------------------------
    def _pipeline_loop(self):
        alert_manager = self.alert_manager
        engine = self.engine
        camera = self.camera

        while self._running:
            frame = camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            # Apply behavior toggle settings
            for key, var in self._behavior_vars.items():
                b = engine.config.get("behaviors", {})
                if key in b:
                    b[key]["enabled"] = var.get()

            # Run detection
            result = engine.process_frame(frame)

            # Update UI canvas from main thread
            self.after(0, self._update_feed, result)

            # Fire alerts for significant detections
            for det in result.detections:
                if det.label != "person":
                    alert_manager.handle(det, frame)

    # ------------------------------------------------------------------
    def _update_feed(self, result):
        """Update canvas and info bar (called on main thread via after())."""
        if not self._running:
            return

        self.feed_canvas.update_frame(result.frame)
        if result.tamper_detected:
            self.feed_canvas.show_tamper_warning()

        self._fps_label.configure(text=f"FPS: {result.fps:.1f}")
        w, h = self.camera.resolution
        self._res_label.configure(text=f"RES: {w}×{h}")

        visible_dets = [d for d in result.detections if d.label != "person"]
        self._det_label.configure(
            text=f"DETECTIONS: {len(visible_dets)}",
            fg=PALETTE["red"] if visible_dets else PALETTE["text_dim"],
        )

        now = datetime.now().strftime("%H:%M:%S")
        self._time_label.configure(text=now)

    # ------------------------------------------------------------------
    def _on_new_alert(self, record):
        """Called by AlertManager when a new alert fires (any thread)."""
        self.after(0, self._add_alert_row, record)
        self.after(0, self._update_stats)

    def _add_alert_row(self, record):
        row = AlertRow(self._alert_inner, record)
        row.pack(fill="x", pady=1)
        self._alert_rows.append(row)

        # Keep only last 50 rows
        if len(self._alert_rows) > 50:
            oldest = self._alert_rows.pop(0)
            oldest.destroy()

        # Scroll to bottom
        self._alert_canvas.yview_moveto(0.0)  # scroll to top (newest first)

    def _update_stats(self):
        stats = self.alert_manager.get_stats()
        for sev, lbl in self._stat_labels.items():
            count = stats["by_severity"].get(sev, 0)
            lbl.configure(text=str(count))

    # ------------------------------------------------------------------
    def _on_conf_change(self, val):
        self.engine.config["confidence_threshold"] = float(val)

    def _export_log(self):
        path = filedialog.asksaveasfilename(
            defaultextension=".log",
            filetypes=[("Log files", "*.log"), ("Text files", "*.txt")],
            initialfile=f"securoserv_export_{datetime.now():%Y%m%d_%H%M%S}.log",
        )
        if not path:
            return
        history = self.alert_manager.get_history(limit=500)
        with open(path, "w", encoding="utf-8") as f:
            f.write("SecuroServ Alert Export\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write("=" * 60 + "\n\n")
            for r in reversed(history):
                f.write(r.to_log_line() + "\n")
        messagebox.showinfo("Export Complete", f"Log exported to:\n{path}")

    def _clear_alerts(self):
        self.alert_manager.clear_history()
        for row in self._alert_rows:
            row.destroy()
        self._alert_rows.clear()
        for lbl in self._stat_labels.values():
            lbl.configure(text="0")

    def _save_snapshot(self):
        frame = self.camera.read()
        if frame is None:
            messagebox.showwarning("No Feed", "No active camera feed to snapshot.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            initialfile=f"snapshot_{datetime.now():%Y%m%d_%H%M%S}.jpg",
        )
        if path:
            cv2.imwrite(path, frame)
            messagebox.showinfo("Saved", f"Snapshot saved to:\n{path}")

    # ------------------------------------------------------------------
    def _start_ui_refresh(self):
        """Periodic UI refresh for clock etc."""
        def _refresh():
            now = datetime.now().strftime("%H:%M:%S")
            self._time_label.configure(text=now)
            self.after(1000, _refresh)
        self.after(1000, _refresh)

    def on_close(self):
        self._stop_surveillance()
        self.destroy()


# ---------------------------------------------------------------------------
# Helper widget functions
# ---------------------------------------------------------------------------

def section_label(parent, text):
    tk.Frame(parent, bg=PALETTE["border"], height=1).pack(fill="x", padx=8, pady=(10, 0))
    tk.Label(
        parent, text=text,
        bg=PALETTE["bg_sidebar"],
        fg=PALETTE["text_dim"],
        font=("Courier", 8, "bold"),
        anchor="w",
    ).pack(fill="x", padx=14, pady=(4, 2))


def sidebar_btn(parent, text, command, color=None):
    color = color or PALETTE["accent"]
    btn = tk.Button(
        parent,
        text=text,
        command=command,
        bg=PALETTE["bg_card"],
        fg=color,
        activebackground=PALETTE["border"],
        activeforeground=color,
        font=("Courier", 10, "bold"),
        relief="flat",
        cursor="hand2",
        anchor="w",
        padx=12, pady=6,
        bd=0,
    )
    btn.pack(fill="x", padx=12, pady=3)
    return btn
