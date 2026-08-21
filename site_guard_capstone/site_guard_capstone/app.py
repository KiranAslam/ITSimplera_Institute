"""
PPE Compliance Detection - Capstone Dashboard
-----------------------------------------------
Live webcam feed -> YOLOv8 detection -> ByteTrack tracking -> per-track
violation logging (deduped) -> Flask dashboard (video + logs + stats).

No auth / login / user management by design (single-user academic demo).

Run:
    python app.py

Then open http://127.0.0.1:5000 in your browser.
"""

import csv
import os
import threading
import time
from collections import Counter
from datetime import datetime

import cv2
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = os.path.join("models", "best.pt")
CAMERA_INDEX = 0                # change if your webcam isn't device 0
CONF_THRESHOLD = 0.4
TRACKER_CONFIG = "bytetrack.yaml"   # ships with ultralytics, no extra install needed
LOG_CSV_PATH = os.path.join("logs", "violations.csv")

# Classes that represent a PPE violation (must match your data.yaml names).
# Adjust this set if your trained model's class names differ.
VIOLATION_CLASSES = {"no_helmet", "no_goggle", "no_gloves", "no_vest", "none"}

# --------------------------------------------------------------------------
# Flask app
# --------------------------------------------------------------------------
app = Flask(__name__)

# --------------------------------------------------------------------------
# Global state (single model + single camera, shared across requests)
# Loaded ONCE at startup to avoid the "shared model/tracker across loops"
# and reloader-double-init issues.
# --------------------------------------------------------------------------
model = None
camera = None
camera_lock = threading.Lock()

logged_violations = set()      # {(track_id, class_name)} -> dedup guard
violation_log = []             # list of dicts, most recent first (in-memory, mirrors CSV)
log_lock = threading.Lock()

stats = {
    "unique_people_tracked": 0,
    "total_violations_logged": 0,
    "violations_by_class": Counter(),
}
stats_lock = threading.Lock()

_seen_track_ids = set()        # to count unique people/objects tracked


def init_model_and_camera():
    global model, camera
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained weights not found at '{MODEL_PATH}'. "
            f"Copy your trained best.pt into the models/ folder first."
        )
    model = YOLO(MODEL_PATH)

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        raise RuntimeError(
            f"Could not open webcam at index {CAMERA_INDEX}. "
            f"Check that no other app is using the camera."
        )


def ensure_log_csv():
    os.makedirs("logs", exist_ok=True)
    if not os.path.exists(LOG_CSV_PATH):
        with open(LOG_CSV_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "track_id", "violation_class", "confidence"])


def log_violation(track_id, class_name, confidence):
    """Log a violation once per (track_id, class_name) pair."""
    key = (track_id, class_name)
    with log_lock:
        if key in logged_violations:
            return
        logged_violations.add(key)

        entry = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "track_id": int(track_id),
            "violation_class": class_name,
            "confidence": round(float(confidence), 3),
        }
        violation_log.insert(0, entry)  # newest first

        with open(LOG_CSV_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [entry["timestamp"], entry["track_id"], entry["violation_class"], entry["confidence"]]
            )

    with stats_lock:
        stats["total_violations_logged"] += 1
        stats["violations_by_class"][class_name] += 1


def track_id_seen(track_id):
    with stats_lock:
        if track_id not in _seen_track_ids:
            _seen_track_ids.add(track_id)
            stats["unique_people_tracked"] = len(_seen_track_ids)


# --------------------------------------------------------------------------
# Box color helper: red for violation classes, green for compliant/other
# --------------------------------------------------------------------------
def box_color(class_name):
    return (0, 0, 255) if class_name in VIOLATION_CLASSES else (0, 200, 0)


# --------------------------------------------------------------------------
# Video generator: reads frames, runs tracked inference, draws boxes,
# logs violations, yields MJPEG stream.
# --------------------------------------------------------------------------
def generate_frames():
    while True:
        with camera_lock:
            success, frame = camera.read()
        if not success:
            time.sleep(0.05)
            continue

        results = model.track(
            frame,
            persist=True,
            tracker=TRACKER_CONFIG,
            conf=CONF_THRESHOLD,
            verbose=False,
        )

        r = results[0]
        if r.boxes is not None and r.boxes.id is not None:
            boxes = r.boxes.xyxy.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            class_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()

            for box, tid, cid, conf in zip(boxes, track_ids, class_ids, confs):
                class_name = model.names[int(cid)]
                x1, y1, x2, y2 = map(int, box)
                color = box_color(class_name)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"ID{tid} {class_name} {conf:.2f}"
                cv2.putText(
                    frame, label, (x1, max(y1 - 8, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
                )

                track_id_seen(tid)
                if class_name in VIOLATION_CLASSES:
                    log_violation(tid, class_name, conf)

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/api/logs")
def api_logs():
    with log_lock:
        return jsonify(violation_log[:100])  # most recent 100


@app.route("/api/stats")
def api_stats():
    with stats_lock:
        return jsonify(
            {
                "unique_people_tracked": stats["unique_people_tracked"],
                "total_violations_logged": stats["total_violations_logged"],
                "violations_by_class": dict(stats["violations_by_class"]),
            }
        )


if __name__ == "__main__":
    ensure_log_csv()
    init_model_and_camera()
    # use_reloader=False -> avoids double model/camera init from Werkzeug's
    # debug reloader spawning a second process.
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
