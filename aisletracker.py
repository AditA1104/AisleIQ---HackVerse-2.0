"""
AisleIQ Tracker — unified entry point.

Two detection sources feed the SAME zone/dwell/IntentEngine pipeline:

  1. Real footage (webcam or mp4) -> YOLOv8 person detection + tracking.
  2. Synthetic scripted shoppers  -> no YOLO, no video file needed. Five
     pre-built trajectories (from the earlier test-data generator) are
     drawn as moving circles and their bounding boxes are fed into the
     exact same code that processes real YOLO boxes. Useful for testing
     the IntentEngine wiring/dashboard without needing footage or model
     weights at all.

Usage:
    python aisleiq_tracker.py --source 0                # webcam
    python aisleiq_tracker.py --source supermarket.mp4   # video file
    python aisleiq_tracker.py --source synthetic         # scripted shoppers
    python aisleiq_tracker.py --source synthetic --headless   # no GUI window
    python aisleiq_tracker.py --source testvid1.mp4 --fps 24  # override
                                                                # playback pacing
                                                                # if auto-detected
                                                                # FPS looks wrong
"""

import argparse
import json
import math
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

import backend

FRAME_W, FRAME_H = 640, 480
ZONE_PTS = np.array([[50, 50], [590, 50], [590, 430], [50, 430]], np.int32)
DWELL_THRESHOLD_SECONDS = 5
NOTIFY_COOLDOWN_SECONDS = 60 # once a shopper is alerted, don't re-notify for this many
                              # more seconds of sustained alert (still logs, just quieter)

# --- Traffic heatmap grid ---
# 16x12 divides evenly into the 640x480 frame (40px cells), giving a decent
# balance of spatial resolution vs a chart that isn't just noise.
HEATMAP_GRID_COLS = 16
HEATMAP_GRID_ROWS = 12
HEATMAP_DATA_PATH = "heatmap_data.json"
HEATMAP_WRITE_INTERVAL_SECONDS = 1.0  # throttle disk writes -- don't hit disk every frame

# --- Live feed mirror (for the Streamlit dashboard's "Live Tracking" tab) ---
# Not a real video stream -- no websockets/WebRTC. We mirror the current
# annotated frame to disk, same pattern as alerts_log.json / heatmap_data.json
# / live_metrics.json, and the dashboard (a separate process) polls + displays
# it. At ~10 writes/sec + a fast dashboard poll, it reads as "live" without
# needing any streaming infrastructure.
LIVE_FRAME_PATH = "live_frame.jpg"
LIVE_FRAME_WRITE_INTERVAL_SECONDS = 0.1  # ~10fps to the dashboard -- smooth
                                          # enough to look live, throttled so
                                          # we're not JPEG-encoding + hitting
                                          # disk on every single inference frame

# --- Playback pacing sanity bounds ---
# Some OpenCV backends/containers (notably macOS's AVFoundation backend on
# certain mp4s) misreport cv2.CAP_PROP_FPS -- e.g. returning a raw container
# timebase value like 90000 instead of the actual frame rate. If trusted
# blindly, that collapses target_frame_time to ~0 and the video plays back
# as fast as the CPU can decode+infer instead of at its real speed. We clamp
# to this range and fall back to a sane default outside it.
MIN_PLAUSIBLE_FPS = 1.0
MAX_PLAUSIBLE_FPS = 120.0
DEFAULT_FPS_FALLBACK = 30.0


def _save_heatmap_data(grid: np.ndarray) -> None:
    """Mirrors the footfall grid to disk (same pattern as live_metrics.json /
    alerts_log.json) so the Streamlit dashboard -- a separate process -- can
    read it. Values are frame-counts per cell: someone standing still keeps
    incrementing the same cell every frame, so this is naturally dwell-weighted,
    not just a raw crossing count."""
    try:
        with open(HEATMAP_DATA_PATH, "w") as f:
            json.dump({
                "grid": grid.tolist(),
                "rows": HEATMAP_GRID_ROWS,
                "cols": HEATMAP_GRID_COLS,
                "frame_w": FRAME_W,
                "frame_h": FRAME_H,
                "last_updated": time.time(),
            }, f)
    except Exception as e:
        print(f"⚠️ Failed to write {HEATMAP_DATA_PATH}: {e}")


def _save_live_frame(frame: np.ndarray) -> None:
    """Mirrors the current annotated frame (zone outline, bounding boxes,
    classification labels -- whatever's already drawn on it) to disk so the
    Streamlit dashboard can display it as a pseudo-live feed by polling and
    re-reading this file. Works identically for webcam, mp4, or synthetic
    mode, since it's called after frame annotation regardless of source."""
    try:
        cv2.imwrite(LIVE_FRAME_PATH, frame)
    except Exception as e:
        print(f"⚠️ Failed to write {LIVE_FRAME_PATH}: {e}")


# ----------------------------------------------------------------------
# IntentEngine (unchanged)
# ----------------------------------------------------------------------
class IntentEngine:
    def __init__(self, buffer_size=100, min_move_px=5.0):
        self.history = {}
        self.buffer_size = buffer_size
        self.min_move_px = min_move_px

    def evaluate_shopper(self, track_id, current_centroid, dwell_time):
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.buffer_size)
        self.history[track_id].append(current_centroid)
        pts = np.array(self.history[track_id])
        if len(pts) < 10:
            return "Gathering Data", 0.0, False
        step_vectors = np.diff(pts, axis=0)
        step_distances = np.linalg.norm(step_vectors, axis=1)
        filtered_distances = step_distances[step_distances >= self.min_move_px]
        total_path = float(np.sum(filtered_distances))
        net_displacement = float(np.linalg.norm(pts[-1] - pts[0]))
        pacing_ratio = total_path / (net_displacement + 1e-5)
        friction_score = dwell_time * pacing_ratio
        if dwell_time < 10:
            classification = "Transient Passerby"
            is_alert = False
        elif pacing_ratio > 2.0 and total_path > 50:
            classification = "Active Hesitation (Pacing Confusion)"
            is_alert = True
        elif dwell_time >= 30 and pacing_ratio <= 2.0 and total_path <= 30:
            classification = "Choice Paralysis (Frozen Confusion)"
            is_alert = True
        elif dwell_time >= 20 and pacing_ratio < 1.3:
            classification = "Relaxed / Passive Browser"
            is_alert = False
        else:
            classification = "Standard Browsing"
            is_alert = False
        return classification, round(friction_score, 2), is_alert


CLASS_COLORS = {
    "Gathering Data": (200, 200, 200),
    "Transient Passerby": (0, 255, 0),
    "Active Hesitation (Pacing Confusion)": (0, 0, 255),
    "Choice Paralysis (Frozen Confusion)": (0, 0, 255),
    "Relaxed / Passive Browser": (255, 200, 0),
    "Standard Browsing": (0, 200, 255),
}


# ----------------------------------------------------------------------
# Shared per-frame pipeline: zone check -> dwell timer -> IntentEngine
# Both the real-video loop and the synthetic loop call this with a list
# of detections in the form (track_id, x1, y1, x2, y2).
# ----------------------------------------------------------------------
def notify_employee(t_id, classification, friction_score, dwell_time):
    """
    This is the ONE place that should ever page a human. Wire your real
    notification (Slack webhook, SMS, dashboard push, whatever) in here.
    It's only called when should_notify() below says the cooldown has
    cleared — never once per frame.
    """
    print(f"🔔 NOTIFY EMPLOYEE -> Customer #{t_id}: {classification} "
          f"(dwell={dwell_time:.1f}s, friction={friction_score})")
    backend.handle_confusion_alert_async(
        customer_id=t_id,
        classification=classification,
        friction_score=friction_score,
        dwell_seconds=dwell_time,
    )


def process_detections(frame, detections, zone_timers, intent_engine, active_alert_ids,
                        notification_state, heatmap_state):
    """
    detections: list of tuples, either
        (t_id, x1, y1, x2, y2)                  -- dwell_time computed from wall clock
        (t_id, x1, y1, x2, y2, dwell_override)  -- dwell_time supplied directly
                                                     (used by synthetic mode, where
                                                     frames are simulated seconds,
                                                     not real elapsed time)
    notification_state: dict of t_id -> dwell_time at which they were last notified.
                         Lives in the caller so it persists across frames.
    heatmap_state: dict {"grid": np.ndarray, "last_write": float} -- lives in the
                    caller so the footfall grid accumulates across frames and the
                    disk write can be throttled independently of the frame rate.
    """
    cv2.polylines(frame, [ZONE_PTS], isClosed=True, color=(0, 255, 255), thickness=2)
    cv2.putText(frame, "Monitored Aisle Zone", (105, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    current_time = time.time()
    current_ids_in_zone = set()
    active_count = len(detections)

    for det in detections:
        if len(det) == 6:
            t_id, x1, y1, x2, y2, dwell_override = det
        else:
            t_id, x1, y1, x2, y2 = det
            dwell_override = None

        center_x = int((x1 + x2) / 2)
        center_y = int((y1 + y2) / 2)
        centroid = (float(center_x), float(center_y))

        # Accumulate footfall for the traffic heatmap -- every detected person,
        # every frame, regardless of whether they're inside the alert zone.
        # This is what makes it dwell-weighted rather than a raw crossing count.
        hm_col = min(max(int(center_x / FRAME_W * HEATMAP_GRID_COLS), 0), HEATMAP_GRID_COLS - 1)
        hm_row = min(max(int(center_y / FRAME_H * HEATMAP_GRID_ROWS), 0), HEATMAP_GRID_ROWS - 1)
        heatmap_state["grid"][hm_row, hm_col] += 1

        is_inside = cv2.pointPolygonTest(ZONE_PTS, (center_x, center_y), False) >= 0

        if is_inside:
            current_ids_in_zone.add(t_id)
            if t_id not in zone_timers:
                zone_timers[t_id] = current_time
            dwell_time = dwell_override if dwell_override is not None else current_time - zone_timers[t_id]
            display_seconds = int(dwell_time)

            classification, friction_score, is_alert = intent_engine.evaluate_shopper(
                t_id, centroid, dwell_time
            )
            if is_alert:
                active_alert_ids.add(t_id)
            else:
                active_alert_ids.discard(t_id)

            color = CLASS_COLORS.get(classification, (0, 255, 0))
            cv2.circle(frame, (center_x, center_y), 4, (0, 255, 255), -1)
            cv2.putText(
                frame,
                f"ID:{t_id} {display_seconds}s | {classification} | F:{friction_score}",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
            )

            if display_seconds == DWELL_THRESHOLD_SECONDS:
                print(f"[ALERT] Customer #{t_id} lingering in Zone for {display_seconds}s!")

            if is_alert:
                last_notified = notification_state.get(t_id)
                should_notify = (
                    last_notified is None
                    or (dwell_time - last_notified) >= NOTIFY_COOLDOWN_SECONDS
                )
                if should_notify:
                    notify_employee(t_id, classification, friction_score, dwell_time)
                    notification_state[t_id] = dwell_time
            else:
                # No longer alerting -> reset, so a FUTURE alert episode
                # notifies immediately instead of inheriting the old cooldown.
                notification_state.pop(t_id, None)
        else:
            if t_id in zone_timers:
                del zone_timers[t_id]
            active_alert_ids.discard(t_id)
            notification_state.pop(t_id, None)
            if t_id in intent_engine.history:
                del intent_engine.history[t_id]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    expired_ids = [t_id for t_id in zone_timers if t_id not in current_ids_in_zone]
    for t_id in expired_ids:
        del zone_timers[t_id]
        active_alert_ids.discard(t_id)
        notification_state.pop(t_id, None)
        if t_id in intent_engine.history:
            del intent_engine.history[t_id]

    dashboard_data = {
        "active_shoppers": int(active_count),
        "alerts_triggered": len(active_alert_ids),
    }
    with open("live_metrics.json", "w") as f:
        json.dump(dashboard_data, f)

    if current_time - heatmap_state["last_write"] >= HEATMAP_WRITE_INTERVAL_SECONDS:
        _save_heatmap_data(heatmap_state["grid"])
        heatmap_state["last_write"] = current_time

    return frame


def _resolve_playback_fps(cap: cv2.VideoCapture, fps_override: Optional[float]) -> float:
    """
    Figures out what FPS to pace file playback at. An explicit --fps
    always wins. Otherwise, trusts cv2.CAP_PROP_FPS only if it falls in a
    plausible range -- some backends/containers report bogus values (e.g.
    a raw timebase like 90000 instead of the real frame rate), which would
    otherwise collapse target_frame_time to ~0 and make the video play
    back as fast as the CPU can decode+infer instead of its real speed.
    """
    if fps_override:
        return fps_override

    reported_fps = cap.get(cv2.CAP_PROP_FPS)
    if (
        not reported_fps
        or math.isnan(reported_fps)
        or not (MIN_PLAUSIBLE_FPS <= reported_fps <= MAX_PLAUSIBLE_FPS)
    ):
        print(
            f"⚠️ Ignoring implausible reported FPS ({reported_fps}); "
            f"falling back to {DEFAULT_FPS_FALLBACK}. If playback still "
            f"looks off, pass --fps <value> with your video's real frame rate."
        )
        return DEFAULT_FPS_FALLBACK

    return reported_fps


# ----------------------------------------------------------------------
# Source 1: real video / webcam via YOLOv8
# ----------------------------------------------------------------------
def run_real_source(video_source, headless=False, fps_override: Optional[float] = None):
    from ultralytics import YOLO  # pyright: ignore[reportPrivateImportUsage]  # lazy import: not needed for synthetic mode

    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)

    # Pace playback to the source's real FPS for FILE sources only. A live
    # webcam already delivers frames in real time (cap.read() blocks until
    # the next frame exists), so pacing there would just add lag. Without
    # this, cap.read() + inference runs as fast as the CPU/GPU allows,
    # which is why mp4 playback can look sped up on faster machines.
    is_file_source = isinstance(video_source, str)
    source_fps = _resolve_playback_fps(cap, fps_override)
    target_frame_time = 1.0 / source_fps

    intent_engine = IntentEngine(buffer_size=100, min_move_px=5.0)
    zone_timers = {}
    active_alert_ids = set()
    notification_state = {}
    heatmap_state = {"grid": np.zeros((HEATMAP_GRID_ROWS, HEATMAP_GRID_COLS), dtype=np.float64), "last_write": 0.0}
    last_live_frame_write = 0.0

    print(f"Starting AisleIQ Tracker on source={video_source} (pacing at {source_fps:.1f} fps) ... Press 'q' to quit.")
    while cap.isOpened():
        loop_start = time.time()

        ret, frame = cap.read()
        if not ret:
            print("End of video stream or camera error.")
            break

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        frame = cv2.flip(frame, 1)

        results = model.track(frame, persist=True, classes=[0], verbose=False)

        detections = []
        boxes_obj = results[0].boxes  # local var: lets pyright narrow Optional[Boxes] -> Boxes
        if boxes_obj is not None and boxes_obj.id is not None:
            boxes = boxes_obj.xyxy.cpu().numpy()  # pyright: ignore[reportAttributeAccessIssue]
            track_ids = boxes_obj.id.cpu().numpy()  # pyright: ignore[reportAttributeAccessIssue]
            for box, track_id in zip(boxes, track_ids):
                x1, y1, x2, y2 = map(int, box)
                detections.append((int(track_id), x1, y1, x2, y2))

        frame = process_detections(frame, detections, zone_timers, intent_engine, active_alert_ids,
                                    notification_state, heatmap_state)

        now = time.time()
        if now - last_live_frame_write >= LIVE_FRAME_WRITE_INTERVAL_SECONDS:
            _save_live_frame(frame)
            last_live_frame_write = now

        remaining = target_frame_time - (time.time() - loop_start) if is_file_source else 0

        if not headless:
            cv2.imshow("AisleIQ - OpenCV Tracking Feed", frame)
            wait_ms = max(1, int(remaining * 1000)) if is_file_source else 1
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break
        elif is_file_source and remaining > 0:
            time.sleep(remaining)

    cap.release()
    if not headless:
        cv2.destroyAllWindows()
    # Final write regardless of the throttle interval, so the dashboard sees
    # the complete accumulated grid once tracking stops, not a stale snapshot.
    _save_heatmap_data(heatmap_state["grid"])


# ----------------------------------------------------------------------
# Source 2: synthetic scripted shoppers (no YOLO, no video file needed)
# Trajectories match the same shapes used in the earlier standalone
# IntentEngine test-data script, repositioned so they sit inside ZONE_PTS.
# ----------------------------------------------------------------------
def traj_transient_passerby(n=12):
    xs = np.linspace(80, 560, n)
    ys = np.full(n, 240.0) + np.random.normal(0, 1.5, n)
    return list(zip(xs, ys))


def traj_active_hesitation(n=25):
    t = np.arange(n)
    xs = 320 + 90 * np.sin(t * 0.9)
    ys = 150 + np.random.normal(0, 2.0, n)
    return list(zip(xs, ys))


def traj_choice_paralysis(n=35):
    xs = 200 + np.random.normal(0, 1.2, n)
    ys = 300 + np.random.normal(0, 1.2, n)
    return list(zip(xs, ys))


def traj_relaxed_browser(n=25):
    xs = np.linspace(400, 500, n) + np.random.normal(0, 0.5, n)
    ys = np.linspace(120, 160, n) + np.random.normal(0, 0.5, n)
    return list(zip(xs, ys))


def traj_standard_browsing(n=15):
    xs, ys = [480.0], [380.0]
    for _ in range(n - 1):
        xs.append(xs[-1] + np.random.normal(3, 4))
        ys.append(ys[-1] + np.random.normal(0, 3))
    return list(zip(xs, ys))


# (track_id, generator, fps) — fps controls how many seconds of dwell
# elapse per rendered frame, same trick as the standalone test script so
# "Transient Passerby" (dwell < 10s) can actually be observed.
SYNTHETIC_SHOPPERS = [
    (101, traj_transient_passerby, 3.0),
    (102, traj_active_hesitation, 1.0),
    (103, traj_choice_paralysis, 1.0),
    (104, traj_relaxed_browser, 1.0),
    (105, traj_standard_browsing, 1.0),
]
BLOB_RADIUS = 14


def run_synthetic_source(headless=False):
    np.random.seed(0)
    intent_engine = IntentEngine(buffer_size=100, min_move_px=5.0)
    zone_timers = {}
    active_alert_ids = set()
    notification_state = {}
    heatmap_state = {"grid": np.zeros((HEATMAP_GRID_ROWS, HEATMAP_GRID_COLS), dtype=np.float64), "last_write": 0.0}
    last_live_frame_write = 0.0

    tracks = [(tid, gen(), fps) for tid, gen, fps in SYNTHETIC_SHOPPERS]
    max_len = max(len(pts) for _, pts, _ in tracks)

    print("Starting AisleIQ Tracker on source=synthetic ... Press 'q' to quit.")
    for frame_idx in range(max_len):
        frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)

        detections = []
        for tid, pts, fps in tracks:
            if frame_idx >= len(pts):
                continue  # this shopper has "left" — lets exit-cleanup logic run
            x, y = pts[frame_idx]
            x1, y1 = int(x - BLOB_RADIUS), int(y - BLOB_RADIUS)
            x2, y2 = int(x + BLOB_RADIUS), int(y + BLOB_RADIUS)
            cv2.circle(frame, (int(x), int(y)), BLOB_RADIUS, (180, 180, 180), -1)
            # Simulated dwell time: (frames elapsed for this shopper) / fps.
            # This is what lets a short/fast trajectory land in "Transient
            # Passerby" instead of waiting on real wall-clock seconds.
            simulated_dwell = round((frame_idx + 1) / fps, 1)
            detections.append((tid, x1, y1, x2, y2, simulated_dwell))

        frame = process_detections(frame, detections, zone_timers, intent_engine, active_alert_ids,
                                    notification_state, heatmap_state)

        now = time.time()
        if now - last_live_frame_write >= LIVE_FRAME_WRITE_INTERVAL_SECONDS:
            _save_live_frame(frame)
            last_live_frame_write = now

        if not headless:
            cv2.imshow("AisleIQ - OpenCV Tracking Feed", frame)
            key = cv2.waitKey(150) & 0xFF  # slow playback so it's watchable
            if key == ord('q'):
                break

    if not headless:
        cv2.destroyAllWindows()
    print("Synthetic run complete.")
    # Final write regardless of the throttle interval -- in headless mode
    # especially, the loop can finish inside the same 1s window and never
    # otherwise trigger a write.
    _save_heatmap_data(heatmap_state["grid"])


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="AisleIQ Tracker")
    parser.add_argument(
        "--source", default="0",
        help="Webcam index (e.g. 0), path to an mp4 file, or 'synthetic'."
    )
    parser.add_argument(
        "--headless", action="store_true",
        help="Skip cv2.imshow/waitKey — useful for servers/CI without a display."
    )
    parser.add_argument(
        "--fps", type=float, default=None,
        help="Override video playback pacing (frames per second). Use this "
             "if a video file plays back too fast or too slow -- some "
             "containers/backends report an incorrect FPS value that "
             "can't be reliably auto-detected."
    )
    args = parser.parse_args()

    if args.source.lower() == "synthetic":
        run_synthetic_source(headless=args.headless)
    else:
        # numeric string -> webcam index; otherwise treat as a file path
        source = int(args.source) if args.source.isdigit() else args.source
        run_real_source(source, headless=args.headless, fps_override=args.fps)


if __name__ == "__main__":
    main()
