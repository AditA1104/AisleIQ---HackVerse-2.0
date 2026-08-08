import cv2
import numpy as np
from ultralytics import YOLO
import time
import json

def run_store_tracker(video_source=0):
    model = YOLO("yolov8n.pt")
    cap = cv2.VideoCapture(video_source)
    
    # Define monitored store zone polygon coordinates
    zone_pts = np.array([[50, 50], [590, 50], [590, 430], [50, 430]], np.int32)
    
    # Dictionary to track dwell times: {track_id: enter_timestamp}
    zone_timers = {}
    DWELL_THRESHOLD_SECONDS = 5  # Alert trigger threshold

    print("Starting AisleIQ Tracker... Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or camera error.")
            break

        # Resize frame for smooth performance
        frame = cv2.resize(frame, (640, 480))
        
        # 1. Flip the frame horizontally so it acts like a mirror
        frame = cv2.flip(frame, 1)

        # Run YOLOv8 tracking (persists unique IDs across frames)
        results = model.track(frame, persist=True, classes=[0], verbose=False)
        
        # Draw the target store zone on the frame
        cv2.polylines(frame, [zone_pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(frame, "Monitored Aisle Zone", (105, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        active_count = 0
        current_time = time.time()
        
        # Keep track of IDs seen inside the zone in this current frame
        current_ids_in_zone = set()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy()
            active_count = len(track_ids)

            for box, track_id in zip(boxes, track_ids):
                t_id = int(track_id)
                x1, y1, x2, y2 = map(int, box)
                
                # Use the true center of the box (easier to trigger than feet)
                center_x = int((x1 + x2) / 2)
                center_y = int((y1 + y2) / 2)

                is_inside = cv2.pointPolygonTest(zone_pts, (center_x, center_y), False) >= 0

                if is_inside:
                    current_ids_in_zone.add(t_id)
                    
                    if t_id not in zone_timers:
                        zone_timers[t_id] = current_time
                    
                    dwell_time = current_time - zone_timers[t_id]
                    display_seconds = int(dwell_time)
                    
                    color = (0, 0, 255) if display_seconds >= DWELL_THRESHOLD_SECONDS else (0, 255, 0)
                    cv2.putText(frame, f"ID: {t_id} | Dwell: {display_seconds}s", (x1, max(y1 - 10, 20)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    
                    if display_seconds == DWELL_THRESHOLD_SECONDS:
                        print(f"[ALERT] Customer #{t_id} lingering in Zone for {display_seconds}s!")
                else:
                    if t_id in zone_timers:
                        del zone_timers[t_id]

                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Remove timers for IDs that have left the zone
        expired_ids = [t_id for t_id in zone_timers if t_id not in current_ids_in_zone]
        for t_id in expired_ids:
            del zone_timers[t_id]

        # Export live metrics for Streamlit dashboard
        dashboard_data = {
            "active_shoppers": int(active_count),
            "alerts_triggered": len(zone_timers)
        }
        
        with open("live_metrics.json", "w") as f:
            json.dump(dashboard_data, f)

        # Display the live window
        cv2.imshow("AisleIQ - OpenCV Tracking Feed", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_store_tracker(0)