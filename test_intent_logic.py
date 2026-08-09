from collections import deque
import numpy as np

class IntentEngine:
    def __init__(self, buffer_size=100, min_move_px=5.0):
        # Stores recent (x, y) centroids per track_id
        self.history = {}
        self.buffer_size = buffer_size
        self.min_move_px = min_move_px

    def evaluate_shopper(self, track_id, current_centroid, dwell_time):
        """
        Evaluates Dual-Vector Intent (Pacing Ratio R + Dwell Time T).
        Returns: (classification, friction_score, is_alert_triggered)
        """
        if track_id not in self.history:
            self.history[track_id] = deque(maxlen=self.buffer_size)

        self.history[track_id].append(current_centroid)
        pts = np.array(self.history[track_id])

        # Require at least 10 history points to evaluate trajectory
        if len(pts) < 10:
            return "Gathering Data", 0.0, False

        # 1. Total Path Length (D) with noise filtering
        step_vectors = np.diff(pts, axis=0)
        step_distances = np.linalg.norm(step_vectors, axis=1)
        # Filter out micro-jitter/noise under min_move_px
        filtered_distances = step_distances[step_distances >= self.min_move_px]
        total_path = float(np.sum(filtered_distances))

        # 2. Net Displacement (N)
        net_displacement = float(np.linalg.norm(pts[-1] - pts[0]))

        # 3. Pacing Ratio (R)
        pacing_ratio = total_path / (net_displacement + 1e-5)

        # 4. Friction Score
        friction_score = dwell_time * pacing_ratio

        # 5. Dual-Vector Classification Matrix
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


def run_test_suite():
    engine = IntentEngine()
    print("==================================================")
    print("    AISLEIO DUAL-VECTOR INTENT ENGINE TEST        ")
    print("==================================================\n")

    # --- Scenario 1: Passerby walking straight past aisle ---
    print("--- Test 1: Straight Passerby ---")
    straight_line = [(100 + i * 10, 200) for i in range(25)]  # Moves 10px per frame
    for frame_idx, pt in enumerate(straight_line):
        dwell = frame_idx * 0.2  # 0.2 seconds per frame
        status, score, alert = engine.evaluate_shopper(track_id=1, current_centroid=pt, dwell_time=dwell)
    print(f"Result -> Status: '{status}' | Score: {score} | Alert Triggered: {alert}\n")

    # --- Scenario 2: Relaxed Browser (Standing still looking at phone) ---
    print("--- Test 2: Relaxed Browser (Low Motion, High Dwell) ---")
    stationary = [(300 + np.random.randint(-1, 2), 200 + np.random.randint(-1, 2)) for _ in range(40)]
    for frame_idx, pt in enumerate(stationary):
        dwell = 25.0 + (frame_idx * 0.2)  # Dwell = ~25s
        status, score, alert = engine.evaluate_shopper(track_id=2, current_centroid=pt, dwell_time=dwell)
    print(f"Result -> Status: '{status}' | Score: {score} | Alert Triggered: {alert}\n")

    # --- Scenario 3: Active Pacing (Oscillating back and forth) ---
    print("--- Test 3: Confused Pacing (High Path, Low Displacement) ---")
    pacing_points = []
    # Moves back and forth between x=200 and x=260
    for cycle in range(5):
        pacing_points.extend([(200 + i * 10, 300) for i in range(6)])
        pacing_points.extend([(260 - i * 10, 300) for i in range(6)])
    for frame_idx, pt in enumerate(pacing_points):
        dwell = 15.0 + (frame_idx * 0.3)  # Dwell = ~18s
        status, score, alert = engine.evaluate_shopper(track_id=3, current_centroid=pt, dwell_time=dwell)
    print(f"Result -> Status: '{status}' | Score: {score} | Alert Triggered: {alert}\n")

    # --- Scenario 4: Frozen Choice Paralysis (Standing still for >30s in front of shelf) ---
    print("--- Test 4: Frozen Choice Paralysis (Long Dwell, Micro-Shifts) ---")
    frozen_points = [(400 + np.random.randint(-1, 2), 150 + np.random.randint(-1, 2)) for _ in range(30)]
    for frame_idx, pt in enumerate(frozen_points):
        dwell = 32.0 + (frame_idx * 0.2)  # Dwell = >30s
        status, score, alert = engine.evaluate_shopper(track_id=4, current_centroid=pt, dwell_time=dwell)
    print(f"Result -> Status: '{status}' | Score: {score} | Alert Triggered: {alert}\n")

if __name__ == "__main__":
    run_test_suite()