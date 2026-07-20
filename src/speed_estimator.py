"""
Speed Estimation Module
Estimates vehicle speed (km/h) using pixel displacement between frames,
converted to real-world distance via a pixels-per-meter calibration factor.

NOTE: For accurate real-world speed, calibrate PIXELS_PER_METER for your
specific camera angle/height (e.g. using known lane width or road markings).
"""

import time
from collections import defaultdict, deque


class SpeedEstimator:
    def __init__(self, pixels_per_meter: float = 8.0, fps: float = 30.0, smoothing_window: int = 5):
        """
        Args:
            pixels_per_meter: calibration factor (pixels that represent 1 real-world meter)
            fps: video frames per second
            smoothing_window: number of past positions to smooth speed estimate
        """
        self.pixels_per_meter = pixels_per_meter
        self.fps = fps
        self.smoothing_window = smoothing_window
        self.track_history = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.speed_cache = {}

    def update(self, track_id: int, centroid: tuple, frame_idx: int):
        """
        Call once per frame per tracked vehicle.
        Args:
            track_id: unique DeepSORT track id
            centroid: (x, y) center of the bounding box in pixels
            frame_idx: current frame number
        Returns:
            speed_kmh (float)
        """
        self.track_history[track_id].append((centroid, frame_idx))

        if len(self.track_history[track_id]) < 2:
            self.speed_cache[track_id] = 0.0
            return 0.0

        (x1, y1), f1 = self.track_history[track_id][0]
        (x2, y2), f2 = self.track_history[track_id][-1]

        frame_gap = max(f2 - f1, 1)
        pixel_dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5

        meters = pixel_dist / self.pixels_per_meter
        seconds = frame_gap / self.fps
        speed_mps = meters / seconds if seconds > 0 else 0.0
        speed_kmh = speed_mps * 3.6

        # Simple smoothing with previous estimate
        prev = self.speed_cache.get(track_id, speed_kmh)
        smoothed = 0.6 * speed_kmh + 0.4 * prev
        self.speed_cache[track_id] = smoothed

        return round(smoothed, 1)

    def get_speed(self, track_id: int) -> float:
        return round(self.speed_cache.get(track_id, 0.0), 1)
