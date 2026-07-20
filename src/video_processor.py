"""
Core video processing pipeline for the Intelligent Traffic Monitoring System.

Combines:
- YOLOv8 vehicle detection (Car, Bike, Bus, Truck)
- DeepSORT tracking (unique ID per vehicle)
- Color detection (OpenCV + KMeans)
- Speed estimation (km/h)
- Total + type-wise vehicle counting
"""

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO

from src.tracker import VehicleTracker
from src.color_detector import detect_vehicle_color
from src.speed_estimator import SpeedEstimator

# Fallback palette cycled through if a class doesn't have an explicit color below
_PALETTE = [
    (255, 128, 0), (0, 200, 255), (0, 0, 255), (255, 0, 255),
    (0, 255, 0), (255, 255, 0), (128, 0, 255), (0, 128, 255),
]

# Preferred colors for the common vehicle classes (used if the class name matches)
_PREFERRED_COLORS = {
    "Car": (255, 128, 0),
    "Bike": (0, 200, 255),
    "Motorcycle": (0, 200, 255),
    "Bus": (0, 0, 255),
    "Truck": (255, 0, 255),
}


class TrafficMonitor:
    def __init__(self, model_path: str = "models/itms_yolov8/weights/best.pt",
                 conf_thresh: float = 0.4, pixels_per_meter: float = 8.0, fps: float = 30.0):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh
        self.tracker = VehicleTracker()
        self.speed_estimator = SpeedEstimator(pixels_per_meter=pixels_per_meter, fps=fps)

        # IMPORTANT: class names are read directly from the loaded model (model.names),
        # not hardcoded. This means the app automatically adapts to whatever classes
        # your trained best.pt actually has, even if that differs from ["Car","Bike","Bus","Truck"].
        self.class_names = list(self.model.names.values())
        self.box_colors = {
            name: _PREFERRED_COLORS.get(name, _PALETTE[i % len(_PALETTE)])
            for i, name in enumerate(self.class_names)
        }

        self.counted_ids = set()
        self.counts = {c: 0 for c in self.class_names}
        self.vehicle_log = []  # rows for the dashboard table
        self.frame_idx = 0

    def process_frame(self, frame: np.ndarray):
        """
        Runs detection -> tracking -> color/speed -> annotation on a single frame.
        Returns:
            annotated_frame (np.ndarray), frame_summary (dict)
        """
        self.frame_idx += 1
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            class_name = self.model.names.get(cls_id, f"class_{cls_id}")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, class_name))

        tracks = self.tracker.update(detections, frame)

        for track in tracks:
            track_id = track.track_id
            class_name = track.get_det_class() or "Car"
            l, t, r, b = map(int, track.to_ltrb())
            l, t = max(l, 0), max(t, 0)
            r, b = min(r, frame.shape[1]), min(b, frame.shape[0])

            crop = frame[t:b, l:r]
            color_name = detect_vehicle_color(crop)

            centroid = ((l + r) / 2, (t + b) / 2)
            speed_kmh = self.speed_estimator.update(track_id, centroid, self.frame_idx)

            # Count each unique vehicle ID only once
            if track_id not in self.counted_ids:
                self.counted_ids.add(track_id)
                self.counts[class_name] = self.counts.get(class_name, 0) + 1
                self.vehicle_log.append({
                    "ID": track_id,
                    "Type": class_name,
                    "Color": color_name,
                    "Speed (km/h)": speed_kmh,
                    "Frame First Seen": self.frame_idx,
                })
            else:
                # keep the log's speed value updated for this ID
                for row in self.vehicle_log:
                    if row["ID"] == track_id:
                        row["Speed (km/h)"] = speed_kmh
                        break

            # Draw annotation
            box_color = self.box_colors.get(class_name, (0, 255, 0))
            cv2.rectangle(frame, (l, t), (r, b), box_color, 2)
            label = f"ID{track_id} {class_name} {color_name} {speed_kmh}km/h"
            cv2.putText(frame, label, (l, max(t - 8, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)

        summary = {
            "total": len(self.counted_ids),
            "counts": dict(self.counts),
        }
        return frame, summary

    def get_log_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.vehicle_log)
