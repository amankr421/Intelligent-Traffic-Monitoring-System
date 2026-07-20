"""
DeepSORT Tracker Wrapper
Wraps deep_sort_realtime to assign persistent unique IDs to detected vehicles
across frames.
"""

from deep_sort_realtime.deepsort_tracker import DeepSort


class VehicleTracker:
    def __init__(self, max_age: int = 30, n_init: int = 3):
        self.tracker = DeepSort(
            max_age=max_age,      # frames to keep a lost track alive
            n_init=n_init,        # frames needed to confirm a new track
            nms_max_overlap=1.0,
            embedder="mobilenet", # built-in appearance embedder for re-ID
            half=True,
            bgr=True,
        )

    def update(self, detections, frame):
        """
        Args:
            detections: list of ([x, y, w, h], confidence, class_name)
            frame: current video frame (BGR numpy array)
        Returns:
            list of confirmed tracks, each with .track_id, .to_ltrb(), .get_det_class()
        """
        tracks = self.tracker.update_tracks(detections, frame=frame)
        confirmed = [t for t in tracks if t.is_confirmed()]
        return confirmed
