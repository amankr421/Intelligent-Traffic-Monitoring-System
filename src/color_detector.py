"""
Color Detection Module
Detects the dominant color of a vehicle crop using HSV color space + KMeans clustering.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans

# HSV ranges for common vehicle colors
COLOR_RANGES = {
    "Red":    [(0, 70, 50), (10, 255, 255)],
    "Red2":   [(170, 70, 50), (180, 255, 255)],  # red wraps around hue circle
    "Orange": [(11, 70, 50), (25, 255, 255)],
    "Yellow": [(26, 70, 50), (34, 255, 255)],
    "Green":  [(35, 40, 40), (85, 255, 255)],
    "Blue":   [(86, 40, 40), (128, 255, 255)],
    "White":  [(0, 0, 200), (180, 40, 255)],
    "Gray":   [(0, 0, 70), (180, 40, 200)],
    "Black":  [(0, 0, 0), (180, 255, 60)],
}


def _closest_named_color(hsv_pixel):
    h, s, v = hsv_pixel
    best_name, best_dist = "Unknown", float("inf")
    for name, (lo, hi) in COLOR_RANGES.items():
        lo = np.array(lo)
        hi = np.array(hi)
        if np.all(hsv_pixel >= lo) and np.all(hsv_pixel <= hi):
            return name.replace("Red2", "Red")
        # fallback: distance to range center
        center = (lo.astype(int) + hi.astype(int)) / 2
        dist = np.linalg.norm(np.array([h, s, v]) - center)
        if dist < best_dist:
            best_dist = dist
            best_name = name.replace("Red2", "Red")
    return best_name


def detect_vehicle_color(vehicle_crop_bgr, k=3):
    """
    Args:
        vehicle_crop_bgr: cropped image (numpy array, BGR) of a single vehicle
        k: number of KMeans clusters to find dominant colors
    Returns:
        color_name (str)
    """
    if vehicle_crop_bgr is None or vehicle_crop_bgr.size == 0:
        return "Unknown"

    # Resize for speed and focus on the central region (avoids background/road pixels)
    img = cv2.resize(vehicle_crop_bgr, (100, 100))
    h, w = img.shape[:2]
    cx0, cx1 = int(w * 0.25), int(w * 0.75)
    cy0, cy1 = int(h * 0.25), int(h * 0.75)
    center_crop = img[cy0:cy1, cx0:cx1]

    hsv = cv2.cvtColor(center_crop, cv2.COLOR_BGR2HSV)
    pixels = hsv.reshape(-1, 3)

    # Remove near-black shadow / near-white glare noise pixels that skew clustering
    mask = (pixels[:, 2] > 20) & (pixels[:, 2] < 250)
    filtered = pixels[mask] if mask.sum() > 10 else pixels

    kmeans = KMeans(n_clusters=min(k, len(filtered)), n_init=5, random_state=42)
    kmeans.fit(filtered)

    # Most frequent cluster = dominant color
    counts = np.bincount(kmeans.labels_)
    dominant_hsv = kmeans.cluster_centers_[np.argmax(counts)]

    return _closest_named_color(dominant_hsv)
