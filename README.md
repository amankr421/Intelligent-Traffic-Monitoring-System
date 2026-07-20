# Intelligent Traffic Monitoring System

**MCA / M.Sc (AI & Data Science) Major Project**

An AI-based system that detects, tracks, and counts vehicles from road videos, and additionally
estimates each vehicle's color and speed — displayed on a real-time Streamlit dashboard.

---

## 1. Problem Statement

Traffic authorities currently rely on manual observation or basic loop-sensors to monitor
road traffic, which is expensive, error-prone, and not scalable. This project builds an
automated, camera-based AI system that:

- Detects vehicles in real time from CCTV/road video footage
- Classifies vehicles into **Car, Bike, Bus, Truck**
- Counts vehicles (total and type-wise) for congestion analysis
- Assigns a unique, persistent ID to every vehicle so it is counted only once
- Estimates each vehicle's dominant color
- Estimates each vehicle's speed (km/h)
- Presents all of this on a live monitoring dashboard

---

## 2. Dataset

- **Source:** Roboflow — "Vehicle_Detection Computer Vision Model" by workspace `ProjectTraffic`
- **Version:** v11 (YOLOv8 format)
- **Size:** 439,902 images, 640×640 resolution
- **Classes (4):** `Car`, `Bike`, `Bus`, `Truck`

Download it via `download_dataset.py` (needs a free Roboflow API key):

```bash
python download_dataset.py --api_key YOUR_ROBOFLOW_API_KEY
```

This places the dataset under `data/Vehicle_Detection/` in YOLOv8 folder structure
(`train/`, `valid/`, `test/`, each with `images/` and `labels/`), matching `data/data.yaml`.

> If Roboflow renames the workspace/project slug, update it inside `download_dataset.py`
> (`rf.workspace(...).project(...)`), or download the dataset manually from the Roboflow
> Universe page in "YOLOv8" export format and place it at the same path.

---

## 3. Model & Approach

| Component | Technology |
|---|---|
| Detection | YOLOv8 (Ultralytics),fine-tuned via transfer learning |
| Tracking / Unique ID | DeepSORT (`deep-sort-realtime`) |
| Color Detection | OpenCV (HSV space) + KMeans clustering |
| Speed Estimation | Pixel-displacement / calibrated pixels-per-meter + FPS |
| Dashboard | Streamlit + Plotly |

### Why transfer learning?
Instead of training YOLOv8 from scratch.
(`yolov8n.pt`) which already understand generic object shapes/edges, then fine-tune
only on our 4-class vehicle dataset. This drastically reduces training time and the
amount of data needed to reach good accuracy.

---

## 4. Project Structure

```
ITMS/
├── data/
│   └── data.yaml                # YOLOv8 dataset config
├── models/                      # trained weights land here after train.py
├── src/
│   ├── color_detector.py        # HSV + KMeans dominant color detection
│   ├── speed_estimator.py       # pixel-displacement based speed (km/h)
│   ├── tracker.py               # DeepSORT wrapper (unique vehicle IDs)
│   └── video_processor.py       # combines detection + tracking + color + speed + counting
├── app.py                       # Streamlit real-time dashboard
├── train.py                     # YOLOv8 transfer-learning training script
├── download_dataset.py          # pulls dataset from Roboflow
├── requirements.txt
└── README.md
```

---

## 4a. Placeholder Model Included

This project ships with a **placeholder `best.pt`** at
`models/itms_yolov8/weights/best.pt`
It lets you run the full pipeline and Streamlit dashboard immediately, even before
training — it just won't detect your project's 4 specific classes accurately until
you replace it with your own trained weights (see `RUN_GUIDE.md`).

> 📘 **For detailed step-by-step instructions** — from first local run, to test
> video requirements, to deploying on Streamlit Community Cloud, plus a
> troubleshooting table of errors — see **`RUN_GUIDE.md`** in this project.

## 5. Setup & Installation

```bash
# 1. Create environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt
```

`requirements.txt` includes:
```
ultralytics, opencv-python, streamlit, numpy, pandas,
deep-sort-realtime, scikit-learn, roboflow, Pillow, plotly
```

---

## 6. Step-by-Step Usage

### Step 1 — Download the dataset
```bash
python download_dataset.py --api_key YOUR_API_KEY
```

### Step 2 — Train YOLOv8 (transfer learning)
```bash
python train.py --data data/data.yaml --epochs 50 --imgsz 640 --batch 16 --model yolov8n.pt --device 0
```
- Use `--device cpu` if no GPU is available (training will be much slower).
- Best weights are saved to `models/itms_yolov8/weights/best.pt`.
- Given the dataset size (~440K images), consider first prototyping on a subset
  or fewer epochs, then scaling up once the pipeline is verified end-to-end.

### Step 3 — Run the dashboard
```bash
streamlit run app.py
```
1. In the sidebar, set the path to `best.pt` (defaults to the standard training output path).
2. Calibrate **pixels-per-meter** using a known real-world reference visible in your camera
   (e.g., lane width ≈ 3.5 m) for accurate speed readings.
3. Upload a road video (`.mp4`/`.avi`/`.mov`).
4. Click **Start Monitoring** to see live detection, tracking, counts, and the vehicle log.
5. Download the full report as CSV at the end.

---

## 7. How Each Feature Works

- **Detection:** YOLOv8 predicts bounding boxes + class per frame.
- **Tracking / Unique ID:** DeepSORT matches detections across frames using motion +
  appearance embeddings, giving each vehicle a stable `track_id` — this is what allows
  correct (non-duplicate) counting.
- **Counting:** A vehicle is added to the total/type-wise count the first time its
  `track_id` is seen; later frames only update its speed, not the count.
- **Color Detection:** The vehicle's bounding-box crop is converted to HSV, KMeans finds
  the dominant color cluster (ignoring shadow/glare pixels), which is mapped to a named color.
- **Speed Estimation:** Each track's pixel centroid is recorded per frame; displacement over
  a frame window is converted to meters (via the calibration factor) and divided by elapsed
  time (via FPS) to get km/h, with light smoothing across frames.

---

## 8. Notes & Possible Extensions

- Speed accuracy depends entirely on correct camera calibration (pixels-per-meter) and a
  roughly top-down/fixed camera angle — for angled cameras, a homography-based
  perspective transform would give more accurate results.
- Add congestion-level logic (e.g., vehicles/min threshold → Low/Medium/High).
- Add multi-camera / multi-lane support.
- Deploy `app.py` behind a lightweight backend for live RTSP camera feeds instead of
  uploaded video files.
