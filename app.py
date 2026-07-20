"""
Intelligent Traffic Monitoring System — Streamlit Dashboard (Professional MCA Level)
Run with: streamlit run app.py
"""

import os
import subprocess
import tempfile
import time
import textwrap

import numpy as np
import cv2
import pandas as pd
import streamlit as st
from src.video_processor import TrafficMonitor

st.set_page_config(
    page_title="ITMS - Intelligent Traffic Monitoring System",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# PROFESSIONAL CSS — Glassmorphism + Neon Glow + Animated Elements
# ============================================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Inter:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600&display=swap');

:root {
    --bg-primary: #0a0e17;
    --bg-secondary: #111827;
    --bg-card: rgba(17, 24, 39, 0.85);
    --border-glass: rgba(255, 255, 255, 0.08);
    --text-primary: #f1f5f9;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --accent-cyan: #06b6d4;
    --accent-amber: #f59e0b;
    --accent-green: #10b981;
    --accent-red: #ef4444;
    --accent-purple: #8b5cf6;
    --glow-cyan: 0 0 30px rgba(6, 182, 212, 0.3);
    --glow-amber: 0 0 30px rgba(245, 158, 11, 0.3);
    --glass-bg: rgba(255, 255, 255, 0.03);
    --glass-border: rgba(255, 255, 255, 0.06);
}

/* ---------- Base ---------- */
.stApp {
    background: var(--bg-primary);
    background-image: 
        radial-gradient(ellipse at 20% 50%, rgba(6, 182, 212, 0.05) 0%, transparent 60%),
        radial-gradient(ellipse at 80% 50%, rgba(245, 158, 11, 0.05) 0%, transparent 60%);
}
#MainMenu, footer, header { display: none; }
.block-container { padding-top: 1rem; max-width: 1400px; }

/* ---------- Scrollbar ---------- */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-secondary); }
::-webkit-scrollbar-thumb { background: var(--accent-cyan); border-radius: 10px; }

/* ---------- Glass Card ---------- */
.glass-card {
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--glass-border);
    border-radius: 16px;
    padding: 20px 24px;
    transition: all 0.3s ease;
}
.glass-card:hover {
    border-color: rgba(6, 182, 212, 0.2);
    box-shadow: var(--glow-cyan);
}

/* ---------- Top Navigation ---------- */
.top-nav {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 16px 32px;
    background: var(--glass-bg);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid var(--glass-border);
    border-radius: 16px 16px 0 0;
    margin-bottom: 24px;
}
.brand {
    display: flex;
    align-items: center;
    gap: 16px;
}
.brand-icon {
    width: 48px;
    height: 48px;
    border-radius: 12px;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 24px;
    box-shadow: 0 0 40px rgba(6, 182, 212, 0.3);
}
.brand-text h1 {
    font-family: 'Orbitron', monospace;
    font-weight: 900;
    font-size: 1.4rem;
    background: linear-gradient(135deg, #f1f5f9 0%, #94a3b8 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    line-height: 1.2;
    letter-spacing: 1px;
}
.brand-text .sub {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    color: var(--text-secondary);
    letter-spacing: 3px;
    text-transform: uppercase;
    margin: 0;
    -webkit-text-fill-color: var(--text-secondary);
}
.status-indicator {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 8px 18px;
    border-radius: 30px;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.2);
}
.status-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    background: var(--accent-green);
    animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
    0%, 100% { opacity: 1; transform: scale(1); box-shadow: 0 0 10px rgba(16, 185, 129, 0.5); }
    50% { opacity: 0.5; transform: scale(0.8); box-shadow: 0 0 20px rgba(16, 185, 129, 0.2); }
}
.status-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--accent-green);
    letter-spacing: 1px;
}

/* ---------- Feature Grid ---------- */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 24px;
}
.feature-item {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 14px 18px;
    text-align: center;
    transition: all 0.3s ease;
}
.feature-item:hover {
    border-color: rgba(6, 182, 212, 0.3);
    transform: translateY(-2px);
    box-shadow: var(--glow-cyan);
}
.feature-item .icon { font-size: 22px; display: block; margin-bottom: 4px; }
.feature-item .label { 
    font-family: 'Inter', sans-serif;
    font-size: 0.72rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.feature-item .value {
    font-family: 'Orbitron', monospace;
    font-size: 0.85rem;
    color: var(--text-primary);
    font-weight: 700;
}

/* ---------- Stats Cards ---------- */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    margin-bottom: 16px;
}
.stat-card {
    background: var(--glass-bg);
    backdrop-filter: blur(12px);
    border: 1px solid var(--glass-border);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-card .number {
    font-family: 'Orbitron', monospace;
    font-size: 2.2rem;
    font-weight: 900;
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.stat-card .label {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 2px;
}

/* ---------- Count Row (Vehicle Type) ---------- */
.count-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 14px;
    margin-bottom: 6px;
    border-radius: 8px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    transition: all 0.3s ease;
}
.count-row:hover {
    background: rgba(255,255,255,0.05);
    border-color: rgba(6, 182, 212, 0.15);
}
.count-row .type {
    font-family: 'Inter', sans-serif;
    font-size: 0.85rem;
    color: var(--text-primary);
    display: flex;
    align-items: center;
    gap: 10px;
}
.count-row .type .emoji { font-size: 1.1rem; }
.count-row .num {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent-cyan);
}
.count-bar {
    height: 4px;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-cyan), var(--accent-purple));
    margin-top: 4px;
    transition: width 0.5s ease;
}

/* ---------- Sidebar Styling ---------- */
section[data-testid="stSidebar"] {
    background: rgba(10, 14, 23, 0.95);
    backdrop-filter: blur(20px);
    border-right: 1px solid var(--glass-border);
}
.sidebar-title {
    font-family: 'Orbitron', monospace;
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--text-primary);
    text-align: center;
    padding: 16px 0 8px 0;
    border-bottom: 1px solid var(--glass-border);
    margin-bottom: 16px;
}
.sidebar-section {
    font-family: 'Inter', sans-serif;
    font-size: 0.65rem;
    color: var(--accent-cyan);
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-top: 18px;
    margin-bottom: 6px;
    padding-top: 12px;
    border-top: 1px solid var(--glass-border);
}
.sidebar-note {
    font-family: 'Inter', sans-serif;
    font-size: 0.7rem;
    color: var(--text-muted);
    line-height: 1.5;
    padding: 6px 0;
}
.sidebar-footer {
    font-family: 'Inter', sans-serif;
    font-size: 0.6rem;
    color: var(--text-muted);
    text-align: center;
    padding: 20px 0 10px 0;
    border-top: 1px solid var(--glass-border);
    margin-top: 20px;
    line-height: 1.8;
}

/* ---------- Upload Area ---------- */
[data-testid="stFileUploaderDropzone"] {
    background: rgba(255,255,255,0.02) !important;
    border: 2px dashed rgba(255,255,255,0.08) !important;
    border-radius: 12px !important;
    padding: 20px !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--accent-cyan) !important;
    background: rgba(6, 182, 212, 0.03) !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple)) !important;
    color: #fff !important;
    font-family: 'Orbitron', monospace !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 24px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 30px rgba(6, 182, 212, 0.2) !important;
    letter-spacing: 1px !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 50px rgba(6, 182, 212, 0.4) !important;
}
.stButton > button:active {
    transform: scale(0.98) !important;
}

/* ---------- Metrics ---------- */
[data-testid="stMetric"] {
    background: var(--glass-bg) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid var(--glass-border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
}
[data-testid="stMetricLabel"] {
    color: var(--text-secondary) !important;
    font-size: 0.7rem !important;
    text-transform: uppercase !important;
    letter-spacing: 1px !important;
}
[data-testid="stMetricValue"] {
    color: var(--accent-cyan) !important;
    font-family: 'Orbitron', monospace !important;
    font-size: 1.8rem !important;
    font-weight: 900 !important;
}

/* ---------- Responsive ---------- */
@media (max-width: 768px) {
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
    .stats-grid { grid-template-columns: 1fr 1fr; }
    .top-nav { flex-direction: column; gap: 12px; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================================
# TOP NAVIGATION
# ============================================================================
st.markdown(f"""
<div class="top-nav">
    <div class="brand">
        <div class="brand-icon">🚦</div>
        <div class="brand-text">
            <h1>ITMS</h1>
            <p class="sub">Intelligent Traffic Monitoring System</p>
        </div>
    </div>
    <div class="status-indicator">
        <span class="status-dot"></span>
        <span class="status-text">System Online</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown('<div class="sidebar-title">⚙️ CONTROL CENTER</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">Model Configuration</div>', unsafe_allow_html=True)
    model_path = st.text_input("Weights Path", value="models/itms_yolov8/weights/best.pt")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05,
                            help="Higher = fewer false positives")
    
    st.markdown('<div class="sidebar-section">Speed Calibration</div>', unsafe_allow_html=True)
    pixels_per_meter = st.number_input("Pixels per Meter", value=8.0, step=0.5,
                                       help="Calibrate using a known distance in your video")
    video_fps = st.number_input("Video FPS", value=30.0, step=1.0)
    st.markdown('<div class="sidebar-note">⚡ Speed accuracy depends on correct calibration</div>', unsafe_allow_html=True)
    
    st.markdown('<div class="sidebar-section">Performance</div>', unsafe_allow_html=True)
    frame_skip = st.slider("Frame Skip", 1, 5, 1,
                           help="Higher = faster processing")
    
    st.markdown("""
    <div class="sidebar-footer">
        <strong>MCA / M.Sc (AI &amp; Data Science)</strong><br>
        Major Project · 2025<br>
        YOLOv8 · DeepSORT · Streamlit
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# FEATURE GRID
# ============================================================================
st.markdown("""
<div class="feature-grid">
    <div class="feature-item">
        <span class="icon">🚗</span>
        <div class="label">Detection</div>
        <div class="value">Car · Bike · Bus · Truck</div>
    </div>
    <div class="feature-item">
        <span class="icon">🆔</span>
        <div class="label">Tracking</div>
        <div class="value">Unique ID per Vehicle</div>
    </div>
    <div class="feature-item">
        <span class="icon">🎨</span>
        <div class="label">Color</div>
        <div class="value">Dominant Color Detection</div>
    </div>
    <div class="feature-item">
        <span class="icon">⚡</span>
        <div class="label">Speed</div>
        <div class="value">Real-time km/h</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# UPLOAD SECTION
# ============================================================================
col_upload, col_btn = st.columns([3, 1])
with col_upload:
    uploaded_file = st.file_uploader(
        "Upload Road Video",
        type=["mp4", "avi", "mov"],
        label_visibility="collapsed",
        help="Upload a traffic video for real-time analysis"
    )
with col_btn:
    st.markdown("<br>", unsafe_allow_html=True)
    run_btn = st.button("▶ START MONITORING", use_container_width=True)

# ============================================================================
# MAIN LAYOUT
# ============================================================================
col_video, col_stats = st.columns([2.2, 1])

with col_video:
    st.markdown('<div class="glass-card"><strong>📹 Live Detection Feed</strong></div>', unsafe_allow_html=True)
    video_placeholder = st.empty()

with col_stats:
    st.markdown('<div class="glass-card"><strong>📊 Traffic Statistics</strong></div>', unsafe_allow_html=True)
    stats_placeholder = st.empty()

# ============================================================================
# VEHICLE LOG TABLE
# ============================================================================
st.markdown("---")
st.markdown('<div class="glass-card"><strong>📋 Vehicle Log</strong></div>', unsafe_allow_html=True)
table_placeholder = st.empty()

# ============================================================================
# VIDEO RE-ENCODE HELPER — Browser-Compatible H.264
# ============================================================================
def reencode_to_h264(input_path: str) -> str:
    """
    OpenCV's VideoWriter with the 'mp4v' fourcc produces MPEG-4 Part 2
    encoded video. That is a technically valid .mp4 file, but most
    browsers (Chrome, Edge, Safari) cannot decode it in an HTML5
    <video> tag — they need H.264 (avc1). That's why st.video() shows
    a black player stuck at 0:00 even though the download works fine.

    This function re-encodes the file to H.264 + AAC using ffmpeg so
    it plays correctly inline. If ffmpeg is not available or the
    re-encode fails, it falls back to the original file (download will
    still work even if inline playback doesn't).
    """
    output_path = tempfile.NamedTemporaryFile(delete=False, suffix="_h264.mp4").name
    try:
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", input_path,
                "-vcodec", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "fast",
                "-crf", "23",
                "-movflags", "+faststart",
                "-acodec", "aac",
                output_path
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        return output_path
    except (subprocess.CalledProcessError, FileNotFoundError):
        # ffmpeg missing or failed — fall back to original file
        return input_path


# ============================================================================
# RENDER STATS FUNCTION — SIRF VEHICLE CLASSES
# ============================================================================
def render_stats(total: int, counts: dict):
    """
    Render professional stats with bars - ONLY VEHICLE CLASSES.
    """
    # SIRF YEH CLASSES DIKHENGE (Vehicle classes)
    VEHICLE_MAP = {
        "car": "🚗",
        "bus": "🚌",
        "truck": "🚛",
        "bike": "🏍️",
        "motorcycle": "🏍️",
        "2-wheelers": "🏍️",
        "auto": "🛺",
        "pedestrian": "🚶",
        "person": "🚶"
    }

    # Sirf vehicle classes filter karo
    filtered_counts = {}
    for name, val in counts.items():
        name_lower = name.lower()
        for v_key in VEHICLE_MAP.keys():
            if v_key in name_lower or name_lower in v_key:
                display_name = name.title()
                filtered_counts[display_name] = val
                break

    if not filtered_counts:
        filtered_counts = {}

    items = sorted(filtered_counts.items(), key=lambda x: -x[1])
    max_count = max([v for _, v in items], default=1) or 1

    rows_html = ""
    if not items or all(v == 0 for _, v in items):
        rows_html = '<div style="text-align:center; color:#64748b; padding:20px 0;">No vehicles detected yet</div>'
    else:
        for name, val in items:
            if val == 0:
                continue
            pct = (val / max_count) * 100 if max_count else 0
            icon = "🚘"
            for key, emoji in VEHICLE_MAP.items():
                if key in name.lower():
                    icon = emoji
                    break
            row = f"""
            <div class="count-row">
                <span class="type"><span class="emoji">{icon}</span> {name}</span>
                <span class="num">{val}</span>
            </div>
            <div class="count-bar" style="width:{pct:.1f}%;"></div>
            """
            rows_html += textwrap.dedent(row)

    html = f"""
    <div style="margin-bottom: 12px;">
        <div style="display: flex; justify-content: space-between; align-items: baseline;">
            <span style="font-family: 'Inter', sans-serif; font-size: 0.7rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px;">
                Total Vehicles
            </span>
            <span style="font-family: 'Orbitron', monospace; font-size: 2.4rem; font-weight: 900; color: #06b6d4;">
                {total}
            </span>
        </div>
        <div style="height: 2px; background: linear-gradient(90deg, #06b6d4, #8b5cf6); border-radius: 2px; margin: 8px 0 14px 0;"></div>
        {rows_html}
    </div>
    """
    return textwrap.dedent(html)


# ============================================================================
# PROCESSING LOGIC
# ============================================================================
if run_btn:
    if uploaded_file is None:
        st.warning("⚠️ Please upload a video file first.")
    elif not os.path.exists(model_path):
        st.error(f"❌ Model not found at: {model_path}")
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded_file.read())
        tfile.close()
        video_path = tfile.name

        try:
            monitor = TrafficMonitor(
                model_path=model_path,
                conf_thresh=conf_thresh,
                pixels_per_meter=pixels_per_meter,
                fps=video_fps
            )
        except Exception as e:
            st.error(f"❌ Failed to initialize TrafficMonitor: {e}")
            st.stop()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("❌ Failed to open video file.")
            st.stop()

        frame_count = 0

        # Video writer for output
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        output_fps = max(video_fps / frame_skip, 1) if frame_skip > 0 else video_fps
        output_video_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_video_path, fourcc, output_fps, (frame_w, frame_h))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            try:
                annotated, summary = monitor.process_frame(frame)
            except Exception as e:
                st.error(f"⚠️ Error processing frame {frame_count}: {e}")
                continue

            # Write annotated video
            try:
                if annotated is not None and isinstance(annotated, np.ndarray) and annotated.size > 0:
                    writer.write(annotated)
            except Exception:
                pass

            # Display image with safe check
            display_success = False
            try:
                if annotated is not None and isinstance(annotated, np.ndarray) and annotated.size > 0:
                    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(annotated_rgb, channels="RGB", use_container_width=True)
                    display_success = True
            except Exception:
                pass

            if not display_success:
                try:
                    if frame is not None and isinstance(frame, np.ndarray) and frame.size > 0:
                        video_placeholder.image(frame, channels="BGR", use_container_width=True)
                        display_success = True
                except Exception:
                    pass

            if not display_success:
                video_placeholder.empty()

            # Update stats
            try:
                stats_placeholder.markdown(
                    render_stats(summary.get("total", 0), summary.get("counts", {})),
                    unsafe_allow_html=True
                )
            except Exception:
                stats_placeholder.markdown(
                    render_stats(0, {}),
                    unsafe_allow_html=True
                )

            # Update log table
            try:
                df = monitor.get_log_dataframe()
                if not df.empty:
                    table_placeholder.dataframe(
                        df.sort_values("ID"),
                        use_container_width=True,
                        height=350,
                        key=f"log_{frame_count}"
                    )
            except Exception:
                pass

            time.sleep(0.01)

        cap.release()
        writer.release()

        try:
            os.unlink(video_path)
        except (PermissionError, FileNotFoundError):
            pass

        st.success("✅ Processing complete!")

        # Show output video
        st.markdown("### 🎬 Annotated Video Output")
        if os.path.exists(output_video_path) and os.path.getsize(output_video_path) > 0:
            playable_path = reencode_to_h264(output_video_path)

            with open(playable_path, "rb") as vf:
                video_bytes = vf.read()

            try:
                os.unlink(output_video_path)
            except (PermissionError, FileNotFoundError):
                pass
            if playable_path != output_video_path:
                try:
                    os.unlink(playable_path)
                except (PermissionError, FileNotFoundError):
                    pass

            st.video(video_bytes)
            st.download_button(
                "⬇ Download Annotated Video",
                video_bytes,
                "traffic_analysis.mp4",
                "video/mp4"
            )

        # CSV Report
        try:
            final_df = monitor.get_log_dataframe()
            if not final_df.empty:
                csv = final_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download CSV Report",
                    csv,
                    "traffic_report.csv",
                    "text/csv"
                )
        except Exception:
            pass

else:
    # Idle state
    stats_placeholder.markdown(
        render_stats(0, {}),
        unsafe_allow_html=True
    )
    with video_placeholder.container():
        st.markdown("""
        <div style="
            background: rgba(255,255,255,0.03);
            backdrop-filter: blur(12px);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 16px;
            padding: 80px 20px;
            text-align: center;
            color: #64748b;
        ">
            <div style="font-size: 4rem; margin-bottom: 16px;">🚦</div>
            <div style="font-family: 'Inter', sans-serif; font-size: 1.1rem; font-weight: 600; color: #94a3b8;">
                Ready for Analysis
            </div>
            <div style="font-family: 'Inter', sans-serif; font-size: 0.85rem; color: #64748b; margin-top: 8px;">
                Upload a road video and click <strong>Start Monitoring</strong>
            </div>
        </div>
        """, unsafe_allow_html=True)