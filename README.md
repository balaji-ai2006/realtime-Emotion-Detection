# Real-Time Emotion Detection Engine (Web, PC & Raspberry Pi) 🚀

A high-performance, real-time Computer Vision system designed to run on the **Web (Streamlit Cloud)**, **Desktop PCs**, and low-power **Raspberry Pi** boards (Raspberry Pi 3, 4, 5, Zero 2W).

---

## 🌐 Live Web Demo

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://realtime-emotion-detection-gemude956oggvwyj25huee.streamlit.app)

👉 **Try it Live here:** [https://realtime-emotion-detection-gemude956oggvwyj25huee.streamlit.app](https://realtime-emotion-detection-gemude956oggvwyj25huee.streamlit.app)

---

## 🌟 Key Features & Optimizations

* **🌐 Interactive Streamlit Web App:**
  * **Live WebRTC Camera:** Real-time video stream right inside any web browser.
  * **Snapshot Camera:** Instant photo capture & facial dynamics analysis.
  * **Image Upload:** Upload pictures to inspect ROI variance and eyebrow/mouth metrics.
  * **Dynamic Sensitivity Sliders:** Adjust emotion thresholds live in the browser.
* **🍓 Raspberry Pi Ultra-Fast Engine:**
  * Uses downscaled grayscale processing (`--scale 0.5`) for Haar Cascade detection, achieving **25–30+ FPS** on Raspberry Pi.
  * Default resolution optimized to **640x480** to minimize CPU and RAM overhead.
* **💡 Illumination-Invariant Facial Analysis:**
  * Uses **CLAHE (Contrast Limited Adaptive Histogram Equalization)** on facial ROIs, making emotion detection robust against shadows and uneven lighting.
* **🎯 Temporal Emotion Smoothing:**
  * Built-in frame buffer (`deque`) removes jitter and prevents flickering between emotions.
* **📊 Dynamic Bounding Box & UI Overlay:**
  * Smart label rendering prevents clipping at screen borders.
  * Real-time metrics dashboard: **FPS, CPU Usage, Face Count**.

---

## 🛠️ Requirements & Tech Stack

* **Language:** Python 3.8+
* **Dependencies:**
  * `streamlit`
  * `streamlit-webrtc`
  * `opencv-python-headless` (or `opencv-python`)
  * `numpy`
  * `psutil`
  * `av`
  * `Pillow`

---

## 💻 Local Quick Start

### 1. Run the Web Application locally:
```bash
streamlit run app.py
```

### 2. Run Desktop/Raspberry Pi Engine:
```bash
# On PC:
python main.py

# On Raspberry Pi:
python3 main.py

# Headless on Raspberry Pi (via SSH without monitor):
python3 main.py --headless
```

---

## ⚙️ CLI Options for `main.py`

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--camera` | `0` | Camera device index (`0`, `1`, etc.) |
| `--width` | `640` | Video frame width (640 recommended for Raspberry Pi) |
| `--height` | `480` | Video frame height (480 recommended for Raspberry Pi) |
| `--scale` | `0.5` | Detection scale factor (downscales frame for 4x faster detection) |
| `--headless`| `False` | Run in background/terminal without opening a GUI window |