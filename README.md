# Real-Time Emotion Detection Engine (PC & Raspberry Pi Optimized) 🚀

A lightweight, real-time Computer Vision system designed to run smoothly on both **Desktop PCs** and low-power **Raspberry Pi** boards (Raspberry Pi 3, 4, 5, Zero 2W).

---

## 🌟 Key Features & Optimizations

* **Raspberry Pi Ultra-Fast Engine:**
  * Uses downscaled grayscale processing (`--scale 0.5`) for Haar Cascade detection, achieving **25–30+ FPS** on Raspberry Pi.
  * Default resolution optimized to **640x480** to minimize CPU and RAM overhead.
* **Illumination-Invariant Facial Analysis:**
  * Uses **CLAHE (Contrast Limited Adaptive Histogram Equalization)** on facial ROIs, making emotion detection robust against shadows and uneven lighting.
* **Temporal Emotion Smoothing:**
  * Built-in frame buffer (`deque`) removes jitter and prevents flickering between emotions.
* **Dynamic Bounding Box & UI Overlay:**
  * Smart label rendering prevents clipping at screen borders.
  * Real-time metrics dashboard: **FPS, CPU Usage, Face Count**.
* **Headless Mode Support:**
  * Run remotely on Raspberry Pi via SSH without an attached display using `--headless`.

---

## 🛠️ Requirements & Tech Stack

* **Language:** Python 3.7+
* **Dependencies:**
  * `opencv-python`
  * `numpy`
  * `psutil`

---

## 💻 Installation & Quick Start

### 1. On Windows / Linux / macOS (PC)

```bash
# Clone the repository
git clone https://github.com/balaji-ai2006/realtime-Emotion-Detection.git
cd realtime-Emotion-Detection

# Install dependencies
pip install -r requirements.txt

# Run the detection engine
python main.py
```

### 2. On Raspberry Pi (Raspberry Pi OS)

```bash
# 1. Update system packages
sudo apt update && sudo apt install -y python3-opencv python3-numpy python3-psutil

# 2. Navigate to project folder
cd realtime-Emotion-Detection

# 3. Run the application
python3 main.py

# Optional: Run in headless mode (if connected via SSH without a monitor)
python3 main.py --headless
```

---

## ⚙️ CLI Options & Arguments

| Argument | Default | Description |
| :--- | :--- | :--- |
| `--camera` | `0` | Camera device index (`0`, `1`, etc.) |
| `--width` | `640` | Video frame width (640 recommended for Raspberry Pi) |
| `--height` | `480` | Video frame height (480 recommended for Raspberry Pi) |
| `--scale` | `0.5` | Detection scale factor (downscales frame for 4x faster detection) |
| `--headless`| `False` | Run in background/terminal without opening a GUI window |

---

## 🎮 Exit the Program
* Press **`q`** inside the camera window or **`Ctrl + C`** in your terminal.