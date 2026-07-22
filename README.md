# Real-Time AI Emotion Detection Engine 🚀

A high-performance, real-time Computer Vision system that detects facial expressions and maps structural emotional states dynamically using OpenCV and edge-variance profiling.

---

## 🌟 Key Features

* *Real-time Facial Tracking:* Leverages Haar Cascade Classifiers optimized with scale-factor thresholds to reduce false positives.
* *Edge & Variance Feature Extraction:* Analyzes region-of-interest (ROI) dynamics (Mouth and Eyebrow regions) using Canny Edge Detection and Variance analysis ($np.var$).
* *Dynamic Facial State Classification:* Real-time heuristic mapping for:
  * 😊 *Happy / Smiling*
  * 😰 *Tension / Stress*
  * 🤔 *Thinking / Contemplating*
  * 😔 *Sad / Low Mood*
  * 😐 *Neutral / Focused*
* *Live System Analytics:* Overlay dashboard rendering real-time *FPS* (Frames Per Second), *CPU Usage, and detected **Face Count*.

---

## 🛠️ Tech Stack & Requirements

* *Language:* Python 3.x
* *Core Libraries:*
  * opencv-python
  * numpy
  * psutil

---

## 🚀 Quick Start & Installation

1. *Clone the Repository:*
   ```bash
   git clone [https://github.com/balaji-ai2006/realtime-Emotion-Detection.git](https://github.com/balaji-ai2006/realtime-Emotion-Detection.git)
   cd realtime-Emotion-Detection