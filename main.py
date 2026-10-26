import cv2
import time
import psutil
import os
import argparse
import numpy as np
from collections import deque, Counter

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Emotion Detection (Optimized for PC & Raspberry Pi)")
    parser.add_argument("--camera", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument("--width", type=int, default=640, help="Frame width (default: 640 for Pi optimization)")
    parser.add_argument("--height", type=int, default=480, help="Frame height (default: 480)")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale factor for face detection (default: 0.5 for fast detection)")
    parser.add_argument("--headless", action="store_true", help="Run without GUI window (for SSH/Raspberry Pi headless)")
    return parser.parse_args()

def main():
    args = parse_args()

    xml_file = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_file):
        # Fallback to OpenCV's built-in cascade path if local file not found
        cv_xml = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
        if os.path.exists(cv_xml):
            xml_file = cv_xml
        else:
            print(f"[ERROR] Haar Cascade XML file '{xml_file}' bhetli nahi!")
            return

    print(f"[INFO] Loading Haar Cascade from: {xml_file}")
    face_cascade = cv2.CascadeClassifier(xml_file)

    # Initialize Camera
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"[ERROR] Camera index {args.camera} open karta aali nahi. Device connect ahe ka check kara.")
        return

    # Set frame dimensions (640x480 is ideal for 30+ FPS on Raspberry Pi)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Try setting 30 FPS
    cap.set(cv2.CAP_PROP_FPS, 30)

    window_name = "AI Real-Time Emotion UI (Pi & PC Optimized)"
    if not args.headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # History buffer for emotion temporal smoothing (stops flickering)
    emotion_history = deque(maxlen=7)

    # Local Contrast Enhancement (CLAHE) for illumination invariance
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    prev_frame_time = time.time()
    fps_smooth = 0.0

    print("=" * 60)
    print("🚀 Real-Time Emotion Detection Engine (Raspberry Pi & PC)")
    print("Press 'q' or Ctrl+C in terminal to Exit.")
    print("=" * 60)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[WARNING] Frame capture zali nahi, camera disconnect zali ka?")
                break

            # Calculate FPS
            curr_time = time.time()
            time_diff = curr_time - prev_frame_time
            prev_frame_time = curr_time
            if time_diff > 0:
                current_fps = 1.0 / time_diff
                fps_smooth = 0.9 * fps_smooth + 0.1 * current_fps  # Smooth FPS reading

            cpu_usage = psutil.cpu_percent()

            # Convert to Grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # RASPBERRY PI OPTIMIZATION:
            # Resize grayscale frame for face detection (e.g. 0.5x scale -> 4x fewer pixels to process!)
            scale = args.scale
            if scale < 1.0:
                small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            else:
                small_gray = gray

            # Run Haar Cascade on downscaled image
            faces_detected = face_cascade.detectMultiScale(
                small_gray,
                scaleFactor=1.15,
                minNeighbors=5,
                minSize=(int(50 * scale), int(50 * scale))
            )

            # Re-scale bounding box coordinates back to full-resolution
            faces = []
            for (sx, sy, sw, sh) in faces_detected:
                if scale < 1.0:
                    faces.append((int(sx / scale), int(sy / scale), int(sw / scale), int(sh / scale)))
                else:
                    faces.append((sx, sy, sw, sh))

            for (x, y, w, h) in faces:
                # Boundary checks to prevent array slicing out of bounds
                x = max(0, x)
                y = max(0, y)
                w = min(w, frame.shape[1] - x)
                h = min(h, frame.shape[0] - y)

                if w < 40 or h < 40:
                    continue

                # Draw Face Rectangle
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)

                # Geometry structural regions: Mouth and Eyebrows
                mouth_roi = gray[y + int(h * 0.62) : y + int(h * 0.95), x + int(w * 0.2) : x + int(w * 0.8)]
                eyebrow_roi = gray[y + int(h * 0.12) : y + int(h * 0.35), x + int(w * 0.18) : x + int(w * 0.82)]

                m_var = 0.0
                e_var = 0.0
                m_density = 0.0
                e_density = 0.0

                if mouth_roi.size > 0:
                    # Apply CLAHE to handle uneven lighting / shadows on camera
                    m_norm = clahe.apply(mouth_roi)
                    m_var = float(np.var(m_norm))
                    m_edges = cv2.Canny(m_norm, 50, 130)
                    m_density = float(np.sum(m_edges > 0)) / mouth_roi.size

                if eyebrow_roi.size > 0:
                    e_norm = clahe.apply(eyebrow_roi)
                    e_var = float(np.var(e_norm))
                    e_edges = cv2.Canny(e_norm, 50, 130)
                    e_density = float(np.sum(e_edges > 0)) / eyebrow_roi.size

                # --- INSTANT EMOTION CLASSIFICATION ---
                # 1. HAPPY (Mouth stretch and teeth visibility increase variance & edge density)
                if m_var > 950 or m_density > 0.11:
                    raw_emotion = "Happy / Smiling"
                    color = (0, 255, 255)  # Yellow

                # 2. TENSION / STRESS (Eyebrows furrowed with low mouth variance)
                elif e_density >= 0.095 and m_var < 320:
                    raw_emotion = "Tension / Stress"
                    color = (0, 0, 255)  # Red

                # 3. THINKING (Concentrated look, moderate brow edges)
                elif 0.05 < e_density < 0.095 and m_var < 420:
                    raw_emotion = "Thinking"
                    color = (255, 140, 0)  # Light Blue/Orange

                # 4. SAD / LOW MOOD (Flat mouth edges and low contrast)
                elif m_density < 0.035 and m_var < 230:
                    raw_emotion = "Sad"
                    color = (255, 50, 50)  # Dark Blue

                # 5. NEUTRAL (Relaxed state)
                else:
                    raw_emotion = "Neutral"
                    color = (0, 255, 0)  # Green

                # --- TEMPORAL SMOOTHING (Stops jittering/flicker) ---
                emotion_history.append(raw_emotion)
                smoothed_emotion = Counter(emotion_history).most_common(1)[0][0]

                # Map color to smoothed emotion
                color_map = {
                    "Happy / Smiling": (0, 255, 255),
                    "Tension / Stress": (0, 0, 255),
                    "Thinking": (255, 140, 0),
                    "Sad": (255, 50, 50),
                    "Neutral": (0, 255, 0)
                }
                final_color = color_map.get(smoothed_emotion, (0, 255, 0))

                # --- SMART UI OVERLAY (No screen clipping) ---
                label_text = f"{smoothed_emotion}"
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.55
                thickness = 2
                (text_w, text_h), baseline = cv2.getTextSize(label_text, font, font_scale, thickness)

                # If face is near the top edge, put label inside or below the box
                if y - 30 > 0:
                    box_y1 = y - text_h - 14
                    box_y2 = y
                    text_y = y - 7
                else:
                    box_y1 = y + h
                    box_y2 = y + h + text_h + 14
                    text_y = y + h + text_h + 7

                box_x1 = x
                box_x2 = min(frame.shape[1], x + text_w + 14)

                # Draw label background box & text
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), final_color, -1)
                cv2.putText(frame, label_text, (box_x1 + 7, text_y), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

            # --- BOTTOM STATUS DASHBOARD ---
            h_frame, w_frame, _ = frame.shape
            dashboard_h = 42
            cv2.rectangle(frame, (0, h_frame - dashboard_h), (w_frame, h_frame), (25, 25, 25), -1)

            db_text = f"FPS: {int(fps_smooth)} | CPU: {cpu_usage}% | Faces: {len(faces)} | Pi-Optimized"
            cv2.putText(frame, db_text, (15, h_frame - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)

            if not args.headless:
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("[INFO] 'q' dabala, programme band hot ahe...")
                    break
            else:
                # In headless mode, log status periodically
                if int(fps_smooth) > 0 and int(curr_time) % 2 == 0:
                    print(f"\r[STATUS] {db_text}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt (Ctrl+C) detected. Exiting gracefully...")

    finally:
        cap.release()
        if not args.headless:
            cv2.destroyAllWindows()
        print("[INFO] Cleanup complete. Program exit.")

if __name__ == "__main__":
    main()
