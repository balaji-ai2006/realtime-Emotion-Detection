import cv2
import time
import psutil
import os
import numpy as np

xml_file = "haarcascade_frontalface_default.xml"
if not os.path.exists(xml_file):
    print(f"Error: '{xml_file}' bhetli nahi.")
    exit()

face_cascade = cv2.CascadeClassifier(xml_file)
cap = cv2.VideoCapture(0)

# Aspect Frame Stretching to Max Bounds
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cv2.namedWindow('AI Real-Time Emotion UI', cv2.WINDOW_NORMAL)

prev_frame_time = 0

print("Optimized Multi-Filter Engine Running... Press 'q' to Exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Contrast boost clear mapping components sathi
    frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=10)
    new_frame_time = time.time()
    fps = 1 / (new_frame_time - prev_frame_time) if (new_frame_time - prev_frame_time) > 0 else 0
    prev_frame_time = new_frame_time
    cpu_usage = psutil.cpu_percent()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # False detection clear layers remove karava mhanun scale thresholds vadhवले
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.12, 
        minNeighbors=6, 
        minSize=(120, 120)
    )

    for (x, y, w, h) in faces:
        # Avoid minor textures profiling / Mobile surface mapping reject filters
        if w > 450 or h > 450:
            continue
            
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
        
        # Geometry structure coordinates extraction mapping
        mouth_roi = gray[y+int(h/1.6):y+h, x+int(w/5):x+int(4*w/5)]
        eyebrow_roi = gray[y+int(h/7):y+int(h/3), x+int(w/5):x+int(4*w/5)]
        
        m_var = np.var(mouth_roi) if mouth_roi.size > 0 else 0
        e_var = np.var(eyebrow_roi) if eyebrow_roi.size > 0 else 0
        
        # Adaptive Threshold processing tracking zones variations mapping
        m_edges = cv2.Canny(mouth_roi, 50, 130) if mouth_roi.size > 0 else np.array([])
        e_edges = cv2.Canny(eyebrow_roi, 50, 130) if eyebrow_roi.size > 0 else np.array([])
        
        m_density = np.sum(m_edges > 0) / mouth_roi.size if mouth_roi.size > 0 else 0
        e_density = np.sum(e_edges > 0) / eyebrow_roi.size if eyebrow_roi.size > 0 else 0

        # --- RE-BALANCED HIGHER ACCURACY THRESHOLDS ---
        # 1. HAPPY (Requires significantly higher variance gradient expansion)
        if m_var > 980 or m_density > 0.11:
            emotion = "Happy / Smiling 😊"
            color = (0, 255, 255) # Yellow
            
        # 2. TENSION / STRESS (Eyebrows dynamic contraction analysis tracking)
        elif e_density >= 0.095 and m_var < 300:
            emotion = "Tension / Stress 😰"
            color = (0, 0, 255) # Red
            
        # 3. THINKING (Vichar Karat Ahe - Fine structural variance logic checks)
        elif 0.05 < e_density < 0.095 and m_var < 400:
            emotion = "Thinking / Vichar Karat Ahe 🤔"
            color = (255, 100, 0) # Light Blue
            
        # 4. SAD / DEPRESSION (Flat variance contours metrics parameters mapping)
        elif m_density < 0.035 and m_var < 220:
            emotion = "Sad / Depression 😔"
            color = (255, 0, 0) # Dark Blue
            
        # 5. NEUTRAL / FOCUSED
        else:
            emotion = "Neutral / Focused 😐"
            color = (0, 255, 0) # Green

        # State Indicator layout render
        cv2.rectangle(frame, (x, y - 32), (x + 300, y), color, -1)
        cv2.putText(frame, f"State: {emotion}", (x + 5, y - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 2, cv2.LINE_AA)

    # Bottom Fixed status layout dashboard configuration metrics map text
    h_frame, w_frame, _ = frame.shape
    cv2.rectangle(frame, (0, h_frame - 50), (w_frame, h_frame), (20, 20, 20), -1)
    db_text = f"REAL-TIME ENGINE | FPS: {int(fps)} | CPU: {cpu_usage}% | Faces Count: {len(faces)}"
    cv2.putText(frame, db_text, (20, h_frame - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('AI Real-Time Emotion UI', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()