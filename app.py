import os
import cv2
import numpy as np
import streamlit as st
import psutil
from collections import deque, Counter
from PIL import Image

# Check if streamlit_webrtc is available
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

# Page Configuration
st.set_page_config(
    page_title="AI Emotion Detection Engine",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Dark Theme, Glassmorphism, Gradient Titles)
st.markdown("""
<style>
    .main {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        color: #f8fafc;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-val {
        font-size: 26px;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 13px;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .stButton>button {
        background: linear-gradient(90deg, #38bdf8 0%, #6366f1 100%);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 24px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache Haar Cascade Classifier
@st.cache_resource
def load_cascade():
    xml_file = "haarcascade_frontalface_default.xml"
    if not os.path.exists(xml_file):
        xml_file = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    return cv2.CascadeClassifier(xml_file)

face_cascade = load_cascade()
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Color Mapping for Emotions (BGR for OpenCV)
EMOTION_COLORS = {
    "Happy / Smiling 😊": (0, 255, 255),    # Yellow
    "Tension / Stress 😰": (0, 0, 255),      # Red
    "Thinking 🤔": (255, 140, 0),            # Orange
    "Sad / Low Mood 😔": (255, 60, 60),      # Blue
    "Neutral / Focused 😐": (0, 255, 0)      # Green
}

def analyze_face(face_gray, w, h, happy_thr, tension_thr, thinking_thr):
    """Calculates ROI variance, edge density, and infers emotion."""
    mouth_roi = face_gray[int(h * 0.62) : int(h * 0.95), int(w * 0.2) : int(w * 0.8)]
    eyebrow_roi = face_gray[int(h * 0.12) : int(h * 0.35), int(w * 0.18) : int(w * 0.82)]

    m_var, e_var, m_density, e_density = 0.0, 0.0, 0.0, 0.0

    if mouth_roi.size > 0:
        m_norm = clahe.apply(mouth_roi)
        m_var = float(np.var(m_norm))
        m_edges = cv2.Canny(m_norm, 50, 130)
        m_density = float(np.sum(m_edges > 0)) / mouth_roi.size

    if eyebrow_roi.size > 0:
        e_norm = clahe.apply(eyebrow_roi)
        e_var = float(np.var(e_norm))
        e_edges = cv2.Canny(e_norm, 50, 130)
        e_density = float(np.sum(e_edges > 0)) / eyebrow_roi.size

    # Heuristic Emotion Mapping
    if m_var > happy_thr or m_density > 0.11:
        emotion = "Happy / Smiling 😊"
    elif e_density >= tension_thr and m_var < 320:
        emotion = "Tension / Stress 😰"
    elif 0.05 < e_density < thinking_thr and m_var < 420:
        emotion = "Thinking 🤔"
    elif m_density < 0.035 and m_var < 230:
        emotion = "Sad / Low Mood 😔"
    else:
        emotion = "Neutral / Focused 😐"

    return emotion, m_var, m_density, e_var, e_density, mouth_roi, eyebrow_roi

def process_frame(frame, happy_thr=950, tension_thr=0.095, thinking_thr=0.095, scale=0.5):
    """Processes an RGB frame, draws bounding boxes, labels, and returns results."""
    bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)

    # Downscale for ultra-fast processing
    if scale < 1.0:
        small_gray = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    else:
        small_gray = gray

    faces_detected = face_cascade.detectMultiScale(
        small_gray,
        scaleFactor=1.15,
        minNeighbors=5,
        minSize=(int(50 * scale), int(50 * scale))
    )

    detected_faces = []
    for (sx, sy, sw, sh) in faces_detected:
        if scale < 1.0:
            x, y, w, h = int(sx / scale), int(sy / scale), int(sw / scale), int(sh / scale)
        else:
            x, y, w, h = sx, sy, sw, sh

        # Ensure valid bounds
        x, y = max(0, x), max(0, y)
        w = min(w, frame.shape[1] - x)
        h = min(h, frame.shape[0] - y)

        if w < 40 or h < 40:
            continue

        face_gray = gray[y : y + h, x : x + w]
        emotion, m_var, m_dens, e_var, e_dens, m_roi, e_roi = analyze_face(
            face_gray, w, h, happy_thr, tension_thr, thinking_thr
        )
        color = EMOTION_COLORS.get(emotion, (0, 255, 0))

        # Draw Face Rectangle
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color, 2)

        # Smart text rendering
        label = emotion
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

        if y - 30 > 0:
            by1, by2, ty = y - text_h - 14, y, y - 7
        else:
            by1, by2, ty = y + h, y + h + text_h + 14, y + h + text_h + 7

        bx1, bx2 = x, min(frame.shape[1], x + text_w + 14)
        cv2.rectangle(bgr, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(bgr, label, (bx1 + 7, ty), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)

        detected_faces.append({
            "emotion": emotion,
            "box": (x, y, w, h),
            "m_var": m_var,
            "m_density": m_dens,
            "e_var": e_var,
            "e_density": e_dens,
            "mouth_roi": m_roi,
            "eyebrow_roi": e_roi
        })

    rgb_result = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb_result, detected_faces

# Sidebar Controls
st.sidebar.title("⚙️ Engine Controls")
mode = st.sidebar.radio(
    "Select Input Mode:",
    ["📹 Live WebRTC Camera", "📸 Snapshot Camera", "🖼️ Upload Image"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Emotion Sensitivity")
happy_thr = st.sidebar.slider("Smile / Happy Variance", 600, 1500, 950, step=25)
tension_thr = st.sidebar.slider("Tension Eyebrow Density", 0.05, 0.15, 0.095, step=0.005)
thinking_thr = st.sidebar.slider("Thinking Eyebrow Density", 0.05, 0.12, 0.095, step=0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Hardware")
cpu_p = psutil.cpu_percent()
mem_p = psutil.virtual_memory().percent
st.sidebar.progress(cpu_p / 100.0, text=f"CPU: {cpu_p}%")
st.sidebar.progress(mem_p / 100.0, text=f"RAM: {mem_p}%")

# Main Header
st.title("🚀 Real-Time AI Emotion Detection Engine")
st.markdown("Computer Vision system with facial tracking, edge-variance profiling, and dynamic emotional state mapping.")

# App Modes
if mode == "📹 Live WebRTC Camera":
    if not HAS_WEBRTC:
        st.warning("⚠️ `streamlit-webrtc` सापडले नाही. कृपया '📸 Snapshot Camera' पर्याय निवडा किंवा `pip install streamlit-webrtc` रन करा.")
    else:
        st.info("💡 **टीप:** कॅमेरा सुरू करण्यासाठी खालील 'START' बटनावर क्लिक करा आणि ब्राउझरमध्ये Camera Permission अलाऊ (Allow) करा.")

        class EmotionVideoTransformer(VideoTransformerBase):
            def __init__(self):
                self.history = deque(maxlen=7)

            def transform(self, frame):
                img = frame.to_ndarray(format="bgr24")
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                processed_rgb, faces = process_frame(
                    rgb, happy_thr=happy_thr, tension_thr=tension_thr, thinking_thr=thinking_thr
                )
                return cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)

        # STUN Server configuration for robust cloud streaming
        rtc_config = RTCConfiguration({
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        })

        webrtc_streamer(
            key="emotion-detection-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_transformer_factory=EmotionVideoTransformer,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True
        )

elif mode == "📸 Snapshot Camera":
    st.subheader("📸 Live Webcam Snapshot")
    img_file = st.camera_input("चेहऱ्याचा फोटो काढा:")

    if img_file is not None:
        pil_image = Image.open(img_file)
        np_frame = np.array(pil_image)

        with st.spinner("Analyzing Facial Dynamics..."):
            processed, faces = process_frame(
                np_frame, happy_thr=happy_thr, tension_thr=tension_thr, thinking_thr=thinking_thr
            )

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(processed, caption="Detected Emotion State", use_container_width=True)

        with col2:
            st.subheader("📋 Detection Analytics")
            st.metric("Total Faces Detected", len(faces))

            for i, face in enumerate(faces):
                st.markdown(f"### Face #{i+1}: **{face['emotion']}**")
                subcol1, subcol2 = st.columns(2)
                with subcol1:
                    st.metric("Mouth Variance", f"{face['m_var']:.1f}")
                    st.metric("Mouth Density", f"{face['m_density']:.3f}")
                with subcol2:
                    st.metric("Eyebrow Variance", f"{face['e_var']:.1f}")
                    st.metric("Eyebrow Density", f"{face['e_density']:.3f}")

                # Display Cropped ROIs
                if face['mouth_roi'].size > 0:
                    st.image(face['mouth_roi'], caption="Mouth ROI (Edges)", width=120)

elif mode == "🖼️ Upload Image":
    st.subheader("🖼️ Upload a Photo for Emotion Analysis")
    uploaded = st.file_uploader("JPG / PNG इमेज निवडा:", type=["jpg", "jpeg", "png"])

    if uploaded is not None:
        pil_image = Image.open(uploaded)
        np_frame = np.array(pil_image)

        # Convert RGBA to RGB if necessary
        if np_frame.ndim == 3 and np_frame.shape[2] == 4:
            np_frame = cv2.cvtColor(np_frame, cv2.COLOR_RGBA2RGB)

        with st.spinner("Processing image..."):
            processed, faces = process_frame(
                np_frame, happy_thr=happy_thr, tension_thr=tension_thr, thinking_thr=thinking_thr
            )

        col1, col2 = st.columns([3, 2])
        with col1:
            st.image(processed, caption="Analyzed Image", use_container_width=True)

        with col2:
            st.subheader("📋 Facial Analytics")
            st.metric("Detected Faces", len(faces))
            if faces:
                for idx, face in enumerate(faces):
                    st.success(f"Face #{idx+1} State: **{face['emotion']}**")
                    st.write(f"- **Mouth Variance:** {face['m_var']:.1f}")
                    st.write(f"- **Mouth Edge Density:** {face['m_density']:.3f}")
                    st.write(f"- **Eyebrow Density:** {face['e_density']:.3f}")
            else:
                st.info("कोणताही चेहरा स्पष्ट दिसला नाही. कृपया चांगल्या प्रकाशातील फोटो अपलोड करा.")
