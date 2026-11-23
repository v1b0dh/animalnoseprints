import io
import os
import sys
import json
import time
import tempfile
import numpy as np
from PIL import Image
import streamlit as st

# Safe path imports from parent app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "app")))
# Robust directory and path resolution
CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector

from video_engine import VideoRegistrationEngine
from matcher import PureCosineMatcher

GALLERY_V2_DIR = "data/gallery_v2"
CHECKPOINTS_DIR = "checkpoints"
NOSE_DETECTOR_PATH = os.path.join(CHECKPOINTS_DIR, "nose_detector.onnx")
GALLERY_V2_DIR = os.path.join(CUR_DIR, "data", "gallery_v2")
NOSE_DETECTOR_PATH = os.path.join(CUR_DIR, "checkpoints", "nose_detector.onnx")

st.set_page_config(
    page_title="DogID v2 (Video + Hybrid)",
    page_icon="??",
    layout="wide"
)

# Custom Styling
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #3b82f6; margin-bottom: 0.2rem; }
    .sub-header { color: #94a3b8; font-size: 1.05rem; margin-bottom: 1.5rem; }
    .card { background-color: #1e293b; padding: 1.25rem; border-radius: 0.5rem; border: 1px solid #334155; margin-bottom: 1rem; }
    .metric-val { font-size: 1.4rem; font-weight: 600; color: #38bdf8; }
</style>
""", unsafe_allow_html=True)

# -- Cached Resources --
@st.cache_resource
def load_backbone():
    pipeline = CLAHEPipeline(image_size=224)
    target_ckpt = None
    for name in ["teacher_best.pth", "dnnet_v3_best.pth"]:
        p = os.path.join(CHECKPOINTS_DIR, name)
        if os.path.exists(p):
            target_ckpt = p
            break

    if not target_ckpt:
        raise RuntimeError(f"FATAL: Trained weights not found in {CHECKPOINTS_DIR}. Hard-lock active.")

    import torch
    model = DNNetV3(pretrained=True, use_head=True)
    ckpt = torch.load(target_ckpt, map_location="cpu")
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    return model, pipeline

@st.cache_resource
def load_detector():
    return NoseDetector(weights_path=NOSE_DETECTOR_PATH)

def extract_embeddings(img_pil: Image.Image, nose_pil: Optional[Image.Image], model, pipeline):
    import torch
    emb_nose = None
    emb_face = None
    with torch.no_grad():
        if nose_pil is not None:
            t_nose, _ = pipeline(nose_pil)
            emb_nose = model(t_nose.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
        if img_pil is not None:
            t_face, _ = pipeline(img_pil)
            emb_face = model(t_face.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
    return emb_nose, emb_face

# -- Sidebar Info --
with st.sidebar:
    st.image("https://img.icons8.com/isometric/512/dog.png", width=70)
    st.title("DogID v2 Engine")
    st.markdown("**Dual-Stream Biometrics**\n* 85% Nose Macro-Texture\n* 15% Face / Head Shape")
    st.markdown("**Few-Shot Classifier**\n* Fast Logistic Boundary Head\n* Open-Set Cosine Gate")
    st.markdown("---")
    
    detector = load_detector()
    if detector.has_model:
        st.success("?? YOLO Nose Detector: ACTIVE")
    else:
        st.info("?? Nose Detector: Center-Crop Fallback")

    st.info("?? Matching Mode: Pure Cosine Similarity (Zero Retraining Required)")

# -- Main Tabs --
st.markdown('<div class="main-header">?? DogID v2: Video Registration & Hybrid Biometrics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Register dogs via quick video clips  Identify dogs from single query photos</div>', unsafe_allow_html=True)

tab_register, tab_identify, tab_gallery = st.tabs([
    "?? Video Registration", 
    "?? Single-Photo Identification", 
    "?? Gallery & Model Inspector"
])

model, pipeline = load_backbone()
matcher = PureCosineMatcher(gallery_dir=GALLERY_V2_DIR, nose_weight=0.80, body_weight=0.20)

# --------------------------------------------------------------------------------
# TAB 1: DUAL VIDEO REGISTRATION (NOSE + BODY)
# --------------------------------------------------------------------------------
with tab_register:
    st.subheader("Register a New Dog via Dual Videos")
    st.caption("Upload separate video clips for high-resolution Snout/Nose macro-texture and Full-Body views. Automatically extracts 40-50 diverse frames from each.")

    col1, col2 = st.columns([1, 1])
    with col1:
        dog_name = st.text_input("Dog Name *", placeholder="e.g. Max, Bella, Bruno").strip()
        breed = st.text_input("Breed", placeholder="e.g. Golden Retriever, Beagle").strip()
        gender = st.selectbox("Gender", ["Unknown", "Male", "Female"])
        age = st.number_input("Age (years)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
        color = st.text_input("Color / Markings", placeholder="e.g. Golden, Brown / White patch")

    with col2:
        uploaded_nose_video = st.file_uploader(
            "1. Upload Nose/Face Video (tagged: nose_<dog_name>.mp4)",
            type=["mp4", "mov", "avi"],
            key="nose_vid_uploader"
        )
        uploaded_body_video = st.file_uploader(
            "2. Upload Body Video (tagged: body_<dog_name>.mp4)",
            type=["mp4", "mov", "avi"],
            key="body_vid_uploader"
        )

    if dog_name and (uploaded_nose_video or uploaded_body_video):
        if st.button("🚀 Process Dual Videos & Register Dog", type="primary"):
            progress_bar = st.progress(0.0)
            status_text = st.empty()

            dog_dir = os.path.join(GALLERY_V2_DIR, dog_name.replace(" ", "_"))
            os.makedirs(dog_dir, exist_ok=True)

            engine = VideoRegistrationEngine(
                target_fps=4.0,
                min_sharpness=65.0,
                min_body_sharpness=35.0,
                target_k_frames=45
            )

            nose_frames_count = 0
            body_frames_count = 0

            # 1. Process Nose Video if provided
            if uploaded_nose_video:
                tfile_nose = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile_nose.write(uploaded_nose_video.read())
                tfile_nose.flush()
                tfile_nose.close()

                status_text.text("Extracting 40-50 sharp snout frames from nose video...")
                selected_nose_frames, stats_nose = engine.process_video(
                    tfile_nose.name,
                    nose_detector_fn=detector.detect_and_crop,
                    progress_callback=lambda pct, msg: progress_bar.progress(pct * 0.45)
                )

                for idx, frame_obj in enumerate(selected_nose_frames, start=1):
                    prefix = f"{idx:04d}"
                    emb_nose, emb_face = extract_embeddings(frame_obj.full_image, frame_obj.nose_crop, model, pipeline)
                    np.save(os.path.join(dog_dir, f"{prefix}.npy"), emb_nose)
                    np.save(os.path.join(dog_dir, f"face_{prefix}.npy"), emb_face)
                    frame_obj.nose_crop.save(os.path.join(dog_dir, f"{prefix}.jpg"), "JPEG", quality=95)
                    frame_obj.full_image.save(os.path.join(dog_dir, f"original_{prefix}.jpg"), "JPEG", quality=90)

                nose_frames_count = len(selected_nose_frames)
                if os.path.exists(tfile_nose.name):
                    os.unlink(tfile_nose.name)

            # 2. Process Body Video if provided (direct frames, bypasses nose detector)
            if uploaded_body_video:
                tfile_body = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                tfile_body.write(uploaded_body_video.read())
                tfile_body.flush()
                tfile_body.close()

                status_text.text("Splitting body video into 40-50 sharp full-body images...")
                selected_body_frames, stats_body = engine.process_body_video(
                    tfile_body.name,
                    progress_callback=lambda pct, msg: progress_bar.progress(0.45 + pct * 0.45)
                )

                for idx, b_frame in enumerate(selected_body_frames, start=1):
                    prefix = f"body_{idx:04d}"
                    _, emb_body = extract_embeddings(b_frame.full_image, None, model, pipeline)
                    np.save(os.path.join(dog_dir, f"{prefix}.npy"), emb_body)
                    b_frame.full_image.save(os.path.join(dog_dir, f"{prefix}.jpg"), "JPEG", quality=90)

                body_frames_count = len(selected_body_frames)
                if os.path.exists(tfile_body.name):
                    os.unlink(tfile_body.name)

            # 3. Save Master Metadata
            master_meta = {
                "dog_name": dog_name,
                "breed": breed,
                "gender": gender,
                "age_years": age,
                "color": color,
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "nose_frames_enrolled": nose_frames_count,
                "body_frames_enrolled": body_frames_count,
                "has_nose_stream": nose_frames_count > 0,
                "has_body_stream": body_frames_count > 0
            }
            with open(os.path.join(dog_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(master_meta, f, indent=2)

            progress_bar.progress(1.0)
            status_text.text("Registration Complete! Dual-modality gallery saved.")
            st.success(f"🎉 Registered **{dog_name}** with {nose_frames_count} Nose frames + {body_frames_count} Body frames!")

# --------------------------------------------------------------------------------
# TAB 2: IDENTIFICATION (WITH MULTI-MODAL WEIGHT RULES)
# --------------------------------------------------------------------------------
with tab_identify:
    st.subheader("Identify Dog from Photo")
    st.caption("Upload either a close-up Snout photo, a Full-Body photo, or both. Weighted rules: 80% Nose / 20% Body when both present; 100% fallback if only one is available.")

    col_q1, col_q2 = st.columns(2)
    with col_q1:
        query_nose_file = st.file_uploader("Upload Nose/Face Photo", type=["jpg", "jpeg", "png"], key="q_nose")
    with col_q2:
        query_body_file = st.file_uploader("Upload Full Body Photo (Optional)", type=["jpg", "jpeg", "png"], key="q_body")

    if query_nose_file or query_body_file:
        if st.button("🔍 Run Identification", type="primary"):
            with st.spinner("Extracting embeddings and applying biometric decision rules..."):
                emb_nose, emb_face, emb_body = None, None, None

                if query_nose_file:
                    img_nose = Image.open(query_nose_file).convert("RGB")
                    crop_res = detector.detect_and_crop(img_nose)
                    emb_nose, emb_face = extract_embeddings(img_nose, crop_res.crop, model, pipeline)

                if query_body_file:
                    img_body = Image.open(query_body_file).convert("RGB")
                    _, emb_body = extract_embeddings(img_body, None, model, pipeline)

                top_match, all_candidates = matcher.match(
                    query_nose_emb=emb_nose,
                    query_face_emb=emb_face,
                    query_body_emb=emb_body
                )

            if top_match:
                st.success(f"## ✅ Match Found: **{top_match.dog_name}**")
                st.metric("Composite Score", f"{top_match.cosine_score * 100:.1f}%", help=f"Rule: {top_match.rule_applied}")
                
                cols_sc = st.columns(3)
                if top_match.nose_score is not None:
                    cols_sc[0].metric("Nose Similarity", f"{top_match.nose_score * 100:.1f}%")
                if top_match.body_score is not None:
                    cols_sc[1].metric("Body Similarity", f"{top_match.body_score * 100:.1f}%")
                cols_sc[2].info(f"Rule Applied: `{top_match.rule_applied}`")

                matched_thumb_p = os.path.join(GALLERY_V2_DIR, top_match.dog_name, top_match.best_matching_frame)
                if os.path.exists(matched_thumb_p):
                    st.image(matched_thumb_p, caption=f"Best Match ({top_match.best_matching_frame})", width=220)
            elif all_candidates and all_candidates[0].status == "LOW_CONFIDENCE":
                cand = all_candidates[0]
                st.warning(f"## ⚠️ Low Confidence Match: **{cand.dog_name}** ({cand.hybrid_score*100:.1f}%)")
                st.write(f"Applied Rule: `{cand.rule_applied}`. Score below acceptance threshold.")
            else:
                st.error("## ❌ Unknown Dog / No Match Found")

            if all_candidates:
                st.write("---")
                st.write("#### Detailed Candidate Ranking")
                table_data = []
                for c in all_candidates:
                    table_data.append({
                        "Dog": c.dog_name,
                        "Status": c.status,
                        "Score": f"{c.cosine_score*100:.1f}%",
                        "Nose": f"{c.nose_score*100:.1f}%" if c.nose_score is not None else "N/A",
                        "Body": f"{c.body_score*100:.1f}%" if c.body_score is not None else "N/A",
                        "Rule": c.rule_applied
                    })
                st.table(table_data)

# --------------------------------------------------------------------------------
# TAB 3: GALLERY & MODEL INSPECTOR
# --------------------------------------------------------------------------------
with tab_gallery:
    st.subheader("Registered Dogs & Linear Head Status")
    
    if not os.path.exists(GALLERY_V2_DIR):
        st.info("Gallery is empty.")
    else:
        dogs = [d for d in sorted(os.listdir(GALLERY_V2_DIR)) 
                if os.path.isdir(os.path.join(GALLERY_V2_DIR, d)) and not d.startswith("_")]
        
        st.write(f"Total Registered Dogs: **{len(dogs)}**")
        
        for dog in dogs:
            dog_p = os.path.join(GALLERY_V2_DIR, dog)
            meta_p = os.path.join(dog_p, "meta.json")
            meta = {}
            if os.path.exists(meta_p):
                with open(meta_p, "r") as f:
                    meta = json.load(f)

            with st.expander(f"?? {meta.get('dog_name', dog)}  ({meta.get('breed', 'Unknown')}, {meta.get('frame_count', '?')} frames)"):
                st.json(meta)
                
                # Show thumbnails
                jpgs = [f for f in sorted(os.listdir(dog_p)) if f.endswith(".jpg") and not f.startswith("original_")]
                t_cols = st.columns(min(6, max(1, len(jpgs))))
                for i, jf in enumerate(jpgs[:12]):
                    with t_cols[i % 6]:
                        st.image(os.path.join(dog_p, jf), caption=jf, width=120)
