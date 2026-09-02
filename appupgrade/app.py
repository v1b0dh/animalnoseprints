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
from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector

from video_engine import VideoRegistrationEngine
from classifier_head import FastLinearClassifier
from matcher import HybridBiometricMatcher

GALLERY_V2_DIR = "data/gallery_v2"
CHECKPOINTS_DIR = "checkpoints"
NOSE_DETECTOR_PATH = os.path.join(CHECKPOINTS_DIR, "nose_detector.onnx")

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
    model = DNNetV3(pretrained=True, use_head=True)
    model.eval()
    return model, pipeline

@st.cache_resource
def load_detector():
    return NoseDetector(weights_path=NOSE_DETECTOR_PATH)

def extract_embeddings(img_pil: Image.Image, nose_pil: Image.Image, model, pipeline):
    t_nose, _ = pipeline(nose_pil)
    t_face, _ = pipeline(img_pil)
    import torch
    with torch.no_grad():
        emb_nose = model(t_nose.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
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

    classifier = FastLinearClassifier(os.path.join(GALLERY_V2_DIR, "_model"))
    if classifier.is_trained():
        st.success(f"?? Linear Classifier: ACTIVE ({len(classifier.class_names)} dogs)")
    else:
        st.warning(f"?? Linear Classifier: Standby (Need >= 2 dogs)")

    if st.button("?? Force Retrain Classifier"):
        res = classifier.train_from_gallery(GALLERY_V2_DIR)
        if res["success"]:
            st.success(res["message"])
        else:
            st.error(res["message"])

# -- Main Tabs --
st.markdown('<div class="main-header">?? DogID v2: Video Registration & Hybrid Biometrics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Register dogs via quick video clips  Identify dogs from single query photos</div>', unsafe_allow_html=True)

tab_register, tab_identify, tab_gallery = st.tabs([
    "?? Video Registration", 
    "?? Single-Photo Identification", 
    "?? Gallery & Model Inspector"
])

model, pipeline = load_backbone()
matcher = HybridBiometricMatcher(gallery_dir=GALLERY_V2_DIR)

# --------------------------------------------------------------------------------
# TAB 1: VIDEO REGISTRATION
# --------------------------------------------------------------------------------
with tab_register:
    st.subheader("Register a New Dog via Video")
    st.caption("Upload a 5-15 second video showing the dog's snout and head from different angles.")

    col1, col2 = st.columns([1, 1])
    with col1:
        dog_name = st.text_input("Dog Name *", placeholder="e.g. Max, Bella, Bruno").strip()
        breed = st.text_input("Breed", placeholder="e.g. Golden Retriever, Beagle").strip()
        age = st.number_input("Age (years)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
        color = st.text_input("Color / Markings", placeholder="e.g. Golden, Brown / White patch")

    with col2:
        uploaded_video = st.file_uploader("Upload Registration Video (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])

    if uploaded_video and dog_name:
        st.video(uploaded_video)
        
        if st.button("?? Process Video & Register Dog", type="primary"):
            # Save temporary video file for OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            tfile.flush()
            tfile.close()

            progress_bar = st.progress(0.0)
            status_text = st.empty()

            def p_cb(pct, msg):
                progress_bar.progress(pct)
                status_text.text(msg)

            engine = VideoRegistrationEngine(
                target_fps=4.0,
                min_sharpness=70.0,
                target_k_frames=10
            )

            with st.spinner("Extracting sharp candidate nose frames..."):
                selected_frames, stats = engine.process_video(
                    tfile.name,
                    nose_detector_fn=detector.detect_and_crop,
                    progress_callback=p_cb
                )

            os.unlink(tfile.name)

            if not selected_frames:
                st.error("? No sharp frames with detectable noses could be extracted from this video. Please upload a clearer video.")
            else:
                progress_bar.progress(0.90)
                status_text.text("Computing biometric embeddings and saving gallery...")

                dog_dir = os.path.join(GALLERY_V2_DIR, dog_name.replace(" ", "_"))
                os.makedirs(dog_dir, exist_ok=True)

                sharpness_list = []
                for idx, frame_obj in enumerate(selected_frames, start=1):
                    prefix = f"{idx:04d}"
                    emb_nose, emb_face = extract_embeddings(frame_obj.full_image, frame_obj.nose_crop, model, pipeline)
                    
                    # Save embeddings and thumbnails
                    np.save(os.path.join(dog_dir, f"{prefix}.npy"), emb_nose)
                    np.save(os.path.join(dog_dir, f"face_{prefix}.npy"), emb_face)
                    frame_obj.nose_crop.save(os.path.join(dog_dir, f"{prefix}.jpg"), "JPEG", quality=95)
                    frame_obj.full_image.save(os.path.join(dog_dir, f"original_{prefix}.jpg"), "JPEG", quality=90)

                    # Per-frame metadata
                    meta_frame = {
                        "frame_idx": frame_obj.frame_idx,
                        "timestamp_sec": frame_obj.timestamp_sec,
                        "sharpness": frame_obj.sharpness,
                        "detector_confidence": frame_obj.detector_confidence,
                        "detector_source": frame_obj.detector_source,
                        "bbox": frame_obj.bbox
                    }
                    with open(os.path.join(dog_dir, f"{prefix}.json"), "w") as f:
                        json.dump(meta_frame, f, indent=2)

                    sharpness_list.append(frame_obj.sharpness)

                # Save dog master metadata
                master_meta = {
                    "dog_name": dog_name,
                    "breed": breed,
                    "age_years": age,
                    "color": color,
                    "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "frame_count": len(selected_frames),
                    "avg_sharpness": float(np.mean(sharpness_list)),
                    "video_duration_sec": stats["duration_sec"]
                }
                with open(os.path.join(dog_dir, "meta.json"), "w") as f:
                    json.dump(master_meta, f, indent=2)

                # Trigger Fast Linear Retraining
                progress_bar.progress(0.98)
                status_text.text("Updating Fast Linear Boundary Classifier...")
                clf_res = classifier.train_from_gallery(GALLERY_V2_DIR)

                progress_bar.progress(1.0)
                status_text.text("Registration Complete!")

                st.success(f"?? Successfully registered **{dog_name}** with {len(selected_frames)} diverse high-quality frames!")
                st.info(f"?? Stats: {stats['valid_sharp_frames']} valid sharp frames found from {stats['total_frames_sampled']} sampled. {stats['rejected_frames']} blurry/bad frames discarded.")

                st.write("### Stored Optimal Frame Gallery")
                cols = st.columns(min(5, len(selected_frames)))
                for idx, frame_obj in enumerate(selected_frames):
                    with cols[idx % 5]:
                        st.image(frame_obj.nose_crop, caption=f"Frame #{frame_obj.frame_idx} (Sharp: {frame_obj.sharpness:.1f})")

# --------------------------------------------------------------------------------
# TAB 2: SINGLE-PHOTO IDENTIFICATION
# --------------------------------------------------------------------------------
with tab_identify:
    st.subheader("Identify Dog from Photo")
    st.caption("Upload a single query photo to identify the dog using the 2-Stage Hybrid matching engine.")

    query_file = st.file_uploader("Upload Query Photo (JPG / PNG)", type=["jpg", "jpeg", "png"], key="query_uploader")

    if query_file:
        col_img, col_match = st.columns([1, 1.2])

        with col_img:
            img = Image.open(query_file).convert("RGB")
            st.image(img, caption="Query Upload", use_container_width=True)

            # Crop nose
            crop_res = detector.detect_and_crop(img)
            st.image(crop_res.crop, caption=f"Detected Nose Crop ({crop_res.source}, Conf: {crop_res.confidence:.2f})", width=224)

        with col_match:
            if st.button("?? Run Hybrid Identification", type="primary"):
                with st.spinner("Embedding query and querying hybrid classifier..."):
                    emb_nose, emb_face = extract_embeddings(img, crop_res.crop, model, pipeline)
                    top_match, all_candidates = matcher.match(emb_nose, emb_face)

                if top_match:
                    st.success(f"## ? Match Found: **{top_match.dog_name}**")
                    
                    st.metric("Hybrid Confidence Score", f"{top_match.hybrid_score * 100:.1f}%")
                    
                    c1, c2 = st.columns(2)
                    c1.metric("Ensemble Cosine Score", f"{top_match.cosine_score * 100:.1f}%")
                    c2.metric("Classifier Probability", f"{top_match.classifier_prob * 100:.1f}%" if top_match.classifier_prob >= 0 else "N/A")

                    # Load thumbnail
                    matched_thumb_p = os.path.join(GALLERY_V2_DIR, top_match.dog_name, top_match.best_matching_frame)
                    if os.path.exists(matched_thumb_p):
                        st.image(matched_thumb_p, caption=f"Best Matching Gallery Frame ({top_match.best_matching_frame})", width=200)

                elif all_candidates and all_candidates[0].status == "LOW_CONFIDENCE":
                    cand = all_candidates[0]
                    st.warning(f"## ?? Low Confidence Match: **{cand.dog_name}** ({cand.hybrid_score*100:.1f}%)")
                    st.write("Score is below acceptance threshold. Please provide a sharper photo.")
                else:
                    st.error("## ? Unknown Dog / No Match Found")
                    st.write("The query sample does not match any registered dog in the database above the Open-Set Cosine Gate.")

                if all_candidates:
                    st.write("---")
                    st.write("#### Detailed Candidate Ranking")
                    table_data = []
                    for c in all_candidates:
                        table_data.append({
                            "Dog": c.dog_name,
                            "Status": c.status,
                            "Hybrid Score": f"{c.hybrid_score*100:.1f}%",
                            "Cosine Sim (85/15)": f"{c.cosine_score*100:.1f}%",
                            "Classifier Prob": f"{c.classifier_prob*100:.1f}%" if c.classifier_prob >= 0 else "N/A"
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
