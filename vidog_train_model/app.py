import os
import re
import sys
import json
import time
import tempfile
from typing import Optional
import numpy as np
from PIL import Image
import streamlit as st

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(CUR_DIR)

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector
from video_engine import VideoRegistrationEngine
from classifier_head import FastLinearClassifier
from matcher import HybridBiometricMatcher

GALLERY_V2_DIR = os.path.join(CUR_DIR, "data", "gallery_v2")
CHECKPOINTS_DIR = os.path.join(CUR_DIR, "checkpoints")
NOSE_DETECTOR_PATH = os.path.join(CHECKPOINTS_DIR, "nose_detector.onnx")

os.makedirs(GALLERY_V2_DIR, exist_ok=True)

st.set_page_config(
    page_title="DogID v2 (Single Video + Hybrid Classifier)",
    page_icon="🐾",
    layout="wide",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #3b82f6; margin-bottom: 0.2rem; }
    .sub-header { color: #94a3b8; font-size: 1.05rem; margin-bottom: 1.5rem; }
</style>
""",
    unsafe_allow_html=True,
)


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
    ckpt = torch.load(target_ckpt, map_location="cpu", weights_only=True)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    model.eval()
    return model, pipeline


@st.cache_resource
def load_detector():
    return NoseDetector(weights_path=NOSE_DETECTOR_PATH)


def extract_embeddings(img_pil: Optional[Image.Image], nose_pil: Optional[Image.Image], model, pipeline):
    import torch

    emb_nose, emb_face = None, None
    with torch.no_grad():
        if nose_pil is not None:
            t_nose, _ = pipeline(nose_pil)
            emb_nose = model(t_nose.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
        if img_pil is not None:
            t_face, _ = pipeline(img_pil)
            emb_face = model(t_face.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
    return emb_nose, emb_face


def save_npz_atomic(target_path: str, **arrays: np.ndarray) -> None:
    tmp_path = os.path.join(os.path.dirname(target_path), f".tmp_{os.path.basename(target_path)}")
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, target_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def safe_dog_dir(dog_name: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "_", dog_name.strip()).lstrip("._") or "unknown_dog"
    gallery_real = os.path.realpath(GALLERY_V2_DIR)
    target_real = os.path.realpath(os.path.join(gallery_real, cleaned))
    if not target_real.startswith(gallery_real + os.sep):
        raise ValueError("Invalid dog name path.")
    return target_real


# -- Sidebar Info --
with st.sidebar:
    st.title("DogID v2 Hybrid Engine")
    st.markdown("**Single-Video Auto-Split**\n* Auto-detects Nose -> `nose_vectors.npz`\n* Extracts Body -> `body_vectors.npz`")
    st.markdown("**2-Stage Matching**\n* Stage 1: Cosine Gate (80% Nose / 20% Body)\n* Stage 2: Fast Linear Boundary Head")
    st.markdown("---")

    detector = load_detector()
    if detector.has_model:
        st.success("YOLO Nose Detector: ACTIVE")
    else:
        st.info("Nose Detector: Center-Crop Fallback")

    classifier = FastLinearClassifier(os.path.join(GALLERY_V2_DIR, "_model"))
    if classifier.is_trained():
        st.success(f"Linear Classifier: ACTIVE ({len(classifier.class_names)} dogs)")
    else:
        st.warning("Linear Classifier: Standby (Need >= 2 dogs with nose vectors)")

    if st.button("Force Retrain Classifier"):
        res = classifier.train_from_gallery(GALLERY_V2_DIR)
        if res["success"]:
            st.success(res["message"])
        else:
            st.error(res["message"])


model, pipeline = load_backbone()
matcher = HybridBiometricMatcher(gallery_dir=GALLERY_V2_DIR)

st.markdown('<div class="main-header">DogID v2: Single-Video Auto-Split & Hybrid Biometrics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload 1 video: automatically extracts Nose + Body vectors into .npz format & retrains linear head</div>', unsafe_allow_html=True)

tab_register, tab_identify, tab_gallery = st.tabs([
    "Video Registration",
    "Single-Photo Identification",
    "Gallery & Model Inspector",
])

# --------------------------------------------------------------------------------
# TAB 1: SINGLE-VIDEO REGISTRATION (AUTO NOSE + BODY EXTRACTION)
# --------------------------------------------------------------------------------
with tab_register:
    st.subheader("Register a Dog via Single Video")
    st.caption("Upload one video. The engine automatically detects nose frames (nose_vectors.npz) and full-body frames (body_vectors.npz).")

    col1, col2 = st.columns([1, 1])
    with col1:
        dog_name = st.text_input("Case ID / Dog Name *", placeholder="e.g. case_1042 or Max").strip()
        breed = st.text_input("Breed", placeholder="e.g. Indie, Golden Retriever").strip()
        gender = st.selectbox("Gender", ["Unknown", "Male", "Female"])
        age = st.number_input("Age (years)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
        color = st.text_input("Color / Markings", placeholder="e.g. Golden, Brown / White patch")

    with col2:
        uploaded_video = st.file_uploader("Upload Registration Video (MP4 / MOV / AVI)", type=["mp4", "mov", "avi"])

    if uploaded_video and dog_name:
        st.video(uploaded_video)

        if st.button("Process Video & Register Dog", type="primary"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            try:
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
                    min_sharpness=65.0,
                    min_body_sharpness=35.0,
                    target_k_frames=45,
                )

                with st.spinner("Auto-splitting video into sharp Nose & Body frames..."):
                    selected_nose, selected_body, stats = engine.process_video_auto(
                        tfile.name,
                        nose_detector_fn=detector.detect_and_crop,
                        progress_callback=p_cb,
                    )
            finally:
                if os.path.exists(tfile.name):
                    os.unlink(tfile.name)

            if not selected_nose and not selected_body:
                st.error("No sharp frames could be extracted from this video. Please upload clearer footage.")
            else:
                progress_bar.progress(0.88)
                status_text.text("Computing 1024-d embeddings and saving .npz matrices...")

                dog_dir = safe_dog_dir(dog_name)
                os.makedirs(dog_dir, exist_ok=True)
                case_id = os.path.basename(dog_dir)

                if selected_nose:
                    nose_vecs, face_vecs = [], []
                    for frame_obj in selected_nose:
                        emb_n, emb_f = extract_embeddings(frame_obj.full_image, frame_obj.nose_crop, model, pipeline)
                        nose_vecs.append(emb_n)
                        face_vecs.append(emb_f)
                    save_npz_atomic(
                        os.path.join(dog_dir, "nose_vectors.npz"),
                        nose_vectors=np.array(nose_vecs, dtype=np.float32),
                        face_vectors=np.array(face_vecs, dtype=np.float32),
                    )

                if selected_body:
                    body_vecs = []
                    for b_frame in selected_body:
                        _, emb_b = extract_embeddings(b_frame.full_image, None, model, pipeline)
                        body_vecs.append(emb_b)
                    save_npz_atomic(
                        os.path.join(dog_dir, "body_vectors.npz"),
                        body_vectors=np.array(body_vecs, dtype=np.float32),
                    )

                meta_path = os.path.join(dog_dir, "meta.json")
                existing_meta = {}
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            existing_meta = json.load(f)
                    except Exception:
                        pass

                has_nose = os.path.exists(os.path.join(dog_dir, "nose_vectors.npz"))
                has_body = os.path.exists(os.path.join(dog_dir, "body_vectors.npz"))
                final_nose_count = len(selected_nose) if selected_nose else existing_meta.get("nose_frames_enrolled", 0)
                final_body_count = len(selected_body) if selected_body else existing_meta.get("body_frames_enrolled", 0)

                master_meta = {
                    "case_id": case_id,
                    "dog_name": dog_name,
                    "breed": breed,
                    "gender": gender,
                    "age_years": age,
                    "color": color,
                    "created_at": existing_meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
                    "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "nose_frames_enrolled": final_nose_count,
                    "body_frames_enrolled": final_body_count,
                    "has_nose_stream": has_nose,
                    "has_body_stream": has_body,
                    "video_duration_sec": stats.get("duration_sec", 0.0),
                }
                tmp_meta = os.path.join(dog_dir, ".tmp_meta.json")
                with open(tmp_meta, "w", encoding="utf-8") as f:
                    json.dump(master_meta, f, indent=2)
                os.replace(tmp_meta, meta_path)

                progress_bar.progress(0.95)
                status_text.text("Updating Fast Linear Boundary Classifier...")
                clf_res = classifier.train_from_gallery(GALLERY_V2_DIR)
                matcher.reload_classifier()

                progress_bar.progress(1.0)
                status_text.text("Registration Complete!")

                st.success(
                    f"Registered **{dog_name}** (`{case_id}`) — "
                    f"{len(selected_nose)} Nose vectors + {len(selected_body)} Body vectors saved to `.npz`."
                )
                st.info(clf_res["message"])

# --------------------------------------------------------------------------------
# TAB 2: SINGLE-PHOTO IDENTIFICATION
# --------------------------------------------------------------------------------
with tab_identify:
    st.subheader("Identify Dog from Photo")
    st.caption("Upload a single query photo. Auto-detects nose if present (80% Nose / 20% Body + Linear Head); falls back to 100% Body if no nose detected.")

    query_file = st.file_uploader("Upload Query Photo (JPG / PNG)", type=["jpg", "jpeg", "png"], key="query_uploader")

    if query_file:
        col_img, col_match = st.columns([1, 1.2])

        with col_img:
            img = Image.open(query_file).convert("RGB")
            st.image(img, caption="Query Upload", use_container_width=True)

            crop_res = detector.detect_and_crop(img)
            has_snout = crop_res.source == "model" and crop_res.confidence >= 0.40
            if has_snout:
                st.image(crop_res.crop, caption=f"Detected Nose Crop (YOLO Conf: {crop_res.confidence:.2f})", width=224)
            else:
                st.info("No close-up nose detected by YOLO; using Body-Only fallback.")

        with col_match:
            if st.button("Run Hybrid Identification", type="primary"):
                with st.spinner("Embedding query and querying hybrid classifier..."):
                    emb_nose, emb_face = extract_embeddings(img, crop_res.crop, model, pipeline) if has_snout else (None, None)
                    _, emb_body = extract_embeddings(img, None, model, pipeline)
                    top_match, all_candidates = matcher.match(
                        query_nose_emb=emb_nose,
                        query_face_emb=emb_face,
                        query_body_emb=emb_body,
                    )

                if top_match:
                    st.success(f"## Match Found: **{top_match.dog_name}**")
                    st.metric("Hybrid Confidence Score", f"{top_match.hybrid_score * 100:.1f}%")
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Cosine Score", f"{top_match.cosine_score * 100:.1f}%")
                    c2.metric("Classifier Prob", f"{top_match.classifier_prob * 100:.1f}%" if top_match.classifier_prob >= 0 else "N/A")
                    c3.info(f"Rule: `{top_match.rule_applied}`")
                elif all_candidates and all_candidates[0].status == "LOW_CONFIDENCE":
                    cand = all_candidates[0]
                    st.warning(f"## Low Confidence Match: **{cand.dog_name}** ({cand.hybrid_score*100:.1f}%)")
                    st.write(f"Rule: `{cand.rule_applied}`. Score is below acceptance threshold.")
                else:
                    st.error("## Unknown Dog / No Match Found")

                if all_candidates:
                    st.write("---")
                    st.write("#### Detailed Candidate Ranking")
                    table_data = [
                        {
                            "Case / Dog": c.dog_name,
                            "Status": c.status,
                            "Hybrid": f"{c.hybrid_score*100:.1f}%",
                            "Cosine": f"{c.cosine_score*100:.1f}%",
                            "Classifier": f"{c.classifier_prob*100:.1f}%" if c.classifier_prob >= 0 else "N/A",
                            "Rule": c.rule_applied,
                        }
                        for c in all_candidates
                    ]
                    st.table(table_data)

# --------------------------------------------------------------------------------
# TAB 3: GALLERY & MODEL INSPECTOR
# --------------------------------------------------------------------------------
with tab_gallery:
    st.subheader("Registered Dogs & Linear Head Status")

    if not os.path.exists(GALLERY_V2_DIR):
        st.info("Gallery is empty.")
    else:
        dogs = [
            d for d in sorted(os.listdir(GALLERY_V2_DIR))
            if os.path.isdir(os.path.join(GALLERY_V2_DIR, d)) and not d.startswith("_")
        ]
        st.write(f"Total Registered Dogs: **{len(dogs)}**")

        for dog in dogs:
            dog_p = os.path.join(GALLERY_V2_DIR, dog)
            meta_p = os.path.join(dog_p, "meta.json")
            meta = {}
            if os.path.exists(meta_p):
                with open(meta_p, "r", encoding="utf-8") as f:
                    meta = json.load(f)

            nose_cnt = meta.get("nose_frames_enrolled", "?")
            body_cnt = meta.get("body_frames_enrolled", "?")
            with st.expander(f"{meta.get('dog_name', dog)} — ({meta.get('breed', 'Unknown')}, {nose_cnt} nose / {body_cnt} body vectors)"):
                st.json(meta)
