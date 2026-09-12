"""
app.py — DogID System: Dog Nose Biometric Recognition
=======================================================================
Folder-based gallery (no database).

Storage layout (under GALLERY_DIR, default "data/gallery/"):

    <dog_name>/
        0001.npy        # 1024-d teacher embedding (float32, L2-normalized)
        0001.jpg        # thumbnail of the source photo
        0002.npy
        0002.jpg
        ...
        meta.json       # {breed, age_years, weight_kg, color, registered_at,
                        #  emb_count, avg_sharpness, best_sharpness}

Workflow:
  REGISTER : Upload one or more nose photos → embed → save as .npy + .jpg.
  IDENTIFY : Upload one or more nose photos → ensemble max-cosine match.
"""

import io
import os
import json
import time
import base64
from typing import Dict, List, Tuple, Optional

import streamlit as st
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageStat

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector

GALLERY_DIR    = "data/gallery"
MATCH_THRESH   = 0.82
QUALITY_WARN   = 100.0
REGISTER_MIN_SHARPNESS = 100.0
IDENTIFY_MIN_SHARPNESS = 60.0
MIN_NOSE_CROP_SIDE = 80
NOSE_DETECTOR_PATH = "checkpoints/nose_detector.onnx"
NOSE_WEIGHT = 0.85
FACE_WEIGHT = 0.15
EMBED_DIM      = 1024
SAFE_NAME_REPL = str.maketrans({c: "_" for c in r'<>:"/\|?*'})


# ==============================================================================
# Page config + CSS
# ==============================================================================

st.set_page_config(page_title="DogID System", layout="wide", page_icon="🐾")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

  *, *::before, *::after { box-sizing: border-box; }
  html, body, [data-testid="stAppViewContainer"] {
    font-family: 'Inter', sans-serif !important;
  }

  .stApp { background: #0c1017; color: #d1d5db; }

  [data-testid="stSidebar"] {
    background: #0c1017 !important;
    border-right: 1px solid #1e2530 !important;
  }

  .stButton > button {
    border-radius: 6px !important;
    font-weight: 500 !important;
    transition: background 0.15s ease, border-color 0.15s ease !important;
    border: 1px solid #2d3748 !important;
    background: #151d2b !important;
    color: #d1d5db !important;
  }
  .stButton > button:hover {
    border-color: #4f5b8a !important;
    background: #1a2338 !important;
  }
  .stButton > button[kind="primary"] {
    background: #3b4fd4 !important;
    border-color: #3b4fd4 !important;
    color: #fff !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #4a5fe0 !important;
    border-color: #4a5fe0 !important;
  }

  .stTextInput > div > div > input,
  .stNumberInput > div > div > input {
    border-radius: 6px !important;
    background: #111827 !important;
    border: 1px solid #1e2d3d !important;
    color: #d1d5db !important;
  }

  [data-testid="stExpander"] {
    border: 1px solid #1e2530 !important;
    border-radius: 8px !important;
    background: #111520 !important;
  }

  img { border-radius: 6px !important; }

  .stat-card {
    background: #111520;
    border: 1px solid #1e2530;
    border-radius: 8px;
    padding: 24px 16px;
    text-align: center;
  }
  .stat-number {
    font-size: 2.2rem;
    font-weight: 600;
    color: #7b8cde;
  }
  .stat-label {
    font-size: 0.75rem;
    color: #4b5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-top: 4px;
  }

  .section-hdr {
    font-size: 0.8rem;
    font-weight: 600;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 28px 0 10px 0;
  }

  .stProgress > div > div {
    background: #3b4fd4 !important;
    border-radius: 3px !important;
  }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# Gallery I/O (folder of .npy files)
# ==============================================================================

def safe_dirname(name: str) -> str:
    """Make a dog name filesystem-safe."""
    s = (name or "").strip().translate(SAFE_NAME_REPL)
    while "__" in s:
        s = s.replace("__", "_")
    return s.strip("._") or "unnamed"


def ensure_gallery_root() -> str:
    os.makedirs(GALLERY_DIR, exist_ok=True)
    return GALLERY_DIR


def dog_dir(name: str) -> str:
    return os.path.join(ensure_gallery_root(), safe_dirname(name))


def list_dogs() -> List[str]:
    root = ensure_gallery_root()
    out = []
    for entry in sorted(os.listdir(root)):
        p = os.path.join(root, entry)
        if os.path.isdir(p):
            out.append(entry)
    return out


def list_breeds() -> List[str]:
    """Collect distinct breeds from all dog meta.json files."""
    breeds = set()
    for d in list_dogs():
        meta = read_meta(d)
        if meta and meta.get("breed"):
            breeds.add(meta["breed"])
    return sorted(breeds)


def read_meta(name: str) -> Optional[dict]:
    p = os.path.join(dog_dir(name), "meta.json")
    if not os.path.exists(p):
        return None
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def write_meta(name: str, **kwargs) -> None:
    p = os.path.join(dog_dir(name), "meta.json")
    existing = {}
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = {}
    existing.update({k: v for k, v in kwargs.items() if v is not None})
    existing["name"] = name
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)


def next_index(name: str) -> int:
    """Return the next available 4-digit index for a dog's embeddings."""
    d = dog_dir(name)
    if not os.path.isdir(d):
        return 1
    used = []
    for fn in os.listdir(d):
        if fn.endswith(".npy"):
            try:
                used.append(int(fn[:4]))
            except ValueError:
                pass
    return (max(used) + 1) if used else 1


def save_embedding(name: str, embedding: torch.Tensor, pil_img: Image.Image,
                   sharpness: float, original_img: Optional[Image.Image] = None,
                   face_embedding: Optional[torch.Tensor] = None,
                   crop_bbox: Optional[Tuple[int, int, int, int]] = None,
                   detector_confidence: Optional[float] = None,
                   detector_source: Optional[str] = None) -> str:
    """Save one nose embedding + thumbnail; return the index used."""
    d = dog_dir(name)
    os.makedirs(d, exist_ok=True)
    idx = next_index(name)
    npy_path = os.path.join(d, f"{idx:04d}.npy")
    jpg_path = os.path.join(d, f"{idx:04d}.jpg")
    np.save(npy_path, embedding.detach().cpu().numpy().astype(np.float32))

    if face_embedding is not None:
        face_path = os.path.join(d, f"face_{idx:04d}.npy")
        np.save(face_path, face_embedding.detach().cpu().numpy().astype(np.float32))

    thumb = pil_img.copy()
    thumb.thumbnail((256, 256))
    thumb.save(jpg_path, format="JPEG", quality=85)

    if original_img is not None:
        original_path = os.path.join(d, f"original_{idx:04d}.jpg")
        original_thumb = original_img.copy()
        original_thumb.thumbnail((768, 768))
        original_thumb.save(original_path, format="JPEG", quality=88)

    frame_meta_path = os.path.join(d, f"{idx:04d}.json")
    with open(frame_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "idx": f"{idx:04d}",
            "sharpness": float(sharpness),
            "crop_bbox": crop_bbox,
            "detector_confidence": detector_confidence,
            "detector_source": detector_source,
            "nose_weight": NOSE_WEIGHT,
            "face_weight": FACE_WEIGHT,
            "has_face_embedding": face_embedding is not None,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f, indent=2)

    write_meta(name, last_sharpness=float(sharpness))
    return f"{idx:04d}"


def list_dog_frames(name: str) -> List[dict]:
    """Return [{'idx': '0001', 'emb': np.ndarray, 'jpg': path, 'sharpness': float}, ...]"""
    d = dog_dir(name)
    if not os.path.isdir(d):
        return []
    frames = []
    for fn in sorted(os.listdir(d)):
        if not (fn.endswith(".npy") and fn[:4].isdigit()):
            continue
        idx = fn[:4]
        try:
            arr = np.load(os.path.join(d, fn))
        except Exception:
            continue
        jpg = os.path.join(d, f"{idx}.jpg")
        if not os.path.exists(jpg):
            jpg = None
        face_path = os.path.join(d, f"face_{idx}.npy")
        face_emb = None
        if os.path.exists(face_path):
            try:
                face_emb = np.load(face_path).astype(np.float32)
            except Exception:
                face_emb = None
        frame_meta_path = os.path.join(d, f"{idx}.json")
        frame_meta = {}
        if os.path.exists(frame_meta_path):
            try:
                with open(frame_meta_path, "r", encoding="utf-8") as f:
                    frame_meta = json.load(f)
            except Exception:
                frame_meta = {}
        # No per-frame sharpness stored; approximate from meta (not ideal but ok).
        frames.append({
            "idx": idx,
            "emb": arr.astype(np.float32),
            "face_emb": face_emb,
            "jpg": jpg,
            "meta": frame_meta,
        })
    # Fill sharpness from meta if present (single global value, used as fallback)
    meta = read_meta(name) or {}
    for fr in frames:
        fr["sharpness"] = fr.get("meta", {}).get("sharpness", meta.get("last_sharpness", 0.0))
    return frames


def delete_dog(name: str) -> None:
    import shutil
    d = dog_dir(name)
    if os.path.isdir(d):
        shutil.rmtree(d)


def clear_dog_embeddings(name: str) -> None:
    d = dog_dir(name)
    if not os.path.isdir(d):
        return
    for fn in os.listdir(d):
        if fn.endswith(".npy") or fn.endswith(".jpg") or (fn[:4].isdigit() and fn.endswith(".json")):
            try:
                os.remove(os.path.join(d, fn))
            except Exception:
                pass


def build_gallery(model_type: str = "teacher",
                  breed: Optional[str] = None
                  ) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """
    Return {dog_name: {"nose": [tensor, ...], "face": [tensor, ...]}}.
    `model_type` and `breed` filters are accepted for API compatibility with
    the old SQLite version. The folder layout only stores the teacher
    model, so `model_type` is informational. `breed` filters by meta.json.
    """
    out: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for name in list_dogs():
        meta = read_meta(name) or {}
        if breed and meta.get("breed") != breed:
            continue
        nose_embs = []
        face_embs = []
        for fr in list_dog_frames(name):
            nose_embs.append(torch.from_numpy(fr["emb"].copy()))
            if fr.get("face_emb") is not None:
                face_embs.append(torch.from_numpy(fr["face_emb"].copy()))
        if nose_embs:
            out[name] = {"nose": nose_embs, "face": face_embs}
    return out


def dog_stats(name: str) -> dict:
    frames = list_dog_frames(name)
    sharp = [f["sharpness"] for f in frames if f.get("sharpness", 0) > 0]
    return {
        "emb_count": len(frames),
        "avg_sharpness": float(np.mean(sharp)) if sharp else 0.0,
        "best_sharpness": float(np.max(sharp)) if sharp else 0.0,
    }


# ==============================================================================
# Model
# ==============================================================================

@st.cache_resource
def load_model():
    ckpt_path = "checkpoints/teacher_best.pth"
    if os.path.exists(ckpt_path):
        m = DNNetV3(pretrained=True, use_head=True)
        ckpt       = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state_dict)
    else:
        m = DNNetV3(pretrained=True, use_head=False)
    m.eval()
    p = CLAHEPipeline(image_size=224)
    return m, p


model, pipeline = load_model()


@st.cache_resource
def load_nose_detector():
    return NoseDetector(weights_path=NOSE_DETECTOR_PATH)


nose_detector = load_nose_detector()


# ==============================================================================
# Inference helpers
# ==============================================================================

def embed_pil(img: Image.Image):
    """Embed a single PIL image -> (tensor, sharpness)."""
    t, s = pipeline(img)
    with torch.no_grad():
        vec = model(t.unsqueeze(0)).squeeze(0)
    return vec, s


def image_quality(img: Image.Image, sharpness: float, min_sharpness: float) -> Tuple[bool, List[str]]:
    """Basic crop quality checks used before embedding."""
    reasons = []
    w, h = img.size
    if min(w, h) < MIN_NOSE_CROP_SIDE:
        reasons.append(f"nose crop is too small ({w}x{h})")
    if sharpness < min_sharpness:
        reasons.append(f"sharpness is low ({sharpness:.0f})")

    gray = img.convert("L")
    stat = ImageStat.Stat(gray)
    brightness = stat.mean[0]
    contrast = stat.stddev[0]
    if brightness < 25:
        reasons.append("nose crop is too dark")
    elif brightness > 235:
        reasons.append("nose crop is too bright")
    if contrast < 8:
        reasons.append("nose crop has very low contrast")

    return len(reasons) == 0, reasons


def prepare_sample(img: Image.Image, min_sharpness: float) -> dict:
    """Detect/crop nose, score quality, and prepare embeddings."""
    original = img.convert("RGB")
    detected = nose_detector.detect_and_crop(original)
    nose_vec, sharpness = embed_pil(detected.crop)
    face_vec, _ = embed_pil(original)
    quality_ok, quality_reasons = image_quality(detected.crop, sharpness, min_sharpness)

    return {
        "original": original,
        "nose_crop": detected.crop,
        "nose_vec": nose_vec,
        "face_vec": face_vec,
        "sharpness": sharpness,
        "bbox": detected.bbox,
        "detector_confidence": detected.confidence,
        "detector_source": detected.source,
        "quality_ok": quality_ok,
        "quality_reasons": quality_reasons,
    }


def show_prepared_samples(samples: List[dict], max_show: int = 6) -> None:
    subset = samples[:max_show]
    if not subset:
        return
    cols = st.columns(len(subset))
    for col, sample in zip(cols, subset):
        with col:
            st.image(sample["nose_crop"], width='stretch')
            status = "OK" if sample["quality_ok"] else "WARN"
            st.caption(
                f"{status} sharp {sample['sharpness']:.0f} | "
                f"{sample['detector_source']} {sample['detector_confidence']:.2f}"
            )


def max_cosine(query_vecs, emb_list) -> float:
    best = 0.0
    for qv in query_vecs:
        for ev in emb_list:
            sim = F.cosine_similarity(qv.unsqueeze(0), ev.unsqueeze(0)).item()
            best = max(best, sim)
    return best


def weighted_identity_score(nose_score: float, face_score: Optional[float]) -> float:
    if face_score is None:
        return nose_score
    return (NOSE_WEIGHT * nose_score) + (FACE_WEIGHT * face_score)


def match_to_gallery(query_samples, gallery):
    """Weighted ensemble match. Nose is primary; face is secondary when present."""
    results = []
    query_nose = [s["nose_vec"] for s in query_samples]
    query_face = [s["face_vec"] for s in query_samples if s.get("face_vec") is not None]
    for name, emb_groups in gallery.items():
        nose_score = max_cosine(query_nose, emb_groups["nose"])
        face_score = None
        if query_face and emb_groups.get("face"):
            face_score = max_cosine(query_face, emb_groups["face"])
        final_score = weighted_identity_score(nose_score, face_score)
        results.append((name, final_score, nose_score, face_score))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def render_frame_strip(pil_sharp_pairs, max_show: int = 8):
    subset = pil_sharp_pairs[:max_show]
    if not subset:
        return
    cols = st.columns(len(subset))
    for col, (pil_img, score) in zip(cols, subset):
        with col:
            st.image(pil_img, width='stretch')
            icon = "OK" if score > 300 else (".." if score > 150 else "!!")
            st.caption(f"{icon} {score:.0f}")


# ==============================================================================
# Sidebar
# ==============================================================================

with st.sidebar:
    st.markdown(
        "<div style='padding:16px 0 12px 0; font-size:1.1rem; font-weight:600; color:#9ca3af;'>DogID</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    choice = st.radio(
        "nav",
        ["Identify", "Register", "Manage", "Database"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("DNNetV3 - TinyViT-21M - MagFace")


# ==============================================================================
# Page: IDENTIFY
# ==============================================================================

if choice == "Identify":
    st.header("Identify Dog")
    st.caption("Upload one or more dog photos. The app crops the nose before matching.")

    query_samples: list = []

    uploaded_photos = st.file_uploader(
        "Upload one or more dog images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="id_img_up",
    )
    if uploaded_photos:
        imgs = [Image.open(f).convert("RGB") for f in uploaded_photos]
        thumb_cols = st.columns(min(len(imgs), 6))
        for col, img in zip(thumb_cols, imgs):
            col.image(img, width='stretch')
        with st.spinner("Processing photos..."):
            for img in imgs:
                sample = prepare_sample(img, IDENTIFY_MIN_SHARPNESS)
                if not sample["quality_ok"]:
                    st.caption("Image quality warning: " + "; ".join(sample["quality_reasons"]))
                query_samples.append(sample)
        st.markdown('<div class="section-hdr">Detected Nose Crops</div>', unsafe_allow_html=True)
        show_prepared_samples(query_samples)
        if not nose_detector.has_model:
            st.info("No nose detector weights found yet, so the app is using a center-crop fallback.")
        st.caption(f"{len(imgs)} photo(s) loaded.")

    if query_samples:
        st.divider()

        st.markdown('<div class="section-hdr">Filter Search (Optional)</div>', unsafe_allow_html=True)
        breeds = list_breeds()
        filter_breed = None
        if breeds:
            c_filter, _ = st.columns([1, 1])
            with c_filter:
                filter_breed = st.selectbox("Narrow search by breed", ["All Breeds"] + breeds)
                if filter_breed == "All Breeds":
                    filter_breed = None

        gallery = build_gallery(breed=filter_breed)

        if not gallery:
            st.error("No dogs registered yet. Head to **Register** to add one.")
        else:
            with st.spinner("Matching against gallery..."):
                results = match_to_gallery(query_samples, gallery)

            st.subheader("Top Matches")
            top3 = results[:3]
            cols = st.columns(len(top3))
            for i, (name, sim, nose_sim, face_sim) in enumerate(top3):
                with cols[i]:
                    st.metric(label=name, value=f"{sim*100:.1f}%")
                    face_text = "n/a" if face_sim is None else f"{face_sim*100:.1f}%"
                    st.caption(f"nose {nose_sim*100:.1f}% | face {face_text}")
                    if i == 0:
                        if sim >= MATCH_THRESH:
                            st.success("Match")
                        else:
                            st.error("No match")

            top_name, top_sim, _top_nose_sim, _top_face_sim = top3[0]
            st.divider()

            if top_sim >= MATCH_THRESH:
                st.success(f"Best match: **{top_name}** ({top_sim*100:.1f}%)")
                if st.button("Confirm match", type="primary", key="confirm_match"):
                    log_dir = os.path.join(GALLERY_DIR, "_logs")
                    os.makedirs(log_dir, exist_ok=True)
                    log_path = os.path.join(log_dir, "identifications.log")
                    with open(log_path, "a", encoding="utf-8") as f:
                        f.write(f"{int(time.time())}\t{top_name}\t{top_sim:.4f}\t1\n")
                    st.success(f"Identification of **{top_name}** logged.")
            else:
                st.warning(
                    "No confident match found. "
                    "The closest match is below the identification threshold."
                )
                st.markdown(
                    "Know who this is? Link these frames to their record "
                    "to improve future recognition accuracy."
                )
                all_names = list_dogs()
                if all_names:
                    rc1, rc2 = st.columns([3, 1])
                    with rc1:
                        selected = st.selectbox("Select dog", all_names, key="no_match_select")
                    with rc2:
                        st.write(""); st.write("")
                        if st.button("Add Frames", type="primary", key="no_match_add"):
                            for sample in query_samples[:5]:
                                save_embedding(
                                    selected,
                                    sample["nose_vec"],
                                    sample["nose_crop"],
                                    sample["sharpness"],
                                    original_img=sample["original"],
                                    face_embedding=sample["face_vec"],
                                    crop_bbox=sample["bbox"],
                                    detector_confidence=sample["detector_confidence"],
                                    detector_source=sample["detector_source"],
                                )
                            log_dir = os.path.join(GALLERY_DIR, "_logs")
                            os.makedirs(log_dir, exist_ok=True)
                            with open(os.path.join(log_dir, "identifications.log"), "a", encoding="utf-8") as f:
                                f.write(f"{int(time.time())}\t{selected}\t{top_sim:.4f}\t1\n")
                            st.success(f"Frames added to **{selected}**'s record!")
                            st.balloons()
                else:
                    st.info("No dogs in the gallery yet.")


# ==============================================================================
# Page: REGISTER
# ==============================================================================

elif choice == "Register":
    st.header("Register Dog")
    st.caption("Add a new dog. Full-face photos are cropped to the nose before saving.")

    st.markdown('<div class="section-hdr">Dog Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    name   = c1.text_input("Name", placeholder="e.g. Buddy")
    age    = c2.number_input("Age (years)", min_value=0.0, step=0.5)
    c3, c4 = st.columns(2)
    color  = c3.text_input("Color / Markings", placeholder="e.g. Golden")
    weight = c4.number_input("Weight (kg)", min_value=0.0, step=0.5)

    st.markdown('<div class="section-hdr">Breed</div>', unsafe_allow_html=True)
    existing_breeds = list_breeds()
    nb_toggle = st.toggle("Add a new breed")
    breed: Optional[str] = None

    if nb_toggle:
        nb1, nb2, nb3 = st.columns(3)
        nb_name   = nb1.text_input("Breed Name", key="nb_name")
        nb_origin = nb2.text_input("Origin",     key="nb_origin")
        nb_weight = nb3.text_input("Typical Weight", key="nb_weight")
        if nb_name:
            breed = nb_name.strip()
            st.success(f"Breed '{breed}' will be saved.")
    else:
        if existing_breeds:
            breed = st.selectbox("Breed", existing_breeds)
        else:
            st.caption("No breeds yet - toggle above to add one.")

    st.markdown('<div class="section-hdr">Upload</div>', unsafe_allow_html=True)
    reg_frames: list = []

    uploaded_photos = st.file_uploader(
        "Upload one or more dog images",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=True,
        key="reg_img_up",
    )
    if uploaded_photos:
        imgs = [Image.open(f).convert("RGB") for f in uploaded_photos]
        thumb_cols = st.columns(min(len(imgs), 6))
        for col, img in zip(thumb_cols, imgs):
            col.image(img, width='stretch')
        for img in imgs:
            sample = prepare_sample(img, REGISTER_MIN_SHARPNESS)
            if not sample["quality_ok"]:
                st.caption("Registration quality warning: " + "; ".join(sample["quality_reasons"]))
            reg_frames.append(sample)
        st.markdown('<div class="section-hdr">Detected Nose Crops</div>', unsafe_allow_html=True)
        show_prepared_samples(reg_frames)
        if not nose_detector.has_model:
            st.info("No nose detector weights found yet, so the app is using a center-crop fallback.")
        st.caption(f"{len(imgs)} photo(s) loaded.")

    st.divider()
    if reg_frames:
        good_frames = [f for f in reg_frames if f["quality_ok"]]
        st.markdown(f"**Ready to register with {len(good_frames)} usable frame(s).** Click below to save.")
        if st.button("Register Dog", type="primary", key="reg_save"):
            if not name:
                st.error("Please enter a dog name first.")
            elif not good_frames:
                st.error("No uploaded image passed the registration quality checks.")
            else:
                safe = safe_dirname(name)
                # Write / update meta.json first so the directory is initialized.
                write_meta(
                    safe,
                    breed=breed,
                    age_years=age or None,
                    weight_kg=weight or None,
                    color=color or None,
                    registered_at=time.strftime("%Y-%m-%d %H:%M:%S"),
                    nose_weight=NOSE_WEIGHT,
                    face_weight=FACE_WEIGHT,
                    nose_detector_path=NOSE_DETECTOR_PATH,
                    nose_detector_active=nose_detector.has_model,
                )
                prog = st.progress(0, text="Embedding frames...")
                for i, sample in enumerate(good_frames):
                    save_embedding(
                        safe,
                        sample["nose_vec"],
                        sample["nose_crop"],
                        sample["sharpness"],
                        original_img=sample["original"],
                        face_embedding=sample["face_vec"],
                        crop_bbox=sample["bbox"],
                        detector_confidence=sample["detector_confidence"],
                        detector_source=sample["detector_source"],
                    )
                    prog.progress((i + 1) / len(good_frames), text=f"Saving frame {i+1}/{len(good_frames)}...")
                prog.empty()
                st.success(f"{name} registered with {len(good_frames)} frame(s) -> data/gallery/{safe}/")
                st.balloons()
    else:
        st.caption("Upload photos above to enable registration.")


# ==============================================================================
# Page: MANAGE
# ==============================================================================

elif choice == "Manage":
    st.header("Manage Dogs")
    st.caption("View and manage dog records and their stored embeddings.")

    dogs = list_dogs()
    if not dogs:
        st.info("No dogs registered yet. Head to **Register** to add one.")
    else:
        st.markdown(f"**{len(dogs)} dog(s) in the gallery**")
        st.write("")

        for name in dogs:
            meta = read_meta(name) or {}
            stats = dog_stats(name)
            breed_str = meta.get("breed") or "Unknown breed"
            label = (
                f"{name} - {breed_str} - "
                f"{stats['emb_count']} frame(s) - "
                f"Avg sharpness: {stats['avg_sharpness']:.1f}"
            )
            with st.expander(label):
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Frame Embeddings", stats["emb_count"])
                mc2.metric("Avg Sharpness",    f"{stats['avg_sharpness']:.1f}")
                mc3.metric("Best Sharpness",   f"{stats['best_sharpness']:.1f}")
                st.caption(f"Registered: {meta.get('registered_at', '-')}")

                # Show a small strip of thumbnails.
                frames = list_dog_frames(name)
                jpgs = [f["jpg"] for f in frames if f["jpg"]]
                if jpgs:
                    imgs = [Image.open(j) for j in jpgs[:8]]
                    strip = st.columns(len(imgs))
                    for col, img in zip(strip, imgs):
                        col.image(img, width='stretch')

                st.divider()
                b1, b2 = st.columns(2)
                with b1:
                    if st.button("Clear Embeddings Only",
                                 key=f"clr_{name}",
                                 help="Remove all stored frames but keep the dog record."):
                        clear_dog_embeddings(name)
                        st.warning(f"All embeddings cleared for **{name}**. "
                                   "Record is preserved.")
                        st.rerun()
                with b2:
                    if st.button(f"Delete {name} Entirely",
                                 key=f"del_{name}",
                                 help="Permanently delete this dog and all their embeddings."):
                        delete_dog(name)
                        st.rerun()


# ==============================================================================
# Page: DATABASE
# ==============================================================================

elif choice == "Database":
    st.header("Gallery")
    st.caption(f"Folder: `{GALLERY_DIR}/`")

    dogs   = list_dogs()
    breeds = list_breeds()
    n_embs = 0
    for d in dogs:
        n_embs += dog_stats(d)["emb_count"]
    n_logs = 0
    log_path = os.path.join(GALLERY_DIR, "_logs", "identifications.log")
    if os.path.exists(log_path):
        with open(log_path, "r", encoding="utf-8") as f:
            n_logs = sum(1 for _ in f)

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in zip(
        [c1, c2, c3, c4],
        [len(dogs), len(breeds), n_embs, n_logs],
        ["Dogs", "Breeds", "Frame Embeddings", "Identifications"],
    ):
        col.markdown(
            f'<div class="stat-card">'
            f'<div class="stat-number">{num}</div>'
            f'<div class="stat-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("Per-Dog Summary")
    if dogs:
        import pandas as pd
        rows = []
        for d in dogs:
            meta  = read_meta(d) or {}
            stats = dog_stats(d)
            rows.append({
                "Name":          d,
                "Breed":         meta.get("breed", "-"),
                "Frames Stored": stats["emb_count"],
                "Avg Sharpness": round(stats["avg_sharpness"], 1),
                "Best Sharpness": round(stats["best_sharpness"], 1),
                "Registered At": meta.get("registered_at", "-"),
            })
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True)
    else:
        st.info("No data yet.")
