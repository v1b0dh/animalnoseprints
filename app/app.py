"""
app.py — DogID System: 360° Video-Based Dog Nose Biometric Recognition
=======================================================================
New workflow:
  REGISTER : Upload a short 360° nose/face video → auto-extract frames →
             quality-filter by sharpness → embed all good frames → store.
  IDENTIFY : Upload a short video (or single photo) → multi-frame query →
             ensemble max-cosine-similarity against all stored embeddings.
"""

import streamlit as st
import sqlite3
import torch
import torch.nn.functional as F
import numpy as np
import io
import os
import pandas as pd
from PIL import Image

from det5 import DNNetV3, CLAHEPipeline
from video_utils import pil_frames_from_upload

DB_NAME          = "data/dog_biometrics.db"
MATCH_THRESH     = 0.82       # Minimum cosine similarity to call a match
QUALITY_WARN     = 100.0      # Sharpness below this → warn user
VIDEO_MAX_FRAMES = 30         # Frames to evenly sample from video
VIDEO_MIN_SHARP  = 100.0      # Minimum Laplacian score to keep a frame
VIDEO_TOP_N      = 20         # Maximum frames to store per session
VIDEO_MIN_PASS   = 5          # Minimum good frames required to proceed

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DogID System", layout="wide", page_icon="🐾")

# ── Global CSS ─────────────────────────────────────────────────────────────────
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
# DATABASE HELPERS
# ==============================================================================

def get_conn():
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn


def get_breeds():
    conn = get_conn()
    rows = conn.execute("SELECT id, breed_name FROM breeds ORDER BY breed_name").fetchall()
    conn.close()
    return rows


def add_breed(name, origin="", weight=""):
    conn = get_conn()
    try:
        conn.execute(
            "INSERT INTO breeds (breed_name, origin, typical_weight_kg) VALUES (?,?,?)",
            (name, origin, weight)
        )
        conn.commit()
    except Exception:
        pass
    row = conn.execute("SELECT id FROM breeds WHERE breed_name=?", (name,)).fetchone()
    conn.close()
    return row["id"] if row else None


def get_owners():
    conn = get_conn()
    rows = conn.execute("SELECT id, name FROM owners ORDER BY name").fetchall()
    conn.close()
    return rows


def add_owner(name, phone="", email="", address=""):
    conn = get_conn()
    cur  = conn.execute(
        "INSERT INTO owners (name, phone, email, address) VALUES (?,?,?,?)",
        (name, phone, email, address)
    )
    oid = cur.lastrowid
    conn.commit()
    conn.close()
    return oid


def get_or_create_dog(name, breed_id=None, owner_id=None, age=None, weight=None, color=None):
    conn = get_conn()
    row  = conn.execute("SELECT id FROM dogs WHERE name=?", (name,)).fetchone()
    if row:
        dog_id = row["id"]
    else:
        cur    = conn.execute(
            "INSERT INTO dogs (name, breed_id, owner_id, age_years, weight_kg, color) VALUES (?,?,?,?,?,?)",
            (name, breed_id, owner_id, age, weight, color)
        )
        dog_id = cur.lastrowid
        conn.commit()
    conn.close()
    return dog_id


def add_embedding(dog_id, embedding, photo_bytes, sharpness, model_type="teacher"):
    blob = embedding.detach().cpu().numpy().astype(np.float32).tobytes()
    conn = get_conn()
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute(
        "INSERT INTO embeddings (dog_id, embedding, photo, sharpness, model_type) VALUES (?,?,?,?,?)",
        (dog_id, blob, photo_bytes, float(sharpness), model_type)
    )
    conn.commit()
    conn.close()


def log_identification(dog_id, similarity, confirmed):
    conn = get_conn()
    conn.execute(
        "INSERT INTO identifications (dog_id, similarity, confirmed) VALUES (?,?,?)",
        (dog_id, float(similarity), confirmed)
    )
    conn.commit()
    conn.close()


def get_dog_gallery(model_type="teacher", breed_id=None):
    """Return {dog_name: [embedding_tensor, ...]} from the DB."""
    conn = get_conn()
    
    if breed_id is not None:
        rows = conn.execute(
            "SELECT d.name, e.embedding FROM dogs d "
            "JOIN embeddings e ON e.dog_id = d.id WHERE e.model_type = ? AND d.breed_id = ?",
            (model_type, breed_id)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT d.name, e.embedding FROM dogs d "
            "JOIN embeddings e ON e.dog_id = d.id WHERE e.model_type = ?",
            (model_type,)
        ).fetchall()
        
    conn.close()

    gallery: dict = {}
    for r in rows:
        vec = np.frombuffer(r["embedding"], dtype=np.float32)
        gallery.setdefault(r["name"], []).append(torch.from_numpy(vec.copy()))
    return gallery


def get_all_dog_names():
    conn = get_conn()
    rows = conn.execute("SELECT name FROM dogs ORDER BY name").fetchall()
    conn.close()
    return [r["name"] for r in rows]


def img_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    thumb.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


# ==============================================================================
# MODEL
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


# ==============================================================================
# INFERENCE HELPERS
# ==============================================================================

def embed_pil(img: Image.Image):
    """Embed a single PIL image → (tensor, sharpness)."""
    t, s = pipeline(img)
    with torch.no_grad():
        vec = model(t.unsqueeze(0)).squeeze(0)
    return vec, s


def embed_frame_list(frames_with_scores):
    """
    Embed a list of (PIL.Image, sharpness) frames.
    Returns list of (embedding_tensor, sharpness, PIL.Image).
    """
    results = []
    for pil_img, sharpness in frames_with_scores:
        vec, _ = embed_pil(pil_img)
        results.append((vec, sharpness, pil_img))
    return results


def match_to_gallery(query_vecs, gallery):
    """
    Ensemble max-cosine-similarity: for each registered dog,
    take the best similarity across ALL (query_frame × stored_embedding) pairs.

    Returns a list of (name, best_sim) sorted descending.
    """
    results = []
    for name, emb_list in gallery.items():
        best = 0.0
        for qv in query_vecs:
            for ev in emb_list:
                sim  = F.cosine_similarity(qv.unsqueeze(0), ev.unsqueeze(0)).item()
                best = max(best, sim)
        results.append((name, best))
    results.sort(key=lambda x: x[1], reverse=True)
    return results


def render_frame_strip(frames_with_scores, max_show: int = 8):
    """Render a horizontal thumbnail strip with sharpness badges."""
    subset = frames_with_scores[:max_show]
    if not subset:
        return
    cols = st.columns(len(subset))
    for col, (pil_img, score) in zip(cols, subset):
        with col:
            st.image(pil_img, use_container_width=True)
            icon = "🟢" if score > 300 else ("🟡" if score > 150 else "🔴")
            st.caption(f"{icon} {score:.0f}")


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================

with st.sidebar:
    st.markdown(
        "<div style='padding:16px 0 12px 0; font-size:1.1rem; font-weight:600; color:#9ca3af;'>DogID</div>",
        unsafe_allow_html=True,
    )
    st.divider()
    choice = st.radio(
        "nav",
        ["🔍 Identify", "📝 Register", "⚙️ Manage", "📊 Database"],
        label_visibility="collapsed"
    )
    st.divider()
    st.caption("DNNetV3 · TinyViT-21M · MagFace")


# ==============================================================================
# PAGE — IDENTIFY
# ==============================================================================

if choice == "🔍 Identify":
    st.header("Identify Dog")
    st.caption("Upload a video or photos of the dog's nose to find a match.")

    tab_vid, tab_photo = st.tabs(["Video", "Photo Upload"])

    query_vecs:   list = []
    query_frames: list = []   # (PIL, sharpness)

    # ── VIDEO TAB ─────────────────────────────────────────────────────────────
    with tab_vid:
        vid_file = st.file_uploader(
            "Upload a short nose/face video",
            type=["mp4", "mov", "avi", "mkv"],
            key="id_vid",
            help="5–15 seconds is ideal. Keep the camera steady."
        )
        if vid_file:
            with st.spinner("Extracting frames and scoring quality…"):
                good_frames, total_extracted = pil_frames_from_upload(
                    vid_file.read(),
                    max_frames=VIDEO_MAX_FRAMES,
                    min_sharpness=VIDEO_MIN_SHARP,
                    top_n=VIDEO_TOP_N,
                )

            qa1, qa2, qa3 = st.columns(3)
            qa1.metric("Frames Sampled",   total_extracted)
            qa2.metric("Quality Passed",   len(good_frames))
            qa3.metric("Querying With",    min(len(good_frames), VIDEO_TOP_N))

            if len(good_frames) < VIDEO_MIN_PASS:
                st.error(
                    f"❌ Only **{len(good_frames)}** usable frame(s) found "
                    f"(minimum {VIDEO_MIN_PASS}). "
                    "Try recording in better lighting or holding the camera steadier."
                )
            else:
                st.success(f"✅ {len(good_frames)} quality frame(s) ready.")
                st.markdown("**Sample frames (best sharpness first):**")
                render_frame_strip(good_frames, max_show=8)

                with st.spinner("Generating embeddings…"):
                    embedded     = embed_frame_list(good_frames)
                    query_vecs   = [e[0] for e in embedded]
                    query_frames = good_frames

    # ── PHOTO TAB ─────────────────────────────────────────────────────────────
    with tab_photo:
        uploaded_photos = st.file_uploader(
            "Upload one or more nose images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key="id_img_up",
        )
        if uploaded_photos:
            imgs = [Image.open(f).convert("RGB") for f in uploaded_photos]
            # Show thumbnails
            thumb_cols = st.columns(min(len(imgs), 6))
            for col, img in zip(thumb_cols, imgs):
                col.image(img, use_container_width=True)
            # Embed all uploaded photos
            with st.spinner("Processing photos…"):
                for img in imgs:
                    vec, s = embed_pil(img)
                    if s < QUALITY_WARN:
                        st.caption(f"Low sharpness ({s:.0f}) on one image — may affect accuracy.")
                    query_vecs.append(vec)
                    query_frames.append((img, s))
            st.caption(f"{len(imgs)} photo(s) loaded.")

    # ── FILTER & MATCHING ─────────────────────────────────────────────────────
    if query_vecs:
        st.divider()
        
        st.markdown('<div class="section-hdr">Filter Search (Optional)</div>', unsafe_allow_html=True)
        breeds_rows = get_breeds()
        b_names     = [r["breed_name"] for r in breeds_rows]
        b_ids       = [r["id"]         for r in breeds_rows]
        
        filter_breed_id = None
        if b_names:
            c_filter, _ = st.columns([1, 1])
            with c_filter:
                filter_b = st.selectbox("Narrow search by breed", ["All Breeds"] + b_names)
                if filter_b != "All Breeds":
                    filter_breed_id = b_ids[b_names.index(filter_b)]
                    
        gallery = get_dog_gallery(breed_id=filter_breed_id)

        if not gallery:
            st.error("No dogs registered yet. Head to **📝 Register** to add one.")
        else:
            with st.spinner("Matching against gallery…"):
                results = match_to_gallery(query_vecs, gallery)

            st.subheader("Top Matches")
            top3 = results[:3]
            cols = st.columns(3)
            for i, (name, sim) in enumerate(top3):
                with cols[i]:
                    st.metric(label=name, value=f"{sim*100:.1f}%")
                    if i == 0:
                        if sim >= MATCH_THRESH:
                            st.success("Match")
                        else:
                            st.error("No match")

            top_name, top_sim = top3[0]
            st.divider()

            if top_sim >= MATCH_THRESH:
                if st.button("✅ Confirm & Log Match", type="primary", key="confirm_match"):
                    conn = get_conn()
                    row  = conn.execute("SELECT id FROM dogs WHERE name=?", (top_name,)).fetchone()
                    if row:
                        log_identification(row["id"], top_sim, 1)
                        st.success(f"Identification of **{top_name}** logged.")
                    conn.close()
            else:
                st.warning(
                    "❌ **No confident match found.** "
                    "The closest match is below the identification threshold."
                )
                st.markdown(
                    "**Know who this is?** Link these frames to their record "
                    "to improve future recognition accuracy."
                )
                all_names = get_all_dog_names()
                if all_names:
                    rc1, rc2 = st.columns([3, 1])
                    with rc1:
                        selected = st.selectbox("Select dog", all_names, key="no_match_select")
                    with rc2:
                        st.write(""); st.write("")
                        if st.button("➕ Add Frames", type="primary", key="no_match_add"):
                            conn = get_conn()
                            row  = conn.execute("SELECT id FROM dogs WHERE name=?", (selected,)).fetchone()
                            if row:
                                dog_id = row["id"]
                                for vec, sharpness, pil_img in embed_frame_list(query_frames)[:5]:
                                    add_embedding(dog_id, vec, img_to_bytes(pil_img), sharpness)
                                log_identification(dog_id, top_sim, 1)
                                st.success(f"✅ Frames added to **{selected}**'s record!")
                                st.balloons()
                            conn.close()
                else:
                    st.info("No dogs in the database yet.")


# ==============================================================================
# PAGE — REGISTER
# ==============================================================================

elif choice == "📝 Register":
    st.header("Register Dog")
    st.caption("Add a new dog to the system using a video or photos of their nose.")

    # ── Dog Details ────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Dog Details</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    name   = c1.text_input("Name", placeholder="e.g. Buddy")
    age    = c2.number_input("Age (years)", min_value=0.0, step=0.5)
    c3, c4 = st.columns(2)
    color  = c3.text_input("Color / Markings", placeholder="e.g. Golden")
    weight = c4.number_input("Weight (kg)", min_value=0.0, step=0.5)

    # ── Breed ──────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Breed</div>', unsafe_allow_html=True)
    breeds_rows = get_breeds()
    b_names     = [r["breed_name"] for r in breeds_rows]
    b_ids       = [r["id"]         for r in breeds_rows]
    nb_toggle   = st.toggle("Add a new breed")
    breed_id    = None

    if nb_toggle:
        nb1, nb2, nb3 = st.columns(3)
        nb_name   = nb1.text_input("Breed Name", key="nb_name")
        nb_origin = nb2.text_input("Origin",     key="nb_origin")
        nb_weight = nb3.text_input("Typical Weight", key="nb_weight")
        if nb_name:
            breed_id = add_breed(nb_name, nb_origin, nb_weight)
            st.success(f"Breed '{nb_name}' added.")
    else:
        if b_names:
            sel_b    = st.selectbox("Breed", b_names)
            breed_id = b_ids[b_names.index(sel_b)]
        else:
            st.caption("No breeds yet — toggle above to add one.")

    owner_id = None

    # ── Upload ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hdr">Upload</div>', unsafe_allow_html=True)

    reg_frames: list = []   # (PIL.Image, sharpness)

    vid_tab, photo_tab = st.tabs(["Video", "Photo Upload"])

    with vid_tab:
        vid_file = st.file_uploader(
            "Upload video (mp4, mov, avi, mkv)",
            type=["mp4", "mov", "avi", "mkv"],
            key="reg_vid",
        )
        if vid_file:
            with st.spinner(f"Sampling up to {VIDEO_MAX_FRAMES} frames and checking quality…"):
                good_frames, total_extracted = pil_frames_from_upload(
                    vid_file.read(),
                    max_frames=VIDEO_MAX_FRAMES,
                    min_sharpness=VIDEO_MIN_SHARP,
                    top_n=VIDEO_TOP_N,
                )

            qa1, qa2, qa3 = st.columns(3)
            qa1.metric("Frames Sampled", total_extracted)
            qa2.metric("Quality Passed", len(good_frames),
                       delta=f"min {VIDEO_MIN_PASS} needed")
            qa3.metric("Will Store",     min(len(good_frames), VIDEO_TOP_N))

            if len(good_frames) < VIDEO_MIN_PASS:
                st.error(
                    f"❌ Only **{len(good_frames)}** usable frame(s) "
                    f"(need at least {VIDEO_MIN_PASS}). "
                    "Record in brighter conditions or hold the camera steadier."
                )
            else:
                st.success(f"✅ {len(good_frames)} quality frame(s) extracted!")
                st.markdown("**Preview — best frames by sharpness:**")
                render_frame_strip(good_frames, max_show=8)
                reg_frames = good_frames

    with photo_tab:
        uploaded_photos = st.file_uploader(
            "Upload one or more nose images",
            type=["jpg", "png", "jpeg"],
            accept_multiple_files=True,
            key="reg_img_up",
        )
        if uploaded_photos:
            imgs = [Image.open(f).convert("RGB") for f in uploaded_photos]
            thumb_cols = st.columns(min(len(imgs), 6))
            for col, img in zip(thumb_cols, imgs):
                col.image(img, use_container_width=True)
            for img in imgs:
                _, s = pipeline(img)
                if s < QUALITY_WARN:
                    st.caption(f"Low sharpness ({s:.0f}) on one image.")
                reg_frames.append((img, s))
            st.caption(f"{len(imgs)} photo(s) loaded.")

    # ── Register Button ───────────────────────────────────────────────────────
    st.divider()
    if reg_frames:
        n = len(reg_frames)
        st.markdown(f"**Ready to register with {n} frame(s).** Click below to process and save.")
        if st.button("Register Dog", type="primary", key="reg_save"):
            if not name:
                st.error("Please enter a dog name first.")
            else:
                dog_id = get_or_create_dog(
                    name,
                    breed_id,
                    None,
                    age    or None,
                    weight or None,
                    color  or None,
                )
                prog = st.progress(0, text="Embedding frames…")
                for i, (pil_img, sharpness) in enumerate(reg_frames):
                    t, _ = pipeline(pil_img)
                    with torch.no_grad():
                        vec = model(t.unsqueeze(0)).squeeze(0)
                    add_embedding(dog_id, vec, img_to_bytes(pil_img), sharpness)
                    prog.progress((i + 1) / n, text=f"Embedding frame {i+1}/{n}…")
                prog.empty()
                st.success(f"{name} registered with {n} frame(s).")
                st.balloons()
    else:
        st.caption("Upload a video or photos above to enable registration.")


# ==============================================================================
# PAGE — MANAGE
# ==============================================================================

elif choice == "⚙️ Manage":
    st.header("Manage Dogs")
    st.caption("View and manage dog records and their stored embeddings.")

    conn = get_conn()
    dogs = conn.execute("""
        SELECT
            d.id,
            d.name,
            d.registered_at,
            COUNT(e.id)               AS emb_count,
            ROUND(AVG(e.sharpness),1) AS avg_sharpness,
            ROUND(MAX(e.sharpness),1) AS best_sharpness
        FROM dogs d
        LEFT JOIN embeddings e ON e.dog_id = d.id AND e.model_type = 'teacher'
        GROUP BY d.id
        ORDER BY d.name
    """).fetchall()
    conn.close()

    if not dogs:
        st.info("No dogs registered yet. Head to **📝 Register** to add one.")
    else:
        st.markdown(f"**{len(dogs)} dog(s) in the database**")
        st.write("")

        for d in dogs:
            label = (
                f"🐕 **{d['name']}**  —  "
                f"{d['emb_count']} frame(s) stored  |  "
                f"Avg sharpness: {d['avg_sharpness'] or 0:.1f}"
            )
            with st.expander(label):
                mc1, mc2, mc3 = st.columns(3)
                mc1.metric("Frame Embeddings", d["emb_count"])
                mc2.metric("Avg Sharpness",    f"{d['avg_sharpness'] or 0:.1f}")
                mc3.metric("Best Sharpness",   f"{d['best_sharpness'] or 0:.1f}")
                st.caption(f"Registered: {d['registered_at']}")

                st.divider()
                b1, b2 = st.columns(2)

                with b1:
                    if st.button(
                        "🗑️ Clear Embeddings Only",
                        key=f"clr_{d['id']}",
                        help="Remove all stored frames but keep the dog record."
                    ):
                        c = get_conn()
                        c.execute(
                            "DELETE FROM embeddings WHERE dog_id=? AND model_type='teacher'",
                            (d["id"],)
                        )
                        c.commit(); c.close()
                        st.warning(
                            f"All embeddings cleared for **{d['name']}**. "
                            "Record is preserved — re-register a video to restore."
                        )
                        st.rerun()

                with b2:
                    if st.button(
                        f"❌ Delete {d['name']} Entirely",
                        key=f"del_{d['id']}",
                        help="Permanently delete this dog and all their embeddings."
                    ):
                        c = get_conn()
                        c.execute("PRAGMA foreign_keys = ON")
                        c.execute("DELETE FROM dogs WHERE id=?", (d["id"],))
                        c.commit(); c.close()
                        st.rerun()


# ==============================================================================
# PAGE — DATABASE
# ==============================================================================

elif choice == "📊 Database":
    st.header("Database")

    conn    = get_conn()
    n_dogs  = conn.execute("SELECT COUNT(*) FROM dogs").fetchone()[0]
    n_own   = conn.execute("SELECT COUNT(*) FROM owners").fetchone()[0]
    n_embs  = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model_type='teacher'"
    ).fetchone()[0]
    n_ids   = conn.execute("SELECT COUNT(*) FROM identifications").fetchone()[0]
    conn.close()

    c1, c2, c3, c4 = st.columns(4)
    for col, num, label in zip(
        [c1, c2, c3, c4],
        [n_dogs, n_own, n_embs, n_ids],
        ["Dogs", "Owners", "Frame Embeddings", "Identifications"],
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

    conn = get_conn()
    rows = conn.execute("""
        SELECT
            d.name                         AS "Name",
            COUNT(e.id)                    AS "Frames Stored",
            ROUND(AVG(e.sharpness), 1)     AS "Avg Sharpness",
            ROUND(MAX(e.sharpness), 1)     AS "Best Sharpness",
            d.registered_at                AS "Registered At"
        FROM dogs d
        LEFT JOIN embeddings e ON e.dog_id = d.id AND e.model_type='teacher'
        GROUP BY d.id
        ORDER BY COUNT(e.id) DESC
    """).fetchall()
    conn.close()

    if rows:
        df = pd.DataFrame([dict(r) for r in rows])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.info("No data yet.")
