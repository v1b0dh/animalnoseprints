"""
app.py — Dog Nose Biometrics (DNNetV3 SOTA) — Streamlit Frontend
================================================================
Connects to:
  • det5.py  →  DNNetV3 model (TinyViT-21M, 1024-d embeddings)
                CLAHEPipeline (CLAHE + sharpness scoring)
  • dog_biometrics.db  →  SQLite (name, embedding BLOB, photo BLOB, timestamp)
"""

import streamlit as st
import sqlite3
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os
from datetime import datetime

# ── Model imports (det5 is the SOTA module) ────────────────────────────────────
from det5 import DNNetV3, CLAHEPipeline

# ── Constants ──────────────────────────────────────────────────────────────────
DB_NAME       = "dog_biometrics.db"
EMBED_DIM     = DNNetV3.EMBED_DIM          # 1024
MATCH_THRESH  = 0.82                        # Cosine similarity threshold
QUALITY_WARN  = 100.0                       # Laplacian variance below → warn user


# ==============================================================================
# DATABASE LAYER
# ==============================================================================

def init_db():
    """Create table if it doesn't exist. Schema is forward-compatible."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dogs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL,
            embedding     BLOB    NOT NULL,
            photo         BLOB,
            registered_at TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.commit()
    conn.close()


def register_dog_to_db(name: str, embedding: torch.Tensor, photo_bytes: bytes):
    """Insert one dog record. embedding must be 1024-d float32 tensor."""
    blob = embedding.detach().cpu().numpy().astype(np.float32).tobytes()
    conn = sqlite3.connect(DB_NAME)
    conn.execute(
        "INSERT INTO dogs (name, embedding, photo) VALUES (?, ?, ?)",
        (name, blob, photo_bytes),
    )
    conn.commit()
    conn.close()


def get_all_registered_dogs() -> list[dict]:
    """Return list of {id, name, vector, registered_at} dicts."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.execute(
        "SELECT id, name, embedding, registered_at FROM dogs ORDER BY registered_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()

    dogs = []
    for row_id, name, blob, ts in rows:
        vec_np = np.frombuffer(blob, dtype=np.float32)
        if len(vec_np) != EMBED_DIM:
            st.warning(
                f"⚠️ Skipping **{name}** (id={row_id}): "
                f"stored {len(vec_np)}-d vector, expected {EMBED_DIM}-d. "
                f"Delete and re-register this dog."
            )
            continue
        dogs.append({
            "id"           : row_id,
            "name"         : name,
            "vector"       : torch.from_numpy(vec_np.copy()),
            "registered_at": ts,
        })
    return dogs


def delete_dog_by_id(dog_id: int):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("DELETE FROM dogs WHERE id = ?", (dog_id,))
    conn.commit()
    conn.close()


def wipe_database():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)


def get_dog_count() -> int:
    if not os.path.exists(DB_NAME):
        return 0
    conn = sqlite3.connect(DB_NAME)
    count = conn.execute("SELECT COUNT(*) FROM dogs").fetchone()[0]
    conn.close()
    return count


# ==============================================================================
# MODEL LAYER
# ==============================================================================

@st.cache_resource(show_spinner="Loading DNNetV3 (TinyViT-21M)…")
def load_model_and_pipeline():
    """Load once, reuse across reruns. Cached until app restarts."""
    model    = DNNetV3(pretrained=True)
    model.eval()
    pipeline = CLAHEPipeline(image_size=224)
    return model, pipeline


@torch.no_grad()
def get_embedding(model: DNNetV3, pipeline: CLAHEPipeline, pil_img: Image.Image):
    """
    Returns:
        embedding  : (1024,) L2-normalized tensor on CPU
        sharpness  : float — Laplacian variance (quality score)
    """
    tensor, sharpness = pipeline(pil_img)
    tensor    = tensor.unsqueeze(0)          # (1, 3, 224, 224)
    embedding = model(tensor).squeeze(0)     # (1024,)
    return embedding, sharpness


# ==============================================================================
# APP
# ==============================================================================

st.set_page_config(
    page_title = "DNNetV3 | Dog Nose Biometrics",
    page_icon  = "🐾",
    layout     = "centered",
)

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()
model, pipeline = load_model_and_pipeline()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/dog.png", width=64)
    st.title("🐾 DNNetV3")
    st.caption("SOTA Dog Nose Biometrics")
    st.markdown("---")
    menu   = ["🔍 Identify Dog", "📝 Register Dog", "⚙️ Manage Database"]
    choice = st.radio("Navigation", menu, label_visibility="collapsed")
    st.markdown("---")
    st.metric("Dogs in DB", get_dog_count())
    st.caption("Model: TinyViT-21M | Embed: 1024-d | Loss: MagFace")

# ==============================================================================
# PAGE: IDENTIFY
# ==============================================================================
if choice == "🔍 Identify Dog":
    st.title("🔍 Identify Dog")
    st.markdown("Upload a nose photo to match it against the registered database.")

    file = st.file_uploader(
        "Upload Nose Photo", type=["jpg", "jpeg", "png"], key="identify_upload"
    )

    if file:
        img = Image.open(file).convert("RGB")
        col1, col2 = st.columns([1, 2])
        col1.image(img, caption="Query Image", use_container_width=True)

        with st.spinner("Running CLAHE → TinyViT-21M embedding…"):
            query_vec, sharpness = get_embedding(model, pipeline, img)

        # Quality warning
        quality_pct = min(sharpness / 500.0, 1.0)   # normalize to 0–1 (500 = sharp)
        col2.markdown("**Image Quality**")
        col2.progress(quality_pct, text=f"Sharpness: {sharpness:.0f}")
        if sharpness < QUALITY_WARN:
            col2.warning("⚠️ Low quality image — results may be less accurate.")

        registered = get_all_registered_dogs()

        if not registered:
            st.error("Database is empty. Register dogs first via **Register Dog**.")
        else:
            # Cosine similarity against all stored embeddings
            sims = [
                (dog["name"], dog["id"],
                 F.cosine_similarity(query_vec.unsqueeze(0),
                                     dog["vector"].unsqueeze(0)).item())
                for dog in registered
            ]
            sims.sort(key=lambda x: x[2], reverse=True)
            best_name, best_id, best_sim = sims[0]

            st.markdown("---")
            if best_sim >= MATCH_THRESH:
                st.success(f"✅ **Match Found: {best_name}**")
                st.progress(float(best_sim), text=f"Confidence: {best_sim:.4f}")
            else:
                st.error("❌ **No Match Found** — dog may not be registered.")
                st.info(
                    f"Closest: **{best_name}** — similarity {best_sim:.4f} "
                    f"(threshold: {MATCH_THRESH})"
                )

            with st.expander("📊 Full Ranking"):
                for name, _, sim in sims:
                    bar_pct = max(0.0, float(sim))
                    st.write(f"`{name}` — {sim:.4f}")
                    st.progress(bar_pct)


# ==============================================================================
# PAGE: REGISTER
# ==============================================================================
elif choice == "📝 Register Dog":
    st.title("📝 Register New Dog")
    st.markdown("Capture a clear nose photo for biometric enrollment.")

    dog_name = st.text_input("Dog's Name", placeholder="e.g. Buddy")
    file     = st.file_uploader(
        "Upload Clear Nose Photo", type=["jpg", "jpeg", "png"], key="register_upload"
    )

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Preview", width=260)

        # Show quality indicator before committing
        with st.spinner("Analysing image quality…"):
            _, sharpness = get_embedding(model, pipeline, img)   # quick pre-check
        q_pct = min(sharpness / 500.0, 1.0)
        st.progress(q_pct, text=f"Sharpness score: {sharpness:.0f}")
        if sharpness < QUALITY_WARN:
            st.warning("⚠️ Image is blurry — consider uploading a sharper photo for better accuracy.")

    can_register = bool(dog_name and file)
    if st.button("✅ Register Dog", disabled=not can_register, type="primary"):
        with st.spinner(f"Extracting {EMBED_DIM}-d biometric fingerprint…"):
            img       = Image.open(file).convert("RGB")
            emb, sharpness = get_embedding(model, pipeline, img)

            # Store photo as JPEG bytes for display in Manage page
            buf = io.BytesIO()
            img.thumbnail((256, 256))
            img.save(buf, format="JPEG", quality=85)
            photo_bytes = buf.getvalue()

            register_dog_to_db(dog_name.strip(), emb, photo_bytes)

        st.success(f"🎉 **{dog_name}** registered successfully!")
        st.metric("Embedding Dimension", f"{EMBED_DIM}-d")
        st.metric("Image Sharpness", f"{sharpness:.0f}")
        st.balloons()


# ==============================================================================
# PAGE: MANAGE DATABASE
# ==============================================================================
elif choice == "⚙️ Manage Database":
    st.title("⚙️ Manage Database")

    dogs = get_all_registered_dogs()
    st.metric("Total Registered Dogs", len(dogs))

    if not dogs:
        st.info("No dogs registered yet.")
    else:
        st.markdown("---")
        for dog in dogs:
            with st.expander(f"🐶 {dog['name']}  —  id={dog['id']}"):
                col1, col2 = st.columns([1, 2])
                col1.caption(f"Registered: {dog['registered_at']}")
                col1.caption(f"Embedding: {EMBED_DIM}-d float32")
                col1.caption(f"Norm: {dog['vector'].norm().item():.3f}")

                # Per-dog delete button
                if col2.button(
                    f"🗑️ Delete {dog['name']}",
                    key=f"del_{dog['id']}",
                    type="secondary",
                ):
                    delete_dog_by_id(dog["id"])
                    st.success(f"Deleted **{dog['name']}** (id={dog['id']}).")
                    st.rerun()

    st.markdown("---")
    st.subheader("🚨 Danger Zone")
    with st.expander("Wipe entire database"):
        st.warning(
            "This permanently deletes **all** registered dogs and cannot be undone."
        )
        confirm = st.text_input(
            'Type  `WIPE`  to confirm', key="wipe_confirm", placeholder="WIPE"
        )
        if st.button("Delete All Dogs", type="primary", disabled=(confirm != "WIPE")):
            wipe_database()
            init_db()   # Recreate empty DB so app doesn't crash
            st.cache_resource.clear()
            st.success("Database wiped. All entries deleted.")
            st.rerun()