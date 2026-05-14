"""
app.py — Dog Nose Biometrics (DNNetV3 SOTA) — Streamlit Frontend
================================================================
v2.0 — Multi-Embedding Architecture (Strategy B: Max Similarity)

Connects to:
  • det5.py  →  DNNetV3 model (TinyViT-21M, 1024-d embeddings)
                CLAHEPipeline (CLAHE + sharpness scoring)
  • dog_biometrics.db  →  SQLite with two-table schema:
        dogs       : id, name, registered_at
        embeddings : id, dog_id (FK), embedding BLOB, photo BLOB, sharpness, added_at

Matching Strategy: B (Max Similarity)
  → For each dog, compare query against ALL stored embeddings.
  → The dog's score = highest cosine similarity across all its embeddings.
"""

import streamlit as st
import sqlite3
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os

# ── Model imports (det5 is the SOTA module) ────────────────────────────────────
from det5 import DNNetV3, CLAHEPipeline

# ── Constants ──────────────────────────────────────────────────────────────────
DB_NAME          = "dog_biometrics.db"
EMBED_DIM        = DNNetV3.EMBED_DIM          # 1024
MATCH_THRESH     = 0.82                        # Cosine similarity threshold
ENROLL_THRESH    = 0.88                        # Minimum confidence for auto-enrollment
QUALITY_WARN     = 100.0                       # Laplacian variance below → warn user
QUALITY_MIN_ENR  = 150.0                       # Minimum sharpness for auto-enrollment
MAX_EMB_PER_DOG  = 20                          # Cap embeddings per identity


# ==============================================================================
# DATABASE LAYER — Two-Table Schema
# ==============================================================================

def init_db():
    """Create the two-table schema if it doesn't exist."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS dogs (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            name          TEXT    NOT NULL UNIQUE,
            registered_at TEXT    DEFAULT (datetime('now','localtime'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS embeddings (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            dog_id    INTEGER NOT NULL,
            embedding BLOB    NOT NULL,
            photo     BLOB,
            sharpness REAL    DEFAULT 0.0,
            added_at  TEXT    DEFAULT (datetime('now','localtime')),
            FOREIGN KEY (dog_id) REFERENCES dogs(id) ON DELETE CASCADE
        )
    """)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.commit()
    conn.close()


def get_or_create_dog(name: str) -> int:
    """
    Return the dog_id for the given name.
    Creates a new dog row if the name doesn't exist yet.
    """
    conn = sqlite3.connect(DB_NAME)
    row = conn.execute("SELECT id FROM dogs WHERE name = ?", (name,)).fetchone()
    if row:
        dog_id = row[0]
    else:
        cursor = conn.execute("INSERT INTO dogs (name) VALUES (?)", (name,))
        dog_id = cursor.lastrowid
        conn.commit()
    conn.close()
    return dog_id


def add_embedding(dog_id: int, embedding: torch.Tensor, photo_bytes: bytes, sharpness: float):
    """
    Insert a new embedding for the given dog.
    If the dog already has MAX_EMB_PER_DOG embeddings, drop the one with the
    lowest sharpness score to keep the gallery clean.
    """
    blob = embedding.detach().cpu().numpy().astype(np.float32).tobytes()
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")

    # Check current count
    count = conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE dog_id = ?", (dog_id,)
    ).fetchone()[0]

    if count >= MAX_EMB_PER_DOG:
        # Drop the worst-quality embedding
        worst = conn.execute(
            "SELECT id FROM embeddings WHERE dog_id = ? ORDER BY sharpness ASC LIMIT 1",
            (dog_id,),
        ).fetchone()
        if worst:
            conn.execute("DELETE FROM embeddings WHERE id = ?", (worst[0],))

    conn.execute(
        "INSERT INTO embeddings (dog_id, embedding, photo, sharpness) VALUES (?, ?, ?, ?)",
        (dog_id, blob, photo_bytes, sharpness),
    )
    conn.commit()
    conn.close()


def get_all_dog_names() -> list[str]:
    """Return a sorted list of all registered dog names."""
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute("SELECT name FROM dogs ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_dog_gallery() -> dict:
    """
    Returns a dict: { dog_name: [list of 1024-d tensors] }
    Used by Strategy B (Max Similarity) matching.
    """
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute("""
        SELECT d.name, e.embedding
        FROM dogs d
        JOIN embeddings e ON e.dog_id = d.id
        ORDER BY d.name
    """).fetchall()
    conn.close()

    gallery = {}
    for name, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        if len(vec) != EMBED_DIM:
            continue
        gallery.setdefault(name, []).append(torch.from_numpy(vec.copy()))
    return gallery


def get_dog_details() -> list[dict]:
    """
    Returns detailed info for the Manage page:
    [ { id, name, registered_at, embedding_count, embeddings: [{emb_id, sharpness, added_at}] } ]
    """
    conn = sqlite3.connect(DB_NAME)
    dogs = conn.execute("SELECT id, name, registered_at FROM dogs ORDER BY name").fetchall()
    result = []
    for dog_id, name, reg_at in dogs:
        embs = conn.execute(
            "SELECT id, sharpness, added_at FROM embeddings WHERE dog_id = ? ORDER BY added_at DESC",
            (dog_id,),
        ).fetchall()
        result.append({
            "id"             : dog_id,
            "name"           : name,
            "registered_at"  : reg_at,
            "embedding_count": len(embs),
            "embeddings"     : [{"emb_id": e[0], "sharpness": e[1], "added_at": e[2]} for e in embs],
        })
    conn.close()
    return result


def delete_dog_by_id(dog_id: int):
    """Delete a dog and all its embeddings (CASCADE)."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("DELETE FROM dogs WHERE id = ?", (dog_id,))
    conn.commit()
    conn.close()


def delete_embedding_by_id(emb_id: int):
    """Delete a single embedding."""
    conn = sqlite3.connect(DB_NAME)
    conn.execute("DELETE FROM embeddings WHERE id = ?", (emb_id,))
    conn.commit()
    conn.close()


def get_dog_count() -> int:
    if not os.path.exists(DB_NAME):
        return 0
    conn = sqlite3.connect(DB_NAME)
    count = conn.execute("SELECT COUNT(*) FROM dogs").fetchone()[0]
    conn.close()
    return count


def get_embedding_count() -> int:
    if not os.path.exists(DB_NAME):
        return 0
    conn = sqlite3.connect(DB_NAME)
    count = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    conn.close()
    return count


def wipe_database():
    if os.path.exists(DB_NAME):
        os.remove(DB_NAME)


# ==============================================================================
# MATCHING ENGINE — Strategy B: Max Similarity
# ==============================================================================

def match_query(query_vec: torch.Tensor, gallery: dict) -> list[tuple]:
    """
    Strategy B: Max Similarity.

    For each dog, compute cosine similarity against ALL its stored embeddings.
    The dog's score = the HIGHEST similarity found.

    Returns sorted list: [(dog_name, best_sim, best_emb_index), ...]
    """
    results = []
    for name, emb_list in gallery.items():
        best_sim = -1.0
        best_idx = 0
        for i, stored_vec in enumerate(emb_list):
            sim = F.cosine_similarity(
                query_vec.unsqueeze(0), stored_vec.unsqueeze(0)
            ).item()
            if sim > best_sim:
                best_sim = sim
                best_idx = i
        results.append((name, best_sim, best_idx))

    results.sort(key=lambda x: x[1], reverse=True)
    return results


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
def get_embedding_from_image(model: DNNetV3, pipeline: CLAHEPipeline, pil_img: Image.Image):
    """
    Returns:
        embedding  : (1024,) L2-normalized tensor on CPU
        sharpness  : float — Laplacian variance (quality score)
    """
    tensor, sharpness = pipeline(pil_img)
    tensor    = tensor.unsqueeze(0)
    embedding = model(tensor).squeeze(0)
    return embedding, sharpness


def image_to_jpeg_bytes(img: Image.Image) -> bytes:
    """Thumbnail + JPEG compress a PIL image for storage."""
    buf = io.BytesIO()
    thumb = img.copy()
    thumb.thumbnail((256, 256))
    thumb.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


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
    col_a, col_b = st.columns(2)
    col_a.metric("Dogs", get_dog_count())
    col_b.metric("Photos", get_embedding_count())
    st.caption("Strategy B: Max Similarity")
    st.caption("Model: TinyViT-21M | 1024-d | MagFace")


# ==============================================================================
# PAGE: IDENTIFY — with Confirm-to-Enroll
# ==============================================================================
if choice == "🔍 Identify Dog":
    st.title("🔍 Identify Dog")
    st.markdown("Upload a nose photo to match against all registered dogs.")

    file = st.file_uploader(
        "Upload Nose Photo", type=["jpg", "jpeg", "png"], key="identify_upload"
    )

    if file:
        img = Image.open(file).convert("RGB")
        col1, col2 = st.columns([1, 2])
        col1.image(img, caption="Query Image", width="stretch")

        with st.spinner("Running CLAHE → TinyViT-21M embedding…"):
            query_vec, sharpness = get_embedding_from_image(model, pipeline, img)

        # Quality indicator
        quality_pct = min(sharpness / 500.0, 1.0)
        col2.markdown("**Image Quality**")
        col2.progress(quality_pct, text=f"Sharpness: {sharpness:.0f}")
        if sharpness < QUALITY_WARN:
            col2.warning("⚠️ Low quality — results may be less accurate.")

        gallery = get_dog_gallery()

        if not gallery:
            st.error("Database is empty. Register dogs first via **Register Dog**.")
        else:
            # Strategy B matching
            rankings = match_query(query_vec, gallery)
            best_name, best_sim, _ = rankings[0]

            st.markdown("---")

            if best_sim >= MATCH_THRESH:
                st.success(f"✅ **Match Found: {best_name}**")
                st.progress(float(best_sim), text=f"Confidence: {best_sim:.4f}")
                emb_count = len(gallery.get(best_name, []))
                st.caption(f"Matched against {emb_count} stored embedding(s) for {best_name}")

                # ── Confirm-to-Enroll ──────────────────────────────────
                can_enroll = (best_sim >= ENROLL_THRESH and sharpness >= QUALITY_MIN_ENR)
                st.markdown("---")
                st.markdown("**Is this identification correct?**")

                c1, c2 = st.columns(2)
                if c1.button("✅ Yes, correct!", key="confirm_yes", type="primary",
                             disabled=not can_enroll):
                    dog_id = get_or_create_dog(best_name)
                    photo_bytes = image_to_jpeg_bytes(img)
                    add_embedding(dog_id, query_vec, photo_bytes, sharpness)
                    st.success(
                        f"📸 Auto-enrolled this photo into **{best_name}**'s gallery! "
                        f"(now {emb_count + 1} embeddings)"
                    )
                    st.balloons()

                if not can_enroll and best_sim >= MATCH_THRESH:
                    if best_sim < ENROLL_THRESH:
                        st.caption(
                            f"ℹ️ Auto-enrollment requires ≥{ENROLL_THRESH:.0%} confidence "
                            f"(current: {best_sim:.0%})"
                        )
                    elif sharpness < QUALITY_MIN_ENR:
                        st.caption(
                            f"ℹ️ Auto-enrollment requires sharpness ≥{QUALITY_MIN_ENR:.0f} "
                            f"(current: {sharpness:.0f})"
                        )

                if c2.button("❌ No, wrong match", key="confirm_no"):
                    st.info("No changes made. Consider registering this dog if it's new.")

            else:
                st.error("❌ **No Match Found** — dog may not be registered.")
                st.info(
                    f"Closest: **{best_name}** — similarity {best_sim:.4f} "
                    f"(threshold: {MATCH_THRESH})"
                )

            # Full ranking
            with st.expander("📊 Full Ranking (Strategy B: Max Similarity)"):
                for name, sim, _ in rankings:
                    emb_count = len(gallery.get(name, []))
                    st.write(f"`{name}` — {sim:.4f}  ({emb_count} embedding{'s' if emb_count != 1 else ''})")
                    st.progress(max(0.0, float(sim)))


# ==============================================================================
# PAGE: REGISTER — New Dog / Existing Dog toggle
# ==============================================================================
elif choice == "📝 Register Dog":
    st.title("📝 Register Dog")

    # ── Mode toggle ──
    mode = st.radio(
        "Registration Mode",
        ["🆕 New Dog", "📂 Add Photo to Existing Dog"],
        horizontal=True,
    )

    if mode == "🆕 New Dog":
        st.markdown("Register a brand new dog into the system.")
        dog_name = st.text_input("Dog's Name", placeholder="e.g. Buddy", key="new_name")
        existing_names = get_all_dog_names()

        # Warn on duplicate
        if dog_name and dog_name.strip() in existing_names:
            st.warning(
                f"⚠️ A dog named **{dog_name.strip()}** already exists! "
                f"Did you mean to **Add Photo to Existing Dog** instead?"
            )
    else:
        st.markdown("Add another nose photo to strengthen an existing dog's profile.")
        existing_names = get_all_dog_names()
        if not existing_names:
            st.error("No dogs registered yet. Use **New Dog** mode first.")
            dog_name = None
        else:
            dog_name = st.selectbox("Select Dog", existing_names, key="existing_select")

    file = st.file_uploader(
        "Upload Clear Nose Photo", type=["jpg", "jpeg", "png"], key="register_upload"
    )

    if file:
        img = Image.open(file).convert("RGB")
        st.image(img, caption="Preview", width=260)

        with st.spinner("Analysing image quality…"):
            _, sharpness = get_embedding_from_image(model, pipeline, img)
        q_pct = min(sharpness / 500.0, 1.0)
        st.progress(q_pct, text=f"Sharpness score: {sharpness:.0f}")
        if sharpness < QUALITY_WARN:
            st.warning("⚠️ Image is blurry — consider a sharper photo.")

    can_register = bool(dog_name and file)
    btn_label = "✅ Register New Dog" if mode == "🆕 New Dog" else "📸 Add Photo"

    if st.button(btn_label, disabled=not can_register, type="primary"):
        with st.spinner(f"Extracting {EMBED_DIM}-d biometric fingerprint…"):
            img = Image.open(file).convert("RGB")
            emb, sharpness = get_embedding_from_image(model, pipeline, img)
            photo_bytes = image_to_jpeg_bytes(img)

            dog_id = get_or_create_dog(dog_name.strip())
            add_embedding(dog_id, emb, photo_bytes, sharpness)

        if mode == "🆕 New Dog":
            st.success(f"🎉 **{dog_name}** registered successfully!")
        else:
            st.success(f"📸 New photo added to **{dog_name}**'s gallery!")

        st.metric("Embedding Dimension", f"{EMBED_DIM}-d")
        st.metric("Image Sharpness", f"{sharpness:.0f}")
        st.balloons()


# ==============================================================================
# PAGE: MANAGE DATABASE
# ==============================================================================
elif choice == "⚙️ Manage Database":
    st.title("⚙️ Manage Database")

    details = get_dog_details()
    col_a, col_b = st.columns(2)
    col_a.metric("Total Dogs", len(details))
    col_b.metric("Total Embeddings", sum(d["embedding_count"] for d in details))

    if not details:
        st.info("No dogs registered yet.")
    else:
        st.markdown("---")
        for dog in details:
            label = f"🐶 {dog['name']}  —  {dog['embedding_count']} photo(s)"
            with st.expander(label):
                st.caption(f"ID: {dog['id']}  |  Registered: {dog['registered_at']}")
                st.caption(f"Embeddings: {dog['embedding_count']} / {MAX_EMB_PER_DOG} max")

                # List individual embeddings
                if dog["embeddings"]:
                    st.markdown("**Stored Photos:**")
                    for emb in dog["embeddings"]:
                        ecol1, ecol2 = st.columns([3, 1])
                        ecol1.caption(
                            f"  #{emb['emb_id']}  |  "
                            f"Sharpness: {emb['sharpness']:.0f}  |  "
                            f"Added: {emb['added_at']}"
                        )
                        if ecol2.button("🗑️", key=f"del_emb_{emb['emb_id']}"):
                            delete_embedding_by_id(emb["emb_id"])
                            # If this was the last embedding, delete the dog too
                            if dog["embedding_count"] <= 1:
                                delete_dog_by_id(dog["id"])
                                st.success(f"Deleted **{dog['name']}** (last photo removed).")
                            else:
                                st.success(f"Deleted embedding #{emb['emb_id']}.")
                            st.rerun()

                st.markdown("---")
                if st.button(
                    f"🗑️ Delete {dog['name']} entirely",
                    key=f"del_dog_{dog['id']}",
                    type="secondary",
                ):
                    delete_dog_by_id(dog["id"])
                    st.success(f"Deleted **{dog['name']}** and all embeddings.")
                    st.rerun()

    st.markdown("---")
    st.subheader("🚨 Danger Zone")
    with st.expander("Wipe entire database"):
        st.warning("This permanently deletes **all** dogs and **all** embeddings.")
        confirm = st.text_input(
            'Type  `WIPE`  to confirm', key="wipe_confirm", placeholder="WIPE"
        )
        if st.button("Delete All Dogs", type="primary", disabled=(confirm != "WIPE")):
            wipe_database()
            init_db()
            st.cache_resource.clear()
            st.success("Database wiped. All entries deleted.")
            st.rerun()