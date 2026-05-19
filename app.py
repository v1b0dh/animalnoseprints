"""
app.py — Dog Nose Biometrics (DNNetV3 SOTA) — Minimal Frontend
================================================================
v2.1 — Simple 2D Interface (No Graphics/Animations)

Features:
  • Two-table SQLite Schema
  • Strategy B: Max Similarity Matching
  • Quality-gated Auto-enrollment
  • Per-embedding Management
"""

import streamlit as st
import sqlite3
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import io
import os

# -- Model Imports --
from det5 import DNNetV3, CLAHEPipeline

# -- Configuration --
DB_NAME          = "dog_biometrics.db"
MATCH_THRESH     = 0.82
ENROLL_THRESH    = 0.88
QUALITY_WARN     = 100.0
QUALITY_MIN_ENR  = 150.0
MAX_EMB_PER_DOG  = 20

# ==============================================================================
# DATABASE LAYER
# ==============================================================================

def init_db():
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
    blob = embedding.detach().cpu().numpy().astype(np.float32).tobytes()
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON")
    count = conn.execute("SELECT COUNT(*) FROM embeddings WHERE dog_id = ?", (dog_id,)).fetchone()[0]
    if count >= MAX_EMB_PER_DOG:
        worst = conn.execute("SELECT id FROM embeddings WHERE dog_id = ? ORDER BY sharpness ASC LIMIT 1", (dog_id,)).fetchone()
        if worst: conn.execute("DELETE FROM embeddings WHERE id = ?", (worst[0],))
    conn.execute("INSERT INTO embeddings (dog_id, embedding, photo, sharpness) VALUES (?, ?, ?, ?)", (dog_id, blob, photo_bytes, sharpness))
    conn.commit()
    conn.close()

def get_all_dog_names():
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute("SELECT name FROM dogs ORDER BY name").fetchall()
    conn.close()
    return [r[0] for r in rows]

def get_dog_gallery():
    conn = sqlite3.connect(DB_NAME)
    rows = conn.execute("SELECT d.name, e.embedding FROM dogs d JOIN embeddings e ON e.dog_id = d.id").fetchall()
    conn.close()
    gallery = {}
    for name, blob in rows:
        vec = np.frombuffer(blob, dtype=np.float32)
        if len(vec) == EMBED_DIM:
            gallery.setdefault(name, []).append(torch.from_numpy(vec.copy()))
    return gallery

def get_dog_details():
    conn = sqlite3.connect(DB_NAME)
    dogs = conn.execute("SELECT id, name, registered_at FROM dogs ORDER BY name").fetchall()
    result = []
    for d_id, name, reg_at in dogs:
        embs = conn.execute("SELECT id, sharpness, added_at FROM embeddings WHERE dog_id = ? ORDER BY added_at DESC", (d_id,)).fetchall()
        result.append({"id": d_id, "name": name, "registered_at": reg_at, "count": len(embs), "embs": embs})
    conn.close()
    return result

def delete_dog(d_id):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("PRAGMA foreign_keys = ON; DELETE FROM dogs WHERE id = ?", (d_id,))
    conn.commit()
    conn.close()

def delete_emb(e_id):
    conn = sqlite3.connect(DB_NAME)
    conn.execute("DELETE FROM embeddings WHERE id = ?", (e_id,))
    conn.commit()
    conn.close()

# ==============================================================================
# MATCHING LOGIC
# ==============================================================================

def match_query(query_vec, gallery):
    results = []
    for name, emb_list in gallery.items():
        best_sim = max([F.cosine_similarity(query_vec.unsqueeze(0), v.unsqueeze(0)).item() for v in emb_list])
        results.append((name, best_sim))
    results.sort(key=lambda x: x[1], reverse=True)
    return results

# ==============================================================================
# UTILS
# ==============================================================================

@st.cache_resource
def load_model():
    # If custom trained weights exist, load them and use the head.
    # Otherwise, bypass the head and use raw 576-d backbone features for SOTA ImageNet zero-shot matching.
    ckpt_path = "checkpoints/teacher_best.pth"
    if os.path.exists(ckpt_path):
        m = DNNetV3(pretrained=True, use_head=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        m.load_state_dict(state_dict)
    else:
        m = DNNetV3(pretrained=True, use_head=False)
    m.eval()
    p = CLAHEPipeline(image_size=224)
    return m, p

def img_to_bytes(img):
    buf = io.BytesIO()
    t = img.copy(); t.thumbnail((256, 256))
    t.save(buf, format="JPEG", quality=85)
    return buf.getvalue()

def get_embedding_from_image(m, p, img):
    t, s = p(img)
    with torch.no_grad():
        emb = m(t.unsqueeze(0)).squeeze(0)
    return emb, s

# ==============================================================================
# APP INTERFACE
# ==============================================================================

st.set_page_config(page_title="Dog Biometrics", layout="centered")
init_db()
model, pipeline = load_model()
EMBED_DIM = 576 if not model.use_head else 1024

# Sidebar
choice = st.sidebar.radio("Navigation", ["Identify", "Register", "Manage"])

# PAGE: IDENTIFY
if choice == "Identify":
    st.header("Identify Dog")
    
    tab_upload, tab_camera = st.tabs(["📁 Upload or Paste Image", "📷 Take Live Photo"])
    
    img = None
    with tab_upload:
        st.markdown("💡 **Tip:** You can browse, drag & drop, or click below and press **Ctrl+V** to paste an image directly from your clipboard!")
        file = st.file_uploader("Upload or Paste nose image", type=["jpg", "png", "jpeg"], key="identify_upload")
        if file:
            img = Image.open(file).convert("RGB")
            
    with tab_camera:
        camera_file = st.camera_input("Capture Nose Print", key="identify_camera")
        if camera_file:
            img = Image.open(camera_file).convert("RGB")
            
    if img is not None:
        st.image(img, width=220, caption="Selected Image")
        with st.spinner("Processing..."):
            vec, sharp = get_embedding_from_image(model, pipeline, img)
        st.text(f"Image Sharpness: {sharp:.1f}")
        if sharp < QUALITY_WARN: st.warning("Warning: Low quality image.")
        
        gallery = get_dog_gallery()
        if not gallery:
            st.error("Database empty.")
        else:
            rankings = match_query(vec, gallery)
            name, sim = rankings[0]
            if sim >= MATCH_THRESH:
                st.success(f"Match: {name} (Score: {sim:.4f})")
                if st.button("Confirm Match and Save Photo"):
                    if sim >= ENROLL_THRESH and sharp >= QUALITY_MIN_ENR:
                        add_embedding(get_or_create_dog(name), vec, img_to_bytes(img), sharp)
                        st.success("Photo successfully added to database!")
                    else:
                        st.error("Score or quality too low for auto-save.")
            else:
                st.error(f"No match found. Closest: {name} ({sim:.4f})")

# PAGE: REGISTER
elif choice == "Register":
    st.header("Register Dog")
    mode = st.radio("Mode", ["New Dog", "Existing Dog"])
    if mode == "New Dog":
        name = st.text_input("Name")
    else:
        names = get_all_dog_names()
        name = st.selectbox("Select Dog", names) if names else None
    
    tab_upload, tab_camera = st.tabs(["📁 Upload or Paste Image", "📷 Take Live Photo"])
    
    img = None
    with tab_upload:
        st.markdown("💡 **Tip:** You can browse, drag & drop, or click below and press **Ctrl+V** to paste an image directly from your clipboard!")
        file = st.file_uploader("Upload or Paste nose image", type=["jpg", "png", "jpeg"], key="register_upload")
        if file:
            img = Image.open(file).convert("RGB")
            
    with tab_camera:
        camera_file = st.camera_input("Capture Nose Print", key="register_camera")
        if camera_file:
            img = Image.open(camera_file).convert("RGB")
            
    if img is not None and name:
        st.image(img, width=220, caption="Preview Image")
        if st.button("Save to Database"):
            vec, sharp = get_embedding_from_image(model, pipeline, img)
            add_embedding(get_or_create_dog(name), vec, img_to_bytes(img), sharp)
            st.success(f"Successfully saved {name} to database!")

# PAGE: MANAGE
elif choice == "Manage":
    st.header("Manage Database")
    details = get_dog_details()
    st.write(f"Total Dogs: {len(details)}")
    for d in details:
        with st.expander(f"{d['name']} ({d['count']} photos)"):
            st.text(f"Registered: {d['registered_at']}")
            for e in d['embs']:
                c1, c2 = st.columns([4, 1])
                c1.text(f"Photo ID: {e[0]} | Sharpness: {e[1]:.1f}")
                if c2.button("Delete", key=f"e_{e[0]}"):
                    delete_emb(e[0])
                    if d['count'] <= 1: delete_dog(d['id'])
                    st.rerun()
            if st.button(f"Delete Entire Record: {d['name']}", key=f"d_{d['id']}"):
                delete_dog(d['id'])
                st.rerun()
    
    st.write("---")
    if st.button("Wipe Entire Database"):
        if os.path.exists(DB_NAME): os.remove(DB_NAME)
        init_db()
        st.rerun()
