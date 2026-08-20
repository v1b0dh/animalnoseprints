# DogID System — Full Project Overview

## What Is This?

**DogID** is a dog nose biometric recognition system — analogous to fingerprint or face recognition, but using a dog's **nose print** as the unique identifier. Each dog's nose has a distinct pattern of ridges and pores, just like a human fingerprint. The system embeds nose images into a high-dimensional vector space and uses cosine similarity to match query images against a gallery of registered dogs.

The app is a **Streamlit web application** running locally, backed by a **SQLite database**.

---

## Project Files

| File | Purpose |
|------|---------|
| [`app.py`](file:///c:/Users/papu_/Downloads/dogshi/app.py) | Main Streamlit UI — all 4 pages |
| [`det5.py`](file:///c:/Users/papu_/Downloads/dogshi/det5.py) | Core ML: model architecture, loss function, training loop |
| [`video_utils.py`](file:///c:/Users/papu_/Downloads/dogshi/video_utils.py) | Video frame extraction & quality filtering |
| [`dog_biometrics.db`](file:///c:/Users/papu_/Downloads/dogshi/dog_biometrics.db) | SQLite database (dogs, owners, breeds, embeddings, identifications) |
| [`requirements.txt`](file:///c:/Users/papu_/Downloads/dogshi/requirements.txt) | Python dependencies |
| `batch_reenroll_student.py` | Utility: re-embed all dogs using the student (mobile) model |
| `clean_safe.py` | Utility: database cleanup/maintenance script |
| `checkpoints/teacher_best.pth` | Trained model weights *(not in repo — placed manually)* |

---

## Architecture Deep Dive

### 1. Backbone — TinyViT-21M

The feature extractor is **TinyViT-21M**, a lightweight Vision Transformer pretrained on ImageNet-22K.

- Input: `224×224` RGB image
- Output: **576-dimensional** global feature vector (via global average pooling over patch tokens)
- Chosen because it balances strong representation learning with reasonable inference speed

### 2. Embedding Head (DNNetV3)

On top of the backbone sits a projection head that expands to a richer embedding space:

```
576-d  →  Linear  →  BatchNorm  →  SiLU  →  Dropout(0.1)
       →  Linear  →  BatchNorm  →  L2-normalize  →  1024-d
```

The final embedding is **L2-normalized** (lives on a unit hypersphere). This is required for cosine similarity matching to work correctly, and is standard in all metric learning / face recognition systems.

> The head is only used when a trained checkpoint (`teacher_best.pth`) exists. Without it, the system falls back to raw 576-d backbone features.

### 3. Loss Function — MagFace

> *He et al., CVPR 2021 — "MagFace: A Universal Representation for Face Recognition and Quality Assessment"*

MagFace is an upgrade over the well-known **ArcFace** loss. The key insight is that the **magnitude `||f||` of the feature vector encodes sample quality naturally**:

| Sample quality | Feature norm `||f||` | Margin applied | Effect |
|---|---|---|---|
| Sharp, clear nose | High | Larger | Harder constraint → better separation |
| Blurry nose | Low | Smaller | + regularization pushes `||f||` up |

This means the model **self-learns a quality score** without any extra quality labels. Blurry or poorly-lit samples are automatically down-weighted at training time.

**Parameters used:**
- Scale `s = 64.0`
- Margin range: `[0.45, 0.80]`
- Norm range: `[10.0, 110.0]`
- Regularisation weight `λ = 35.0`

### 4. Preprocessing — CLAHE Pipeline

Before any image goes into the model it passes through:

1. **CLAHE** (Contrast Limited Adaptive Histogram Equalization) applied in **LAB color space** — only the L (luminance) channel is enhanced, preserving natural color. This lifts low-contrast nose ridge patterns that would otherwise be invisible to the model.
2. **Laplacian variance** sharpness scoring — measures blur before CLAHE so the score reflects raw image quality.
3. Standard ImageNet normalization (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`) and resize to `224×224`.

### 5. Knowledge Distillation (Student Model — mobile, not actively used in UI)

A **teacher → student** distillation pipeline exists in `det5.py` and `batch_reenroll_student.py`:

- **Teacher**: DNNetV3 (TinyViT-21M, 1024-d) — full precision, cloud/server
- **Student**: MobileNetV3-Small (~2.5M params, 512-d, ~8ms CPU) — mobile deployment

The distillation uses a combined loss:
```
L = α·MSE(student_emb, teacher_emb)   ← embedding-level KD
  + β·KL(student_logits/T, teacher_logits/T)  ← softened logits (T=4)
  + γ·CrossEntropy(student_logits, labels)     ← hard labels
```

The student model is **not currently used in the main UI** (we're focusing on the teacher model for now).

### 6. Matching — Ensemble Max-Cosine Similarity

At identification time, the system uses **multi-frame ensemble matching**:

```
for each registered dog:
    best_sim = max over all (query_frame × stored_embedding) pairs
               of cosine_similarity(query_vec, stored_vec)

result = dog with highest best_sim
```

This naturally scales with the number of stored frames — the more frames a dog has registered, the more "views" the matcher has to work with.

**Threshold: `0.82`** — similarity must exceed this to call a positive match.

---

## The Video Workflow (New as of Aug 2026)

### Why the change?

Single photos gave limited biometric coverage — only one angle, one lighting condition. A 360° sweep of the dog's face/nose captures multiple angles and lighting variations, making registration **far more robust**.

### How it works

```
USER records 5–15s video of dog's nose/face (360° sweep)
        ↓
video_utils.py: extract_frames()
  └─ OpenCV evenly samples 30 frames across the full video duration
        ↓
video_utils.py: filter_frames_by_quality()
  └─ Laplacian sharpness computed for each frame
  └─ Frames below threshold (100) are discarded
  └─ Top 20 by sharpness are kept
        ↓
app.py: Frame preview strip shown to user (thumbnail + sharpness badge 🟢🟡🔴)
        ↓
app.py: Each good frame → CLAHE pipeline → DNNetV3 → 1024-d embedding
        ↓
SQLite: All embeddings stored (one row per frame, same dog_id)
```

**Minimum 5 good frames** are required to proceed. If fewer pass the quality filter, the user is asked to re-record.

### Identification with video

For identification, the user can also upload a short video. All good frames from the query video are embedded, and the **ensemble max-cosine** matching is run across all (query frame × stored frame) pairs — picking the single highest similarity across everything.

A **single photo fallback** tab is also available on both Register and Identify pages.

---

## Database Schema

The SQLite DB (`dog_biometrics.db`) has 5 tables:

```
breeds        (id, breed_name, origin, typical_weight_kg)
owners        (id, name, phone, email, address)
dogs          (id, name, breed_id→breeds, owner_id→owners, age_years, weight_kg, color, registered_at)
embeddings    (id, dog_id→dogs, embedding BLOB, photo BLOB, sharpness REAL, model_type, created_at)
identifications (id, dog_id→dogs, similarity REAL, confirmed INT, identified_at)
```

Each dog can have **many rows in `embeddings`** — one per registered video frame. The `model_type` column distinguishes `'teacher'` (1024-d, used in UI) from `'student'` (512-d, used by batch script).

---

## Optimizer & Training Setup

| Component | Setting |
|-----------|---------|
| Optimizer | AdamW |
| Backbone LR | `1e-4` (fine-tuning) |
| Head + MagFace LR | `1e-3` (learning from scratch) |
| Weight decay | `1e-4` |
| LR Scheduler | Cosine Annealing Warm Restarts (`T_0=10, T_mult=2, η_min=1e-6`) |
| Mixed precision | `torch.cuda.amp.autocast` + GradScaler |
| Grad clipping | `max_norm=5.0` |

---

## Changes Made (Aug 2026 Pivot)

| What | Before | After |
|------|--------|-------|
| Input modality | Single photo | 360° video → frames |
| Frames per dog | 1 | Up to 20 (quality-filtered) |
| Matching | Single-vec cosine | Multi-frame ensemble max-cosine |
| Identify input | Photo only | Video (recommended) + photo (fallback) |
| Mobile/Android | Full Android app (`dogid-android/`) | **Removed** |
| `requirements.txt` | Included `onnx`, `onnxruntime` | Removed (no longer needed) |
| Single app file | `app.py` + `app_new.py` | `app.py` only |
| New module | — | `video_utils.py` |

---

## How to Run

```powershell
# Install dependencies
pip install -r requirements.txt

# Run the app
python -m streamlit run app.py
```

> Use `python -m streamlit` instead of `streamlit` directly — the `streamlit.exe` launcher may point to a missing Python installation on Windows.

---

## Key Constants (app.py)

| Constant | Value | Meaning |
|----------|-------|---------|
| `MATCH_THRESH` | `0.82` | Min cosine sim for a positive ID |
| `QUALITY_WARN` | `100.0` | Sharpness below this → warn user |
| `VIDEO_MAX_FRAMES` | `30` | Frames evenly sampled from video |
| `VIDEO_MIN_SHARP` | `100.0` | Min Laplacian score to keep a frame |
| `VIDEO_TOP_N` | `20` | Max frames stored per session |
| `VIDEO_MIN_PASS` | `5` | Min good frames required to proceed |
