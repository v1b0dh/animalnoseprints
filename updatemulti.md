# 🔄 Update v2.0: Multi-Embedding Architecture

This document explains all the changes made in the v2.0 update, the matching strategy used, alternatives that can be implemented in the future, and how the system can be scaled.

---

## 📋 What Changed (v1.0 → v2.0)

### Database: Single Table → Two Tables

**Before (v1.0):**
```
dogs: id | name | embedding | photo | registered_at
```
One row per dog. One photo per dog. If you wanted to update a photo, you had to delete and re-register.

**After (v2.0):**
```
dogs:       id | name (UNIQUE) | registered_at
embeddings: id | dog_id (FK)   | embedding | photo | sharpness | added_at
```
One dog can now have **up to 20 photos**. Each photo generates its own 1024-d embedding vector. This is a standard one-to-many relationship.

---

### Registration: New Dog vs. Existing Dog

The Register page now has two modes:

| Mode | What happens |
| :--- | :--- |
| **🆕 New Dog** | You type a name + upload a photo. A new dog entry is created with its first embedding. If the name already exists, you get a warning. |
| **📂 Add Photo to Existing Dog** | You pick a name from a dropdown (populated from the database) and upload another photo. The new embedding is added to that dog's gallery. |

**Why this matters:** More photos = more angles = better accuracy. A dog with 10 reference photos from different lighting conditions will be identified far more reliably than one with a single photo.

---

### Identification: Confirm-to-Enroll

After the system finds a match, it now asks: **"Is this correct? Yes / No"**

| Button | What happens |
| :--- | :--- |
| **✅ Yes, correct!** | The query image's embedding is **automatically added** to the matched dog's gallery. The system gets smarter every time you confirm. |
| **❌ No, wrong match** | Nothing happens. No data is saved. |

**Safety guards on auto-enrollment:**
- Confidence must be ≥ **88%** (not just the 82% match threshold).
- Image sharpness must be ≥ **150** (prevents blurry junk from entering the database).
- Each dog is capped at **20 embeddings**. If the cap is reached, the embedding with the lowest sharpness score is automatically dropped to make room.

---

### Manage Page: Per-Embedding Control

You can now:
- See every individual embedding stored for a dog (with its sharpness score and timestamp).
- Delete a **single embedding** (e.g., remove a bad photo without losing the entire dog).
- Delete an **entire dog** and all its embeddings at once.
- If you delete the last remaining embedding for a dog, the dog entry is also automatically removed.

---

## 🎯 Matching Strategy: B (Max Similarity)

### How It Works
```
For each dog in the database:
    For each of that dog's stored embeddings:
        compute cosine_similarity(query_vector, stored_vector)
    dog_score = MAX of all similarities

Sort all dogs by dog_score (highest first)
Return the top match
```

### Why Max (not Average)?
Averaging multiple embeddings creates a "blurred" centroid that may not match any real photo well. Max Similarity finds the single best-matching reference photo. If a dog has 10 photos and the query matches photo #7 perfectly, that's the score — the other 9 photos don't drag it down.

### Performance
| Dogs | Photos/Dog | Total Comparisons | Time (CPU) |
| :--- | :--- | :--- | :--- |
| 100 | 5 | 500 | ~2ms |
| 1,000 | 10 | 10,000 | ~15ms |
| 10,000 | 15 | 150,000 | ~200ms |
| 100,000 | 20 | 2,000,000 | ~3 seconds |

For up to ~10,000 dogs, brute-force cosine similarity is perfectly fine. Beyond that, see the Scaling section below.

---

## 🔮 Future Strategy Options

### Strategy C: Weighted Max (Next Upgrade)
Same as Strategy B, but multiply each similarity by a quality weight derived from the embedding's sharpness score:
```
score = max(cosine_sim(query, emb_i) × quality_weight(emb_i))
```
A match against a sharp reference photo is worth more than a match against a blurry one. This matters when you have 20+ photos per dog with wildly different quality. Switching from B to C is a one-line code change.

### Strategy D: Centroid + Max Hybrid
1. First, compute the centroid (average) of each dog's embeddings.
2. Use centroids for a fast "coarse" search (top-50 candidates).
3. Then run Strategy B only on those 50 candidates.

This gives you the speed of averaging with the accuracy of max similarity. Useful at 50,000+ dogs.

### Strategy E: Learned Aggregation
Train a small neural network that takes all of a dog's embeddings and produces a single "super-embedding" that is better than any individual one. This is research-level and requires a training dataset.

---

## 📈 How to Scale This System

### Level 1: Current (SQLite + Brute Force) — Up to 10,000 dogs
What you have now. No changes needed. Works on a laptop.

### Level 2: FAISS Vector Index — Up to 1,000,000 dogs
Replace the SQLite cosine similarity loop with **Facebook AI Similarity Search (FAISS)**:
- Uses approximate nearest neighbor (ANN) algorithms.
- Can search 1,000,000 vectors in **<10ms**.
- Store embeddings in a FAISS index file instead of SQLite BLOBs.
- SQLite still holds metadata (name, photo, timestamps).

### Level 3: Distributed Vector DB — Unlimited
For a cloud-deployed system serving millions of requests:
- Use **Milvus**, **Pinecone**, or **Qdrant** as a managed vector database.
- The Streamlit frontend becomes a REST API (FastAPI).
- The model runs on a GPU server or as a serverless function.

### Level 4: On-Device (Mobile App)
- Use the **StudentDNNet** (MobileNetV3-Small) from `det5.py`.
- Export to **ONNX** or **TorchScript** for mobile runtimes.
- FAISS has a mobile-compatible C++ library.
- Total inference: ~11ms on a modern smartphone.

---

## 🛡️ Robustness Improvements (Future Work)

| Feature | What it does | Difficulty |
| :--- | :--- | :--- |
| **Test-Time Augmentation (TTA)** | Take 5 crops of the query image, extract 5 embeddings, average them. Reduces noise by ~0.5% accuracy. | Easy |
| **Hard Negative Mining** | During training, specifically find "look-alike" dogs and force the model to distinguish them. | Medium |
| **Anti-Spoofing** | Detect if someone is holding a printed photo of a dog's nose instead of a real nose. Uses liveness detection. | Hard |
| **Temporal Decay** | Older embeddings contribute less to the matching score. Accounts for nose changes over years. | Easy |
| **Merge Duplicates** | Detect if two different dog names actually have very similar embeddings and prompt the user to merge them. | Medium |
| **Confidence Calibration** | Map raw cosine similarity to a true probability (e.g., "93% means there's a 93% chance this is the same dog"). Uses Platt scaling. | Medium |
| **Batch Enrollment** | Upload a folder of 50 photos of the same dog and register them all at once. | Easy |
| **API Mode** | Replace Streamlit with FastAPI so other apps (mobile, web) can call the biometric engine over HTTP. | Medium |

---

## ⚠️ Important: Delete Old Database

Since the database schema changed from one table to two tables, **you must delete your old `dog_biometrics.db` file** before running the updated app. The new schema will be created automatically on startup.

```bash
# Delete old database (run from the project folder)
del dog_biometrics.db

# Then start the app
streamlit run app.py
```
