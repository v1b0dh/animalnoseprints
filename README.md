# DogID System 🐾

**DogID** is an advanced dog nose biometric recognition system. Analogous to fingerprint or facial recognition for humans, DogID uses a dog's **nose print** as a unique identifier. Each dog's nose has a distinct pattern of ridges and pores. This system embeds nose images into a high-dimensional vector space and uses cosine similarity to match query images against a gallery of registered dogs.

The application is built with **Streamlit** and stores everything on disk as plain files (no database), powered by a state-of-the-art Vision Transformer pipeline.

---

## 🌟 Key Features
- **Multi-Photo Workflow:** Bulk upload multiple photos of a dog's nose to capture different angles and lighting conditions, making registration and identification highly robust.
- **Ensemble Max-Cosine Matching:** Matches queries against all stored embeddings for a dog, dramatically increasing accuracy over single-image comparisons.
- **Advanced Preprocessing:** Uses a custom CLAHE (Contrast Limited Adaptive Histogram Equalization) pipeline in the LAB color space to artificially "pop" the micro-texture of nose ridges, regardless of the dog's coat color or lighting.
- **Quality-Aware Loss (MagFace):** The model was trained using MagFace, meaning the neural network self-learns image quality. Sharp photos generate strong embeddings, while blurry photos are automatically down-weighted.
- **Vision Transformer Backbone:** Built on **TinyViT-21M**, achieving state-of-the-art representation learning while remaining lightweight and fast.

---

## 📂 Project Structure

```text
dogshi/
├── app/
│   ├── app.py                  # Main Streamlit web application
│   └── det5.py                 # Core ML: model architecture & loss function
├── scripts/
│   ├── batch_reenroll_student.py # Utility: add 512-d student embeddings
│   └── eval_gallery.py          # Utility: Rank-1 / mAP / EER on the gallery
├── data/
│   └── gallery/                # Folder gallery (one folder per dog)
└── README.md                   # This file
```

---

## 🛠️ Architecture

1. **Backbone:** TinyViT-21M (pretrained on ImageNet-22K) -> outputs a 576-d global feature vector.
2. **Embedding Head:** Expands the features to a richer 1024-d L2-normalized embedding space for cosine similarity matching.
3. **Storage:** Plain files on disk under `data/gallery/`. One folder per dog:
   ```
   data/gallery/
     Buddy/
       0001.npy        # 1024-d teacher embedding (float32, L2-normalized)
       0001.jpg        # thumbnail of the source photo
       0002.npy
       0002.jpg
       meta.json       # {breed, age_years, weight_kg, color, registered_at, ...}
   ```
   No database server, no migrations — version-controllable, easy to back up.

---

## 🚀 How to Run Locally

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Start the App
Run the Streamlit application from the root directory:
```bash
python -m streamlit run app/app.py
```

### 3. Usage
- **Register:** Head to the "Register" tab, add dog details, and upload one or more clear photos of the dog's nose.
- **Identify:** Head to the "Identify" tab, upload a photo of a dog's nose, and the system will query the database to find the closest match using ensemble max-cosine similarity.

---

## 🤖 Model Deployment (Edge AI)
A teacher-student distillation pipeline exists within the codebase (`det5.py` and `batch_reenroll_student.py`). 
- **Teacher:** DNNetV3 (TinyViT-21M, 1024-d) — used on desktop/server.
- **Student:** MobileNetV3-Small (512-d, ~8ms CPU) — designed for Android/mobile edge deployment.
*Note: The current web UI relies primarily on the high-accuracy Teacher model.*

---

## Nose Detection And Weighted Matching

The web app now sends every uploaded image through a nose-crop step before
creating the main biometric embedding.

Add your trained detector here when it is ready:

```text
checkpoints/nose_detector.onnx
```

Until that file exists, the app uses a center-crop fallback and marks the crop
source as `center_fallback` in the UI and per-frame metadata.

New registrations save:

```text
data/gallery/<dog>/
  0001.npy           # nose embedding
  0001.jpg           # nose crop thumbnail
  face_0001.npy      # secondary full-image/face embedding
  original_0001.jpg  # original upload thumbnail
  0001.json          # crop bbox, sharpness, detector source/confidence
```

Matching uses the nose as the main identity signal and the face/full image as a
secondary signal:

```text
final_score = 0.85 * nose_score + 0.15 * face_score
```

If an older record has no `face_*.npy` files, matching falls back to the nose
score only.
