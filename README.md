# DogID System 🐾

**DogID** is an advanced dog nose biometric recognition system. Analogous to fingerprint or facial recognition for humans, DogID uses a dog's **dermal nose ridge print** as a unique identifier. Each dog's nose exhibits an immutable pattern of ridges, pores, and channels.

This system embeds nose images into a high-dimensional vector space, combining an **integrated YOLOv8 nose detector**, a **Vision Transformer backbone (TinyViT-21M)**, a **LAB-CLAHE micro-texture enhancement pipeline**, and a **hybrid decision engine** (Open-Set Cosine Gate + Few-Shot Linear Classifier).

---

## 🌟 Key Features

- **Integrated YOLOv8 Nose Detector:** Automatically locates and crops dog noses from unconstrained photos and video frames with high precision (`checkpoints/nose_detector.onnx`).
- **Dual Application Workflows:**
  - **V1 App (`app/app.py`):** Multi-photo registration and ensemble max-cosine similarity matching.
  - **V2 App (`appupgrade/app.py`):** Video-based registration (Laplacian blur filtering + temporal diversity selection) with 2-Stage Hybrid matching (Cosine Gate + Fast Linear Boundary Classifier).
- **Dual-Stream Biometrics (85/15):** Fuses macro nose ridge texture (85% weight) with overall head/facial structure (15% weight) for balanced identification.
- **Advanced Preprocessing (LAB-CLAHE):** Luminance-only histogram equalization in LAB color space to pop micro-texture ridge patterns regardless of lighting or coat pigmentation.
- **Quality-Aware Metric Learning (MagFace):** Dynamic margin and regularization loss coupling sample sharpness with feature magnitude ($||f||$).
- **Edge AI Ready:** Knowledge distillation architecture (`MobileNetV3-Small`, 512-d) designed for low-latency (<10ms) edge and mobile deployment.
- **Zero Database Server:** Plain folder-based storage layout on disk (`data/gallery/` and `data/gallery_v2/`), completely portable and version-controllable.

---

## 📂 Project Structure

```text
dogshi/
├── app/                              # V1: Photo-based Streamlit Application
│   ├── app.py                        # Streamlit web app
│   ├── det5.py                       # Core ML: DNNetV3, MagFaceLoss, CLAHEPipeline, StudentDNNet
│   └── nose_detector.py              # YOLOv8 ONNX nose detector loader & inference
├── appupgrade/                       # V2: Video Registration & Hybrid Classifier App
│   ├── app.py                        # Streamlit web app with video ingestion & classifier inspector
│   ├── video_engine.py               # Video frame sampling, Laplacian blur filter & diversity clustering
│   ├── classifier_head.py            # Few-shot Fast Linear Classifier (scikit-learn Logistic Regression)
│   └── matcher.py                    # 2-Stage Hybrid Matcher (Open-Set Gate + Linear Head)
├── checkpoints/
│   └── nose_detector.onnx            # Trained YOLOv8 ONNX nose detector model
├── data/
│   ├── gallery/                      # V1 Gallery store (one folder per dog)
│   └── gallery_v2/                   # V2 Gallery store (embeddings + thumbnails + classifier.pkl)
├── dataset/
│   ├── dog_samples_original/         # Original test dog photographs
│   ├── nose_samples_augmented/       # Synthetic query probes (zoomed in, zoomed out, reshaped)
│   └── video_samples/                # Sample dog registration videos
├── docs/
│   ├── NOSE_DETECTOR_TRAINING_GUIDE.md        # Step-by-step guide to train & export YOLOv8 on Kaggle
│   └── BIOMETRIC_BACKBONE_TRAINING_GUIDE.md  # Step-by-step guide to fine-tune TinyViT with MagFace
├── scripts/
│   ├── eval_augmented_benchmark.py   # Benchmark tool for accuracy, rank-1, rank-3 & open-set FAR
│   ├── batch_reenroll_student.py     # Utility: convert teacher embeddings to student embeddings
│   └── eval_gallery.py               # Utility: Rank-1 / mAP / EER on gallery folders
└── requirements.txt                  # Python dependencies
```

---

## 🛠️ Architecture & Dual-Stage Decision Pipeline

```
════════════════════════════════════════════════════════════════════════════════════
                             REGISTRATION WORKFLOW (VIDEO)
════════════════════════════════════════════════════════════════════════════════════
 [ Dog Video (MP4/MOV) ]
         │
         ▼
 [ Frame Extractor ] ──▶ Adaptive sampling at 3-5 FPS
         │
         ▼
 [ YOLOv8 Nose Detector ] ──▶ Auto-crop bounding box (`nose_detector.onnx`)
         │
         ▼
 [ Laplacian Sharpness Filter ] ──▶ Discard motion-blurred or poorly exposed frames
         │
         ▼
 [ Temporal Diversity Clusterer ] ──▶ Select top K (8–10) distinct, sharp angles
         │
         ▼
 [ Feature Extraction Pipeline ]
    ├── Primary:   LAB-CLAHE + DNNetV3 (Nose Crop)   ──▶ 1024-d nose vector (85%)
    └── Secondary: Resize + DNNetV3 (Full Frame)     ──▶ 1024-d face vector (15%)
         │
         ▼
 [ Gallery Store + Background Classifier Fitting (<0.2s) ] ──▶ `classifier.pkl`

════════════════════════════════════════════════════════════════════════════════════
                           IDENTIFICATION WORKFLOW (PHOTO)
════════════════════════════════════════════════════════════════════════════════════
 [ Query Photo (JPG/PNG) ] ──▶ Nose Crop + Dual Embedding (85% Nose + 15% Face)
         │
         ▼
 [ Stage 1: Open-Set Cosine Gate ]
    ├── If Max-Cosine < 0.64 ────────▶ "Unknown Dog / No Match Found" (Anti-Hallucination)
    └── If Max-Cosine >= 0.64 ───────▶ Proceed to Stage 2
         │
         ▼
 [ Stage 2: Fast Linear Boundary Classifier ]
    └── Evaluates subtle decision boundaries between registered lookalikes
         │
         ▼
 [ Hybrid Score Calculation ]
    Score = 0.60 * Cosine_Score + 0.40 * Classifier_Probability
    └── If Score >= Threshold ──▶ "Match Found: Dog Name"
```

---

## 🚀 How to Run Locally

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

*(For CUDA GPU support, install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)).*

### 2. Launch the Application

- **Run Video Registration + Hybrid Matching App (V2 - Recommended):**
  ```bash
  python -m streamlit run appupgrade/app.py --server.port 8502
  ```

- **Run Photo Registration App (V1):**
  ```bash
  python -m streamlit run app/app.py --server.port 8501
  ```

---

## 📊 Evaluation & Benchmarking

Run the automated synthetic probe benchmark against the dataset:

```bash
# Run benchmark on all available classes with hold-out open-set test
python scripts/eval_augmented_benchmark.py --holdout 5 --threshold 0.64
```

This script evaluates:
- **Rank-1 Identification Accuracy:** Percentage of queries where the top prediction is the exact enrolled dog.
- **Rank-3 Accuracy:** Percentage of queries where the true dog is within the top-3 predictions.
- **Augmentation Breakdown:** Performance across `zoomed_in`, `zoomed_out`, and `reshaped` probes.
- **Open-Set False Acceptance Rejection:** Correct rejection rate for holdout dogs not in the gallery.

---

## 📖 Training Guides & Delegation Documentation

Complete step-by-step guides for training models in free Kaggle / Colab GPU environments:

1. **[YOLOv8 Dog Nose Detector Training Guide](docs/NOSE_DETECTOR_TRAINING_GUIDE.md):**
   - Dataset sourcing from Roboflow Universe.
   - Merging, standardizing, and remapping annotations to class `0: dog_nose`.
   - Training and exporting to `checkpoints/nose_detector.onnx`.

2. **[Biometric Backbone MagFace Fine-Tuning Guide](docs/BIOMETRIC_BACKBONE_TRAINING_GUIDE.md):**
   - Dataset formatting (`dog_id/img.jpg`).
   - One-click script to train `TinyViT-21M` with `MagFaceLoss` for metric separation.
   - Pushing Rank-1 accuracy from generic ImageNet baseline to **>99.0%**.

---

## 🤖 Model Deployment (Edge AI / Mobile)

A teacher-student distillation framework is implemented in [`app/det5.py`](file:///c:/Users/papu_/Downloads/dogshi/app/det5.py) and [`scripts/batch_reenroll_student.py`](file:///c:/Users/papu_/Downloads/dogshi/scripts/batch_reenroll_student.py):
- **Teacher Model:** `DNNetV3` (TinyViT-21M, 1024-d embeddings) — server/desktop.
- **Student Model:** `StudentDNNet` (MobileNetV3-Small, 512-d embeddings, ~8ms CPU) — on-device mobile inference.
