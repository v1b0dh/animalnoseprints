# DogID System 🐾

**DogID** is an advanced dog nose biometric recognition system. Analogous to fingerprint or facial recognition for humans, DogID uses a dog's **dermal nose ridge print** as a unique identifier. Each dog's nose exhibits an immutable pattern of ridges, pores, and channels.

This system embeds nose images into a high-dimensional vector space, combining an **integrated YOLOv8 nose detector**, a **Vision Transformer backbone (TinyViT-21M)**, a **LAB-CLAHE micro-texture enhancement pipeline**, and a **hybrid decision engine** (Open-Set Cosine Gate + Few-Shot Linear Classifier).

---

## 🌟 Key Features

- **Integrated YOLOv8 Nose Detector:** Automatically locates and crops dog noses from unconstrained photos and video frames with high precision (`checkpoints/nose_detector.onnx`).
- **Dual Application Workflows:**
  - **V1 App (`phodog/app.py`):** Multi-photo registration and ensemble max-cosine similarity matching.
  - **V2 App (`vidog/app.py`):** Video-based registration (Laplacian blur filtering + temporal diversity selection) with 2-Stage Hybrid matching (Cosine Gate + Fast Linear Boundary Classifier).
- **Dual-Stream Biometrics (85/15):** Fuses macro nose ridge texture (85% weight) with overall head/facial structure (15% weight) for balanced identification.
- **Advanced Preprocessing (LAB-CLAHE):** Luminance-only histogram equalization in LAB color space to pop micro-texture ridge patterns regardless of lighting or coat pigmentation.
- **Quality-Aware Metric Learning (MagFace):** Dynamic margin and regularization loss coupling sample sharpness with feature magnitude ($||f||$).
- **Edge AI Ready:** Knowledge distillation architecture (`MobileNetV3-Small`, 512-d) designed for low-latency (<10ms) edge and mobile deployment.
- **Zero Database Server:** Plain folder-based storage layout on disk (`data/gallery/` and `data/gallery_v2/`), completely portable and version-controllable.

---

## 🛠️ Requirements

### System Requirements

- **Python:** 3.10+ (developed on 3.14)
- **PyTorch:** 2.1+ (CPU or CUDA 12.1+)
- **Platform:** Windows, macOS, or Linux

### Python Dependencies

```bash
pip install -r requirements.txt
```

**Contents of `requirements.txt`:**

| Category | Package | Version |
|---|---|---|
| Core ML | `torch` | >=2.1.0 |
| Core ML | `torchvision` | >=0.16.0 |
| Core ML | `timm` | >=0.9.12 |
| Image Processing | `Pillow` | >=10.0.0 |
| Image/Video | `opencv-python` | >=4.8.0 |
| Numerical | `numpy` | >=1.24.0 |
| Data | `pandas` | >=2.0.0 |
| Web App | `streamlit` | >=1.35.0 |
| ML (few-shot classifier) | `scikit-learn` | >=1.3.0 |

### GPU Support (Optional)

For CUDA GPU acceleration, replace the default torch install:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

### Additional Requirements

- **ONNX Runtime** (optional, for faster ONNX inference):
  ```bash
  pip install onnxruntime
  ```

---

## 📂 Project Structure

```text
dogshi/
├── phodog/                              # V1: Photo-based Application
│   ├── app.py                                  # Streamlit web app (photo registration + identification)
│   ├── det5.py                                 # Core ML: DNNetV3, MagFaceLoss, CLAHEPipeline, StudentDNNet
│   ├── nose_detector.py                        # YOLOv8 ONNX nose detector loader & inference
│   ├── README.md                               # V1 app-specific docs
│   ├── checkpoints/
│   │   └── nose_detector.onnx                  # Trained YOLOv8 ONNX nose detector model
│   ├── data/                                   # V1 Gallery store (folder per dog)
│   ├── dataset/
│   │   ├── dog_samples_original/               # Original test dog photographs
│   │   └── nose_samples_augmented/             # Synthetic query probes (zoomed_in, zoomed_out, reshaped)
│   ├── scripts/
│   │   └── eval_augmented_benchmark.py         # Benchmark: accuracy, rank-1, rank-3 & open-set FAR
│   └── __pycache__/
├── vidog/                               # V2: Video Registration & Hybrid Classifier
│   ├── app.py                                  # Streamlit web app (video ingestion + classifier inspector)
│   ├── video_engine.py                         # Video frame sampling, Laplacian blur filter & diversity clustering
│   ├── classifier_head.py                      # Few-shot Fast Linear Classifier (scikit-learn Logistic Regression)
│   ├── matcher.py                              # 2-Stage Hybrid Matcher (Open-Set Gate + Linear Head)
│   ├── det5.py                                 # Core ML: DNNetV3, MagFaceLoss, CLAHEPipeline, StudentDNNet
│   ├── nose_detector.py                        # YOLOv8 ONNX nose detector loader & inference
│   ├── checkpoints/
│   │   └── nose_detector.onnx                  # Trained YOLOv8 ONNX nose detector model
│   ├── data/                                   # V2 Gallery store (embeddings + thumbnails + classifier.pkl)
│   ├── dataset/
│   │   └── video_samples/dogo/                 # Sample dog registration videos
│   ├── scripts/
│   │   ├── eval_augmented_benchmark.py         # Benchmark tool for synthetic probe accuracy
│   │   ├── batch_reenroll_student.py           # Utility: convert teacher embeddings to student embeddings
│   │   └── eval_video_benchmark.py             # Benchmark: video enrollment + held-out frame evaluation
│   └── __pycache__/
├── docs/
│   ├── NOSE_DETECTOR_TRAINING_GUIDE.md         # Step-by-step YOLOv8 nose detector training guide
│   └── BIOMETRIC_BACKBONE_TRAINING_GUIDE.md     # Step-by-step TinyViT MagFace fine-tuning guide
├── _archive/                                   # Legacy code (det3.py, det4.py, unified_tester.py)
├── .gitignore
├── requirements.txt                          # Python dependencies (root)
└── README.md                                   # This file
```

> **Note:** Both `phodog/` and `vidog/` have their own copies of `det5.py` and `nose_detector.py`. The `phodog` modules import from the local directory, while `vidog/app.py` imports `det5.py` from `phodog/` as a fallback path. Ensure you run each app from its respective directory, or use the commands below.

---

## 🏃 How to Run Locally

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

*For CUDA GPU support, install PyTorch matching your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/).*

### 2. Launch the Application

There are two app versions available:

#### V2 — Video Registration + Hybrid Matching (Recommended)

```bash
cd vidog && streamlit run app.py --server.port 8502
```

This is the advanced app. Register a dog by uploading a short video clip — it auto-extracts sharp, diverse nose frames, computes dual-stream embeddings (85% nose + 15% face), and fits a few-shot logistic regression classifier.

#### V1 — Photo Registration (Simpler)

```bash
cd phodog && streamlit run app.py --server.port 8501
```

This is the simpler app. Register a dog by uploading 1+ photos — it crops the nose and stores embeddings. Identification uses ensemble max-cosine similarity matching.

---

## 🔄 Architecture & Dual-Stage Decision Pipeline

```
════════════════════════════════════════════════════════════════════════════════════
                             REGISTRATION WORKFLOW (VIDEO - V2)
════════════════════════════════════════════════════════════════════════════════════
 [ Dog Video (MP4/MOV) ]
         │
         ▼
 [ Frame Extractor ] ──▶ Adaptive sampling at 3-5 FPS
         │
         ▼
 [ YOLOv8 Nose Detector ] ──▶ Auto-crop bounding box (nose_detector.onnx)
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
 [ Gallery Store + Background Classifier Fitting (<0.2s) ] ──▶ classifier.pkl
```

```
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

## 📊 Evaluation & Benchmarking

### V2 Video Benchmark

Evaluates video-based registration against held-out unused frames from the same videos, plus open-set rejection on unregistered dogs:

```bash
cd vidog && python scripts/eval_video_benchmark.py
```

This script evaluates:
- **Pure 85/15 Cosine Similarity Rank-1 Accuracy** on held-out frames
- **2-Stage Hybrid (Cosine + Linear Head) Rank-1 Accuracy**
- **Open-Set Rejection:** Correct rejection rate for unknown dogs (Cosine < threshold)

### V1 Synthetic Probe Benchmark

Tests identification against augmented synthetic probes (zoomed in, zoomed out, reshaped):

```bash
cd phodog && python scripts/eval_augmented_benchmark.py --holdout 5 --threshold 0.64
```

This script evaluates:
- **Rank-1 Identification Accuracy:** Percentage of queries where the top prediction is the exact enrolled dog.
- **Rank-3 Accuracy:** Percentage of queries where the true dog is within the top-3 predictions.
- **Augmentation Breakdown:** Performance across `zoomed_in`, `zoomed_out`, and `reshaped` probes.
- **Open-Set False Acceptance Rejection:** Correct rejection rate for holdout dogs not in the gallery.

---

## 📖 Training Guides

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

A teacher-student distillation framework is implemented in [`vidog/det5.py`](vidog/det5.py) and [`vidog/scripts/batch_reenroll_student.py`](vidog/scripts/batch_reenroll_student.py):
- **Teacher Model:** `DNNetV3` — TinyViT-21M backbone, 1024-d L2-normalized embeddings. Cloud/desktop.
- **Student Model:** `StudentDNNet` — MobileNetV3-Small backbone, 512-d L2-normalized embeddings (~8ms CPU). On-device mobile inference.
- **Distillation Loss:** `DistillationLoss` combines Embedding MSE + KL-divergence on softened logits + Hard-label CE.

---

## ⚙️ Key Model Components

| Component | File | Description |
|---|---|---|
| **DNNetV3** | `vidog/det5.py`, `phodog/det5.py` | Teacher model: TinyViT-21M backbone + 1024-d embedding head with BN→SiLU→Dropout |
| **MagFaceLoss** | `vidog/det5.py`, `phodog/det5.py` | Quality-aware margin loss with adaptive margins based on feature norm $\|f\|$ |
| **CLAHEPipeline** | `vidog/det5.py`, `phodog/det5.py` | LAB color space CLAHE preprocessing with Laplacian sharpness scoring |
| **StudentDNNet** | `vidog/det5.py`, `phodog/det5.py` | Mobile student: MobileNetV3-Small, 512-d embeddings + 1024-d projection for KD |
| **NoseDetector** | `vidog/nose_detector.py`, `phodog/nose_detector.py` | YOLOv8 ONNX detector with center-crop fallback |
| **VideoRegistrationEngine** | `vidog/video_engine.py` | Video frame sampling, blur filtering, diversity selection |
| **FastLinearClassifier** | `vidog/classifier_head.py` | Few-shot Logistic Regression classifier head |
| **HybridBiometricMatcher** | `vidog/matcher.py` | 2-Stage matcher: Open-Set Cosine Gate + Linear Boundary Classifier |

---

## 📁 Gallery Storage Layout

### V1 (`phodog/` — Photo-based)

```text
data/
└── gallery/
    └── <dog_name>/
        ├── 0001.npy      # 1024-d nose embedding (float32, L2-normalized)
        ├── 0001.jpg      # Nose crop thumbnail
        ├── face_0001.npy # 1024-d face embedding (optional)
        ├── 0002.npy
        ├── ...
        └── meta.json     # {breed, age, color, registered_at, ...}
```

### V2 (`vidog/` — Video-based)

```text
data/
└── gallery_v2/
    └── <dog_name>/
        ├── 0001.npy      # 1024-d nose embedding
        ├── face_0001.npy # 1024-d face embedding (15% weight)
        ├── 0001.jpg      # Cropped nose thumbnail
        ├── original_0001.jpg
        ├── 0001.json     # Frame metadata: bbox, sharpness, detector_confidence
        ├── meta.json     # Dog master metadata
        └── ...
    └── _model/
        ├── classifier.pkl        # Fitted LogisticRegression (few-shot head)
        └── class_map.json        # Class index ↔ dog name mapping
```

---

## 🎯 Parameters & Thresholds

| Parameter | Value | Where |
|---|---|---|
| Nose weight (dual-stream) | 0.85 | `vidog/matcher.py`, `phodog/app.py` |
| Face weight (dual-stream) | 0.15 | `vidog/matcher.py`, `phodog/app.py` |
| Open-Set Cosine Gate threshold | 0.64 | `vidog/matcher.py`, `phodog/app.py` |
| Match acceptance threshold | 0.78 | `vidog/matcher.py` |
| Hybrid score weights | 60% Cosine + 40% Classifier | `vidog/matcher.py` |
| Match threshold (V1) | 0.82 | `phodog/app.py` |
| Min sharpness (registration, V2) | 70.0 | `vidog/app.py` |
| Min sharpness (identification, V1) | 60.0 | `phodog/app.py` |
| Nose detector confidence threshold | 0.60 | `vidog/nose_detector.py` |
| Target embedding dimension | 1024 (teacher) / 512 (student) | `vidog/det5.py` |
| Image input size | 224×224 | `vidog/det5.py` |
