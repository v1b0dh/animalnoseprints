# Vidog Biometric Engine × PFA Portal (trial-) Integration Guide

This document is the complete, end-to-end technical reference and handoff guide for integrating the **`vidog` Dog Biometric Identification Engine** into the **PFA (People For Animals) Web Portal (`trial-`)**.

---

## 1. System Overview & Architecture

### What is the Goal?
In animal rescue shelters and veterinary hospitals, stray dogs frequently return for follow-up treatments, vaccinations, or post-release ABC (Animal Birth Control) checks. Because stray dogs do not have microchips or collars, identifying duplicate cases or linking a returning dog to its medical history is difficult.

The **`vidog`** engine analyzes dog nose prints (macro-texture rhinarium patterns, which act as unique biometrics similar to human fingerprints) along with facial structure. 

### High-Level Architecture Diagram

```
┌────────────────────────────────────────────────────────────────────────┐
│                   PFA Portal (React 19 / Vite Frontend)                │
│   • Case Reporting (Upload Video/Photo)                                │
│   • Biometric Identification Preview & Duplicate Alert                 │
│   • Animal Profile Biometric Status Badge                              │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ HTTP / REST (Multipart Form)
┌───────────────────────────────────▼────────────────────────────────────┐
│                    PFA Backend (Node.js / Express)                     │
│    POST /api/cases                 POST /api/cases/identify            │
│    (Saves to DB & proxies file)    (Proxies query to Python)           │
└─────────────────┬─────────────────────────────────┬────────────────────┘
                  │ Prisma ORM                      │ HTTP / REST
                  ▼                                 ▼
┌─────────────────────────────────────┐  ┌───────────────────────────────┐
│       PostgreSQL Database           │  │  vidog FastAPI Microservice   │
│  • Cases & Animals tables           │  │  (Python + ONNX + PyTorch)    │
│  • BiometricFrames (Embeddings/     │  │  • Port: 8000                 │
│    Blobs, sharpness, metadata)      │  │  • Snout Detection & Sampling │
│  • Matched Animal references        │  │  • Pure Cosine Matcher        │
└─────────────────────────────────────┘  └───────────────────────────────┘
```

---

## 2. Deployment Strategy: Same Server vs Separate Server?

### Recommended: Co-located on the Same Host / Virtual Machine
Run both services on the same VM (e.g., AWS EC2, DigitalOcean Droplet, Hetzner, or local Linux box):

* **PFA Portal (Node.js / Express)**: Runs on `http://localhost:5000` (or `3000`).
* **PostgreSQL Database**: Runs on `localhost:5432`.
* **Vidog Biometric Service (FastAPI / Uvicorn)**: Runs on `http://localhost:8000`.

### Why this is best:
1. **Low Latency & No Egress Bandwidth Cost**: High-definition video and photo uploads stay on `localhost` without incurring cloud transfer fees or network lag between microservices.
2. **Persistent In-Memory AI Models**: PyTorch (`DNNetV3`) and ONNX Runtime (`best.onnx`, `nose_detector.onnx`) stay warmed up in RAM/VRAM. Inference happens in milliseconds instead of reloading weights from disk every time.
3. **Future Scalability**: If the platform scales and needs a dedicated GPU instance, your friend only needs to update one environment variable in Node.js (`BIOMETRIC_API_URL="https://gpu-service.internal/api/v1"`).

---

## 3. Data Storage & Formats

### Metadata Format (`meta.json` / JSON Column)
Stores master dog properties, registration timestamp, video statistics, and tracking information:
```json
{
  "case_id": "c4b3a1d2-0000-4000-8000-0123456789ab",
  "dog_name": "Tommy",
  "breed": "Indie / Stray",
  "age_years": 3,
  "gender": "Male",
  "color": "Fawn / Light Brown",
  "registered_at": "2026-11-04 22:50:00",
  "frame_count": 5,
  "avg_sharpness": 158.4,
  "video_duration_sec": 8.5
}
```

### Embedding Vectors (`.npy` files or Database Blobs)
* **Nose Embedding (`emb_nose`)**: Extracted from high-resolution, CLAHE-enhanced snout crops. Dimension: `512` (or `1024` depending on head output), float32. Weight: **85%**.
* **Face Embedding (`emb_face`)**: Extracted from head/face crop. Dimension: `512`, float32. Weight: **15%**.
* **Storage Options**:
  1. **Option A (Disk / S3 - Recommended for Simplicity)**: Stored in directory `storage/gallery/{case_id}/0001.npy`, `storage/gallery/{case_id}/face_0001.npy`.
  2. **Option B (PostgreSQL Bytes / Blobs)**: Stored as raw bytes in `biometric_frames.nose_blob` and `biometric_frames.face_blob`.

---

## 4. Part 1: Python Side Changes (`vidog`)

In your `vidog` repository folder, create `api.py` using **FastAPI**. This exposes clean REST endpoints for registration and identification.

### `vidog/api.py`

```python
import os
import io
import time
import shutil
import tempfile
from typing import Optional, List
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector
from video_engine import VideoRegistrationEngine
from matcher import PureCosineMatcher

app = FastAPI(title="Vidog Biometric Identification Service", version="2.0")

# Enable CORS for internal communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(CUR_DIR, "data", "gallery_v2")
CHECKPOINTS_DIR = os.path.join(CUR_DIR, "checkpoints")
NOSE_DETECTOR_PATH = os.path.join(CHECKPOINTS_DIR, "nose_detector.onnx")

os.makedirs(GALLERY_DIR, exist_ok=True)

# 1. Global Model Preloading (Persistent in memory)
print("Loading biometric backbone models into memory...")
pipeline = CLAHEPipeline(image_size=224)
model = DNNetV3(pretrained=True, use_head=True)
model.eval()

nose_detector = NoseDetector(weights_path=NOSE_DETECTOR_PATH)
matcher = PureCosineMatcher(
    gallery_dir=GALLERY_DIR,
    open_set_threshold=0.68,
    match_acceptance_threshold=0.78,
    nose_weight=0.85,
    face_weight=0.15
)
print("Models successfully loaded and ready for inference.")

def extract_features(img_pil: Image.Image, nose_pil: Image.Image):
    import torch
    t_nose, _ = pipeline(nose_pil)
    t_face, _ = pipeline(img_pil)
    with torch.no_grad():
        emb_nose = model(t_nose.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
        emb_face = model(t_face.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
    return emb_nose, emb_face

# --- ENDPOINTS ---

@app.get("/health")
def health():
    return {"status": "ok", "detector_ready": nose_detector.has_model}

@app.post("/api/v1/register")
async def register_dog(
    video: UploadFile = File(...),
    case_id: str = Form(...),
    dog_name: Optional[str] = Form("Unknown"),
    gender: Optional[str] = Form(None),
    breed: Optional[str] = Form(None),
    color: Optional[str] = Form(None)
):
    """
    Accepts an uploaded video, extracts optimal diverse frames,
    computes embeddings, and registers the dog in the gallery.
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_vid:
        shutil.copyfileobj(video.file, tmp_vid)
        tmp_video_path = tmp_vid.name

    try:
        engine = VideoRegistrationEngine(
            target_frames=5,
            laplacian_threshold=60.0,
            sim_threshold=0.82
        )
        selected_frames, stats = engine.sample_sharp_diverse_frames(tmp_video_path)

        if not selected_frames:
            raise HTTPException(
                status_code=400,
                detail=f"Could not extract sharp dog snout frames from video. Reason: {stats.get('reason', 'Too blurry or no snout detected')}"
            )

        # Create folder for this case/dog
        dog_dir = os.path.join(GALLERY_DIR, case_id)
        os.makedirs(dog_dir, exist_ok=True)

        sharpness_list = []
        for idx, frame_obj in enumerate(selected_frames, start=1):
            prefix = f"{idx:04d}"
            emb_nose, emb_face = extract_features(frame_obj.full_image, frame_obj.nose_crop)

            np.save(os.path.join(dog_dir, f"{prefix}.npy"), emb_nose)
            np.save(os.path.join(dog_dir, f"face_{prefix}.npy"), emb_face)
            frame_obj.nose_crop.save(os.path.join(dog_dir, f"{prefix}.jpg"), "JPEG", quality=95)
            frame_obj.full_image.save(os.path.join(dog_dir, f"original_{prefix}.jpg"), "JPEG", quality=90)

            sharpness_list.append(frame_obj.sharpness)

        master_meta = {
            "case_id": case_id,
            "dog_name": dog_name,
            "breed": breed,
            "gender": gender,
            "color": color,
            "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frame_count": len(selected_frames),
            "avg_sharpness": float(np.mean(sharpness_list)),
            "video_duration_sec": stats.get("duration_sec", 0.0)
        }

        with open(os.path.join(dog_dir, "meta.json"), "w", encoding="utf-8") as f:
            import json
            json.dump(master_meta, f, indent=2)

        return {
            "status": "success",
            "case_id": case_id,
            "enrolled_frames": len(selected_frames),
            "metadata": master_meta
        }
    finally:
        if os.path.exists(tmp_video_path):
            os.remove(tmp_video_path)

@app.post("/api/v1/identify")
async def identify_dog(
    image: UploadFile = File(...),
    gender: Optional[str] = Form(None)
):
    """
    Accepts a single query photo, runs snout detection, extracts embedding,
    and returns matching cases ranked by cosine similarity.
    """
    contents = await image.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

    # Detect Snout / Nose
    nose_crop, bbox, conf = nose_detector.detect_and_crop(pil_img)
    if nose_crop is None:
        # Fallback to center crop
        w, h = pil_img.size
        nose_crop = pil_img.crop((int(w * 0.25), int(h * 0.25), int(w * 0.75), int(h * 0.75)))
        conf = 0.0

    q_nose, q_face = extract_features(pil_img, nose_crop)

    best_match, all_candidates = matcher.match(
        query_nose_emb=q_nose,
        query_face_emb=q_face,
        gender=gender
    )

    if not best_match:
        return {
            "status": "UNKNOWN",
            "message": "No match found in gallery or gallery is empty.",
            "top_match": None,
            "candidates": []
        }

    return {
        "status": best_match.status,  # 'MATCH', 'LOW_CONFIDENCE', or 'UNKNOWN'
        "top_match": {
            "case_id": best_match.dog_name,
            "similarity_score": round(best_match.hybrid_score, 4),
            "best_frame": best_match.best_matching_frame
        },
        "candidates": [
            {
                "case_id": c.dog_name,
                "similarity_score": round(c.hybrid_score, 4),
                "status": c.status
            }
            for c in all_candidates[:5]
        ]
    }
```

---

## 5. Part 2: Database Changes for `trial-` (`prisma/schema.prisma`)

Your friend needs to apply these additions to `prisma/schema.prisma` and run `npx prisma migrate dev --name add_biometrics`:

```prisma
// --- prisma/schema.prisma ---

model Animal {
  id           String   @id @default(uuid())
  name         String
  species      String
  breed        String?
  age          Int?
  gender       String?
  healthStatus String?  @map("health_status")
  location     String?
  imageUrl     String?  @map("image_url")
  status       String
  createdAt    DateTime @default(now()) @map("created_at")

  // NEW BIOMETRIC FIELDS:
  biometricEnrolled Boolean @default(false) @map("biometric_enrolled")
  biometricCaseId   String? @map("biometric_case_id")
  biometricMeta     Json?   @map("biometric_meta")

  abcRecords   ABCRecord[]
  adoptions    Adoption[]
  cases        Case[]

  @@map("animals")
}

model Case {
  id           String     @id @default(uuid())
  title        String
  description  String
  location     String
  status       CaseStatus @default(open)
  reportedById String?    @map("reported_by")
  imageUrl     String?    @map("image_url")
  videoUrl     String?    @map("video_url")
  createdAt    DateTime   @default(now()) @map("created_at")

  // NEW BIOMETRIC MATCH FIELDS:
  matchedAnimalId   String?  @map("matched_animal_id")
  matchConfidence   Float?   @map("match_confidence")
  animal            Animal?  @relation(fields: [matchedAnimalId], references: [id])

  reporter          User?    @relation("ReportedBy", fields: [reportedById], references: [id])
  clinicalEntries   ClinicalEntry[]

  @@map("cases")
}
```

---

## 6. Part 3: Backend Node.js Changes for `trial-` (Express)

### Step 1: Create Biometric Bridge Client (`server/services/biometrics.service.ts`)

```typescript
import fs from 'fs';
import FormData from 'form-data';

const BIOMETRIC_API_URL = process.env.BIOMETRIC_API_URL || 'http://localhost:8000/api/v1';

export async function registerCaseVideo(videoPath: string, caseId: string, dogDetails: {
  name?: string;
  breed?: string;
  gender?: string;
  color?: string;
}) {
  const form = new FormData();
  form.append('video', fs.createReadStream(videoPath));
  form.append('case_id', caseId);
  if (dogDetails.name) form.append('dog_name', dogDetails.name);
  if (dogDetails.breed) form.append('breed', dogDetails.breed);
  if (dogDetails.gender) form.append('gender', dogDetails.gender);
  if (dogDetails.color) form.append('color', dogDetails.color);

  const response = await fetch(`${BIOMETRIC_API_URL}/register`, {
    method: 'POST',
    body: form as any,
    headers: form.getHeaders(),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Biometric registration failed: ${errorText}`);
  }

  return await response.json();
}

export async function identifyDogImage(imagePath: string, gender?: string) {
  const form = new FormData();
  form.append('image', fs.createReadStream(imagePath));
  if (gender) form.append('gender', gender);

  const response = await fetch(`${BIOMETRIC_API_URL}/identify`, {
    method: 'POST',
    body: form as any,
    headers: form.getHeaders(),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(`Biometric identification failed: ${errorText}`);
  }

  return await response.json();
}
```

### Step 2: Add Identification Route (`server/routes/case.routes.ts`)

```typescript
// POST /api/cases/identify
router.post('/identify', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'Image file required' });
    }

    const { gender } = req.body;
    const matchResult = await identifyDogImage(req.file.path, gender);

    // If a high-confidence match is returned, fetch existing case & animal info from Postgres
    let matchedCaseData = null;
    if (matchResult.top_match && matchResult.top_match.case_id) {
      matchedCaseData = await prisma.case.findUnique({
        where: { id: matchResult.top_match.case_id },
        include: {
          clinicalEntries: { orderBy: { date: 'desc' }, take: 3 },
          animal: true
        }
      });
    }

    return res.json({
      biometricResult: matchResult,
      matchedCase: matchedCaseData
    });
  } catch (error: any) {
    return sendError(res, error.message || 'Identification failed', 500);
  }
});
```

### Step 3: Trigger Video Registration in Case Creation

In `POST /api/cases`:
```typescript
if (videoUrl && req.file && req.file.mimetype.startsWith('video/')) {
  try {
    // Run registration in background or inline
    const bioResult = await registerCaseVideo(req.file.path, newCase.id, {
      name: newCase.title,
      gender: req.body.gender,
    });
    console.log(`[Biometrics] Successfully enrolled case ${newCase.id}:`, bioResult);
  } catch (err) {
    console.warn(`[Biometrics] Video enrollment skipped/failed:`, err);
  }
}
```

---

## 7. Part 4: Frontend Changes in React (`trial-`)

### In `src/pages/NewCasePage.tsx`:

1. **Add Biometric Scan Button**:
   When the user selects a photo, show an **"Analyze & Check Existing Records"** button next to the preview.
2. **Match Alert Banner**:
   If `matchResult.status === 'MATCH'` (similarity $\ge 0.78$):
   * Display a warning card:
     > **Match Found (84% Confidence)**: This dog matches **Case #1042 ("Sheru")** admitted on Oct 12, 2026.
     > `[View Existing Case]` or `[Link This Case to Sheru]`
3. **Registration Feedback**:
   When uploading a video, show:
   * *"Analyzing video frames with Vidog AI... Enrolling nose-print into shelter database."*

---

## 8. Summary of Developer Handoff Checklist

Share this final punchlist with your friend:

| Component | Responsibility / Action |
| :--- | :--- |
| **`vidog/api.py`** | You run this FastAPI server on port `8000`. |
| **Environment Variable** | Add `BIOMETRIC_API_URL="http://localhost:8000/api/v1"` into `.env` in `trial-`. |
| **Prisma Schema** | Add `biometricEnrolled`, `matchedAnimalId`, `matchConfidence` to `prisma/schema.prisma` and run `npx prisma migrate dev`. |
| **Express Route** | Add `POST /api/cases/identify` that passes the image to port 8000. |
| **React UI** | Add the match preview card in `NewCasePage.tsx` when an image is uploaded. |

