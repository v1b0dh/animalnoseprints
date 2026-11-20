import os
import io
import time
import json
import shutil
import tempfile
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector
from video_engine import VideoRegistrationEngine
from matcher import PureCosineMatcher

app = FastAPI(
    title="Vidog Zero-Friction Biometric Service",
    description="Drop-in 2-endpoint REST API for PFA Portal. No video or image files are saved to disk; only compact vector matrices and metadata are stored.",
    version="3.0"
)

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

# 1. Warm-up models in memory
print("[INIT] Pre-loading TinyViT biometric backbone & YOLO snout detector...")
pipeline = CLAHEPipeline(image_size=224)
model = DNNetV3(pretrained=True, use_head=True)
model.eval()

nose_detector = NoseDetector(weights_path=NOSE_DETECTOR_PATH)
matcher = PureCosineMatcher(
    gallery_dir=GALLERY_DIR,
    open_set_threshold=0.68,
    match_acceptance_threshold=0.78,
    body_acceptance_threshold=0.82,
    nose_weight=0.80,
    body_weight=0.20
)
print("[INIT] Ready for real-time registration and identification.")

def extract_features(img_pil: Image.Image, nose_pil: Optional[Image.Image] = None):
    import torch
    emb_nose, emb_face = None, None
    with torch.no_grad():
        if nose_pil is not None:
            t_nose, _ = pipeline(nose_pil)
            emb_nose = model(t_nose.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
        if img_pil is not None:
            t_face, _ = pipeline(img_pil)
            emb_face = model(t_face.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
    return emb_nose, emb_face

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "engine": "vidog_pure_cosine_similarity",
        "detector_loaded": nose_detector.has_model,
        "gallery_dogs_count": len([d for d in os.listdir(GALLERY_DIR) if os.path.isdir(os.path.join(GALLERY_DIR, d)) and not d.startswith('_')]) if os.path.exists(GALLERY_DIR) else 0
    }

# ==============================================================================
# CALL 1: REGISTRATION ENDPOINT
# ==============================================================================
@app.post("/api/v1/register")
async def register(
    request: Request,
    case_id: str = Form(..., description="Unique case identifier from portal"),
    dog_name: Optional[str] = Form("Unknown"),
    metadata: Optional[str] = Form(None, description="Optional raw JSON string of form details"),
    # Direct field bindings
    nose_video: Optional[UploadFile] = File(None),
    body_video: Optional[UploadFile] = File(None),
    # Flexible multi-file list in case website form submits multiple files under 'videos' or 'files'
    videos: Optional[List[UploadFile]] = File(None)
):
    """
    Call 1: Registration
    -------------------
    Accepts:
      - case_id, dog_name, and any extra form metadata.
      - 2 video uploads distinguished by form key OR filename tag ('nose_*' vs 'body_*').
    Processing:
      - Nose video -> YOLO crop snout -> CLAHE -> 40-50 sharp vector embeddings.
      - Body video -> Direct frame splitting -> 40-50 sharp vector embeddings (NO YOLO).
      - NO videos or images are saved to disk. Only compact vector files (nose_vectors.npz, body_vectors.npz) & meta.json are stored.
    """
    # 1. Parse additional metadata if passed as JSON string
    form_data: Dict[str, Any] = {}
    if metadata:
        try:
            form_data = json.loads(metadata)
        except Exception:
            form_data = {"raw_metadata": metadata}

    # 2. Sort files into Nose Video and Body Video based on key or filename prefix
    actual_nose_vid: Optional[UploadFile] = nose_video
    actual_body_vid: Optional[UploadFile] = body_video

    # Check flexible file list if provided
    all_uploads = []
    if videos:
        all_uploads.extend(videos)

    for up in all_uploads:
        fname = (up.filename or "").lower()
        if ("nose" in fname or "snout" in fname or "face" in fname) and actual_nose_vid is None:
            actual_nose_vid = up
        elif ("body" in fname or "full" in fname) and actual_body_vid is None:
            actual_body_vid = up
        elif actual_nose_vid is None:
            actual_nose_vid = up
        elif actual_body_vid is None:
            actual_body_vid = up

    if actual_nose_vid is None and actual_body_vid is None:
        raise HTTPException(
            status_code=400,
            detail="Registration failed: Please provide at least one video (tagged with 'nose' or 'body')."
        )

    dog_folder = os.path.join(GALLERY_DIR, case_id)
    os.makedirs(dog_folder, exist_ok=True)

    engine = VideoRegistrationEngine(
        target_fps=4.0,
        min_sharpness=65.0,
        min_body_sharpness=35.0,
        target_k_frames=45  # 40-50 diverse frames
    )

    enrolled_nose_count = 0
    enrolled_body_count = 0

    # 3. Process Nose Video (Cropped Snout -> Biometric Vectors)
    if actual_nose_vid is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_nose:
            shutil.copyfileobj(actual_nose_vid.file, tmp_nose)
            tmp_path = tmp_nose.name

        try:
            selected_nose_frames, stats_n = engine.process_video(
                tmp_path,
                nose_detector_fn=nose_detector.detect_and_crop
            )
            nose_vecs = []
            face_vecs = []
            for frame_obj in selected_nose_frames:
                emb_n, emb_f = extract_features(frame_obj.full_image, frame_obj.nose_crop)
                nose_vecs.append(emb_n)
                face_vecs.append(emb_f)

            if nose_vecs:
                # Save as single compact compressed matrix
                np.savez_compressed(
                    os.path.join(dog_folder, "nose_vectors.npz"),
                    nose_vectors=np.array(nose_vecs, dtype=np.float32),
                    face_vectors=np.array(face_vecs, dtype=np.float32)
                )
                enrolled_nose_count = len(nose_vecs)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

    # 4. Process Body Video (Direct Frame Splitting -> Body Vectors, NO YOLO)
    if actual_body_vid is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_body:
            shutil.copyfileobj(actual_body_vid.file, tmp_body)
            tmp_body_path = tmp_body.name

        try:
            selected_body_frames, stats_b = engine.process_body_video(tmp_body_path)
            body_vecs = []
            for b_frame in selected_body_frames:
                _, emb_b = extract_features(b_frame.full_image, None)
                body_vecs.append(emb_b)

            if body_vecs:
                # Save as single compact compressed matrix
                np.savez_compressed(
                    os.path.join(dog_folder, "body_vectors.npz"),
                    body_vectors=np.array(body_vecs, dtype=np.float32)
                )
                enrolled_body_count = len(body_vecs)
        finally:
            if os.path.exists(tmp_body_path):
                os.remove(tmp_body_path)

    # 5. Save Combined Master Metadata
    master_meta = {
        "case_id": case_id,
        "dog_name": dog_name,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nose_frames_enrolled": enrolled_nose_count,
        "body_frames_enrolled": enrolled_body_count,
        "has_nose_stream": enrolled_nose_count > 0,
        "has_body_stream": enrolled_body_count > 0,
        "form_metadata": form_data
    }

    with open(os.path.join(dog_folder, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(master_meta, f, indent=2)

    return {
        "success": True,
        "message": f"Successfully enrolled dog '{dog_name}' for case '{case_id}'.",
        "case_id": case_id,
        "dog_name": dog_name,
        "nose_frames_enrolled": enrolled_nose_count,
        "body_frames_enrolled": enrolled_body_count,
        "metadata": master_meta
    }

# ==============================================================================
# CALL 2: IDENTIFICATION ENDPOINT
# ==============================================================================
@app.post("/api/v1/identify")
async def identify(
    image: Optional[UploadFile] = File(None, description="Query image of the dog"),
    photo: Optional[UploadFile] = File(None),  # Alias
    file: Optional[UploadFile] = File(None),   # Alias
    gender: Optional[str] = Form(None)
):
    """
    Call 2: Identification
    ----------------------
    Accepts:
      - A single uploaded query image (under field 'image', 'photo', or 'file').
    Processing:
      - Detects snout crop. If detected -> extracts nose + face embeddings.
      - Also computes full image body embedding.
      - Matches against all registered dogs using Pure Cosine weighted rules:
          • Both available in gallery: 80% Nose + 20% Body.
          • Only Nose in gallery: 100% Nose.
          • Only Body in gallery: 100% Body.
    Returns:
      - Status ('MATCH', 'LOW_CONFIDENCE', or 'UNKNOWN').
      - Top matched case details and candidate ranking with similarity scores.
    """
    actual_img = image or photo or file
    if actual_img is None:
        raise HTTPException(status_code=400, detail="Identification failed: Query image is required.")

    img_bytes = await actual_img.read()
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file uploaded.")

    # 1. Detect Snout
    crop_res = nose_detector.detect_and_crop(pil_img)
    has_snout = crop_res.crop is not None and crop_res.confidence >= 0.40

    emb_nose, emb_face = None, None
    if has_snout:
        emb_nose, emb_face = extract_features(pil_img, crop_res.crop)

    # 2. Extract full-view body embedding as well
    _, emb_body = extract_features(pil_img, None)

    # 3. Match against gallery
    best_match, all_candidates = matcher.match(
        query_nose_emb=emb_nose,
        query_face_emb=emb_face,
        query_body_emb=emb_body,
        gender=gender
    )

    if not best_match:
        return {
            "success": True,
            "status": "UNKNOWN",
            "message": "No match found above acceptance threshold.",
            "top_match": None,
            "candidates": []
        }

    return {
        "success": True,
        "status": best_match.status,  # 'MATCH', 'LOW_CONFIDENCE', 'UNKNOWN'
        "rule_applied": best_match.rule_applied,
        "top_match": {
            "case_id": best_match.dog_name,
            "similarity_score": round(best_match.cosine_score, 4),
            "nose_score": round(best_match.nose_score, 4) if best_match.nose_score is not None else None,
            "body_score": round(best_match.body_score, 4) if best_match.body_score is not None else None,
            "best_matching_frame": best_match.best_matching_frame
        },
        "candidates": [
            {
                "case_id": c.dog_name,
                "similarity_score": round(c.cosine_score, 4),
                "nose_score": round(c.nose_score, 4) if c.nose_score is not None else None,
                "body_score": round(c.body_score, 4) if c.body_score is not None else None,
                "rule_applied": c.rule_applied,
                "status": c.status
            }
            for c in all_candidates[:5]
        ]
    }
