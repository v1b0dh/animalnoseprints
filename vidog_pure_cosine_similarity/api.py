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

app = FastAPI(title="Vidog Dual-Stream Biometric Service", version="2.5")

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

# Preload models into memory
print("[INIT] Loading DNNetV3 backbone and YOLO snout detector...")
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
print("[INIT] Models ready for dual-stream registration and identification.")

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
    return {"status": "ok", "detector_ready": nose_detector.has_model}

@app.post("/api/v1/register")
async def register_dog(
    nose_video: Optional[UploadFile] = File(None),
    body_video: Optional[UploadFile] = File(None),
    video: Optional[UploadFile] = File(None),  # Fallback: checks filename prefix
    case_id: str = Form(...),
    dog_name: Optional[str] = Form("Unknown"),
    gender: Optional[str] = Form(None),
    breed: Optional[str] = Form(None),
    color: Optional[str] = Form(None)
):
    """
    Registers a dog using dual video streams:
      1. nose_video (or file prefixed with nose_): 40-50 sharp snout frames + embeddings.
      2. body_video (or file prefixed with body_): 40-50 sharp full body frames (direct splitting).
    """
    # Auto-sort if a single or combined video upload was passed with naming conventions
    actual_nose_video = nose_video
    actual_body_video = body_video

    if video is not None:
        fname = (video.filename or "").lower()
        if "nose" in fname and actual_nose_video is None:
            actual_nose_video = video
        elif "body" in fname and actual_body_video is None:
            actual_body_video = video
        elif actual_nose_video is None:
            actual_nose_video = video

    if actual_nose_video is None and actual_body_video is None:
        raise HTTPException(status_code=400, detail="At least one video (nose_video or body_video) must be provided.")

    dog_dir = os.path.join(GALLERY_DIR, case_id)
    os.makedirs(dog_dir, exist_ok=True)

    engine = VideoRegistrationEngine(
        target_fps=4.0,
        min_sharpness=65.0,
        min_body_sharpness=35.0,
        target_k_frames=45  # Extract 40-50 frames automatically
    )

    enrolled_nose_frames = 0
    enrolled_body_frames = 0

    # 1. Process Nose Video
    if actual_nose_video is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_nose:
            shutil.copyfileobj(actual_nose_video.file, tmp_nose)
            tmp_nose_path = tmp_nose.name

        try:
            selected_frames, _ = engine.process_video(
                tmp_nose_path,
                nose_detector_fn=nose_detector.detect_and_crop
            )
            for idx, frame_obj in enumerate(selected_frames, start=1):
                prefix = f"{idx:04d}"
                emb_n, emb_f = extract_features(frame_obj.full_image, frame_obj.nose_crop)
                np.save(os.path.join(dog_dir, f"{prefix}.npy"), emb_n)
                np.save(os.path.join(dog_dir, f"face_{prefix}.npy"), emb_f)
                frame_obj.nose_crop.save(os.path.join(dog_dir, f"{prefix}.jpg"), "JPEG", quality=95)
                frame_obj.full_image.save(os.path.join(dog_dir, f"original_{prefix}.jpg"), "JPEG", quality=90)
            enrolled_nose_frames = len(selected_frames)
        finally:
            if os.path.exists(tmp_nose_path):
                os.remove(tmp_nose_path)

    # 2. Process Body Video (Direct frame extraction, bypasses nose detector)
    if actual_body_video is not None:
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp_body:
            shutil.copyfileobj(actual_body_video.file, tmp_body)
            tmp_body_path = tmp_body.name

        try:
            selected_body_frames, _ = engine.process_body_video(tmp_body_path)
            for idx, b_frame in enumerate(selected_body_frames, start=1):
                prefix = f"body_{idx:04d}"
                _, emb_b = extract_features(b_frame.full_image, None)
                np.save(os.path.join(dog_dir, f"{prefix}.npy"), emb_b)
                b_frame.full_image.save(os.path.join(dog_dir, f"{prefix}.jpg"), "JPEG", quality=90)
            enrolled_body_frames = len(selected_body_frames)
        finally:
            if os.path.exists(tmp_body_path):
                os.remove(tmp_body_path)

    master_meta = {
        "case_id": case_id,
        "dog_name": dog_name,
        "breed": breed,
        "gender": gender,
        "color": color,
        "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "nose_frames_enrolled": enrolled_nose_frames,
        "body_frames_enrolled": enrolled_body_frames,
        "has_nose_stream": enrolled_nose_frames > 0,
        "has_body_stream": enrolled_body_frames > 0
    }

    with open(os.path.join(dog_dir, "meta.json"), "w", encoding="utf-8") as f:
        import json
        json.dump(master_meta, f, indent=2)

    return {
        "status": "success",
        "case_id": case_id,
        "nose_frames_enrolled": enrolled_nose_frames,
        "body_frames_enrolled": enrolled_body_frames,
        "metadata": master_meta
    }

@app.post("/api/v1/identify")
async def identify_dog(
    nose_image: Optional[UploadFile] = File(None),
    body_image: Optional[UploadFile] = File(None),
    image: Optional[UploadFile] = File(None),  # fallback query
    gender: Optional[str] = Form(None)
):
    """
    Multi-modal identification endpoint.
    Applies strict weighting rules:
      - Both present: 80% Nose + 20% Body
      - Only Nose: 100% Nose
      - Only Body: 100% Body
    """
    actual_nose_img = nose_image
    actual_body_img = body_image

    if image is not None:
        fname = (image.filename or "").lower()
        if "body" in fname and actual_body_img is None:
            actual_body_img = image
        elif actual_nose_img is None:
            actual_nose_img = image

    if actual_nose_img is None and actual_body_img is None:
        raise HTTPException(status_code=400, detail="Provide at least one query image (nose_image or body_image).")

    emb_nose, emb_face, emb_body = None, None, None

    if actual_nose_img is not None:
        bytes_n = await actual_nose_img.read()
        pil_n = Image.open(io.BytesIO(bytes_n)).convert("RGB")
        crop_res = nose_detector.detect_and_crop(pil_n)
        emb_nose, emb_face = extract_features(pil_n, crop_res.crop)

    if actual_body_img is not None:
        bytes_b = await actual_body_img.read()
        pil_b = Image.open(io.BytesIO(bytes_b)).convert("RGB")
        _, emb_body = extract_features(pil_b, None)

    best_match, all_candidates = matcher.match(
        query_nose_emb=emb_nose,
        query_face_emb=emb_face,
        query_body_emb=emb_body,
        gender=gender
    )

    if not best_match:
        return {
            "status": "UNKNOWN",
            "message": "No biometric match found above acceptance gate.",
            "top_match": None,
            "candidates": []
        }

    return {
        "status": best_match.status,
        "rule_applied": best_match.rule_applied,
        "top_match": {
            "case_id": best_match.dog_name,
            "composite_score": round(best_match.cosine_score, 4),
            "nose_score": round(best_match.nose_score, 4) if best_match.nose_score is not None else None,
            "body_score": round(best_match.body_score, 4) if best_match.body_score is not None else None,
            "best_matching_frame": best_match.best_matching_frame
        },
        "candidates": [
            {
                "case_id": c.dog_name,
                "composite_score": round(c.cosine_score, 4),
                "nose_score": round(c.nose_score, 4) if c.nose_score is not None else None,
                "body_score": round(c.body_score, 4) if c.body_score is not None else None,
                "rule_applied": c.rule_applied,
                "status": c.status
            }
            for c in all_candidates[:5]
        ]
    }
