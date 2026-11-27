import os
import io
import re
import time
import json
import shutil
import tempfile
import threading
from typing import Optional, List
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Thread lock to guarantee atomic, race-condition-free writes across concurrent requests
FILE_WRITE_LOCK = threading.Lock()

# Security limits
MAX_VIDEO_BYTES = int(os.getenv("MAX_VIDEO_BYTES", 150 * 1024 * 1024))  # 150 MB
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", 15 * 1024 * 1024))   # 15 MB
CASE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,127}$")

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector
from video_engine import VideoRegistrationEngine
from matcher import PureCosineMatcher

app = FastAPI(
    title="Vidog Simple Biometrics Engine",
    description="Simple 2-endpoint REST API for PFA Portal. Handles Registration (Case ID + Videos) and Identification (Query Photo).",
    version="3.1"
)

ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS",
        "http://localhost:3000,http://localhost:5000,http://127.0.0.1:3000,http://127.0.0.1:5000",
    ).split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
GALLERY_DIR = os.path.join(CUR_DIR, "data", "gallery_v2")
CHECKPOINTS_DIR = os.path.join(CUR_DIR, "checkpoints")
NOSE_DETECTOR_PATH = os.path.join(CHECKPOINTS_DIR, "nose_detector.onnx")

os.makedirs(GALLERY_DIR, exist_ok=True)

# 1. Warm-up models in memory once at startup
print("[INIT] Loading DNNetV3 biometric backbone & YOLO snout detector...")
pipeline = CLAHEPipeline(image_size=224)

# STRICT HARD-LOCK: Enforce 1024-d fine-tuned MagFace weights
# This prevents silent dimension mismatches (576 vs 1024) across server sessions.
TEACHER_CKPT = os.path.join(CHECKPOINTS_DIR, "teacher_best.pth")
FALLBACK_CKPT = os.path.join(CHECKPOINTS_DIR, "dnnet_v3_best.pth")

target_ckpt = None
if os.path.exists(TEACHER_CKPT):
    target_ckpt = TEACHER_CKPT
elif os.path.exists(FALLBACK_CKPT):
    target_ckpt = FALLBACK_CKPT

if not target_ckpt:
    raise RuntimeError(
        f"FATAL: Trained checkpoint not found in {CHECKPOINTS_DIR}! "
        f"Expected 'teacher_best.pth' or 'dnnet_v3_best.pth'. "
        f"Hard-lock active to prevent vector dimension mismatches."
    )

import torch
model = DNNetV3(pretrained=True, use_head=True)
ckpt = torch.load(target_ckpt, map_location="cpu", weights_only=True)
sd = ckpt.get("model_state_dict", ckpt)
model.load_state_dict(sd)
model.eval()
print(f"[INIT] Successfully locked and loaded fine-tuned 1024-d backbone weights from {target_ckpt}.")

nose_detector = NoseDetector(weights_path=NOSE_DETECTOR_PATH)
matcher = PureCosineMatcher(
    gallery_dir=GALLERY_DIR,
    open_set_threshold=0.68,
    match_acceptance_threshold=0.78,
    body_acceptance_threshold=0.82,
    nose_weight=0.80,
    body_weight=0.20
)
print("[INIT] Ready.")

def _safe_case_dir(case_id: str) -> str:
    """Validate case_id and ensure the resolved directory stays strictly inside GALLERY_DIR."""
    cleaned = case_id.strip()
    if not CASE_ID_PATTERN.match(cleaned):
        raise HTTPException(
            status_code=400,
            detail="Invalid case_id. Use 1-128 alphanumeric characters, hyphens, or underscores (cannot start with '.' or '_').",
        )
    gallery_real = os.path.realpath(GALLERY_DIR)
    target_real = os.path.realpath(os.path.join(gallery_real, cleaned))
    if not target_real.startswith(gallery_real + os.sep):
        raise HTTPException(status_code=400, detail="Invalid case_id path.")
    return target_real


def _save_upload_to_temp(upload: UploadFile, max_bytes: int) -> str:
    """Stream an uploaded file to a temporary file while enforcing a strict byte limit."""
    total_bytes = 0
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    try:
        while True:
            chunk = upload.file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=f"Uploaded video exceeds maximum allowed size ({max_bytes // (1024 * 1024)} MB).",
                )
            tmp.write(chunk)
        tmp.close()
        return tmp.name
    except Exception:
        tmp.close()
        if os.path.exists(tmp.name):
            os.remove(tmp.name)
        raise


def _save_npz_atomic(target_path: str, **arrays: np.ndarray) -> None:
    """Atomically write compressed .npz arrays; temp name must end in .npz so NumPy doesn't append another .npz."""
    tmp_path = os.path.join(os.path.dirname(target_path), f".tmp_{os.path.basename(target_path)}")
    with FILE_WRITE_LOCK:
        try:
            np.savez_compressed(tmp_path, **arrays)
            os.replace(tmp_path, target_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


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
    return {"status": "ok"}

# ==============================================================================
# 1. REGISTRATION ENDPOINT (Initial Create OR Edit Form)
# Input: case_id + 1 or 2 videos
# ==============================================================================
@app.post("/api/v1/register")
async def register(
    case_id: str = Form(..., description="Unique case ID from the website"),
    nose_video: Optional[UploadFile] = File(None),
    body_video: Optional[UploadFile] = File(None),
    # Flexible fallback: accepts multiple files under 'videos' with 'nose' or 'body' in filenames
    videos: Optional[List[UploadFile]] = File(None)
):
    """
    Call 1: Registration (Works for Initial Registration and Edit Case)
    Input:
      - case_id (Required)
      - nose_video (Optional if body_video provided)
      - body_video (Optional if nose_video provided)
    """
    actual_nose_vid = nose_video
    actual_body_vid = body_video

    # Check flexible list if passed
    if videos:
        for v in videos:
            fn = (v.filename or "").lower()
            if ("nose" in fn or "snout" in fn or "face" in fn) and actual_nose_vid is None:
                actual_nose_vid = v
            elif ("body" in fn or "full" in fn) and actual_body_vid is None:
                actual_body_vid = v
            elif actual_nose_vid is None:
                actual_nose_vid = v
            elif actual_body_vid is None:
                actual_body_vid = v

    if actual_nose_vid is None and actual_body_vid is None:
        raise HTTPException(status_code=400, detail="Provide at least one video (nose_video or body_video).")

    # Folder dedicated to this case_id (path-traversal protected)
    dog_folder = _safe_case_dir(case_id)
    os.makedirs(dog_folder, exist_ok=True)

    engine = VideoRegistrationEngine(
        target_fps=4.0,
        min_sharpness=65.0,
        min_body_sharpness=35.0,
        target_k_frames=45
    )

    enrolled_nose_count = 0
    enrolled_body_count = 0

    # Process Nose Video: YOLO detect -> crop -> extract 40-50 vectors
    if actual_nose_vid is not None:
        tmp_nose_path = _save_upload_to_temp(actual_nose_vid, MAX_VIDEO_BYTES)
        try:
            selected_nose_frames, _ = engine.process_video(
                tmp_nose_path,
                nose_detector_fn=nose_detector.detect_and_crop
            )
            nose_vecs = []
            face_vecs = []
            for frame_obj in selected_nose_frames:
                emb_n, emb_f = extract_features(frame_obj.full_image, frame_obj.nose_crop)
                nose_vecs.append(emb_n)
                face_vecs.append(emb_f)

            if nose_vecs:
                _save_npz_atomic(
                    os.path.join(dog_folder, "nose_vectors.npz"),
                    nose_vectors=np.array(nose_vecs, dtype=np.float32),
                    face_vectors=np.array(face_vecs, dtype=np.float32),
                )
                enrolled_nose_count = len(nose_vecs)
        finally:
            if os.path.exists(tmp_nose_path):
                os.remove(tmp_nose_path)

    # Process Body Video: direct frame sampling (No YOLO) -> extract 40-50 vectors
    if actual_body_vid is not None:
        tmp_body_path = _save_upload_to_temp(actual_body_vid, MAX_VIDEO_BYTES)
        try:
            selected_body_frames, _ = engine.process_body_video(tmp_body_path)
            body_vecs = []
            for b_frame in selected_body_frames:
                _, emb_b = extract_features(b_frame.full_image, None)
                body_vecs.append(emb_b)

            if body_vecs:
                _save_npz_atomic(
                    os.path.join(dog_folder, "body_vectors.npz"),
                    body_vectors=np.array(body_vecs, dtype=np.float32),
                )
                enrolled_body_count = len(body_vecs)
        finally:
            if os.path.exists(tmp_body_path):
                os.remove(tmp_body_path)

    if enrolled_nose_count == 0 and enrolled_body_count == 0:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any sharp frames from the uploaded video(s). Please upload clearer footage.",
        )

    # Atomic update for metadata with threading lock to prevent file corruption
    meta_path = os.path.join(dog_folder, "meta.json")
    tmp_meta_path = meta_path + ".tmp"
    with FILE_WRITE_LOCK:
        existing_meta = {}
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    existing_meta = json.load(f)
            except Exception:
                pass

        # Check if previously enrolled files already exist on disk
        has_existing_nose = os.path.exists(os.path.join(dog_folder, "nose_vectors.npz"))
        has_existing_body = os.path.exists(os.path.join(dog_folder, "body_vectors.npz"))

        final_nose_count = enrolled_nose_count if actual_nose_vid is not None else existing_meta.get("nose_frames_enrolled", 0)
        final_body_count = enrolled_body_count if actual_body_vid is not None else existing_meta.get("body_frames_enrolled", 0)

        meta = {
            "case_id": case_id,
            "created_at": existing_meta.get("created_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "nose_frames_enrolled": final_nose_count,
            "body_frames_enrolled": final_body_count,
            "has_nose_stream": has_existing_nose or (enrolled_nose_count > 0),
            "has_body_stream": has_existing_body or (enrolled_body_count > 0),
            "nose_reupload_recommended": final_nose_count == 10,
            "body_reupload_recommended": final_body_count == 10,
        }
        with open(tmp_meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        os.replace(tmp_meta_path, meta_path)

    return {
        "status": "success",
        "message": f"Vectors generated successfully. Dog for case ID '{case_id}' registered/updated successfully.",
        "case_id": case_id,
        "metadata": meta
    }

# ==============================================================================
# 2. IDENTIFICATION ENDPOINT
# Input: Single query image
# ==============================================================================
@app.post("/api/v1/identify")
async def identify(
    image: Optional[UploadFile] = File(None),
    photo: Optional[UploadFile] = File(None),
    file: Optional[UploadFile] = File(None)
):
    """
    Call 2: Identification
    Input:
      - Single query photo of the dog (sent under 'image', 'photo', or 'file')
    Output:
      - Pure cosine similarity match against all registered cases.
    """
    actual_img = image or photo or file
    if actual_img is None:
        raise HTTPException(status_code=400, detail="Query image is required.")

    img_bytes = await actual_img.read(MAX_IMAGE_BYTES + 1)
    if len(img_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Uploaded image exceeds maximum allowed size ({MAX_IMAGE_BYTES // (1024 * 1024)} MB).",
        )
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image uploaded.")

    # 1. Check for snout (only trust actual YOLO detections, not center_fallback)
    crop_res = nose_detector.detect_and_crop(pil_img)
    has_snout = crop_res.source == "model" and crop_res.crop is not None and crop_res.confidence >= 0.40

    emb_nose, emb_face = None, None
    if has_snout:
        emb_nose, emb_face = extract_features(pil_img, crop_res.crop)

    # 2. Body embedding
    _, emb_body = extract_features(pil_img, None)

    # 3. Pure Cosine Matching
    best_match, all_candidates = matcher.match(
        query_nose_emb=emb_nose,
        query_face_emb=emb_face,
        query_body_emb=emb_body
    )

    if not best_match:
        return {
            "status": "UNKNOWN",
            "message": "No matching dog found.",
            "top_match": None,
            "candidates": []
        }

    def _load_meta(case_name: str) -> Optional[dict]:
        p = os.path.join(GALLERY_DIR, case_name, "meta.json")
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass
        return None

    top_meta = _load_meta(best_match.dog_name)

    return {
        "status": best_match.status,  # "MATCH", "LOW_CONFIDENCE", or "UNKNOWN"
        "top_match": {
            "case_id": best_match.dog_name,
            "similarity_score": round(best_match.cosine_score, 4),
            "rule_applied": best_match.rule_applied,
            "metadata": top_meta
        },
        "candidates": [
            {
                "case_id": c.dog_name,
                "similarity_score": round(c.cosine_score, 4),
                "status": c.status,
                "metadata": _load_meta(c.dog_name)
            }
            for c in all_candidates[:5]
        ]
    }
