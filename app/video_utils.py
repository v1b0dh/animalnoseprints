"""
video_utils.py — Video Frame Extraction & Quality Filtering for DogID System
=============================================================================
Handles all video-to-frame logic so app.py stays clean.

Pipeline:
  1. extract_frames()          — evenly sample N frames from a video file
  2. filter_frames_by_quality() — Laplacian sharpness filter + top-N selection
  3. pil_frames_from_upload()  — convenience wrapper for Streamlit uploaded bytes
"""

import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from typing import List, Tuple


# ==============================================================================
# CORE EXTRACTION
# ==============================================================================

def extract_frames(video_path: str, max_frames: int = 30) -> List[np.ndarray]:
    """
    Evenly sample up to `max_frames` frames from a video file.

    Args:
        video_path : Absolute path to the video file.
        max_frames : Maximum number of frames to sample.

    Returns:
        List of BGR numpy arrays (OpenCV format).
        Empty list if the video cannot be opened.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Evenly spaced frame indices across the full video duration
    n_sample = min(max_frames, total_frames)
    indices  = np.linspace(0, total_frames - 1, n_sample, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


# ==============================================================================
# QUALITY SCORING
# ==============================================================================

def compute_sharpness(bgr_frame: np.ndarray) -> float:
    """
    Laplacian variance sharpness score.

    Returns a float in [0, ∞). Higher = sharper.
    Typical thresholds: <100 = blurry, >300 = sharp, >500 = very sharp.
    """
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def filter_frames_by_quality(
    frames      : List[np.ndarray],
    min_sharpness: float = 100.0,
    top_n       : int   = 20,
) -> List[Tuple[Image.Image, float]]:
    """
    Filter a list of BGR frames by Laplacian sharpness and return the top-N.

    Args:
        frames        : List of BGR numpy arrays from extract_frames().
        min_sharpness : Minimum sharpness score to keep a frame.
        top_n         : Maximum number of frames to return (best first).

    Returns:
        List of (PIL.Image [RGB], sharpness_score) tuples,
        sorted by sharpness descending.
    """
    scored: List[Tuple[Image.Image, float]] = []

    for frame in frames:
        score = compute_sharpness(frame)
        if score >= min_sharpness:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            scored.append((pil, score))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_n]


# ==============================================================================
# STREAMLIT-FRIENDLY ENTRY POINT
# ==============================================================================

def pil_frames_from_upload(
    video_bytes  : bytes,
    max_frames   : int   = 30,
    min_sharpness: float = 100.0,
    top_n        : int   = 20,
) -> Tuple[List[Tuple[Image.Image, float]], int]:
    """
    Full pipeline for a Streamlit `st.file_uploader` result.

    Saves uploaded bytes to a temp file, extracts frames, filters by
    sharpness, and cleans up the temp file before returning.

    Args:
        video_bytes   : Raw bytes from `uploaded_file.read()`.
        max_frames    : How many frames to evenly sample from the video.
        min_sharpness : Minimum Laplacian score to accept a frame.
        top_n         : Maximum number of frames to return after filtering.

    Returns:
        (good_frames, total_extracted) where:
            good_frames     — list of (PIL.Image, sharpness) sorted best-first
            total_extracted — number of raw frames sampled before quality filter
    """
    # Write to a named temp file (OpenCV needs a real file path)
    suffix = ".mp4"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        raw_frames      = extract_frames(tmp_path, max_frames=max_frames)
        total_extracted = len(raw_frames)
        good_frames     = filter_frames_by_quality(
            raw_frames,
            min_sharpness=min_sharpness,
            top_n=top_n,
        )
    finally:
        os.unlink(tmp_path)   # Always clean up

    return good_frames, total_extracted


# ==============================================================================
# QUICK TEST (run directly: python video_utils.py <path_to_video>)
# ==============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python video_utils.py <video_path>")
        sys.exit(1)

    path = sys.argv[1]
    print(f"[video_utils] Processing: {path}")

    frames, total = pil_frames_from_upload(
        open(path, "rb").read(),
        max_frames=30,
        min_sharpness=100.0,
        top_n=20,
    )

    print(f"[video_utils] Total sampled : {total}")
    print(f"[video_utils] Quality passed: {len(frames)}")
    for i, (img, score) in enumerate(frames[:5]):
        print(f"  Frame {i+1}: {img.size} | sharpness={score:.1f}")
