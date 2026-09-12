import os
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

@dataclass
class ExtractedFrame:
    frame_idx: int
    timestamp_sec: float
    full_image: Image.Image
    nose_crop: Image.Image
    bbox: Tuple[int, int, int, int]
    sharpness: float
    detector_confidence: float
    detector_source: str
    is_valid: bool = True
    rejection_reason: str = ""

def compute_sharpness(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

class VideoRegistrationEngine:
    """
    Handles video ingestion, frame extraction, quality checks,
    nose cropping, and greedy diversity filtering.
    """
    def __init__(
        self,
        target_fps: float = 4.0,
        min_sharpness: float = 75.0,
        min_luminance: float = 30.0,
        max_luminance: float = 230.0,
        target_k_frames: int = 10
    ):
        self.target_fps = target_fps
        self.min_sharpness = min_sharpness
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.target_k_frames = target_k_frames

    def process_video(
        self,
        video_path: str,
        nose_detector_fn: Callable[[Image.Image], any],
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[List[ExtractedFrame], dict]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], {"error": f"Failed to open video file: {video_path}"}

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / max(video_fps, 1.0)
        
        sample_interval = max(1, int(round(video_fps / self.target_fps)))
        
        raw_candidates: List[ExtractedFrame] = []
        rejected_count = 0
        current_frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if current_frame_idx % sample_interval == 0:
                timestamp = current_frame_idx / video_fps
                if progress_callback:
                    pct = min(0.65, current_frame_idx / max(1, total_frames))
                    progress_callback(pct, f"Inspecting frame {current_frame_idx}/{total_frames} ({timestamp:.1f}s)...")

                # Convert to RGB & PIL
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                mean_lum = float(np.mean(gray))

                # Step 1: Run Nose Detection
                crop_res = nose_detector_fn(frame_pil)
                nose_pil = crop_res.crop
                bbox = crop_res.bbox
                conf = crop_res.confidence
                source = crop_res.source

                # Step 2: Quality score on the nose crop specifically
                nose_np = np.array(nose_pil.convert("RGB"))
                nose_gray = cv2.cvtColor(nose_np, cv2.COLOR_RGB2GRAY)
                sharpness = compute_sharpness(nose_gray)

                # Step 3: Thresholding checks
                is_valid = True
                rejection_reason = ""

                if sharpness < self.min_sharpness:
                    is_valid = False
                    rejection_reason = f"Low sharpness ({sharpness:.1f} < {self.min_sharpness})"
                    rejected_count += 1
                elif mean_lum < self.min_luminance:
                    is_valid = False
                    rejection_reason = f"Underexposed (lum: {mean_lum:.1f})"
                    rejected_count += 1
                elif mean_lum > self.max_luminance:
                    is_valid = False
                    rejection_reason = f"Overexposed (lum: {mean_lum:.1f})"
                    rejected_count += 1

                extracted = ExtractedFrame(
                    frame_idx=current_frame_idx,
                    timestamp_sec=timestamp,
                    full_image=frame_pil,
                    nose_crop=nose_pil,
                    bbox=bbox,
                    sharpness=sharpness,
                    detector_confidence=conf,
                    detector_source=source,
                    is_valid=is_valid,
                    rejection_reason=rejection_reason
                )
                raw_candidates.append(extracted)

            current_frame_idx += 1

        cap.release()

        # Step 4: Diversity & Selection of Top K
        valid_candidates = [f for f in raw_candidates if f.is_valid]
        
        if progress_callback:
            progress_callback(0.85, f"Filtering {len(valid_candidates)} sharp frames for angle diversity...")

        selected_frames = self._select_diverse_frames(valid_candidates, k=self.target_k_frames)

        stats = {
            "total_frames_sampled": len(raw_candidates),
            "valid_sharp_frames": len(valid_candidates),
            "rejected_frames": rejected_count,
            "selected_frames": len(selected_frames),
            "duration_sec": duration_sec
        }

        return selected_frames, stats

    def _select_diverse_frames(self, frames: List[ExtractedFrame], k: int) -> List[ExtractedFrame]:
        if len(frames) <= k:
            return sorted(frames, key=lambda f: f.sharpness, reverse=True)

        # Sort by sharpness descending
        sorted_by_sharp = sorted(frames, key=lambda f: f.sharpness, reverse=True)
        selected = [sorted_by_sharp[0]]
        
        # Greedy temporal / index spacing to capture varied movement angles
        remaining = sorted_by_sharp[1:]
        min_frame_gap = max(2, len(frames) // (k * 2))

        for cand in remaining:
            if len(selected) >= k:
                break
            # Ensure cand is spaced out from already selected frames
            if all(abs(cand.frame_idx - s.frame_idx) >= min_frame_gap for s in selected):
                selected.append(cand)

        # If still need more to reach k, fill with highest sharpness
        if len(selected) < k:
            for cand in remaining:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= k:
                    break

        return sorted(selected, key=lambda f: f.frame_idx)
