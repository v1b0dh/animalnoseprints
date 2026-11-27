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


@dataclass
class ExtractedBodyFrame:
    frame_idx: int
    timestamp_sec: float
    full_image: Image.Image
    sharpness: float
    is_valid: bool = True
    rejection_reason: str = ""


def compute_sharpness(img_gray: np.ndarray) -> float:
    return float(cv2.Laplacian(img_gray, cv2.CV_64F).var())


class VideoRegistrationEngine:
    """
    Single-video dual-stream extractor for vidog_train_model.
    From one uploaded video, automatically separates:
      1. Detected sharp nose crops -> nose_vectors.npz
      2. Sharp full-body frames   -> body_vectors.npz
    """

    def __init__(
        self,
        target_fps: float = 4.0,
        min_sharpness: float = 65.0,
        min_body_sharpness: float = 35.0,
        min_luminance: float = 30.0,
        max_luminance: float = 230.0,
        min_nose_confidence: float = 0.40,
        target_k_frames: int = 45,
    ):
        self.target_fps = target_fps
        self.min_sharpness = min_sharpness
        self.min_body_sharpness = min_body_sharpness
        self.min_luminance = min_luminance
        self.max_luminance = max_luminance
        self.min_nose_confidence = min_nose_confidence
        self.target_k_frames = target_k_frames

    def process_video_auto(
        self,
        video_path: str,
        nose_detector_fn: Callable[[Image.Image], any],
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Tuple[List[ExtractedFrame], List[ExtractedBodyFrame], dict]:
        """
        Reads a single video once and automatically extracts both:
          - Sharp frames where YOLO detected a dog nose
          - Sharp full-body frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [], [], {"error": f"Failed to open video file: {video_path}"}

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = total_frames / max(video_fps, 1.0)
        sample_interval = max(1, int(round(video_fps / self.target_fps)))

        nose_candidates: List[ExtractedFrame] = []
        body_candidates: List[ExtractedBodyFrame] = []
        sampled_count = 0
        rejected_nose = 0
        rejected_body = 0
        current_frame_idx = 0

        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break

            if current_frame_idx % sample_interval == 0:
                sampled_count += 1
                timestamp = current_frame_idx / video_fps
                if progress_callback:
                    pct = min(0.75, current_frame_idx / max(1, total_frames))
                    progress_callback(
                        pct,
                        f"Inspecting frame {current_frame_idx}/{total_frames} ({timestamp:.1f}s)...",
                    )

                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
                mean_lum = float(np.mean(gray))
                exposure_ok = self.min_luminance <= mean_lum <= self.max_luminance

                # 1. Body candidate check on full frame
                body_sharpness = compute_sharpness(gray)
                if exposure_ok and body_sharpness >= self.min_body_sharpness:
                    body_candidates.append(
                        ExtractedBodyFrame(
                            frame_idx=current_frame_idx,
                            timestamp_sec=timestamp,
                            full_image=frame_pil,
                            sharpness=body_sharpness,
                        )
                    )
                else:
                    rejected_body += 1

                # 2. Nose candidate check (only if actual YOLO model detected a nose)
                crop_res = nose_detector_fn(frame_pil)
                if (
                    crop_res.source == "model"
                    and crop_res.confidence >= self.min_nose_confidence
                ):
                    nose_np = np.array(crop_res.crop.convert("RGB"))
                    nose_gray = cv2.cvtColor(nose_np, cv2.COLOR_RGB2GRAY)
                    nose_sharpness = compute_sharpness(nose_gray)

                    if exposure_ok and nose_sharpness >= self.min_sharpness:
                        nose_candidates.append(
                            ExtractedFrame(
                                frame_idx=current_frame_idx,
                                timestamp_sec=timestamp,
                                full_image=frame_pil,
                                nose_crop=crop_res.crop,
                                bbox=crop_res.bbox,
                                sharpness=nose_sharpness,
                                detector_confidence=crop_res.confidence,
                                detector_source=crop_res.source,
                            )
                        )
                    else:
                        rejected_nose += 1

            current_frame_idx += 1

        cap.release()

        if progress_callback:
            progress_callback(
                0.85,
                f"Selecting diverse frames ({len(nose_candidates)} nose, {len(body_candidates)} body)...",
            )

        selected_nose = self._select_diverse(nose_candidates, k=self.target_k_frames)
        selected_body = self._select_diverse(body_candidates, k=self.target_k_frames)

        stats = {
            "total_frames_sampled": sampled_count,
            "valid_nose_frames": len(nose_candidates),
            "valid_body_frames": len(body_candidates),
            "rejected_nose_frames": rejected_nose,
            "rejected_body_frames": rejected_body,
            "selected_nose_frames": len(selected_nose),
            "selected_body_frames": len(selected_body),
            "duration_sec": duration_sec,
        }
        return selected_nose, selected_body, stats

    @staticmethod
    def _select_diverse(frames: list, k: int) -> list:
        if len(frames) <= k:
            return sorted(frames, key=lambda f: f.frame_idx)

        sorted_by_sharp = sorted(frames, key=lambda f: f.sharpness, reverse=True)
        selected = [sorted_by_sharp[0]]
        remaining = sorted_by_sharp[1:]
        min_frame_gap = max(2, len(frames) // (k * 2))

        for cand in remaining:
            if len(selected) >= k:
                break
            if all(abs(cand.frame_idx - s.frame_idx) >= min_frame_gap for s in selected):
                selected.append(cand)

        if len(selected) < k:
            for cand in remaining:
                if cand not in selected:
                    selected.append(cand)
                if len(selected) >= k:
                    break

        return sorted(selected, key=lambda f: f.frame_idx)
