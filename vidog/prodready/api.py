"""
DogID API for production handoff.
Provides a simple Python interface for registering dogs from video and identifying from photos.
"""

import os
import sys
import json
import time
import tempfile
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from PIL import Image

# Ensure vidog package is importable
VIDOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, VIDOG_DIR)

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector
from video_engine import VideoRegistrationEngine
from classifier_head import FastLinearClassifier
from matcher import HybridBiometricMatcher


class DogIDApi:
    """
    Production-ready DogID API.
    Handles dog registration from video and identification from single photos.
    """

    def __init__(self, gallery_dir: Optional[str] = None, checkpoint_dir: Optional[str] = None):
        """
        Initialize DogID API.

        Args:
            gallery_dir: Path to gallery directory (default: vidog/data/gallery_v2)
            checkpoint_dir: Path to checkpoint directory (default: vidog/checkpoints)
        """
        self.vidog_dir = VIDOG_DIR
        self.gallery_dir = gallery_dir or os.path.join(VIDOG_DIR, "data", "gallery_v2")
        self.checkpoint_dir = checkpoint_dir or os.path.join(VIDOG_DIR, "checkpoints")
        self.nose_detector_path = os.path.join(self.checkpoint_dir, "nose_detector.onnx")

        # Initialize ML components
        self.pipeline = CLAHEPipeline(image_size=224)
        self.model = DNNetV3(pretrained=True, use_head=True)
        self.model.eval()
        self.detector = NoseDetector(weights_path=self.nose_detector_path)
        self.classifier = FastLinearClassifier(os.path.join(self.gallery_dir, "_model"))
        self.matcher = HybridBiometricMatcher(gallery_dir=self.gallery_dir)

    def _extract_embeddings(self, img_pil: Image.Image, nose_pil: Image.Image) -> Tuple[np.ndarray, np.ndarray]:
        """Extract nose and face embeddings from images."""
        t_nose, _ = self.pipeline(nose_pil)
        t_face, _ = self.pipeline(img_pil)
        import torch
        with torch.no_grad():
            emb_nose = self.model(t_nose.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
            emb_face = self.model(t_face.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
        # L2 normalize
        nose_norm = np.linalg.norm(emb_nose)
        if nose_norm > 1e-6:
            emb_nose = emb_nose / nose_norm
        face_norm = np.linalg.norm(emb_face)
        if face_norm > 1e-6:
            emb_face = emb_face / face_norm
        return emb_nose, emb_face

    def register_dog(
        self,
        video_path: str,
        dog_name: str,
        gender: str = "Unknown",
        breed: str = "",
        age: float = 2.0,
        color: str = ""
    ) -> Dict[str, Any]:
        """
        Register a new dog from a video file.

        Args:
            video_path: Path to registration video (MP4/MOV/AVI)
            dog_name: Name of the dog
            gender: Gender (Female/Male/Unknown)
            breed: Breed description
            age: Age in years
            color: Color/markings description

        Returns:
            Dict with success status and message
        """
        if not os.path.exists(video_path):
            return {"success": False, "message": f"Video file not found: {video_path}"}

        if not dog_name.strip():
            return {"success": False, "message": "Dog name is required"}

        # Process video
        tfile = None
        try:
            # Save temporary video file for OpenCV
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            with open(video_path, "rb") as f:
                tfile.write(f.read())
            tfile.flush()
            tfile.close()

            # Registration engine
            engine = VideoRegistrationEngine(
                target_fps=4.0,
                min_sharpness=70.0,
                target_k_frames=10
            )

            selected_frames, stats = engine.process_video(
                tfile.name,
                nose_detector_fn=self.detector.detect_and_crop,
                progress_callback=None  # No progress callback in API
            )

            os.unlink(tfile.name)
            tfile = None

            if not selected_frames:
                return {"success": False, "message": "No sharp frames with detectable noses could be extracted"}

            # Save gallery data
            dog_dir = os.path.join(self.gallery_dir, dog_name.replace(" ", "_"))
            os.makedirs(dog_dir, exist_ok=True)

            sharpness_list = []
            for idx, frame_obj in enumerate(selected_frames, start=1):
                prefix = f"{idx:04d}"
                emb_nose, emb_face = self._extract_embeddings(frame_obj.full_image, frame_obj.nose_crop)

                # Save embeddings and thumbnails
                np.save(os.path.join(dog_dir, f"{prefix}.npy"), emb_nose)
                np.save(os.path.join(dog_dir, f"face_{prefix}.npy"), emb_face)
                frame_obj.nose_crop.save(os.path.join(dog_dir, f"{prefix}.jpg"), "JPEG", quality=95)
                frame_obj.full_image.save(os.path.join(dog_dir, f"original_{prefix}.jpg"), "JPEG", quality=90)

                # Per-frame metadata
                meta_frame = {
                    "frame_idx": frame_obj.frame_idx,
                    "timestamp_sec": frame_obj.timestamp_sec,
                    "sharpness": frame_obj.sharpness,
                    "detector_confidence": frame_obj.detector_confidence,
                    "detector_source": frame_obj.detector_source,
                    "bbox": frame_obj.bbox
                }
                with open(os.path.join(dog_dir, f"{prefix}.json"), "w") as f:
                    json.dump(meta_frame, f, indent=2)

                sharpness_list.append(frame_obj.sharpness)

            # Save master metadata
            master_meta = {
                "dog_name": dog_name,
                "gender": gender,
                "breed": breed,
                "age_years": age,
                "color": color,
                "registered_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "frame_count": len(selected_frames),
                "avg_sharpness": float(np.mean(sharpness_list)),
                "video_duration_sec": stats.get("duration_sec", 0.0)
            }
            with open(os.path.join(dog_dir, "meta.json"), "w") as f:
                json.dump(master_meta, f, indent=2)

            # Retrain classifier
            clf_res = self.classifier.train_from_gallery(self.gallery_dir)
            self.classifier.load()  # Reload updated classifier

            return {
                "success": True,
                "message": f"Successfully registered {dog_name} with {len(selected_frames)} frames",
                "stats": stats
            }

        except Exception as e:
            if tfile and os.path.exists(tfile.name):
                os.unlink(tfile.name)
            return {"success": False, "message": f"Registration failed: {str(e)}"}

    def identify_dog(
        self,
        image_path: str,
        gender_filter: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Identify a dog from a single image.

        Args:
            image_path: Path to query image (JPG/PNG)
            gender_filter: Filter by gender (Female/Male/Unknown/None for all)

        Returns:
            Dict with identification results
        """
        if not os.path.exists(image_path):
            return {"success": False, "message": f"Image file not found: {image_path}"}

        try:
            # Load and preprocess image
            img = Image.open(image_path).convert("RGB")

            # Detect nose
            crop_res = self.detector.detect_and_crop(img)
            if crop_res.crop is None:
                return {"success": False, "message": "No nose detected in image"}

            # Extract embeddings
            emb_nose, emb_face = self._extract_embeddings(img, crop_res.crop)

            # Perform matching
            top_match, all_candidates = self.matcher.match(
                query_nose_emb=emb_nose,
                query_face_emb=emb_face,
                gender=gender_filter
            )

            # Format results
            if top_match:
                result = {
                    "success": True,
                    "match": {
                        "dog_name": top_match.dog_name,
                        "gender": top_match.gender,
                        "hybrid_score": top_match.hybrid_score,
                        "cosine_score": top_match.cosine_score,
                        "classifier_prob": top_match.classifier_prob if top_match.classifier_prob >= 0 else None,
                        "status": top_match.status,
                        "best_matching_frame": top_match.best_matching_frame
                    },
                    "candidates": [
                        {
                            "dog_name": c.dog_name,
                            "gender": c.gender,
                            "hybrid_score": c.hybrid_score,
                            "cosine_score": c.cosine_score,
                            "classifier_prob": c.classifier_prob if c.classifier_prob >= 0 else None,
                            "status": c.status,
                            "best_matching_frame": c.best_matching_frame
                        }
                        for c in all_candidates[:5]  # Top 5 candidates
                    ]
                }
            else:
                result = {
                    "success": True,
                    "match": None,
                    "candidates": [
                        {
                            "dog_name": c.dog_name,
                            "gender": c.gender,
                            "hybrid_score": c.hybrid_score,
                            "cosine_score": c.cosine_score,
                            "classifier_prob": c.classifier_prob if c.classifier_prob >= 0 else None,
                            "status": c.status,
                            "best_matching_frame": c.best_matching_frame
                        }
                        for c in all_candidates[:5]
                    ],
                    "message": "No match found above acceptance threshold"
                }

            return result

        except Exception as e:
            return {"success": False, "message": f"Identification failed: {str(e)}"}

    def get_gallery_info(self) -> Dict[str, Any]:
        """Get information about registered dogs in the gallery."""
        if not os.path.exists(self.gallery_dir):
            return {"dogs": [], "count": 0}

        dogs = []
        for dog_name in sorted(os.listdir(self.gallery_dir)):
            dog_dir = os.path.join(self.gallery_dir, dog_name)
            if os.path.isdir(dog_dir) and not dog_name.startswith("_"):
                meta_path = os.path.join(dog_dir, "meta.json")
                meta = {}
                if os.path.exists(meta_path):
                    try:
                        with open(meta_path, "r") as f:
                            meta = json.load(f)
                    except Exception:
                        pass
                dogs.append({
                    "dog_name": meta.get("dog_name", dog_name.replace("_", " ")),
                    "gender": meta.get("gender", "Unknown"),
                    "breed": meta.get("breed", ""),
                    "age_years": meta.get("age_years", 0.0),
                    "color": meta.get("color", ""),
                    "frame_count": meta.get("frame_count", 0),
                    "registered_at": meta.get("registered_at", "")
                })

        return {
            "dogs": dogs,
            "count": len(dogs)
        }


# Convenience function for simple usage
def create_api(gallery_dir: Optional[str] = None, checkpoint_dir: Optional[str] = None) -> DogIDApi:
    """Create and return a DogIDApi instance."""
    return DogIDApi(gallery_dir=gallery_dir, checkpoint_dir=checkpoint_dir)