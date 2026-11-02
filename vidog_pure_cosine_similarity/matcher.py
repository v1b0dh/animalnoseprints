import os
import json
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class MatchCandidate:
    dog_name: str
    cosine_score: float
    hybrid_score: float
    best_matching_frame: str
    status: str  # 'MATCH', 'LOW_CONFIDENCE', 'UNKNOWN'
    gender: Optional[str] = None

class PureCosineMatcher:
    def __init__(
        self,
        gallery_dir: str = 'data/gallery_v2',
        open_set_threshold: float = 0.68,
        match_acceptance_threshold: float = 0.78,
        nose_weight: float = 0.85,
        face_weight: float = 0.15
    ):
        self.gallery_dir = gallery_dir
        self.open_set_threshold = open_set_threshold
        self.match_acceptance_threshold = match_acceptance_threshold
        self.nose_weight = nose_weight
        self.face_weight = face_weight

    def _get_dog_gender(self, dog: str) -> Optional[str]:
        meta_path = os.path.join(self.gallery_dir, dog, 'meta.json')
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as f:
                    meta = json.load(f)
                return meta.get('gender')
            except Exception:
                pass
        return None

    def match(
        self,
        query_nose_emb: np.ndarray,
        query_face_emb: Optional[np.ndarray] = None,
        gender: Optional[str] = None
    ) -> Tuple[Optional[MatchCandidate], List[MatchCandidate]]:
        if not os.path.exists(self.gallery_dir):
            return None, []

        dogs = [d for d in sorted(os.listdir(self.gallery_dir))
                if os.path.isdir(os.path.join(self.gallery_dir, d)) and not d.startswith('_')]

        if not dogs:
            return None, []

        # Gender filtering
        if gender is not None:
            filtered_dogs = []
            for dog in dogs:
                dog_gender = self._get_dog_gender(dog)
                if dog_gender is None or dog_gender.lower() == gender.lower():
                    filtered_dogs.append(dog)
            if filtered_dogs:
                dogs = filtered_dogs

        q_nose = query_nose_emb.flatten().astype(np.float32)
        q_nose_norm = np.linalg.norm(q_nose)
        if q_nose_norm > 1e-6:
            q_nose = q_nose / q_nose_norm

        has_face = query_face_emb is not None
        if has_face:
            q_face = query_face_emb.flatten().astype(np.float32)
            q_face_norm = np.linalg.norm(q_face)
            if q_face_norm > 1e-6:
                q_face = q_face / q_face_norm

        # Compute max cosine similarity per dog across all stored frames
        dog_cosine_scores: Dict[str, float] = {}
        dog_best_frames: Dict[str, str] = {}

        for dog in dogs:
            dog_path = os.path.join(self.gallery_dir, dog)
            npy_files = [f for f in sorted(os.listdir(dog_path))
                         if f.endswith('.npy') and not f.startswith('face_') and not f.endswith('_student.npy')]
            
            max_sim = -1.0
            best_frame = ''

            for npy_f in npy_files:
                base_idx = npy_f[:-4]
                nose_p = os.path.join(dog_path, npy_f)
                face_p = os.path.join(dog_path, f'face_{base_idx}.npy')

                try:
                    g_nose = np.load(nose_p).flatten().astype(np.float32)
                    g_norm = np.linalg.norm(g_nose)
                    if g_norm > 1e-6:
                        g_nose = g_nose / g_norm

                    nose_sim = float(np.dot(q_nose, g_nose))

                    if has_face and os.path.exists(face_p):
                        g_face = np.load(face_p).flatten().astype(np.float32)
                        f_norm = np.linalg.norm(g_face)
                        if f_norm > 1e-6:
                            g_face = g_face / f_norm
                        face_sim = float(np.dot(q_face, g_face))
                        composite_sim = self.nose_weight * nose_sim + self.face_weight * face_sim
                    else:
                        composite_sim = nose_sim

                    if composite_sim > max_sim:
                        max_sim = composite_sim
                        best_frame = f'{base_idx}.jpg'

                except Exception as e:
                    print(f'[WARN] Error loading {nose_p}: {e}')

            dog_cosine_scores[dog] = max_sim
            dog_best_frames[dog] = best_frame

        max_overall_cosine = max(dog_cosine_scores.values()) if dog_cosine_scores else -1.0

        candidates: List[MatchCandidate] = []
        for dog in sorted(dogs, key=lambda d: dog_cosine_scores.get(d, -1.0), reverse=True):
            score = dog_cosine_scores.get(dog, 0.0)
            if score >= self.match_acceptance_threshold:
                status = 'MATCH'
            elif score >= self.open_set_threshold:
                status = 'LOW_CONFIDENCE'
            else:
                status = 'UNKNOWN'

            candidates.append(MatchCandidate(
                dog_name=dog,
                cosine_score=score,
                hybrid_score=score,
                best_matching_frame=dog_best_frames.get(dog, ''),
                status=status,
                gender=self._get_dog_gender(dog)
            ))

        if not candidates or candidates[0].status == 'UNKNOWN':
            return None, candidates

        return candidates[0], candidates
