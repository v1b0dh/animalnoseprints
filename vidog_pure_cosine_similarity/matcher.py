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
    nose_score: Optional[float] = None
    body_score: Optional[float] = None
    rule_applied: str = "DEFAULT"

class PureCosineMatcher:
    """
    Adaptive Biometric Matcher supporting Dual Modalities (Nose Biometrics + Full Body).
    
    Rules:
      1. Both Nose and Body available:
         Composite Score = (nose_weight * nose_score) + (body_weight * body_score) [Default: 0.80 Nose / 0.20 Body]
      2. Nose available, Body NOT available:
         Composite Score = 1.0 * nose_score
      3. Body available, Nose NOT available:
         Composite Score = 1.0 * body_score (with stricter threshold)
    """
    def __init__(
        self,
        gallery_dir: str = 'data/gallery_v2',
        open_set_threshold: float = 0.68,
        match_acceptance_threshold: float = 0.78,
        body_acceptance_threshold: float = 0.82,
        nose_weight: float = 0.80,
        body_weight: float = 0.20
    ):
        self.gallery_dir = gallery_dir
        self.open_set_threshold = open_set_threshold
        self.match_acceptance_threshold = match_acceptance_threshold
        self.body_acceptance_threshold = body_acceptance_threshold
        self.nose_weight = nose_weight
        self.body_weight = body_weight

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

    def _normalize_vec(self, vec: np.ndarray) -> Optional[np.ndarray]:
        v = vec.flatten().astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 1e-6:
            return v / norm
        return None

    def match(
        self,
        query_nose_emb: Optional[np.ndarray] = None,
        query_face_emb: Optional[np.ndarray] = None,
        query_body_emb: Optional[np.ndarray] = None,
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

        q_nose = self._normalize_vec(query_nose_emb) if query_nose_emb is not None else None
        q_face = self._normalize_vec(query_face_emb) if query_face_emb is not None else None
        q_body = self._normalize_vec(query_body_emb) if query_body_emb is not None else None

        has_query_nose = q_nose is not None
        has_query_body = q_body is not None

        if not has_query_nose and not has_query_body:
            return None, []

        dog_composite_scores: Dict[str, float] = {}
        dog_nose_scores: Dict[str, float] = {}
        dog_body_scores: Dict[str, float] = {}
        dog_best_frames: Dict[str, str] = {}
        dog_rules: Dict[str, str] = {}

        for dog in dogs:
            dog_path = os.path.join(self.gallery_dir, dog)
            all_files = sorted(os.listdir(dog_path))

            # Multi-vector bundle support
            nose_npz_p = os.path.join(dog_path, 'nose_vectors.npz')
            body_npz_p = os.path.join(dog_path, 'body_vectors.npz')

            # --- 1. Compute Best Nose Score ---
            best_nose_sim = -1.0
            best_nose_frame = ''
            if has_query_nose:
                if os.path.exists(nose_npz_p):
                    try:
                        data = np.load(nose_npz_p)
                        nose_mat = data.get('nose_vectors')  # (N, D)
                        face_mat = data.get('face_vectors')  # (N, D)
                        if nose_mat is not None and len(nose_mat) > 0:
                            # Normalize gallery nose vectors along axis 1
                            norms = np.linalg.norm(nose_mat, axis=1, keepdims=True)
                            norms[norms < 1e-6] = 1.0
                            norm_nose_mat = nose_mat / norms
                            sims = np.dot(norm_nose_mat, q_nose)  # (N,)

                            if q_face is not None and face_mat is not None and len(face_mat) == len(nose_mat):
                                fnorms = np.linalg.norm(face_mat, axis=1, keepdims=True)
                                fnorms[fnorms < 1e-6] = 1.0
                                norm_face_mat = face_mat / fnorms
                                fsims = np.dot(norm_face_mat, q_face)
                                sims = 0.85 * sims + 0.15 * fsims

                            best_idx = int(np.argmax(sims))
                            best_nose_sim = float(sims[best_idx])
                            best_nose_frame = f'frame_{best_idx+1:04d}'
                    except Exception as e:
                        print(f'[WARN] Error loading {nose_npz_p}: {e}')
                else:
                    # Legacy fallback to separate individual .npy files
                    nose_npy_files = [f for f in all_files if f.endswith('.npy') and not f.startswith(('face_', 'body_')) and not f.endswith('_student.npy')]
                    for npy_f in nose_npy_files:
                        base_idx = npy_f[:-4]
                        nose_p = os.path.join(dog_path, npy_f)
                        face_p = os.path.join(dog_path, f'face_{base_idx}.npy')
                        try:
                            g_nose = self._normalize_vec(np.load(nose_p))
                            if g_nose is not None:
                                n_sim = float(np.dot(q_nose, g_nose))
                                if q_face is not None and os.path.exists(face_p):
                                    g_face = self._normalize_vec(np.load(face_p))
                                    if g_face is not None:
                                        f_sim = float(np.dot(q_face, g_face))
                                        n_sim = 0.85 * n_sim + 0.15 * f_sim
                                if n_sim > best_nose_sim:
                                    best_nose_sim = n_sim
                                    best_nose_frame = f'{base_idx}.jpg'
                        except Exception as e:
                            print(f'[WARN] Error loading {nose_p}: {e}')

            # --- 2. Compute Best Body Score ---
            best_body_sim = -1.0
            best_body_frame = ''
            if has_query_body:
                if os.path.exists(body_npz_p):
                    try:
                        data = np.load(body_npz_p)
                        body_mat = data.get('body_vectors')  # (M, D)
                        if body_mat is not None and len(body_mat) > 0:
                            bnorms = np.linalg.norm(body_mat, axis=1, keepdims=True)
                            bnorms[bnorms < 1e-6] = 1.0
                            norm_body_mat = body_mat / bnorms
                            bsims = np.dot(norm_body_mat, q_body)
                            best_bidx = int(np.argmax(bsims))
                            best_body_sim = float(bsims[best_bidx])
                            best_body_frame = f'body_frame_{best_bidx+1:04d}'
                    except Exception as e:
                        print(f'[WARN] Error loading {body_npz_p}: {e}')
                else:
                    # Legacy fallback to separate individual body_*.npy files
                    body_npy_files = [f for f in all_files if f.startswith('body_') and f.endswith('.npy')]
                    for b_file in body_npy_files:
                        body_p = os.path.join(dog_path, b_file)
                        try:
                            g_body = self._normalize_vec(np.load(body_p))
                            if g_body is not None:
                                b_sim = float(np.dot(q_body, g_body))
                                if b_sim > best_body_sim:
                                    best_body_sim = b_sim
                                    best_body_frame = b_file.replace('.npy', '.jpg')
                        except Exception as e:
                            print(f'[WARN] Error loading {body_p}: {e}')

            # --- 3. Apply Decision Weighting Rules ---
            has_nose_match = best_nose_sim >= 0.0
            has_body_match = best_body_sim >= 0.0

            if has_nose_match and has_body_match:
                # Rule 1: Both Nose & Body available -> 80% Nose, 20% Body
                composite = (self.nose_weight * best_nose_sim) + (self.body_weight * best_body_sim)
                rule = "NOSE_80_BODY_20"
                chosen_frame = best_nose_frame or best_body_frame
            elif has_nose_match and not has_body_match:
                # Rule 2: Only Nose available -> 100% Nose
                composite = best_nose_sim
                rule = "NOSE_ONLY_100"
                chosen_frame = best_nose_frame
            elif not has_nose_match and has_body_match:
                # Rule 3: Only Body available -> 100% Body
                composite = best_body_sim
                rule = "BODY_ONLY_100"
                chosen_frame = best_body_frame
            else:
                composite = -1.0
                rule = "NO_DATA"
                chosen_frame = ''

            dog_composite_scores[dog] = composite
            dog_nose_scores[dog] = best_nose_sim
            dog_body_scores[dog] = best_body_sim
            dog_best_frames[dog] = chosen_frame
            dog_rules[dog] = rule

        candidates: List[MatchCandidate] = []
        for dog in sorted(dogs, key=lambda d: dog_composite_scores.get(d, -1.0), reverse=True):
            score = dog_composite_scores.get(dog, 0.0)
            rule = dog_rules.get(dog, 'DEFAULT')

            # Determine acceptance based on modality rule
            acc_threshold = self.body_acceptance_threshold if rule == "BODY_ONLY_100" else self.match_acceptance_threshold

            if score >= acc_threshold:
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
                gender=self._get_dog_gender(dog),
                nose_score=dog_nose_scores.get(dog) if dog_nose_scores.get(dog, -1.0) >= 0 else None,
                body_score=dog_body_scores.get(dog) if dog_body_scores.get(dog, -1.0) >= 0 else None,
                rule_applied=rule
            ))

        if not candidates or candidates[0].status == 'UNKNOWN':
            return None, candidates

        return candidates[0], candidates
