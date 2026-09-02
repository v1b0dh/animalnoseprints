import os
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from classifier_head import FastLinearClassifier

@dataclass
class MatchCandidate:
    dog_name: str
    cosine_score: float
    classifier_prob: float
    hybrid_score: float
    best_matching_frame: str
    status: str  # 'MATCH', 'LOW_CONFIDENCE', 'UNKNOWN'

class HybridBiometricMatcher:
    """
    Two-Stage Hybrid Matcher:
      Stage 1: Open-Set Gate (Cosine Similarity over 85% Nose + 15% Face)
      Stage 2: Few-Shot Linear Boundary Classifier
    """
    def __init__(
        self,
        gallery_dir: str = "data/gallery_v2",
        open_set_threshold: float = 0.68,
        match_acceptance_threshold: float = 0.78,
        nose_weight: float = 0.85,
        face_weight: float = 0.15,
        cosine_hybrid_weight: float = 0.60,
        classifier_hybrid_weight: float = 0.40
    ):
        self.gallery_dir = gallery_dir
        self.open_set_threshold = open_set_threshold
        self.match_acceptance_threshold = match_acceptance_threshold
        self.nose_weight = nose_weight
        self.face_weight = face_weight
        self.cosine_hybrid_weight = cosine_hybrid_weight
        self.classifier_hybrid_weight = classifier_hybrid_weight
        self.classifier = FastLinearClassifier(os.path.join(gallery_dir, "_model"))

    def reload_classifier(self):
        self.classifier.load()

    def match(
        self,
        query_nose_emb: np.ndarray,
        query_face_emb: Optional[np.ndarray] = None
    ) -> Tuple[Optional[MatchCandidate], List[MatchCandidate]]:
        if not os.path.exists(self.gallery_dir):
            return None, []

        dogs = [d for d in sorted(os.listdir(self.gallery_dir))
                if os.path.isdir(os.path.join(self.gallery_dir, d)) and not d.startswith("_")]

        if not dogs:
            return None, []

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

        # -- Step 1: Compute Ensemble Cosine Similarities across all Dogs --
        dog_cosine_scores: Dict[str, float] = {}
        dog_best_frames: Dict[str, str] = {}

        for dog in dogs:
            dog_path = os.path.join(self.gallery_dir, dog)
            npy_files = [f for f in sorted(os.listdir(dog_path))
                         if f.endswith(".npy") and not f.startswith("face_") and not f.endswith("_student.npy")]
            
            max_sim = -1.0
            best_frame = ""

            for npy_f in npy_files:
                base_idx = npy_f[:-4]
                nose_p = os.path.join(dog_path, npy_f)
                face_p = os.path.join(dog_path, f"face_{base_idx}.npy")

                try:
                    g_nose = np.load(nose_p).flatten().astype(np.float32)
                    g_nose_norm = np.linalg.norm(g_nose)
                    if g_nose_norm > 1e-6:
                        g_nose = g_nose / g_nose_norm

                    nose_sim = float(np.dot(q_nose, g_nose))

                    if has_face and os.path.exists(face_p):
                        g_face = np.load(face_p).flatten().astype(np.float32)
                        g_face_norm = np.linalg.norm(g_face)
                        if g_face_norm > 1e-6:
                            g_face = g_face / g_face_norm
                        face_sim = float(np.dot(q_face, g_face))
                        composite_sim = self.nose_weight * nose_sim + self.face_weight * face_sim
                    else:
                        composite_sim = nose_sim

                    if composite_sim > max_sim:
                        max_sim = composite_sim
                        best_frame = f"{base_idx}.jpg"

                except Exception as e:
                    print(f"[WARN] Error comparing against {nose_p}: {e}")

            dog_cosine_scores[dog] = max_sim
            dog_best_frames[dog] = best_frame

        max_overall_cosine = max(dog_cosine_scores.values()) if dog_cosine_scores else -1.0

        # -- Step 2: Open-Set Cosine Gate --
        # If max cosine is below gate threshold, reject immediately as Unknown Dog
        if max_overall_cosine < self.open_set_threshold:
            candidates = [
                MatchCandidate(
                    dog_name=dog,
                    cosine_score=dog_cosine_scores[dog],
                    classifier_prob=0.0,
                    hybrid_score=dog_cosine_scores[dog],
                    best_matching_frame=dog_best_frames[dog],
                    status="UNKNOWN"
                )
                for dog in sorted(dogs, key=lambda d: dog_cosine_scores[d], reverse=True)
            ]
            return None, candidates

        # -- Step 3: Run Fast Linear Classifier --
        clf_probs = self.classifier.predict_proba(q_nose)
        use_clf = self.classifier.is_trained() and bool(clf_probs)

        candidates: List[MatchCandidate] = []
        for dog in dogs:
            c_score = dog_cosine_scores.get(dog, 0.0)
            p_score = clf_probs.get(dog, 0.0) if use_clf else c_score

            if use_clf:
                h_score = (self.cosine_hybrid_weight * c_score) + (self.classifier_hybrid_weight * p_score)
            else:
                h_score = c_score

            if h_score >= self.match_acceptance_threshold:
                status = "MATCH"
            elif h_score >= self.open_set_threshold:
                status = "LOW_CONFIDENCE"
            else:
                status = "UNKNOWN"

            candidates.append(MatchCandidate(
                dog_name=dog,
                cosine_score=c_score,
                classifier_prob=p_score if use_clf else -1.0,
                hybrid_score=h_score,
                best_matching_frame=dog_best_frames.get(dog, ""),
                status=status
            ))

        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
        top_match = candidates[0] if candidates and candidates[0].status == "MATCH" else None

        return top_match, candidates
