import os
import json
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
    gender: Optional[str] = None
    nose_score: Optional[float] = None
    body_score: Optional[float] = None
    rule_applied: str = "DEFAULT"


class HybridBiometricMatcher:
    """
    Two-Stage Hybrid Matcher with Dual-Modality (.npz) Support:
      Stage 1: Open-Set Cosine Gate (80% Nose + 20% Body when both present; 100% fallback)
      Stage 2: Few-Shot Linear Boundary Classifier Fusion (60% Cosine + 40% Classifier)
    """

    def __init__(
        self,
        gallery_dir: str = "data/gallery_v2",
        open_set_threshold: float = 0.68,
        match_acceptance_threshold: float = 0.78,
        body_acceptance_threshold: float = 0.82,
        nose_weight: float = 0.80,
        body_weight: float = 0.20,
        cosine_hybrid_weight: float = 0.60,
        classifier_hybrid_weight: float = 0.40,
    ):
        self.gallery_dir = gallery_dir
        self.open_set_threshold = open_set_threshold
        self.match_acceptance_threshold = match_acceptance_threshold
        self.body_acceptance_threshold = body_acceptance_threshold
        self.nose_weight = nose_weight
        self.body_weight = body_weight
        self.cosine_hybrid_weight = cosine_hybrid_weight
        self.classifier_hybrid_weight = classifier_hybrid_weight
        self.classifier = FastLinearClassifier(os.path.join(gallery_dir, "_model"))

    def reload_classifier(self):
        self.classifier.load()

    def _get_dog_gender(self, dog: str) -> Optional[str]:
        meta_path = os.path.join(self.gallery_dir, dog, "meta.json")
        if os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                return meta.get("gender")
            except Exception:
                pass
        return None

    @staticmethod
    def _normalize_vec(vec: np.ndarray) -> Optional[np.ndarray]:
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
        gender: Optional[str] = None,
    ) -> Tuple[Optional[MatchCandidate], List[MatchCandidate]]:
        if not os.path.exists(self.gallery_dir):
            return None, []

        dogs = [
            d for d in sorted(os.listdir(self.gallery_dir))
            if os.path.isdir(os.path.join(self.gallery_dir, d)) and not d.startswith("_")
        ]
        if not dogs:
            return None, []

        if gender is not None:
            filtered = [
                d for d in dogs
                if (g := self._get_dog_gender(d)) is None or g.lower() == gender.lower()
            ]
            if filtered:
                dogs = filtered

        q_nose = self._normalize_vec(query_nose_emb) if query_nose_emb is not None else None
        q_face = self._normalize_vec(query_face_emb) if query_face_emb is not None else None
        q_body = self._normalize_vec(query_body_emb) if query_body_emb is not None else None

        if q_nose is None and q_body is None:
            return None, []

        dog_cosine_scores: Dict[str, float] = {}
        dog_nose_scores: Dict[str, float] = {}
        dog_body_scores: Dict[str, float] = {}
        dog_best_frames: Dict[str, str] = {}
        dog_rules: Dict[str, str] = {}

        for dog in dogs:
            dog_path = os.path.join(self.gallery_dir, dog)
            all_files = sorted(os.listdir(dog_path))
            nose_npz_p = os.path.join(dog_path, "nose_vectors.npz")
            body_npz_p = os.path.join(dog_path, "body_vectors.npz")

            # --- 1. Nose Similarity ---
            best_nose_sim = -1.0
            best_nose_frame = ""
            if q_nose is not None:
                if os.path.exists(nose_npz_p):
                    try:
                        data = np.load(nose_npz_p)
                        nose_mat = data.get("nose_vectors")
                        face_mat = data.get("face_vectors")
                        if nose_mat is not None and len(nose_mat) > 0:
                            if nose_mat.shape[1] != q_nose.shape[0]:
                                print(f"[WARN] Nose dim mismatch for '{dog}': {nose_mat.shape[1]} vs {q_nose.shape[0]}")
                            else:
                                norms = np.linalg.norm(nose_mat, axis=1, keepdims=True)
                                norms[norms < 1e-6] = 1.0
                                sims = np.dot(nose_mat / norms, q_nose)

                                if q_face is not None and face_mat is not None and len(face_mat) == len(nose_mat):
                                    fnorms = np.linalg.norm(face_mat, axis=1, keepdims=True)
                                    fnorms[fnorms < 1e-6] = 1.0
                                    fsims = np.dot(face_mat / fnorms, q_face)
                                    sims = 0.85 * sims + 0.15 * fsims

                                best_idx = int(np.argmax(sims))
                                best_nose_sim = float(sims[best_idx])
                                best_nose_frame = f"frame_{best_idx+1:04d}"
                    except Exception as e:
                        print(f"[WARN] Error loading {nose_npz_p}: {e}")
                else:
                    nose_npy = [
                        f for f in all_files
                        if f.endswith(".npy") and not f.startswith(("face_", "body_")) and not f.endswith("_student.npy")
                    ]
                    for npy_f in nose_npy:
                        base_idx = npy_f[:-4]
                        try:
                            g_nose = self._normalize_vec(np.load(os.path.join(dog_path, npy_f)))
                            if g_nose is not None and g_nose.shape[0] == q_nose.shape[0]:
                                n_sim = float(np.dot(q_nose, g_nose))
                                face_p = os.path.join(dog_path, f"face_{base_idx}.npy")
                                if q_face is not None and os.path.exists(face_p):
                                    g_face = self._normalize_vec(np.load(face_p))
                                    if g_face is not None and g_face.shape[0] == q_face.shape[0]:
                                        n_sim = 0.85 * n_sim + 0.15 * float(np.dot(q_face, g_face))
                                if n_sim > best_nose_sim:
                                    best_nose_sim = n_sim
                                    best_nose_frame = f"{base_idx}.jpg"
                        except Exception as e:
                            print(f"[WARN] Error comparing {npy_f}: {e}")

            # --- 2. Body Similarity ---
            best_body_sim = -1.0
            best_body_frame = ""
            if q_body is not None and os.path.exists(body_npz_p):
                try:
                    data = np.load(body_npz_p)
                    body_mat = data.get("body_vectors")
                    if body_mat is not None and len(body_mat) > 0:
                        if body_mat.shape[1] != q_body.shape[0]:
                            print(f"[WARN] Body dim mismatch for '{dog}': {body_mat.shape[1]} vs {q_body.shape[0]}")
                        else:
                            bnorms = np.linalg.norm(body_mat, axis=1, keepdims=True)
                            bnorms[bnorms < 1e-6] = 1.0
                            bsims = np.dot(body_mat / bnorms, q_body)
                            best_bidx = int(np.argmax(bsims))
                            best_body_sim = float(bsims[best_bidx])
                            best_body_frame = f"body_frame_{best_bidx+1:04d}"
                except Exception as e:
                    print(f"[WARN] Error loading {body_npz_p}: {e}")

            # --- 3. Weighting Rules ---
            has_nose = best_nose_sim >= 0.0
            has_body = best_body_sim >= 0.0

            if has_nose and has_body:
                composite = (self.nose_weight * best_nose_sim) + (self.body_weight * best_body_sim)
                rule = "NOSE_80_BODY_20"
                chosen_frame = best_nose_frame or best_body_frame
            elif has_nose:
                composite = best_nose_sim
                rule = "NOSE_ONLY_100"
                chosen_frame = best_nose_frame
            elif has_body:
                composite = best_body_sim
                rule = "BODY_ONLY_100"
                chosen_frame = best_body_frame
            else:
                composite = -1.0
                rule = "NO_DATA"
                chosen_frame = ""

            dog_cosine_scores[dog] = composite
            dog_nose_scores[dog] = best_nose_sim
            dog_body_scores[dog] = best_body_sim
            dog_best_frames[dog] = chosen_frame
            dog_rules[dog] = rule

        max_overall_cosine = max(dog_cosine_scores.values()) if dog_cosine_scores else -1.0

        # Stage 1: Open-Set Cosine Gate
        if max_overall_cosine < self.open_set_threshold:
            candidates = [
                MatchCandidate(
                    dog_name=dog,
                    cosine_score=dog_cosine_scores[dog],
                    classifier_prob=0.0,
                    hybrid_score=dog_cosine_scores[dog],
                    best_matching_frame=dog_best_frames[dog],
                    status="UNKNOWN",
                    gender=self._get_dog_gender(dog),
                    nose_score=dog_nose_scores[dog] if dog_nose_scores[dog] >= 0 else None,
                    body_score=dog_body_scores[dog] if dog_body_scores[dog] >= 0 else None,
                    rule_applied=dog_rules[dog],
                )
                for dog in sorted(dogs, key=lambda d: dog_cosine_scores[d], reverse=True)
            ]
            return None, candidates

        # Stage 2: Fast Linear Classifier (only when nose query is present)
        clf_probs = self.classifier.predict_proba(q_nose) if q_nose is not None else {}
        use_clf = self.classifier.is_trained() and bool(clf_probs)

        candidates: List[MatchCandidate] = []
        for dog in dogs:
            c_score = dog_cosine_scores.get(dog, 0.0)
            rule = dog_rules.get(dog, "DEFAULT")
            p_score = clf_probs.get(dog, 0.0) if use_clf else c_score

            if use_clf and rule != "BODY_ONLY_100":
                h_score = (self.cosine_hybrid_weight * c_score) + (self.classifier_hybrid_weight * p_score)
            else:
                h_score = c_score

            acc_threshold = (
                self.body_acceptance_threshold
                if rule == "BODY_ONLY_100"
                else self.match_acceptance_threshold
            )

            if h_score >= acc_threshold:
                status = "MATCH"
            elif h_score >= self.open_set_threshold:
                status = "LOW_CONFIDENCE"
            else:
                status = "UNKNOWN"

            candidates.append(
                MatchCandidate(
                    dog_name=dog,
                    cosine_score=c_score,
                    classifier_prob=p_score if use_clf else -1.0,
                    hybrid_score=h_score,
                    best_matching_frame=dog_best_frames.get(dog, ""),
                    status=status,
                    gender=self._get_dog_gender(dog),
                    nose_score=dog_nose_scores[dog] if dog_nose_scores[dog] >= 0 else None,
                    body_score=dog_body_scores[dog] if dog_body_scores[dog] >= 0 else None,
                    rule_applied=rule,
                )
            )

        candidates.sort(key=lambda c: c.hybrid_score, reverse=True)
        top_match = candidates[0] if candidates and candidates[0].status == "MATCH" else None
        return top_match, candidates
