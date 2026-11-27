import os
import json
import numpy as np
from typing import Dict, List, Optional
from sklearn.linear_model import LogisticRegression


class FastLinearClassifier:
    """
    Fast few-shot linear classifier head trained on L2-normalized nose embeddings.
    Persists weights safely as NumPy arrays + JSON (no pickle deserialization).
    """

    def __init__(self, model_dir: str = "data/gallery_v2/_model"):
        self.model_dir = model_dir
        self.weights_path = os.path.join(model_dir, "classifier_weights.npz")
        self.map_path = os.path.join(model_dir, "class_map.json")
        self.coef: Optional[np.ndarray] = None
        self.intercept: Optional[np.ndarray] = None
        self.class_names: List[str] = []
        self.load()

    def load(self) -> bool:
        if os.path.exists(self.weights_path) and os.path.exists(self.map_path):
            try:
                data = np.load(self.weights_path)
                self.coef = data["coef"].astype(np.float32)
                self.intercept = data["intercept"].astype(np.float32)
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.class_names = json.load(f)
                return True
            except Exception as e:
                print(f"[WARN] Failed to load classifier weights: {e}")
                self.coef = None
                self.intercept = None
                self.class_names = []
        return False

    def is_trained(self) -> bool:
        return (
            self.coef is not None
            and self.intercept is not None
            and len(self.class_names) >= 2
        )

    @staticmethod
    def _load_dog_nose_embeddings(dog_path: str) -> List[np.ndarray]:
        """Load nose embeddings from unified nose_vectors.npz (with legacy .npy fallback)."""
        vecs: List[np.ndarray] = []
        npz_path = os.path.join(dog_path, "nose_vectors.npz")
        if os.path.exists(npz_path):
            try:
                data = np.load(npz_path)
                mat = data.get("nose_vectors")
                if mat is not None and len(mat) > 0:
                    for row in mat:
                        v = row.flatten().astype(np.float32)
                        norm = np.linalg.norm(v)
                        if norm > 1e-6:
                            vecs.append(v / norm)
                    return vecs
            except Exception as e:
                print(f"[WARN] Error reading {npz_path}: {e}")

        # Legacy per-frame .npy fallback
        npy_files = [
            f for f in sorted(os.listdir(dog_path))
            if f.endswith(".npy")
            and not f.startswith(("face_", "body_"))
            and not f.endswith("_student.npy")
        ]
        for npy_file in npy_files:
            emb_path = os.path.join(dog_path, npy_file)
            try:
                emb = np.load(emb_path).flatten().astype(np.float32)
                norm = np.linalg.norm(emb)
                if norm > 1e-6:
                    vecs.append(emb / norm)
            except Exception as e:
                print(f"[WARN] Error reading {emb_path}: {e}")
        return vecs

    def train_from_gallery(self, gallery_dir: str = "data/gallery_v2") -> Dict[str, any]:
        if not os.path.exists(gallery_dir):
            return {"success": False, "message": "Gallery directory does not exist"}

        dogs = [
            d for d in sorted(os.listdir(gallery_dir))
            if os.path.isdir(os.path.join(gallery_dir, d)) and not d.startswith("_")
        ]

        X_list: List[np.ndarray] = []
        y_list: List[int] = []
        class_names: List[str] = []

        for dog_name in dogs:
            dog_path = os.path.join(gallery_dir, dog_name)
            embs = self._load_dog_nose_embeddings(dog_path)
            if not embs:
                continue
            class_idx = len(class_names)
            class_names.append(dog_name)
            for emb in embs:
                X_list.append(emb)
                y_list.append(class_idx)

        if len(class_names) < 2 or len(X_list) < 2:
            self.coef = None
            self.intercept = None
            self.class_names = class_names
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.map_path, "w", encoding="utf-8") as f:
                json.dump(self.class_names, f, indent=2)
            if os.path.exists(self.weights_path):
                os.remove(self.weights_path)
            return {
                "success": True,
                "dog_count": len(class_names),
                "sample_count": len(X_list),
                "message": "Fewer than 2 dogs with nose vectors; linear classifier on standby (using Cosine Gate).",
            }

        X = np.vstack(X_list)
        y = np.array(y_list)

        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X, y)

        os.makedirs(self.model_dir, exist_ok=True)
        tmp_weights = os.path.join(self.model_dir, ".tmp_classifier_weights.npz")
        tmp_map = os.path.join(self.model_dir, ".tmp_class_map.json")
        np.savez_compressed(
            tmp_weights,
            coef=clf.coef_.astype(np.float32),
            intercept=clf.intercept_.astype(np.float32),
        )
        os.replace(tmp_weights, self.weights_path)

        with open(tmp_map, "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)
        os.replace(tmp_map, self.map_path)

        self.coef = clf.coef_.astype(np.float32)
        self.intercept = clf.intercept_.astype(np.float32)
        self.class_names = class_names

        return {
            "success": True,
            "dog_count": len(class_names),
            "sample_count": len(X),
            "message": f"Successfully fit Linear Classifier on {len(X)} nose samples across {len(class_names)} dogs.",
        }

    def predict_proba(self, query_emb: Optional[np.ndarray]) -> Dict[str, float]:
        if query_emb is None or not self.is_trained():
            return {}

        emb = query_emb.flatten().astype(np.float32)
        if emb.shape[0] != self.coef.shape[1]:
            return {}
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm

        try:
            logits = np.dot(self.coef, emb) + self.intercept
            if len(self.class_names) == 2 and logits.shape[0] == 1:
                p1 = float(1.0 / (1.0 + np.exp(-np.clip(logits[0], -50.0, 50.0))))
                probs = [1.0 - p1, p1]
            else:
                shifted = logits - np.max(logits)
                exp_l = np.exp(shifted)
                probs = (exp_l / np.sum(exp_l)).tolist()

            return {
                self.class_names[idx]: float(prob)
                for idx, prob in enumerate(probs)
                if idx < len(self.class_names)
            }
        except Exception as e:
            print(f"[ERROR] Classifier prediction error: {e}")
            return {}
