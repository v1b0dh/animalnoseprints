import os
import json
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.linear_model import LogisticRegression

class FastLinearClassifier:
    """
    Production-grade fast few-shot linear classifier head.
    Trains on 1024-d L2-normalized embeddings from registered dogs.
    """
    def __init__(self, model_dir: str = "data/gallery_v2/_model"):
        self.model_dir = model_dir
        self.model_path = os.path.join(model_dir, "classifier.pkl")
        self.map_path = os.path.join(model_dir, "class_map.json")
        self.model: Optional[LogisticRegression] = None
        self.class_names: List[str] = []
        self.load()

    def load(self) -> bool:
        if os.path.exists(self.model_path) and os.path.exists(self.map_path):
            try:
                with open(self.model_path, "rb") as f:
                    self.model = pickle.load(f)
                with open(self.map_path, "r", encoding="utf-8") as f:
                    self.class_names = json.load(f)
                return True
            except Exception as e:
                print(f"[WARN] Failed to load classifier: {e}")
                self.model = None
                self.class_names = []
        return False

    def is_trained(self) -> bool:
        return self.model is not None and len(self.class_names) >= 2

    def train_from_gallery(self, gallery_dir: str = "data/gallery_v2") -> Dict[str, any]:
        X_list = []
        y_list = []
        class_names = []

        if not os.path.exists(gallery_dir):
            return {"success": False, "message": "Gallery directory does not exist"}

        dogs = [d for d in sorted(os.listdir(gallery_dir)) 
                if os.path.isdir(os.path.join(gallery_dir, d)) and not d.startswith("_")]

        if len(dogs) < 2:
            self.model = None
            self.class_names = dogs
            os.makedirs(self.model_dir, exist_ok=True)
            with open(self.map_path, "w", encoding="utf-8") as f:
                json.dump(self.class_names, f, indent=2)
            if os.path.exists(self.model_path):
                os.remove(self.model_path)
            return {
                "success": True, 
                "dog_count": len(dogs), 
                "sample_count": 0,
                "message": "Fewer than 2 dogs in gallery; linear classifier requires >= 2 classes (fallback to Cosine Gate)."
            }

        for class_idx, dog_name in enumerate(dogs):
            dog_path = os.path.join(gallery_dir, dog_name)
            npy_files = [f for f in sorted(os.listdir(dog_path)) 
                         if f.endswith(".npy") and not f.startswith("face_") and not f.endswith("_student.npy")]
            
            for npy_file in npy_files:
                emb_path = os.path.join(dog_path, npy_file)
                try:
                    emb = np.load(emb_path).astype(np.float32)
                    norm = np.linalg.norm(emb)
                    if norm > 1e-6:
                        emb = emb / norm
                    X_list.append(emb)
                    y_list.append(class_idx)
                except Exception as e:
                    print(f"[WARN] Error reading {emb_path}: {e}")

            class_names.append(dog_name)

        if len(X_list) < 2 or len(set(y_list)) < 2:
            return {"success": False, "message": "Insufficient distinct embeddings across dogs to fit classifier."}

        X = np.vstack(X_list)
        y = np.array(y_list)

        clf = LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=300,
            multi_class="multinomial" if len(class_names) > 2 else "auto",
            class_weight="balanced",
            random_state=42
        )
        clf.fit(X, y)

        os.makedirs(self.model_dir, exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump(clf, f)
        with open(self.map_path, "w", encoding="utf-8") as f:
            json.dump(class_names, f, indent=2)

        self.model = clf
        self.class_names = class_names

        return {
            "success": True,
            "dog_count": len(class_names),
            "sample_count": len(X),
            "message": f"Successfully fit Linear Classifier on {len(X)} samples across {len(class_names)} dogs."
        }

    def predict_proba(self, query_emb: np.ndarray) -> Dict[str, float]:
        if not self.is_trained():
            return {}

        emb = query_emb.reshape(1, -1).astype(np.float32)
        norm = np.linalg.norm(emb)
        if norm > 1e-6:
            emb = emb / norm

        try:
            probs = self.model.predict_proba(emb)[0]
            result = {}
            for idx, prob in enumerate(probs):
                if idx < len(self.class_names):
                    result[self.class_names[idx]] = float(prob)
            return result
        except Exception as e:
            print(f"[ERROR] Classifier prediction error: {e}")
            return {}
