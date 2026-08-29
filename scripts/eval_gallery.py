"""
eval_gallery.py
===============
Leave-one-out evaluation of the dog-nose folder gallery at
data/gallery/.

For every stored embedding, treat it as a query and rank all OTHER
embeddings in the gallery by cosine similarity. A correct match means
the top-1 candidate (per-dog, ensemble max-cosine) belongs to the same
dog.

Metrics reported:
  - Rank-1 accuracy (single-embedding query, per-dog ensemble max-cosine)
  - Rank-5 accuracy
  - Mean Average Precision (mAP)
  - EER (Equal Error Rate) on genuine vs impostor similarity distributions
  - Genuine / impostor similarity stats (mu +/- sigma)

Usage:
    python scripts/eval_gallery.py
    python scripts/eval_gallery.py --gallery data/gallery
    python scripts/eval_gallery.py --reembed checkpoints/teacher_best.pth
"""

import argparse
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from det5 import DNNetV3, CLAHEPipeline


DEFAULT_GALLERY = "data/gallery"


# --------------------------------------------------------------------------- #
# Gallery loading
# --------------------------------------------------------------------------- #

def load_gallery(gallery_dir: str):
    """
    Walk every dog/<idx>.npy and build a {dog_name: [emb_array, ...]} dict
    plus a parallel {dog_name: [(idx, jpg_path), ...]} for re-embedding.
    """
    if not os.path.isdir(gallery_dir):
        os.makedirs(gallery_dir, exist_ok=True)
        print(f"[INFO] Created empty gallery at {gallery_dir}")

    emb: dict = defaultdict(list)
    jpgs: dict = defaultdict(list)
    for dog in sorted(os.listdir(gallery_dir)):
        d = os.path.join(gallery_dir, dog)
        if not os.path.isdir(d):
            continue
        for fn in sorted(os.listdir(d)):
            if not (fn.endswith(".npy") and not fn.endswith("_student.npy")):
                continue
            idx = fn[:4]
            try:
                arr = np.load(os.path.join(d, fn))
            except Exception:
                continue
            emb[dog].append(arr.astype(np.float32))
            jpgs[dog].append((idx, os.path.join(d, f"{idx}.jpg")))

    # Keep only dogs with >= 2 embeddings (LOO needs at least 2)
    return {k: np.stack(v) for k, v in emb.items() if len(v) >= 2}, jpgs


def reembed_gallery(jpgs: dict, checkpoint: str, device: str):
    model = DNNetV3(pretrained=True, use_head=True)
    if os.path.exists(checkpoint):
        ckpt = torch.load(checkpoint, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(sd)
            print(f"  [OK] Loaded checkpoint: {checkpoint}")
        except Exception as exc:
            print(f"  [WARN] Checkpoint mismatch ({exc}); using headless backbone.")
            model = DNNetV3(pretrained=True, use_head=False)
    else:
        print(f"  [WARN] No checkpoint at {checkpoint}; using ImageNet-pretrained backbone.")
        model = DNNetV3(pretrained=True, use_head=False)

    model = model.to(device).eval()
    pipeline = CLAHEPipeline(image_size=224)

    out = {}
    for dog, items in jpgs.items():
        vecs = []
        for _idx, jpg in items:
            if not jpg or not os.path.exists(jpg):
                continue
            img = Image.open(jpg).convert("RGB")
            t, _ = pipeline(img)
            t = t.unsqueeze(0).to(device)
            with torch.no_grad():
                v = model(t).squeeze(0).cpu().numpy().astype(np.float32)
            vecs.append(v)
        if len(vecs) >= 2:
            out[dog] = np.stack(vecs)
    return out


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

def per_dog_max_cosine(query_vec, gallery, exclude_name=None):
    scores = []
    for name, vecs in gallery.items():
        if name == exclude_name:
            continue
        sims = vecs @ query_vec
        scores.append((name, float(sims.max())))
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def mean_average_precision(gallery):
    aps = []
    for true_name, q_vecs in gallery.items():
        others = {n: v for n, v in gallery.items() if n != true_name}
        for q in q_vecs:
            ranked = per_dog_max_cosine(q, others)
            hits = 0
            ap = 0.0
            for k, (name, _s) in enumerate(ranked, start=1):
                if name == true_name:
                    hits += 1
                    ap += hits / k
            total_relevant = sum(1 for _ in ranked)
            if total_relevant > 0:
                aps.append(ap / total_relevant)
    return float(np.mean(aps)) if aps else 0.0


def rank_k_accuracy(gallery, k: int = 1) -> float:
    correct = 0
    total = 0
    for true_name, q_vecs in gallery.items():
        for q in q_vecs:
            ranked = per_dog_max_cosine(q, gallery, exclude_name=true_name)
            top_k_names = {n for n, _ in ranked[:k]}
            if true_name in top_k_names:
                correct += 1
            total += 1
    return correct / total if total else 0.0


def eer_and_similarity_stats(gallery):
    names = list(gallery.keys())
    genuine, impostor = [], []
    for i, ni in enumerate(names):
        vi = gallery[ni]
        for a in range(len(vi)):
            for b in range(a + 1, len(vi)):
                genuine.append(float(vi[a] @ vi[b]))
        for nj in names[i + 1:]:
            vj = gallery[nj]
            sims = vi @ vj.T
            impostor.extend(sims.flatten().tolist())

    genuine = np.array(genuine)
    impostor = np.array(impostor)
    if len(genuine) == 0 or len(impostor) == 0:
        return None, genuine, impostor

    thresholds = np.linspace(-1.0, 1.0, 1001)
    far = np.array([(impostor >= t).mean() for t in thresholds])
    frr = np.array([(genuine  <  t).mean() for t in thresholds])
    idx = int(np.argmin(np.abs(far - frr)))
    eer = float((far[idx] + frr[idx]) / 2)
    return eer, genuine, impostor


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=DEFAULT_GALLERY)
    ap.add_argument("--reembed", default=None,
                    help="Path to a checkpoint. If given, re-embed stored JPEGs "
                         "instead of using the existing .npy files.")
    args = ap.parse_args()

    print("=" * 60)
    print(f"  DogID - Gallery Evaluation  ({args.gallery})")
    print("=" * 60)

    print(f"\n[1/3] Loading gallery from {args.gallery} ...")
    gallery, jpgs = load_gallery(args.gallery)
    all_dogs = sum(1 for d in os.listdir(args.gallery)
                   if os.path.isdir(os.path.join(args.gallery, d)))
    print(f"      dogs in gallery folder: {all_dogs}")
    print(f"      dogs with >=2 embeddings (eval-eligible): {len(gallery)}")
    if all_dogs == 0:
        print("\n[!] No dogs found.  Register some dogs first.")
        sys.exit(0)

    if args.reembed:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"\n[2/3] Re-embedding stored JPEGs on {device} ...")
        gallery = reembed_gallery(jpgs, args.reembed, device)
    else:
        print("\n[2/3] Using existing stored embeddings (no re-embed).")

    if len(gallery) < 2:
        print("\n[!] Need at least 2 dogs with >=2 embeddings each to evaluate.")
        sys.exit(0)

    total_embs = sum(len(v) for v in gallery.values())
    print(f"      eval pool: {len(gallery)} dogs / {total_embs} embeddings")

    print("\n[3/3] Computing metrics ...\n")
    r1 = rank_k_accuracy(gallery, k=1)
    r5 = rank_k_accuracy(gallery, k=5)
    m_ap = mean_average_precision(gallery)
    eer, gen, imp = eer_and_similarity_stats(gallery)

    print("-" * 60)
    print(f"  Rank-1 accuracy : {r1 * 100:6.2f}%   (per-dog ensemble max-cosine)")
    print(f"  Rank-5 accuracy : {r5 * 100:6.2f}%")
    print(f"  mAP             : {m_ap * 100:6.2f}%")
    if eer is not None:
        print(f"  EER             : {eer * 100:6.2f}%   (verification threshold)")
        print(f"  Genuine sim     : mu={gen.mean():+.3f}  sigma={gen.std():.3f}  (n={len(gen)})")
        print(f"  Impostor sim    : mu={imp.mean():+.3f}  sigma={imp.std():.3f}  (n={len(imp)})")
    print("-" * 60)

    if r1 == 0.0 and len(gallery) > 0:
        print("\n  [note] Rank-1 = 0% usually means the gallery is too small or")
        print("         all dogs have very few embeddings.  Register more dogs")
        print("         with multiple nose photos each to get a meaningful number.")


if __name__ == "__main__":
    main()
