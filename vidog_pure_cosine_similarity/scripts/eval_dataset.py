"""
eval_dataset.py
===============
Evaluation script for dog datasets (e.g. bigdataset3000 / bigdataset300)
organized as:
    dataset_dir/
      dog_name_1/
        img1.jpg
        img2.jpg
        ...
      dog_name_2/
        ...

Evaluation protocol:
  - For each dog with >= 2 images:
      * Register all images except one (gallery set)
      * Use that held-out image for testing (probe/query)
  - Supports:
      1) --test-mode single: holds out one fixed image per dog (last image by default, or random with --seed).
      2) --test-mode leave-one-out: full round-robin LOO where every image takes turns as the held-out test image.
  - Matches queries against gallery dogs using per-dog ensemble max-cosine similarity
    (and supports average cosine similarity via --match-strategy avg).

Metrics reported (same metrics as eval_gallery.py):
  - Rank-1 accuracy
  - Rank-5 accuracy
  - Mean Average Precision (mAP)
  - Equal Error Rate (EER)
  - Genuine & Impostor similarity distribution statistics (mean +/- std)

Usage:
    python phodog/scripts/eval_dataset.py --dataset bigdataset3000
    python phodog/scripts/eval_dataset.py --dataset bigdataset3000 --detector checkpoints/nose_detector.onnx
    python phodog/scripts/eval_dataset.py --dataset bigdataset3000 --cache-dir data/cache_bigdataset3000
    python phodog/scripts/eval_dataset.py --dataset bigdataset3000 --max-dogs 50
"""

import argparse
import os
import sys
import time
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector


SUPPORTED_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


# --------------------------------------------------------------------------- #
# Dataset Discovery & Image Loading
# --------------------------------------------------------------------------- #

def normalize_identity(name: str) -> str:
    """
    Normalizes dog directory names to consolidate duplicates
    (e.g., 'master_data_adagio' and 'Adagio' -> 'adagio').
    """
    s = name.strip()
    if s.lower().startswith("master_data_"):
        s = s[len("master_data_"):]
    return s.strip().lower()


def discover_dataset(dataset_dir: str, deduplicate: bool = True) -> Dict[str, List[str]]:
    """
    Scans dataset_dir for dog folders containing images.
    If deduplicate=True, merges alias folders (e.g. 'Adagio' and 'master_data_adagio')
    into a single canonical dog identity.
    Returns:
        { dog_name: [img_path_1, img_path_2, ...], ... }
    Only includes identities with at least 2 valid images.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")

    dogs: Dict[str, List[str]] = defaultdict(list)
    subdirs = os.listdir(dataset_dir)
    # Sort with named dogs first, then numeric IDs
    named_dirs = [d for d in subdirs if not d.isdigit()]
    numeric_dirs = [d for d in subdirs if d.isdigit()]
    ordered_dirs = sorted(named_dirs) + sorted(numeric_dirs)

    for name in ordered_dirs:
        folder_path = os.path.join(dataset_dir, name)
        if not os.path.isdir(folder_path):
            continue

        images = []
        for fn in sorted(os.listdir(folder_path)):
            if fn.lower().endswith(SUPPORTED_EXTS):
                images.append(os.path.join(folder_path, fn))

        if not images:
            continue

        key = normalize_identity(name) if deduplicate else name
        dogs[key].extend(images)

    # Return only classes with at least 2 images
    return {k: v for k, v in dogs.items() if len(v) >= 2}


# --------------------------------------------------------------------------- #
# Feature Extraction with Caching
# --------------------------------------------------------------------------- #

def extract_and_cache_embeddings(
    dog_images: Dict[str, List[str]],
    model: torch.nn.Module,
    pipeline: CLAHEPipeline,
    detector: Optional[NoseDetector],
    device: str,
    cache_dir: Optional[str] = None,
    batch_size: int = 16
) -> Dict[str, np.ndarray]:
    """
    Extracts L2-normalized embeddings for all images.
    If cache_dir is specified, embeddings are saved/loaded to/from disk (.npy).
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)

    embeddings_by_dog: Dict[str, List[np.ndarray]] = defaultdict(list)
    total_images = sum(len(imgs) for imgs in dog_images.values())
    processed = 0
    t0 = time.time()

    print(f"      Total images to embed: {total_images} across {len(dog_images)} dogs")

    for dog_idx, (dog_name, img_paths) in enumerate(dog_images.items(), start=1):
        dog_cache_dir = os.path.join(cache_dir, dog_name) if cache_dir else None
        if dog_cache_dir:
            os.makedirs(dog_cache_dir, exist_ok=True)

        for p in img_paths:
            base_fn = os.path.splitext(os.path.basename(p))[0]
            orig_parent = os.path.basename(os.path.dirname(p))
            cache_file = os.path.join(dog_cache_dir, f"{base_fn}.npy") if dog_cache_dir else None
            orig_cache_file = os.path.join(cache_dir, orig_parent, f"{base_fn}.npy") if cache_dir else None

            # Load from cache if valid (check canonical or original parent cache dir)
            loaded_vec = None
            for cf in [cache_file, orig_cache_file]:
                if cf and os.path.exists(cf):
                    try:
                        loaded_vec = np.load(cf).astype(np.float32)
                        break
                    except Exception:
                        pass

            if loaded_vec is not None:
                embeddings_by_dog[dog_name].append(loaded_vec)
                processed += 1
                continue

            # Crop nose if detector is available
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:
                print(f"      [WARN] Could not load image {p}: {e}")
                continue

            if detector is not None and detector.has_model:
                try:
                    crop_res = detector.detect_and_crop(img)
                    crop_img = crop_res.crop
                except Exception:
                    crop_img = img
            else:
                crop_img = img

            # Preprocess & forward through embedding model
            tensor, _ = pipeline(crop_img)
            tensor = tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                feat = model(tensor).squeeze(0).cpu().numpy().astype(np.float32)

            # L2 normalize
            norm = np.linalg.norm(feat)
            if norm > 0:
                feat = feat / norm

            if cache_file:
                try:
                    np.save(cache_file, feat)
                except Exception:
                    pass

            embeddings_by_dog[dog_name].append(feat)
            processed += 1

            if processed % 10 == 0 or processed == total_images:
                elapsed = time.time() - t0
                fps = processed / elapsed if elapsed > 0 else 0
                print(f"      Embedded {processed}/{total_images} images ({fps:.1f} img/s) ...", flush=True)

    print(f"\n      [OK] Finished embedding {processed} images.", flush=True)
    # Return dogs that have at least 2 valid embeddings
    return {k: np.stack(v) for k, v in embeddings_by_dog.items() if len(v) >= 2}


# --------------------------------------------------------------------------- #
# Metrics Calculation
# --------------------------------------------------------------------------- #

def match_query(
    query_vec: np.ndarray,
    gallery: Dict[str, np.ndarray],
    strategy: str = "max"
) -> List[Tuple[str, float]]:
    """
    Ranks gallery dogs by cosine similarity against query_vec.
    strategy:
        - 'max': ensemble max-cosine (best matching image per dog)
        - 'avg': mean cosine similarity per dog
    """
    scores = []
    for dog_name, gallery_vecs in gallery.items():
        sims = gallery_vecs @ query_vec
        score = float(sims.max()) if strategy == "max" else float(sims.mean())
        scores.append((dog_name, score))

    scores.sort(key=lambda x: x[1], reverse=True)
    return scores


def compute_metrics(
    gallery: Dict[str, np.ndarray],
    queries: List[Tuple[str, np.ndarray]],
    match_strategy: str = "max"
) -> Tuple[float, float, float, Optional[float], np.ndarray, np.ndarray]:
    """
    Computes Rank-1, Rank-5, mAP, EER, genuine & impostor similarity distributions.
    queries: list of (true_dog_name, query_vector)
    gallery: dict of {dog_name: gallery_matrix_of_registered_images}
    """
    top1_correct = 0
    top5_correct = 0
    aps = []

    for true_name, q in queries:
        ranked = match_query(q, gallery, strategy=match_strategy)

        # Rank-1 & Rank-5
        top1_name = ranked[0][0] if len(ranked) > 0 else None
        top5_names = {n for n, _ in ranked[:5]}

        if top1_name == true_name:
            top1_correct += 1
        if true_name in top5_names:
            top5_correct += 1

        # Average Precision (AP) for this query
        hits = 0
        ap = 0.0
        for k, (name, _) in enumerate(ranked, start=1):
            if name == true_name:
                hits += 1
                ap += hits / k
        total_relevant = 1  # 1 true dog identity in the gallery
        aps.append(ap / total_relevant)

    total_q = len(queries)
    r1 = top1_correct / total_q if total_q else 0.0
    r5 = top5_correct / total_q if total_q else 0.0
    m_ap = float(np.mean(aps)) if aps else 0.0

    # Verification distribution: Genuine vs Impostor similarity pairs
    genuine = []
    impostor = []

    for true_name, q in queries:
        for dog_name, g_vecs in gallery.items():
            sims = (g_vecs @ q).tolist()
            if dog_name == true_name:
                genuine.extend(sims)
            else:
                impostor.extend(sims)

    genuine = np.array(genuine, dtype=np.float32)
    impostor = np.array(impostor, dtype=np.float32)

    eer = None
    if len(genuine) > 0 and len(impostor) > 0:
        thresholds = np.linspace(-1.0, 1.0, 1001)
        far = np.array([(impostor >= t).mean() for t in thresholds])
        frr = np.array([(genuine < t).mean() for t in thresholds])
        idx = int(np.argmin(np.abs(far - frr)))
        eer = float((far[idx] + frr[idx]) / 2)

    return r1, r5, m_ap, eer, genuine, impostor


# --------------------------------------------------------------------------- #
# Main CLI & Protocol
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(
        description="Dog Nose Biometric Evaluation: Register N-1 images per dog, test on held-out image."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="bigdataset3000",
        help="Path to dataset root folder containing dog subdirectories (e.g., bigdataset3000)."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="phodog/checkpoints/best.pth",
        help="Path to trained checkpoint (.pth). If missing or not found, uses ImageNet pretrained DNNetV3 backbone."
    )
    parser.add_argument(
        "--detector",
        type=str,
        default="phodog/checkpoints/nose_detector.onnx",
        help="Path to nose detector ONNX model. If not found or empty, uses whole images."
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache extracted embeddings (.npy). Recommended for fast re-runs."
    )
    parser.add_argument(
        "--test-mode",
        choices=["single", "leave-one-out"],
        default="single",
        help="single: register all except 1 test image per dog (hold-out test). "
             "leave-one-out: full round-robin cross-validation."
    )
    parser.add_argument(
        "--test-idx",
        type=str,
        default="last",
        help="For --test-mode single: 'last', 'first', 'random', or integer index (e.g. 0, 1)."
    )
    parser.add_argument(
        "--match-strategy",
        choices=["max", "avg"],
        default="max",
        help="Per-dog matching similarity: 'max' (ensemble max-cosine) or 'avg' (mean-cosine)."
    )
    parser.add_argument(
        "--max-dogs",
        type=int,
        default=None,
        help="Limit number of dogs evaluated (for quick benchmarking or debugging)."
    )
    parser.add_argument(
        "--no-deduplicate",
        action="store_true",
        help="Disable automatic deduplication/merging of master_data_<dog> with <dog> folders."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible test splits."
    )

    args = parser.parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    dedup = not args.no_deduplicate

    print("=" * 65)
    print("  DogID - Dataset Evaluation Protocol")
    print(f"  Dataset: {args.dataset}")
    print(f"  Protocol: Register all except 1 -> Test on held-out image")
    print(f"  Deduplicate identities: {dedup}")
    print(f"  Mode: {args.test_mode}")
    print("=" * 65)

    # Resolve dataset path
    dataset_path = args.dataset
    if not os.path.isabs(dataset_path):
        candidates = [
            dataset_path,
            os.path.join(os.getcwd(), dataset_path),
            os.path.join(os.path.dirname(__file__), "..", "..", dataset_path),
            os.path.join(os.path.dirname(__file__), "..", dataset_path)
        ]
        for c in candidates:
            if os.path.exists(c):
                dataset_path = os.path.abspath(c)
                break

    print(f"\n[1/4] Discovering dog folders in {dataset_path} ...")
    dog_images = discover_dataset(dataset_path, deduplicate=dedup)
    print(f"      Found {len(dog_images)} unique dog identities with >= 2 images.")

    if len(dog_images) < 2:
        print("\n[!] Need at least 2 dog classes with >= 2 images each to evaluate.")
        sys.exit(1)

    if args.max_dogs and args.max_dogs < len(dog_images):
        selected_dogs = list(dog_images.keys())[:args.max_dogs]
        dog_images = {k: dog_images[k] for k in selected_dogs}
        print(f"      Subsampled to first {args.max_dogs} dogs for this run.")

    # Setup detector
    detector = None
    det_path = args.detector
    if det_path and not os.path.isabs(det_path):
        for candidate in [det_path, os.path.join("phodog", det_path), os.path.join(os.path.dirname(__file__), "..", det_path)]:
            if os.path.exists(candidate):
                det_path = candidate
                break

    if det_path and os.path.exists(det_path):
        print(f"      [OK] Loading nose detector from {det_path}")
        detector = NoseDetector(weights_path=det_path)
    else:
        print(f"      [NOTE] No detector found at '{args.detector}', using input crops directly.")

    # Setup embedding model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n[2/4] Setting up DNNetV3 embedding model on {device} ...")
    ckpt_path = args.checkpoint
    if ckpt_path and not os.path.isabs(ckpt_path):
        for candidate in [ckpt_path, os.path.join("phodog", ckpt_path), os.path.join(os.path.dirname(__file__), "..", ckpt_path)]:
            if os.path.exists(candidate):
                ckpt_path = candidate
                break

    if ckpt_path and os.path.exists(ckpt_path):
        print(f"      Loading checkpoint: {ckpt_path}")
        model = DNNetV3(pretrained=True, use_head=True)
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(sd)
            print("      [OK] Model weights loaded successfully.")
        except Exception as e:
            print(f"      [WARN] Could not load checkpoint ({e}). Using ImageNet backbone.")
            model = DNNetV3(pretrained=True, use_head=False)
    else:
        print("      [NOTE] No .pth checkpoint found. Using ImageNet-pretrained DNNetV3 backbone.")
        model = DNNetV3(pretrained=True, use_head=False)

    model = model.to(device).eval()
    pipeline = CLAHEPipeline(image_size=224)

    # Auto cache directory if requested
    cache_dir = args.cache_dir
    if cache_dir is None:
        # Default cache inside phodog/data/cache_<dataset_name>
        ds_name = os.path.basename(os.path.normpath(dataset_path))
        cache_dir = os.path.join(os.path.dirname(__file__), "..", "data", f"cache_{ds_name}")

    print(f"\n[3/4] Extracting / loading image embeddings (cache: {cache_dir}) ...")
    dog_embeddings = extract_and_cache_embeddings(
        dog_images=dog_images,
        model=model,
        pipeline=pipeline,
        detector=detector,
        device=device,
        cache_dir=cache_dir
    )

    print(f"\n[4/4] Executing evaluation protocol ({args.test_mode}) ...")

    if args.test_mode == "single":
        # Register all images except one per dog; use the held-out image for testing
        gallery: Dict[str, np.ndarray] = {}
        queries: List[Tuple[str, np.ndarray]] = []

        for dog_name, vecs in dog_embeddings.items():
            n = len(vecs)
            if args.test_idx == "last":
                test_i = n - 1
            elif args.test_idx == "first":
                test_i = 0
            elif args.test_idx == "random":
                test_i = random.randint(0, n - 1)
            else:
                try:
                    test_i = int(args.test_idx) % n
                except ValueError:
                    test_i = n - 1

            # Registration set: all except test_i
            reg_indices = [i for i in range(n) if i != test_i]
            gallery[dog_name] = vecs[reg_indices]
            # Test query: test_i
            queries.append((dog_name, vecs[test_i]))

        r1, r5, m_ap, eer, gen, imp = compute_metrics(gallery, queries, match_strategy=args.match_strategy)

        print("\n" + "=" * 65)
        print("  EVALUATION RESULTS (Hold-Out Test: Register N-1, Test on 1)")
        print("=" * 65)
        print(f"  Dogs in test set : {len(gallery)}")
        print(f"  Queries tested   : {len(queries)} (1 held-out query per dog)")
        print(f"  Registered set   : {sum(len(v) for v in gallery.values())} gallery images")
        print("-" * 65)
        print(f"  Rank-1 accuracy  : {r1 * 100:6.2f}%   ({args.match_strategy}-cosine)")
        print(f"  Rank-5 accuracy  : {r5 * 100:6.2f}%")
        print(f"  mAP              : {m_ap * 100:6.2f}%")
        if eer is not None:
            print(f"  EER              : {eer * 100:6.2f}%   (verification threshold)")
            print(f"  Genuine sim      : mu={gen.mean():+.3f}  sigma={gen.std():.3f}  (n={len(gen)})")
            print(f"  Impostor sim     : mu={imp.mean():+.3f}  sigma={imp.std():.3f}  (n={len(imp)})")
        print("=" * 65)

    elif args.test_mode == "leave-one-out":
        # Full round-robin LOO cross-validation across all dogs & images
        all_r1, all_r5, all_map, all_eer = [], [], [], []
        all_gen, all_imp = [], []

        # Iterate over each dog taking each image as query once
        queries = []
        for dog_name, vecs in dog_embeddings.items():
            for i in range(len(vecs)):
                queries.append((dog_name, i))

        print(f"      Running full leave-one-out across {len(queries)} total queries ...")

        top1_correct = 0
        top5_correct = 0
        aps = []

        for dog_name, q_idx in queries:
            # Gallery is formed with this image held out
            temp_gallery = {}
            for other_dog, vecs in dog_embeddings.items():
                if other_dog == dog_name:
                    rem = [v for j, v in enumerate(vecs) if j != q_idx]
                    temp_gallery[other_dog] = np.stack(rem)
                else:
                    temp_gallery[other_dog] = vecs

            q_vec = dog_embeddings[dog_name][q_idx]
            ranked = match_query(q_vec, temp_gallery, strategy=args.match_strategy)

            top1_name = ranked[0][0] if len(ranked) > 0 else None
            top5_names = {n for n, _ in ranked[:5]}

            if top1_name == dog_name:
                top1_correct += 1
            if dog_name in top5_names:
                top5_correct += 1

            hits = 0
            ap = 0.0
            for k, (name, _) in enumerate(ranked, start=1):
                if name == dog_name:
                    hits += 1
                    ap += hits / k
            aps.append(ap)

        total_q = len(queries)
        r1 = top1_correct / total_q if total_q else 0.0
        r5 = top5_correct / total_q if total_q else 0.0
        m_ap = float(np.mean(aps)) if aps else 0.0

        # Verification distribution
        names = list(dog_embeddings.keys())
        for i, ni in enumerate(names):
            vi = dog_embeddings[ni]
            for a in range(len(vi)):
                for b in range(a + 1, len(vi)):
                    all_gen.append(float(vi[a] @ vi[b]))
            for nj in names[i + 1:]:
                vj = dog_embeddings[nj]
                sims = vi @ vj.T
                all_imp.extend(sims.flatten().tolist())

        gen = np.array(all_gen)
        imp = np.array(all_imp)
        thresholds = np.linspace(-1.0, 1.0, 1001)
        far = np.array([(imp >= t).mean() for t in thresholds])
        frr = np.array([(gen < t).mean() for t in thresholds])
        idx = int(np.argmin(np.abs(far - frr)))
        eer = float((far[idx] + frr[idx]) / 2)

        print("\n" + "=" * 65)
        print("  EVALUATION RESULTS (Full Leave-One-Out Across All Images)")
        print("=" * 65)
        print(f"  Dogs evaluated   : {len(dog_embeddings)}")
        print(f"  Total queries    : {total_q}")
        print("-" * 65)
        print(f"  Rank-1 accuracy  : {r1 * 100:6.2f}%   ({args.match_strategy}-cosine)")
        print(f"  Rank-5 accuracy  : {r5 * 100:6.2f}%")
        print(f"  mAP              : {m_ap * 100:6.2f}%")
        print(f"  EER              : {eer * 100:6.2f}%   (verification threshold)")
        print(f"  Genuine sim      : mu={gen.mean():+.3f}  sigma={gen.std():.3f}  (n={len(gen)})")
        print(f"  Impostor sim     : mu={imp.mean():+.3f}  sigma={imp.std():.3f}  (n={len(imp)})")
        print("=" * 65)


if __name__ == "__main__":
    main()
