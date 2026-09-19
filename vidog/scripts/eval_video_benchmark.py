import os
import sys
import json
import time
import random
import argparse
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import List, Dict, Any

VIDOG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(VIDOG_DIR)

from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector
from video_engine import VideoRegistrationEngine, ExtractedFrame, compute_sharpness
from classifier_head import FastLinearClassifier
from matcher import HybridBiometricMatcher


def extract_embedding(img_pil: Image.Image, model: DNNetV3, pipeline: CLAHEPipeline) -> np.ndarray:
    t, _ = pipeline(img_pil)
    import torch
    with torch.no_grad():
        emb = model(t.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return emb


@dataclass
class SampleResult:
    video: str
    frame_idx: int
    true_label: str
    top1_cosine: str
    top1_hybrid: str
    cosine_score: float
    clf_prob: float
    hybrid_score: float
    is_correct_cosine: bool
    is_correct_hybrid: bool


def run_evaluation(
    video_dir: str,
    phodog_orig_dir: str,
    target_k_frames_per_video: int = 8,
    target_fps: float = 2.5,
    min_sharpness: float = 35.0,
    test_queries_per_video: int = 4,
    open_set_count: int = 10,
    cosine_threshold: float = 0.64,
    top1_prob_floor: float = 0.55,
    random_seed: int = 42,
    cv_iterations: int = 10,
    cv_train_ratio: float = 0.8,
    results_md_path: str = None,
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 85)
    print("   DOGID V2: ALL-7-VIDEO REGISTRATION & HELD-OUT FRAME EVALUATION")
    print("   Cross-Validation: {}x Monte Carlo ({}% train)".format(cv_iterations, int(cv_train_ratio * 100)))
    print("   Ratio: ~33% Frames Enrolled | ~10% Unseen Frames Tested")
    print(f"   Detector: best.onnx (MagFace Fine-Tuned) | Gate Threshold: {cosine_threshold}")
    print("=" * 85)

    # 1. Load ML Pipeline & Nose Detector
    print("\n[1/5] Loading Model Backbone & Fine-Tuned Nose Detector (best.onnx)...")
    onnx_path = os.path.join(VIDOG_DIR, "checkpoints", "best.onnx")
    if not os.path.exists(onnx_path):
        onnx_path = os.path.join(VIDOG_DIR, "checkpoints", "nose_detector.onnx")
        print(f"  [WARN] best.onnx not found, falling back to nose_detector.onnx")
    detector = NoseDetector(weights_path=onnx_path)
    print(f"  [i] Detector: {os.path.basename(onnx_path)} | Loaded: {'ONNX Model (Active)' if detector.has_model else 'Center Fallback'}")

    pipeline = CLAHEPipeline(image_size=224)
    model = DNNetV3(pretrained=True, use_head=True)
    model.eval()

    # 2. Gather All 7 Videos
    videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    print(f"  [i] Found all {len(videos)} videos in {video_dir}")
    if len(videos) == 0:
        print("[ERROR] No MP4 videos found.")
        return

    # Helper function to derive class name from video filename
    def derive_class_name(video_filename: str) -> str:
        base = os.path.splitext(video_filename)[0]
        # Convert "dogid (1)" to "Dogid_1"
        base = base.replace("(", "_").replace(")", "").replace(" ", "_")
        # PascalCase first letter
        return ''.join(word.capitalize() if i == 0 else word for i, word in enumerate(base.split('_')))

    # 3. Process each video: registration + held-out test frames
    print("\n[2/5] Enrolling Frames & Sourcing Held-Out Test Frames...")
    enrolled_frames = []
    enrolled_videos = []
    video_to_class = {}
    test_heldout_frames = []

    for vid in videos:
        vp = os.path.join(video_dir, vid)
        cap = cv2.VideoCapture(vp)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = fc / max(fps, 1.0)
        cap.release()

        reg_k = max(6, int(round(fc / 3.0)))
        test_k = max(2, int(round(fc / 10.0)))
        sample_fps = max(6.0, float(reg_k * 1.5) / max(dur, 1.0))

        engine = VideoRegistrationEngine(
            target_fps=sample_fps,
            min_sharpness=min_sharpness,
            target_k_frames=reg_k
        )

        reg_frames, stats = engine.process_video(vp, nose_detector_fn=detector.detect_and_crop)
        enrolled_indices = set(f.frame_idx for f in reg_frames)
        enrolled_frames.extend(reg_frames)
        enrolled_videos.extend([vid] * len(reg_frames))
        
        # Derive and store class name for this video
        class_name = derive_class_name(vid)
        video_to_class[vid] = class_name

        cap = cv2.VideoCapture(vp)
        total_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        unused_candidates = []
        cur_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            if cur_idx not in enrolled_indices:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                sh = compute_sharpness(gray)
                if sh >= min_sharpness:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    unused_candidates.append((cur_idx, Image.fromarray(rgb), sh))
            cur_idx += 1
        cap.release()

        sample_k = min(test_k, len(unused_candidates))
        chosen = random.sample(unused_candidates, sample_k) if sample_k > 0 else []

        for c_idx, c_pil, c_sh in chosen:
            res = detector.detect_and_crop(c_pil)
            test_heldout_frames.append({
                "video": vid,
                "frame_idx": c_idx,
                "full_image": c_pil,
                "nose_crop": res.crop,
                "bbox": res.bbox,
                "sharpness": c_sh,
                "source": res.source,
                "confidence": res.confidence
            })

        print(f"  [{vid}]: {len(reg_frames)} registered | {len(chosen)} held-out test frames (from {len(unused_candidates)} available)")

    print(f"\n  [OK] Total Enrolled Gallery Frames: {len(enrolled_frames)}")
    print(f"  [OK] Total Held-Out Query Frames: {len(test_heldout_frames)}")

    # 4. Extract Gallery Embeddings
    print("\n[3/5] Computing Biometric Embeddings for Gallery (85% Nose + 15% Face)...")
    dogo_gallery_nose = [extract_embedding(f.nose_crop, model, pipeline) for f in enrolled_frames]
    dogo_gallery_face = [extract_embedding(f.full_image, model, pipeline) for f in enrolled_frames]

    # 5. Build Per-Video Gallery Classes (one class per dog video)
    gallery_classes = {}
    for vid in videos:
        class_name = video_to_class[vid]
        gallery_classes[class_name] = {"nose": [], "face": []}
    for i, f in enumerate(enrolled_frames):
        vid = enrolled_videos[i]
        class_name = video_to_class[vid]
        gallery_classes[class_name]["nose"].append(dogo_gallery_nose[i])
        gallery_classes[class_name]["face"].append(dogo_gallery_face[i])

    # 6. Enroll Negative Dog Classes
    print("\n[4/5] Enrolling Negative Dog Classes from phodog...")
    other_dog_files = sorted(
        [f for f in os.listdir(phodog_orig_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
    )

    neg_gallery_files = other_dog_files[:-open_set_count]
    unknown_dog_files = other_dog_files[-open_set_count:]

    for f in neg_gallery_files:
        d_name = f"Dog_{os.path.splitext(f)[0]}"
        img_p = os.path.join(phodog_orig_dir, f)
        img = Image.open(img_p).convert("RGB")
        res = detector.detect_and_crop(img)
        emb_n = extract_embedding(res.crop, model, pipeline)
        emb_f = extract_embedding(img, model, pipeline)
        gallery_classes[d_name] = {"nose": [emb_n], "face": [emb_f]}

    print(f"  [OK] Total Registered Classes in Gallery: {len(gallery_classes)}")
    print(f"  [OK] Unregistered Holdout Dogs: {len(unknown_dog_files)}")

    # 7. Cross-Validation: Monte Carlo (80/20 split, N iterations)
    print(f"\n[5/5] Running {cv_iterations}x Monte Carlo Cross-Validation (train={int(cv_train_ratio*100)}%/test={int((1-cv_train_ratio)*100)}%)...")

    all_cv_results: List[Dict[str, Any]] = []
    all_sample_results: List[SampleResult] = []
    per_dog_stats = {}  # Track per-dog accuracy

    for iteration in range(cv_iterations):
        iter_seed = random_seed + iteration * 137
        np.random.seed(iter_seed)
        random.seed(iter_seed)

        # Split enrolled Dogo frames into train gallery / test queries
        n_enrolled = len(enrolled_frames)
        indices = list(range(n_enrolled))
        np.random.shuffle(indices)
        split_idx = max(1, int(n_enrolled * cv_train_ratio))
        train_idx = set(indices[:split_idx])
        test_idx = set(indices[split_idx:])

        # Build train gallery
        train_gallery = {}
        for i in train_idx:
            f = enrolled_frames[i]
            vid = enrolled_videos[i]
            class_name = video_to_class[vid]
            if class_name not in train_gallery:
                train_gallery[class_name] = {"nose": [], "face": []}
            train_gallery[class_name]["nose"].append(dogo_gallery_nose[i])
            train_gallery[class_name]["face"].append(dogo_gallery_face[i])

        for d_name in neg_gallery_files:
            dn = f"Dog_{os.path.splitext(d_name)[0]}"
            train_gallery[dn] = gallery_classes[dn]

        # Train classifier on train gallery
        X_train = []
        y_train = []
        class_names = list(train_gallery.keys())

        for idx, (c_name, c_embs) in enumerate(train_gallery.items()):
            for n_emb in c_embs["nose"]:
                X_train.append(n_emb)
                y_train.append(idx)

        X_train = np.vstack(X_train)
        y_train = np.array(y_train)

        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, class_weight="balanced", random_state=iter_seed)
        clf.fit(X_train, y_train)

        # Test on held-out frames (test_heldout_frames + test_idx enrolled frames)
        test_items = list(test_heldout_frames)
        for i in test_idx:
            f = enrolled_frames[i]
            test_items.append({
                "video": enrolled_videos[i],
                "frame_idx": f.frame_idx,
                "full_image": f.full_image,
                "nose_crop": f.nose_crop,
                "bbox": f.bbox,
                "sharpness": f.sharpness,
                "source": f.detector_source,
                "confidence": f.detector_confidence,
                "_from_enrolled": True,
                "_enrolled_idx": i,
            })

        correct_cosine = 0
        correct_hybrid = 0
        iter_samples: List[SampleResult] = []
        positive_cosine = []
        positive_hybrid = []

        for t_item in test_items:
            q_nose = extract_embedding(t_item["nose_crop"], model, pipeline)
            q_face = extract_embedding(t_item["full_image"], model, pipeline)

            # Determine true class for this test item
            true_class_name = video_to_class[t_item["video"]]

            # Cosine similarity scores per class
            class_scores = {}
            for c_name, c_embs in gallery_classes.items():
                sims = [0.85 * float(np.dot(q_nose, gn)) + 0.15 * float(np.dot(q_face, gf))
                        for gn, gf in zip(c_embs["nose"], c_embs["face"])]
                class_scores[c_name] = max(sims)

            sorted_cosine = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
            top1_cos_dog, top1_cos_score = sorted_cosine[0]
            positive_cosine.append(class_scores.get(true_class_name, 0.0))

            # Classifier probabilities
            probs = clf.predict_proba(q_nose.reshape(1, -1))[0]
            hybrid_scores = {}
            for idx, c_name in enumerate(class_names):
                c_score = class_scores[c_name]
                p_score = float(probs[idx])
                hybrid_scores[c_name] = 0.60 * c_score + 0.40 * p_score

            sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
            top1_hyb_dog, top1_hyb_score = sorted_hybrid[0]
            positive_hybrid.append(hybrid_scores.get(true_class_name, 0.0))

            is_correct_c = (top1_cos_dog == true_class_name)
            is_correct_h = (top1_hyb_dog == true_class_name)
            if is_correct_c:
                correct_cosine += 1
            if is_correct_h:
                correct_hybrid += 1

            # Track per-dog stats
            if true_class_name not in per_dog_stats:
                per_dog_stats[true_class_name] = {"cosine_correct": 0, "hybrid_correct": 0, "total": 0}
            per_dog_stats[true_class_name]["cosine_correct"] += int(is_correct_c)
            per_dog_stats[true_class_name]["hybrid_correct"] += int(is_correct_h)
            per_dog_stats[true_class_name]["total"] += 1

            iter_samples.append(SampleResult(
                video=t_item.get("video", "?"),
                frame_idx=t_item.get("frame_idx", -1),
                true_label=true_class_name,
                top1_cosine=top1_cos_dog,
                top1_hybrid=top1_hyb_dog,
                cosine_score=float(top1_cos_score),
                clf_prob=float(probs[class_names.index(true_class_name)]) if true_class_name in class_names else 0.0,
                hybrid_score=float(top1_hyb_score),
                is_correct_cosine=is_correct_c,
                is_correct_hybrid=is_correct_h,
            ))

        n_test = len(test_items)
        cos_acc = (correct_cosine / n_test * 100) if n_test else 0.0
        hyb_acc = (correct_hybrid / n_test * 100) if n_test else 0.0

        # Open-set rejection (same for all iterations)
        unknown_cosine_max = []
        correct_open_cosine = 0
        correct_open_combined = 0

        for f in unknown_dog_files:
            img_p = os.path.join(phodog_orig_dir, f)
            img = Image.open(img_p).convert("RGB")
            res = detector.detect_and_crop(img)
            q_nose = extract_embedding(res.crop, model, pipeline)
            q_face = extract_embedding(img, model, pipeline)

            class_scores = {}
            for c_name, c_embs in gallery_classes.items():
                sims = [0.85 * float(np.dot(q_nose, gn)) + 0.15 * float(np.dot(q_face, gf))
                        for gn, gf in zip(c_embs["nose"], c_embs["face"])]
                class_scores[c_name] = max(sims)

            max_cos = max(class_scores.values())
            unknown_cosine_max.append(max_cos)

            probs = clf.predict_proba(q_nose.reshape(1, -1))[0]
            top1_prob = float(probs.max())
            if max_cos < cosine_threshold:
                correct_open_cosine += 1
            if max_cos < cosine_threshold or top1_prob < top1_prob_floor:
                correct_open_combined += 1

        n_unknown = len(unknown_dog_files)
        far_cosine = (correct_open_cosine / n_unknown * 100) if n_unknown else 0.0
        far_combined = (correct_open_combined / n_unknown * 100) if n_unknown else 0.0

        all_cv_results.append({
            "iteration": iteration,
            "seed": iter_seed,
            "train_size": len(train_idx),
            "test_size": len(test_idx) + len(test_heldout_frames),
            "cosine_accuracy": cos_acc,
            "hybrid_accuracy": hyb_acc,
            "far_cosine": far_cosine,
            "far_combined": far_combined,
            "mean_cosine_pos": float(np.mean(positive_cosine)),
            "mean_hybrid_pos": float(np.mean(positive_hybrid)),
            "mean_unknown_cosine": float(np.mean(unknown_cosine_max)),
        })
        all_sample_results.extend(iter_samples)

        print(f"  Iter {iteration+1:2d}/{cv_iterations}: Cosine Acc={cos_acc:.1f}% | Hybrid Acc={hyb_acc:.1f}% | FAR_cosine={far_cosine:.1f}% | FAR_combined={far_combined:.1f}%")

    # Aggregate CV results
    cos_accs = [r["cosine_accuracy"] for r in all_cv_results]
    hyb_accs = [r["hybrid_accuracy"] for r in all_cv_results]
    far_cosines = [r["far_cosine"] for r in all_cv_results]
    far_combineds = [r["far_combined"] for r in all_cv_results]

    agg = {
        "cosine_accuracy_mean": float(np.mean(cos_accs)),
        "cosine_accuracy_std": float(np.std(cos_accs)),
        "hybrid_accuracy_mean": float(np.mean(hyb_accs)),
        "hybrid_accuracy_std": float(np.std(hyb_accs)),
        "far_cosine_mean": float(np.mean(far_cosines)),
        "far_cosine_std": float(np.std(far_cosines)),
        "far_combined_mean": float(np.mean(far_combineds)),
        "far_combined_std": float(np.std(far_combineds)),
        "iterations": cv_iterations,
        "train_ratio": cv_train_ratio,
    }

    # Calculate per-dog accuracy
    per_dog_accuracy = {}
    for dog_name, stats in per_dog_stats.items():
        if stats["total"] > 0:
            per_dog_accuracy[dog_name] = {
                "cosine_acc": stats["cosine_correct"] / stats["total"] * 100,
                "hybrid_acc": stats["hybrid_correct"] / stats["total"] * 100,
                "total": stats["total"]
            }

    # 8. Report
    print("\n" + "=" * 85)
    print("                    CROSS-VALIDATION RESULTS")
    print("=" * 85)
    print(f"\n  Closed-Set Identification ({cv_iterations} iterations):")
    print(f"    Cosine Rank-1  : {agg['cosine_accuracy_mean']:.2f}% ± {agg['cosine_accuracy_std']:.2f}%")
    print(f"    Hybrid Rank-1  : {agg['hybrid_accuracy_mean']:.2f}% ± {agg['hybrid_accuracy_std']:.2f}%")
    print(f"\n  Open-Set Rejection ({cv_iterations} iterations):")
    print(f"    FAR Cosine Gate: {agg['far_cosine_mean']:.2f}% ± {agg['far_cosine_std']:.2f}%")
    print(f"    FAR Combined   : {agg['far_combined_mean']:.2f}% ± {agg['far_combined_std']:.2f}%")
    print(f"\n  Per-sample confidence tracked: cosine_score, clf_prob, hybrid_score")
    print(f"  Total test samples across all iterations: {len(all_sample_results)}")

    # Print per-dog accuracy
    print(f"\n  Per-Dog Accuracy ({len(per_dog_accuracy)} dogs):")
    for dog_name, acc_data in sorted(per_dog_accuracy.items()):
        print(f"    {dog_name:12s}: Cosine={acc_data['cosine_acc']:5.1f}% | Hybrid={acc_data['hybrid_acc']:5.1f}% | N={acc_data['total']}")

    # 9. Export to Markdown
    md_path = results_md_path or os.path.join(VIDOG_DIR, "results", "eval_results.md")
    md_dir = os.path.dirname(md_path)
    if md_dir:
        os.makedirs(md_dir, exist_ok=True)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Video Benchmark Evaluation Results\n\n")
        f.write(f"**CV Strategy:** Repeated Random Subsampling ({cv_iterations} iterations, {int(cv_train_ratio*100)}/{int((1-cv_train_ratio)*100)} train/test split)\n\n")
        f.write("## Aggregated Metrics\n\n")
        f.write("| Metric | Mean | Std |\n")
        f.write("|--------|------|-----|\n")
        f.write(f"| Cosine Rank-1 Accuracy | {agg['cosine_accuracy_mean']:.2f}% | {agg['cosine_accuracy_std']:.2f}% |\n")
        f.write(f"| Hybrid Rank-1 Accuracy | {agg['hybrid_accuracy_mean']:.2f}% | {agg['hybrid_accuracy_std']:.2f}% |\n")
        f.write(f"| FAR Cosine Gate | {agg['far_cosine_mean']:.2f}% | {agg['far_cosine_std']:.2f}% |\n")
        f.write(f"| FAR Combined | {agg['far_combined_mean']:.2f}% | {agg['far_combined_std']:.2f}% |\n\n")

        f.write("## Per-Iteration Breakdown\n\n")
        f.write("| Iter | Train | Test | Cosine Acc | Hybrid Acc | FAR Cos | FAR Comb |\n")
        f.write("|------|-------|------|------------|------------|---------|----------|\n")
        for r in all_cv_results:
            f.write(f"| {r['iteration']+1:2d} | {r['train_size']:4d} | {r['test_size']:4d} | {r['cosine_accuracy']:.2f}% | {r['hybrid_accuracy']:.2f}% | {r['far_cosine']:.1f}% | {r['far_combined']:.1f}% |\n")
        f.write("\n")

        f.write("## Per-Dog Accuracy\n\n")
        f.write("| Dog | Cosine Accuracy | Hybrid Accuracy | Samples |\n")
        f.write("|-----|----------------|-----------------|---------|\n")
        for dog_name, acc_data in sorted(per_dog_accuracy.items()):
            f.write(f"| {dog_name} | {acc_data['cosine_acc']:.2f}% | {acc_data['hybrid_acc']:.2f}% | {acc_data['total']} |\n")
        f.write("\n")

        f.write("## Per-Sample Confidence Scores (all iterations)\n\n")
        f.write("| Video | Frame | True Label | Top1 Cosine | Top1 Hybrid | Cosine Score | Clf Prob | Hybrid Score | Cos Correct | Hybrid Correct |\n")
        f.write("|-------|-------|------------|-------------|-------------|--------------|----------|--------------|-------------|----------------|\n")
        for s in all_sample_results:
            f.write(f"| {s.video} | {s.frame_idx} | {s.true_label} | {s.top1_cosine} | {s.top1_hybrid} | {s.cosine_score:.4f} | {s.clf_prob:.4f} | {s.hybrid_score:.4f} | {'Y' if s.is_correct_cosine else 'N'} | {'Y' if s.is_correct_hybrid else 'N'} |\n")
        f.write("\n")

        f.write("## Assessment\n\n")
        if agg["hybrid_accuracy_mean"] >= 99.0 and agg["far_combined_mean"] >= 90.0:
            f.write("- [EXCELLENT] Production-ready: 99%+ closed-set accuracy AND strong open-set rejection!\n")
        elif agg["hybrid_accuracy_mean"] >= 95.0:
            f.write("- [GREAT] Very high closed-set accuracy.\n")
        elif agg["hybrid_accuracy_mean"] >= 90.0:
            f.write("- [GOOD] Solid recognition.\n")
        else:
            f.write("- [BASELINE] Backbone needs further fine-tuning.\n")

    print(f"\n  [OK] Results exported to {md_path}")
    print("=" * 85)
    return agg, all_cv_results, all_sample_results


if __name__ == '__main__':
    video_path = os.path.join(VIDOG_DIR, "dataset", "video_samples", "dogo")
    phodog_orig = os.path.join(os.path.dirname(VIDOG_DIR), "phodog", "dataset", "dog_samples_original")

    run_evaluation(
        video_dir=video_path,
        phodog_orig_dir=phodog_orig,
        target_k_frames_per_video=8,
        test_queries_per_video=4,
        min_sharpness=35.0,
        cosine_threshold=0.64,
        top1_prob_floor=0.55,
        cv_iterations=10,
        cv_train_ratio=0.8,
    )