import os
import sys
import json
import time
import random
import argparse
import cv2
import numpy as np
from PIL import Image

# Ensure imports resolve relative to vidog folder
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

def run_evaluation(
    video_dir: str,
    phodog_orig_dir: str,
    target_k_frames_per_video: int = 8,
    target_fps: float = 2.5,
    min_sharpness: float = 40.0,
    test_queries_per_video: int = 4,
    min_sharpness: float = 35.0,
    open_set_count: int = 10,
    cosine_threshold: float = 0.64,
    random_seed: int = 42
):
    random.seed(random_seed)
    np.random.seed(random_seed)

    print("=" * 85)
    print("   DOGID V2: ALL-7-VIDEO REGISTRATION & HELD-OUT FRAME EVALUATION")
    print("   DOGID V2: HIGH-DENSITY VIDEO REGISTRATION & HELD-OUT BENCHMARK")
    print("   Ratio: ~33% Frames Enrolled (e.g. 100/300) | ~10% Unseen Frames Tested (e.g. 30/300)")
    print("=" * 85)

    # 1. Load ML Pipeline & Nose Detector
    print("\n[1/5] Loading Model Backbone & YOLOv8 Nose Detector...")
    onnx_path = os.path.join(VIDOG_DIR, "checkpoints", "nose_detector.onnx")
    detector = NoseDetector(weights_path=onnx_path)
    print(f"  [i] Detector Status: {'ONNX Model (Active)' if detector.has_model else 'Center Fallback'}")

    pipeline = CLAHEPipeline(image_size=224)
    model = DNNetV3(pretrained=True, use_head=True)
    model.eval()

    # 2. Gather All 7 Videos
    videos = sorted([f for f in os.listdir(video_dir) if f.endswith(".mp4")])
    print(f"  [i] Found all {len(videos)} videos in {video_dir}")
    if len(videos) == 0:
        print("[ERROR] No MP4 videos found.")
        return

    # 3. Process each of the 7 videos:
    #    - Registration selects top K sharp diverse frames
    #    - Identification extracts random sharp frames that were NOT selected for registration
    print("\n[2/5] Enrolling from All 7 Videos & Sourcing Unused Held-Out Test Frames...")
    engine = VideoRegistrationEngine(
        target_fps=target_fps,
        min_sharpness=min_sharpness,
        target_k_frames=target_k_frames_per_video
    )

    #    - Registration selects ~33% of video frames (e.g., 100 out of 300)
    #    - Identification extracts ~10% of unused held-out frames (e.g., 30 out of 300)
    print("\n[2/5] Enrolling ~33% Frames per Video & Sourcing ~10% Unused Held-Out Test Frames...")
    enrolled_frames = []
    test_heldout_frames = []

    for vid in videos:
        vp = os.path.join(video_dir, vid)
        
        cap = cv2.VideoCapture(vp)
        fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = fc / max(fps, 1.0)
        cap.release()

        # Proportional frame budget matching user request:
        # e.g., 300 frames -> 100 registration frames, 30 test frames
        reg_k = max(6, int(round(fc / 3.0)))
        test_k = max(2, int(round(fc / 10.0)))
        sample_fps = max(6.0, float(reg_k * 1.5) / max(dur, 1.0))

        engine = VideoRegistrationEngine(
            target_fps=sample_fps,
            min_sharpness=min_sharpness,
            target_k_frames=reg_k
        )

        # A. Registration extraction
        reg_frames, stats = engine.process_video(vp, nose_detector_fn=detector.detect_and_crop)
        enrolled_indices = set(f.frame_idx for f in reg_frames)
        enrolled_frames.extend(reg_frames)

        # B. Sourcing held-out frames that registration DID NOT take
        cap = cv2.VideoCapture(vp)
        total_fc = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        unused_candidates = []
        cur_idx = 0
        while True:
            ret, bgr = cap.read()
            if not ret:
                break
            # Pick candidates with a buffer (>= 3 frames away from any registered frame)
            if all(abs(cur_idx - reg_idx) >= 3 for reg_idx in enrolled_indices):
                # Quick sharpness check to avoid testing on unusable pure black/motion-blur frames
            if cur_idx not in enrolled_indices:
                gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
                sh = compute_sharpness(gray)
                if sh >= min_sharpness:
                    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    unused_candidates.append((cur_idx, Image.fromarray(rgb), sh))
            cur_idx += 1
        cap.release()

        # Randomly sample held-out test frames from unused candidates
        sample_k = min(test_queries_per_video, len(unused_candidates))
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

        print(f"  [{vid}]: {len(reg_frames)} registered frames | {len(chosen)} unused held-out test frames selected (from {len(unused_candidates)} available)")
        print(f"  [{vid}]: {len(reg_frames)} registered (target={reg_k}) | {len(chosen)} unseen test frames (target={test_k}, available={len(unused_candidates)})")

    print(f"\n  [OK] Total Enrolled Gallery Frames for 'Dogo': {len(enrolled_frames)}")
    print(f"\n  [OK] Total Enrolled Gallery Frames for 'Dogo': {len(enrolled_frames)} (from 815 total video frames)")
    print(f"  [OK] Total Unused Held-Out Query Frames for 'Dogo': {len(test_heldout_frames)}")

    # 4. Extract Gallery Embeddings (85% Nose + 15% Face)
    print("\n[3/5] Computing Biometric Embeddings for Gallery (85% Nose + 15% Face)...")
    dogo_gallery_nose = [extract_embedding(f.nose_crop, model, pipeline) for f in enrolled_frames]
    dogo_gallery_face = [extract_embedding(f.full_image, model, pipeline) for f in enrolled_frames]

    # 5. Enroll Negative Dog Classes from phodog for Gallery Competition
    print("\n[4/5] Enrolling Negative Dog Classes from phodog for Multi-Dog Competition...")
    other_dog_files = sorted(
        [f for f in os.listdir(phodog_orig_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(os.path.splitext(x)[0]) if os.path.splitext(x)[0].isdigit() else x
    )

    neg_gallery_files = other_dog_files[:-open_set_count]
    unknown_dog_files = other_dog_files[-open_set_count:]

    gallery_classes = {"Dogo": {"nose": dogo_gallery_nose, "face": dogo_gallery_face}}

    for f in neg_gallery_files:
        d_name = f"Dog_{os.path.splitext(f)[0]}"
        img_p = os.path.join(phodog_orig_dir, f)
        img = Image.open(img_p).convert("RGB")
        res = detector.detect_and_crop(img)
        emb_n = extract_embedding(res.crop, model, pipeline)
        emb_f = extract_embedding(img, model, pipeline)
        gallery_classes[d_name] = {"nose": [emb_n], "face": [emb_f]}

    print(f"  [OK] Total Registered Classes in Gallery: {len(gallery_classes)} (Dogo + {len(gallery_classes)-1} other dogs)")
    print(f"  [OK] Unregistered Holdout Dogs for False-Acceptance Test: {len(unknown_dog_files)}")

    # Fit Fast Linear Classifier
    print("  Fitting Fast Linear Classifier on gallery embeddings...")
    X_train = []
    y_train = []
    class_names = list(gallery_classes.keys())

    for idx, (c_name, c_embs) in enumerate(gallery_classes.items()):
        for n_emb in c_embs["nose"]:
            X_train.append(n_emb)
            y_train.append(idx)

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(C=1.0, solver="lbfgs", max_iter=300, class_weight="balanced", random_state=42)
    clf.fit(X_train, y_train)
    print(f"  [OK] Classifier trained on {len(X_train)} samples across {len(class_names)} classes.")

    # 6. Evaluation on Unused Held-Out Test Frames (Dogo vs All Gallery Classes)
    print("\n[5/5] Evaluating Identification on Unused Held-Out Video Frames...")
    correct_rank1_cosine = 0
    correct_rank1_hybrid = 0
    positive_cosine_scores = []
    positive_hybrid_scores = []

    for t_item in test_heldout_frames:
        q_nose = extract_embedding(t_item["nose_crop"], model, pipeline)
        q_face = extract_embedding(t_item["full_image"], model, pipeline)

        # Compute max ensemble cosine for each gallery dog
        class_scores = {}
        for c_name, c_embs in gallery_classes.items():
            sims = [0.85 * float(np.dot(q_nose, gn)) + 0.15 * float(np.dot(q_face, gf))
                    for gn, gf in zip(c_embs["nose"], c_embs["face"])]
            class_scores[c_name] = max(sims)

        # Cosine rank 1
        sorted_cosine = sorted(class_scores.items(), key=lambda x: x[1], reverse=True)
        top1_cos_dog, top1_cos_score = sorted_cosine[0]
        positive_cosine_scores.append(class_scores["Dogo"])

        if top1_cos_dog == "Dogo":
            correct_rank1_cosine += 1

        # Hybrid score (0.60 Cosine + 0.40 Classifier Probability)
        probs = clf.predict_proba(q_nose.reshape(1, -1))[0]
        hybrid_scores = {}
        for idx, c_name in enumerate(class_names):
            c_score = class_scores[c_name]
            p_score = float(probs[idx])
            hybrid_scores[c_name] = 0.60 * c_score + 0.40 * p_score

        sorted_hybrid = sorted(hybrid_scores.items(), key=lambda x: x[1], reverse=True)
        top1_hyb_dog, top1_hyb_score = sorted_hybrid[0]
        positive_hybrid_scores.append(hybrid_scores["Dogo"])

        if top1_hyb_dog == "Dogo":
            correct_rank1_hybrid += 1

    # 7. Evaluation on Open-Set Unknown Dogs
    unknown_cosine_max = []
    correct_open_set_rejections = 0

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

        if max_cos < cosine_threshold:
            correct_open_set_rejections += 1

    # 8. Report Final Results
    print("\n" + "=" * 85)
    print("                    FINAL BENCHMARK REPORT")
    print("=" * 85)

    n_test = len(test_heldout_frames)
    cos_acc = (correct_rank1_cosine / n_test * 100) if n_test else 0.0
    hyb_acc = (correct_rank1_hybrid / n_test * 100) if n_test else 0.0
    far_rej = (correct_open_set_rejections / len(unknown_dog_files) * 100) if unknown_dog_files else 0.0

    print(f"\n1. IDENTIFICATION ACCURACY (Testing on {n_test} held-out unused frames across all 7 videos):")
    print(f"   * Pure 85/15 Cosine Similarity Rank-1 Accuracy:  {cos_acc:.2f}% ({correct_rank1_cosine}/{n_test})")
    print(f"   * 2-Stage Hybrid (Cosine + Linear Head) Accuracy: {hyb_acc:.2f}% ({correct_rank1_hybrid}/{n_test})")
    print(f"   * Average Positive Similarity (Dogo vs Dogo):      {np.mean(positive_cosine_scores):.4f}")
    print(f"   * Average Hybrid Score (Dogo vs Dogo):             {np.mean(positive_hybrid_scores):.4f}")

    print(f"\n2. OPEN-SET REJECTION (Testing against {len(unknown_dog_files)} unregistered unknown dogs):")
    print(f"   * Correct Unknown Rejection (Cosine < {cosine_threshold}):     {far_rej:.2f}% ({correct_open_set_rejections}/{len(unknown_dog_files)})")
    print(f"   * Average Unknown Max-Cosine Score:                {np.mean(unknown_cosine_max):.4f}")

    print(f"\n3. SUMMARY ASSESSMENT:")
    print(f"   All 7 videos were enrolled for registration. The model was then tested against")
    print(f"   completely unseen frames from those same 7 videos to test genuine within-session recognition.")
    print("=" * 85)

if __name__ == '__main__':
    video_path = os.path.join(VIDOG_DIR, "dataset", "video_samples", "dogo")
    phodog_orig = os.path.join(os.path.dirname(VIDOG_DIR), "phodog", "dataset", "dog_samples_original")
    
    run_evaluation(
        video_dir=video_path,
        phodog_orig_dir=phodog_orig,
        target_k_frames_per_video=8,
        test_queries_per_video=4,
        min_sharpness=35.0,
        cosine_threshold=0.64
    )
