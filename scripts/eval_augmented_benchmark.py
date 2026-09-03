import os
import sys
import json
import time
import argparse
import numpy as np
from PIL import Image
from typing import Dict, List, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from det5 import DNNetV3, CLAHEPipeline
from nose_detector import NoseDetector

DEFAULT_ORIG_DIR = 'dataset/dog_samples_original'
DEFAULT_AUG_DIR  = 'dataset/nose_samples_augmented'
DEFAULT_CHECKPOINT = 'checkpoints/nose_detector.onnx'

def get_dog_id(filename: str) -> str:
    base = os.path.splitext(filename)[0]
    return base.split('_')[0]

def extract_embedding(img_pil: Image.Image, model: DNNetV3, pipeline: CLAHEPipeline) -> np.ndarray:
    t, _ = pipeline(img_pil)
    import torch
    with torch.no_grad():
        emb = model(t.unsqueeze(0)).squeeze(0).cpu().numpy().astype('float32')
    norm = np.linalg.norm(emb)
    if norm > 1e-6:
        emb = emb / norm
    return emb

def run_benchmark(
    orig_dir: str = DEFAULT_ORIG_DIR,
    aug_dir: str = DEFAULT_AUG_DIR,
    detector_path: str = DEFAULT_CHECKPOINT,
    open_set_count: int = 5,
    threshold: float = 0.68
):
    print('=' * 75)
    print('      DOGID BIOMETRIC ACCURACY BENCHMARK (SYNTHETIC PROBES)')
    print('=' * 75)

    if not os.path.isdir(orig_dir):
        print(f'[ERROR] Original samples directory not found: {orig_dir}')
        return
    if not os.path.isdir(aug_dir):
        print(f'[ERROR] Augmented samples directory not found: {aug_dir}')
        return

    print('\n[1/4] Loading ML Pipeline & Nose Detector...')
    pipeline = CLAHEPipeline(image_size=224)
    model = DNNetV3(pretrained=True, use_head=True)
    model.eval()

    detector = NoseDetector(weights_path=detector_path)
    det_status = "ONNX Model" if detector.has_model else "Center Fallback"
    print(f"  [i] Detector Status: {det_status}")

    orig_files = sorted(
        [f for f in os.listdir(orig_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))],
        key=lambda x: int(get_dog_id(x)) if get_dog_id(x).isdigit() else x
    )

    all_dog_ids = list(dict.fromkeys([get_dog_id(f) for f in orig_files]))
    print(f'  [i] Found {len(all_dog_ids)} total unique dogs in dataset.')

    if open_set_count >= len(all_dog_ids):
        open_set_count = max(1, len(all_dog_ids) // 5)

    gallery_dog_ids = all_dog_ids[:-open_set_count]
    unknown_dog_ids = all_dog_ids[-open_set_count:]

    print(f'  [i] Enrolling {len(gallery_dog_ids)} dogs into Gallery (Closed-Set).')
    print(f'  [i] Holding out {len(unknown_dog_ids)} dogs for False-Acceptance Test (Open-Set).')

    print('\n[2/4] Enrolling Reference Gallery...')
    gallery_embeddings: Dict[str, np.ndarray] = {}
    
    for f in orig_files:
        dog_id = get_dog_id(f)
        if dog_id not in gallery_dog_ids:
            continue
        img_path = os.path.join(orig_dir, f)
        img = Image.open(img_path).convert('RGB')
        crop_res = detector.detect_and_crop(img)
        emb = extract_embedding(crop_res.crop, model, pipeline)
        gallery_embeddings[dog_id] = emb

    print(f'  [✓] Enrolled {len(gallery_embeddings)} reference embeddings.')

    print('\n[3/4] Running Identification on Augmented Probes...')
    aug_files = sorted([f for f in os.listdir(aug_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

    total_closed_probes = 0
    correct_rank1 = 0
    correct_rank3 = 0
    closed_scores = []
    
    total_open_probes = 0
    correctly_rejected_unknowns = 0
    open_scores = []

    aug_breakdown = {}

    for f in aug_files:
        dog_id = get_dog_id(f)
        img_path = os.path.join(aug_dir, f)
        img = Image.open(img_path).convert('RGB')
        
        # Image is already cropped nose
        probe_emb = extract_embedding(img, model, pipeline)

        # Cosine similarity against all gallery dogs
        sims = {g_id: float(np.dot(probe_emb, g_emb)) for g_id, g_emb in gallery_embeddings.items()}
        sorted_candidates = sorted(sims.items(), key=lambda x: x[1], reverse=True)
        top1_id, top1_score = sorted_candidates[0]

        aug_type = f.replace(f'{dog_id}_', '').replace('.jpg', '')

        if dog_id in gallery_dog_ids:
            total_closed_probes += 1
            closed_scores.append(top1_score)
            
            is_rank1 = (top1_id == dog_id and top1_score >= threshold)
            if is_rank1:
                correct_rank1 += 1

            rank_candidates = [c[0] for c in sorted_candidates[:3]]
            if dog_id in rank_candidates:
                correct_rank3 += 1

            if aug_type not in aug_breakdown:
                aug_breakdown[aug_type] = {'total': 0, 'correct': 0}
            aug_breakdown[aug_type]['total'] += 1
            if is_rank1:
                aug_breakdown[aug_type]['correct'] += 1

        elif dog_id in unknown_dog_ids:
            total_open_probes += 1
            open_scores.append(top1_score)
            if top1_score < threshold:
                correctly_rejected_unknowns += 1

    print('\n[4/4] Evaluation Summary:')
    print('-' * 75)
    rank1_acc = (correct_rank1 / total_closed_probes * 100) if total_closed_probes else 0.0
    rank3_acc = (correct_rank3 / total_closed_probes * 100) if total_closed_probes else 0.0
    far_rejection = (correctly_rejected_unknowns / total_open_probes * 100) if total_open_probes else 0.0

    print(f'  CLOSED-SET IDENTIFICATION ({total_closed_probes} probes across {len(gallery_dog_ids)} dogs): ')
    print(f'    • Rank-1 Accuracy (Exact Match & Cosine >= {threshold}): {rank1_acc:.2f}% ({correct_rank1}/{total_closed_probes})')
    print(f'    • Rank-3 Accuracy (Within Top 3):                  {rank3_acc:.2f}% ({correct_rank3}/{total_closed_probes})')
    print(f'    • Average Positive Cosine Score:                   {np.mean(closed_scores):.4f}')
    
    print('\n  AUGMENTATION BREAKDOWN (Rank-1 Accuracy):')
    for atype, stats in aug_breakdown.items():
        corr = stats['correct']
        tot = stats['total']
        acc = (corr / tot * 100) if tot else 0.0
        print(f"    - {atype:<15}: {acc:6.2f}% ({corr}/{tot})")

    print(f'\n  OPEN-SET REJECTION ({total_open_probes} unknown dog probes from {len(unknown_dog_ids)} dogs):')
    print(f'    • Correct Unknown Rejection (Cosine < {threshold}):    {far_rejection:.2f}% ({correctly_rejected_unknowns}/{total_open_probes})')
    print(f'    • Average Unknown Max-Cosine Score:                {np.mean(open_scores) if open_scores else 0.0:.4f}')

    print('=' * 75)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run biometric accuracy benchmark on augmented probes')
    parser.add_argument('--orig', default=DEFAULT_ORIG_DIR, help='Original images directory')
    parser.add_argument('--aug', default=DEFAULT_AUG_DIR, help='Augmented images directory')
    parser.add_argument('--detector', default=DEFAULT_CHECKPOINT, help='Nose detector ONNX path')
    parser.add_argument('--threshold', type=float, default=0.68, help='Acceptance cosine threshold')
    parser.add_argument('--holdout', type=int, default=5, help='Number of dogs held out for open-set test')
    args = parser.parse_args()

    run_benchmark(
        orig_dir=args.orig,
        aug_dir=args.aug,
        detector_path=args.detector,
        open_set_count=args.holdout,
        threshold=args.threshold
    )
