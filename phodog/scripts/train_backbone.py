"""
train_backbone.py
=================
Fine-tune the DNNetV3 biometric backbone using MagFace Loss on dog nose images.

Pipeline:
  1. Discovers dog identity classes from dataset (e.g. bigdataset3000)
     with identity deduplication (merging master_data_<dog> with <dog>).
  2. Crops nose regions using nose_detector.onnx (with cache of cropped images
     to avoid repeated detection overhead during training).
  3. Preprocesses with CLAHEPipeline + Data Augmentation (random rotation, slight jitter).
  4. Optimizes DNNetV3 (TinyViT-21M + MagFace projection head) with MagFace loss.
  5. Evaluates Rank-1 accuracy on validation hold-out set after every epoch.
  6. Saves best model weights to checkpoints/dnnet_v3_best.pth and checkpoints/teacher_best.pth.

Usage:
    python phodog/scripts/train_backbone.py --dataset bigdataset3000 --epochs 10 --batch-size 16
    python phodog/scripts/train_backbone.py --dataset bigdataset3000 --epochs 5 --freeze-backbone
"""

import argparse
import os
import sys
import time
import math
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from det5 import DNNetV3, MagFaceLoss, CLAHEPipeline
from nose_detector import NoseDetector
from scripts.eval_dataset import discover_dataset, compute_metrics


# --------------------------------------------------------------------------- #
# Dataset with Nose Crop Caching & Data Augmentation
# --------------------------------------------------------------------------- #

class DogNoseDataset(Dataset):
    def __init__(
        self,
        samples: List[Tuple[str, int]],  # (image_path, class_idx)
        precropped_dir: str,
        is_train: bool = True
    ):
        self.samples = samples
        self.precropped_dir = precropped_dir
        self.is_train = is_train
        self.pipeline = CLAHEPipeline(image_size=224)

        # Augmentation for biometric training: slight rotation & flip
        self.train_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        crop_path, label = self.samples[idx]
        try:
            crop = Image.open(crop_path).convert("RGB")
        except Exception:
            crop = Image.new("RGB", (224, 224), (128, 128, 128))

        if self.is_train:
            crop = self.train_aug(crop)

        tensor, sharpness = self.pipeline(crop)
        return tensor, torch.tensor(sharpness, dtype=torch.float32), label


def precrop_all_images(
    all_samples: List[Tuple[str, int]],
    detector: Optional[NoseDetector],
    crop_dir: str
) -> List[Tuple[str, int]]:
    """
    Pre-crops all nose images once to disk so DataLoader runs at full speed without detector overhead.
    """
    os.makedirs(crop_dir, exist_ok=True)
    out_samples = []
    total = len(all_samples)
    t0 = time.time()

    print(f"      Pre-cropping / verifying {total} nose images in {crop_dir} ...", flush=True)

    for i, (orig_path, label) in enumerate(all_samples, start=1):
        dog_folder = os.path.basename(os.path.dirname(orig_path))
        base_fn = os.path.splitext(os.path.basename(orig_path))[0]
        dog_crop_dir = os.path.join(crop_dir, dog_folder)
        os.makedirs(dog_crop_dir, exist_ok=True)
        crop_path = os.path.join(dog_crop_dir, f"{base_fn}_crop.jpg")

        if not os.path.exists(crop_path):
            try:
                raw_img = Image.open(orig_path).convert("RGB")
                if detector is not None and detector.has_model:
                    res = detector.detect_and_crop(raw_img)
                    crop = res.crop
                else:
                    crop = raw_img
                crop.save(crop_path, quality=95)
            except Exception:
                pass

        out_samples.append((crop_path, label))

        if i % 100 == 0 or i == total:
            elapsed = time.time() - t0
            fps = i / elapsed if elapsed > 0 else 0
            print(f"      Processed {i}/{total} crops ({fps:.1f} img/s) ...", flush=True)

    print(f"      [OK] Ready with {len(out_samples)} nose crops.", flush=True)
    return out_samples


# --------------------------------------------------------------------------- #
# Validation Evaluation Routine
# --------------------------------------------------------------------------- #

def evaluate_model(
    model: DNNetV3,
    val_samples: List[Tuple[str, int]],
    device: str,
    class_names: List[str]
) -> Tuple[float, float, float]:
    """
    Evaluates Rank-1, Rank-5, mAP on validation set using hold-out protocol.
    """
    model.eval()
    val_dataset = DogNoseDataset(val_samples, precropped_dir="", is_train=False)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    # Collect embeddings per identity
    embeddings_by_dog: Dict[str, List[np.ndarray]] = defaultdict(list)

    with torch.no_grad():
        for tensors, _, labels in val_loader:
            tensors = tensors.to(device)
            vecs = model(tensors).cpu().numpy()
            for v, l in zip(vecs, labels.numpy()):
                dog_name = class_names[l]
                embeddings_by_dog[dog_name].append(v)

    # Filter dogs that have >= 2 images in val set
    eligible_gallery: Dict[str, np.ndarray] = {}
    queries: List[Tuple[str, np.ndarray]] = []

    for name, vlist in embeddings_by_dog.items():
        if len(vlist) >= 2:
            arr = np.stack(vlist)
            eligible_gallery[name] = arr[:-1]  # N-1 registered
            queries.append((name, arr[-1]))    # 1 hold-out query

    if not queries or len(eligible_gallery) < 2:
        return 0.0, 0.0, 0.0

    r1, r5, map_, _, _, _ = compute_metrics(eligible_gallery, queries, match_strategy="max")
    return r1, r5, map_


# --------------------------------------------------------------------------- #
# Main Training Function
# --------------------------------------------------------------------------- #

def train():
    parser = argparse.ArgumentParser(description="Fine-tune DNNetV3 on Dog Noses using MagFace Loss")
    parser.add_argument("--dataset", default="bigdataset3000", help="Dataset root directory")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate for head")
    parser.add_argument("--backbone-lr", type=float, default=3e-5, help="Learning rate for backbone")
    parser.add_argument("--freeze-backbone", action="store_true", help="Freeze backbone, only train head")
    parser.add_argument("--output-dir", default="phodog/checkpoints", help="Directory to save weights")
    parser.add_argument("--cache-crops", default="phodog/data/crop_cache", help="Directory to cache nose crops")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation fraction per class")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 65)
    print("  DogID - DNNetV3 Biometric Backbone Fine-Tuning (MagFace)")
    print(f"  Dataset: {args.dataset}")
    print(f"  Epochs: {args.epochs} | Batch size: {args.batch_size}")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[1/5] Using compute device: {device}")

    # Discover and deduplicate dataset
    print(f"[2/5] Loading and deduplicating dataset from {args.dataset} ...")
    dog_dict = discover_dataset(args.dataset, deduplicate=True)
    class_names = sorted(dog_dict.keys())
    num_classes = len(class_names)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    print(f"      Identified {num_classes} unique dog identity classes.")

    # Split train and validation
    train_samples: List[Tuple[str, int]] = []
    val_samples: List[Tuple[str, int]] = []

    for name, img_paths in dog_dict.items():
        c_idx = class_to_idx[name]
        paths = list(img_paths)
        random.shuffle(paths)
        val_count = max(1, int(len(paths) * args.val_split))
        # Ensure at least 1 image remains for train
        if len(paths) > 2:
            val_paths = paths[:val_count]
            tr_paths = paths[val_count:]
        else:
            val_paths = paths[:1]
            tr_paths = paths[1:]

        for p in tr_paths:
            train_samples.append((p, c_idx))
        for p in val_paths:
            val_samples.append((p, c_idx))

    print(f"      Train images: {len(train_samples)} | Validation images: {len(val_samples)}")

    # Load Nose Detector
    detector_path = "phodog/checkpoints/nose_detector.onnx"
    detector = NoseDetector(weights_path=detector_path) if os.path.exists(detector_path) else None
    if detector and detector.has_model:
        print(f"      [OK] Loaded nose detector from {detector_path}")

    # Precrop nose images once for blazing fast training
    print("\n[3/5] Pre-cropping nose regions with ONNX detector ...")
    crop_dir = args.cache_crops
    train_samples = precrop_all_images(train_samples, detector, os.path.join(crop_dir, "train"))
    val_samples = precrop_all_images(val_samples, detector, os.path.join(crop_dir, "val"))

    # Datasets and Loaders
    train_dataset = DogNoseDataset(train_samples, crop_dir, is_train=True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0
    )

    # Initialize Model & MagFace Loss
    print(f"\n[4/5] Initializing DNNetV3 (TinyViT-21M) with MagFace projection head ...")
    model = DNNetV3(pretrained=True, use_head=True).to(device)

    loss_fn = MagFaceLoss(
        num_classes=num_classes,
        embed_dim=DNNetV3.EMBED_DIM,
        scale=64.0,
        margin_lo=0.45,
        margin_hi=0.80,
        norm_lo=10.0,
        norm_hi=110.0,
        lambda_g=35.0
    ).to(device)

    # Optimizer with differential learning rates
    if args.freeze_backbone:
        for p in model.backbone.parameters():
            p.requires_grad = False
        params = [
            {"params": model.head.parameters(), "lr": args.lr},
            {"params": loss_fn.parameters(), "lr": args.lr}
        ]
        print("      Backbone frozen. Training only MagFace projection head.")
    else:
        params = [
            {"params": model.backbone.parameters(), "lr": args.backbone_lr},
            {"params": model.head.parameters(), "lr": args.lr},
            {"params": loss_fn.parameters(), "lr": args.lr}
        ]
        print(f"      Differential LR: Backbone={args.backbone_lr}, Head & MagFace={args.lr}")

    optimizer = torch.optim.AdamW(params, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Initial baseline evaluation
    print("\n[5/5] Evaluating initial baseline (Epoch 0) ...")
    init_r1, init_r5, init_map = evaluate_model(model, val_samples, device, class_names)
    print(f"      Baseline Rank-1: {init_r1*100:5.2f}% | Rank-5: {init_r5*100:5.2f}% | mAP: {init_map*100:5.2f}%\n")

    best_r1 = init_r1
    os.makedirs(args.output_dir, exist_ok=True)
    best_ckpt_path = os.path.join(args.output_dir, "dnnet_v3_best.pth")
    teacher_ckpt_path = os.path.join(args.output_dir, "teacher_best.pth")

    print("[*] Commencing training loop ...")
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()
        num_batches = len(train_loader)

        for step, (tensors, sharpness, labels) in enumerate(train_loader, start=1):
            tensors = tensors.to(device)
            sharpness = sharpness.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            emb_norm, raw_norms = model.get_feature_norm(tensors)
            loss = loss_fn(emb_norm, raw_norms, labels, sharpness=sharpness)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()

            if step % 10 == 0 or step == num_batches:
                elapsed = time.time() - t0
                fps = (step * args.batch_size) / elapsed if elapsed > 0 else 0
                avg_loss = total_loss / step
                print(
                    f"      Epoch [{epoch:02d}/{args.epochs:02d}] Step [{step:03d}/{num_batches:03d}] "
                    f"Loss: {avg_loss:6.3f} | {fps:4.1f} img/s",
                    flush=True
                )

        scheduler.step()
        epoch_loss = total_loss / num_batches

        # Evaluate at epoch end
        val_r1, val_r5, val_map = evaluate_model(model, val_samples, detector, args.cache_crops, device, class_names)
        print("-" * 65)
        print(
            f"  >>> Epoch {epoch:02d} Summary: Train Loss: {epoch_loss:.3f} | "
            f"Val Rank-1: {val_r1*100:5.2f}% | Val Rank-5: {val_r5*100:5.2f}% | Val mAP: {val_map*100:5.2f}%"
        )

        # Save best model
        if val_r1 >= best_r1:
            best_r1 = val_r1
            state = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_r1": val_r1,
                "val_map": val_map,
                "num_classes": num_classes
            }
            torch.save(state, best_ckpt_path)
            torch.save(state, teacher_ckpt_path)
            print(f"      [NEW BEST] Saved checkpoint to {best_ckpt_path}")
        print("-" * 65)

    print("\n" + "=" * 65)
    print("  TRAINING COMPLETE")
    print(f"  Best Val Rank-1 Accuracy: {best_r1*100:5.2f}%")
    print(f"  Saved weights: {best_ckpt_path} and {teacher_ckpt_path}")
    print("=" * 65)


if __name__ == "__main__":
    train()
