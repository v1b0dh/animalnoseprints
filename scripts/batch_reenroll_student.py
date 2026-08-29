"""
batch_reenroll_student.py
=========================
Adds a 512-d student embedding for every teacher embedding in the folder
gallery at data/gallery/. Reads the stored JPEGs, runs StudentDNNet, and
writes a parallel ``<idx>_student.npy`` file next to each ``<idx>.npy``.

After running this, ``<dog>/<idx>.npy``      -> 1024-d teacher
                ``<dog>/<idx>_student.npy``  -> 512-d student

Usage:
    python scripts/batch_reenroll_student.py
    python scripts/batch_reenroll_student.py --gallery data/gallery
    python scripts/batch_reenroll_student.py --dry-run
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from PIL import Image

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'app')))
from det5 import StudentDNNet, CLAHEPipeline


DEFAULT_GALLERY = "data/gallery"
DEFAULT_CKPT    = "checkpoints/student_best.pth"


def load_student(ckpt_path: str) -> StudentDNNet:
    model = StudentDNNet(pretrained=True)
    if os.path.exists(ckpt_path):
        print(f"  [OK] Loading distilled weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
    else:
        print(f"  [WARN] No checkpoint at '{ckpt_path}' -- using ImageNet-pretrained backbone.")
        print(f"         Accuracy will be lower than a distilled model.")
    model.eval()
    return model


def run(gallery_dir: str, dry_run: bool = False):
    print("=" * 60)
    print("  DogID - Batch Student Re-enrollment")
    print("=" * 60)

    if not os.path.isdir(gallery_dir):
        print(f"[ERROR] Gallery not found: {gallery_dir}")
        return

    print("\n[1/4] Loading StudentDNNet ...")
    pipeline = CLAHEPipeline()
    model    = load_student(DEFAULT_CKPT)

    print(f"\n[2/4] Scanning gallery: {gallery_dir}")
    dogs = sorted(d for d in os.listdir(gallery_dir)
                  if os.path.isdir(os.path.join(gallery_dir, d)))
    if not dogs:
        print("[!] No dogs found.")
        return

    skipped, converted, errors = 0, 0, 0
    for dog in dogs:
        d = os.path.join(gallery_dir, dog)
        for fn in sorted(os.listdir(d)):
            if not (fn.endswith(".npy") and not fn.endswith("_student.npy")):
                continue
            idx = fn[:4]
            jpg = os.path.join(d, f"{idx}.jpg")
            out = os.path.join(d, f"{idx}_student.npy")
            if os.path.exists(out):
                skipped += 1
                continue
            if not os.path.exists(jpg):
                print(f"  [ERR] {dog}/{idx} - missing thumbnail")
                errors += 1
                continue
            try:
                img  = Image.open(jpg).convert("RGB")
                t, _ = pipeline(img)
                with torch.no_grad():
                    emb = model(t.unsqueeze(0)).squeeze(0).cpu().numpy().astype("float32")
                if dry_run:
                    print(f"  [DRY] {dog}/{idx} -> would write 512-d student embedding")
                else:
                    import numpy as np
                    np.save(out, emb)
                    print(f"  [OK]  {dog}/{idx} -> 512-d student embedding")
                converted += 1
            except Exception as exc:
                print(f"  [ERR] {dog}/{idx} - {exc}")
                errors += 1

    print("\n" + "=" * 60)
    print(f"  Done!  Converted: {converted}  |  Skipped: {skipped}  |  Errors: {errors}")
    if not dry_run and converted > 0:
        print("\n  [DONE] Student embeddings are now alongside the teacher ones.")
    print("=" * 60)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gallery", default=DEFAULT_GALLERY)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run(args.gallery, dry_run=args.dry_run)
