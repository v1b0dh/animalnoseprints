"""
batch_reenroll_student.py
=========================
Converts existing teacher-enrolled dogs to student embeddings WITHOUT
re-photographing. Reads the stored photos from the database, runs them
through the StudentDNNet model, and inserts new 512-d student embeddings.

After running this, the Streamlit app's "Export Gallery for Mobile"
feature will work immediately.

Usage:
    python batch_reenroll_student.py
    python batch_reenroll_student.py --db dog_biometrics.db
    python batch_reenroll_student.py --dry-run   # Preview only, no writes
"""

import argparse
import io
import sqlite3
import struct
import os
from PIL import Image

import torch
import torch.nn.functional as F
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../app')))
from det5 import StudentDNNet, CLAHEPipeline


# ── Config ────────────────────────────────────────────────────────────────────

DEFAULT_DB  = "data/dog_biometrics.db"
STUDENT_DIM = 512
CKPT_PATH   = "checkpoints/student_best.pth"


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_student(ckpt_path: str) -> StudentDNNet:
    model = StudentDNNet(pretrained=True)
    if os.path.exists(ckpt_path):
        print(f"  [✓] Loading distilled weights from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        sd = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(sd)
    else:
        print(f"  [!] No checkpoint at '{ckpt_path}' -- using ImageNet-pretrained backbone.")
        print(f"      Accuracy will be lower than a distilled model.")
    model.eval()
    return model


def embed_photo_bytes(photo_blob: bytes, model: StudentDNNet, pipeline: CLAHEPipeline) -> list[float]:
    """Decode stored BLOB photo → run student inference → return float list."""
    img = Image.open(io.BytesIO(photo_blob)).convert("RGB")
    tensor, _ = pipeline(img)
    tensor = tensor.unsqueeze(0)          # (1, 3, 224, 224)
    with torch.no_grad():
        emb = model(tensor)               # (1, 512) L2-normalized
    return emb[0].tolist()


def floats_to_blob(floats: list[float]) -> bytes:
    return struct.pack(f"<{len(floats)}f", *floats)


# ── Main ──────────────────────────────────────────────────────────────────────

def run(db_path: str, dry_run: bool = False):
    print("=" * 60)
    print("  DogID — Batch Student Re-enrollment")
    print("=" * 60)

    if not os.path.exists(db_path):
        print(f"[ERROR] Database not found: {db_path}")
        return

    # 1. Load models
    print("\n[1/4] Loading StudentDNNet …")
    pipeline = CLAHEPipeline()
    model    = load_student(CKPT_PATH)

    # 2. Connect to DB
    print(f"\n[2/4] Opening database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON")

    # 3. Fetch all teacher-enrolled rows that have a photo stored
    rows = conn.execute("""
        SELECT e.id, d.name, e.photo, e.sharpness
        FROM   embeddings e
        JOIN   dogs d ON d.id = e.dog_id
        WHERE  e.model_type = 'teacher'
          AND  e.photo IS NOT NULL
          AND  length(e.photo) > 0
        ORDER  BY d.name, e.id
    """).fetchall()

    if not rows:
        print("\n[!] No teacher embeddings with stored photos found.")
        print("    Make sure photos were saved when you registered the dogs.")
        conn.close()
        return

    print(f"\n[3/4] Found {len(rows)} teacher embedding(s) to convert.\n")

    skipped   = 0
    converted = 0
    errors    = 0

    for emb_id, dog_name, photo_blob, sharpness in rows:
        # Skip if a student embedding already exists for this exact embedding row
        existing = conn.execute("""
            SELECT COUNT(*) FROM embeddings
            WHERE dog_id = (SELECT dog_id FROM embeddings WHERE id = ?)
              AND model_type = 'student'
        """, (emb_id,)).fetchone()[0]

        if existing > 0:
            print(f"  [i] {dog_name} (emb #{emb_id}) — student embedding already exists, skipping.")
            skipped += 1
            continue

        try:
            floats = embed_photo_bytes(photo_blob, model, pipeline)
            blob   = floats_to_blob(floats)

            if dry_run:
                print(f"  [DRY] {dog_name} (emb #{emb_id}) -> would insert 512-d student embedding.")
            else:
                conn.execute("""
                    INSERT INTO embeddings (dog_id, embedding, photo, sharpness, model_type)
                    SELECT dog_id, ?, photo, ?, 'student'
                    FROM   embeddings WHERE id = ?
                """, (blob, sharpness, emb_id))
                print(f"  [OK]  {dog_name} (emb #{emb_id}) -> converted to 512-d student embedding.")

            converted += 1

        except Exception as exc:
            print(f"  [ERR] {dog_name} (emb #{emb_id}) - ERROR: {exc}")
            errors += 1

    # 4. Commit & summary
    if not dry_run:
        conn.commit()

    conn.close()

    print("\n" + "=" * 60)
    print(f"  Done!  Converted: {converted}  |  Skipped: {skipped}  |  Errors: {errors}")
    if not dry_run and converted > 0:
        print("\n  [DONE] You can now use 'Export Gallery for Mobile' in the Streamlit app.")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch re-enroll dogs using StudentDNNet")
    parser.add_argument("--db",      default=DEFAULT_DB, help=f"Path to SQLite database (default: {DEFAULT_DB})")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    args = parser.parse_args()
    run(args.db, dry_run=args.dry_run)
