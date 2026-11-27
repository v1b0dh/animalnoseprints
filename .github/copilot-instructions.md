# DogID Repository Instructions

## Project overview

DogID is a Python/Streamlit and FastAPI dog nose biometric system. It uses YOLOv8/ONNX nose detection, LAB-CLAHE image preprocessing, and a `DNNetV3`/TinyViT 1024-dimensional embedding backbone. The repository contains three separate application variants:

- `phodog/`: photo enrollment and ensemble max-cosine identification.
- `vidog_pure_cosine_similarity/`: video enrollment and pure cosine matching, plus the FastAPI integration service.
- `vidog_train_model/`: video enrollment with an open-set cosine gate and a small scikit-learn linear classifier head.

The V2 applications share the same basic dual-stream concept: nose embeddings receive the majority weight and full-face/full-body embeddings provide a secondary stream. Gallery state is stored on disk rather than in a database. For V2, case directories normally contain `.npz` matrices plus `meta.json`; legacy individual `.npy` nose/face embeddings are still supported by the matchers.

## Setup and run commands

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Launch the Streamlit variants from their own directories so local imports and relative gallery paths resolve correctly:

```bash
cd phodog && streamlit run app.py --server.port 8501
cd vidog_pure_cosine_similarity && streamlit run app.py --server.port 8502
cd vidog_train_model && streamlit run app.py --server.port 8503
```

The pure-cosine variant also exposes the HTTP service:

```bash
cd vidog_pure_cosine_similarity
uvicorn api:app --host 0.0.0.0 --port 8000
curl http://localhost:8000/health
```

FastAPI and uvicorn are not currently in `requirements.txt`; when editing or running `api.py`, install them explicitly:

```bash
python -m pip install fastapi uvicorn
```

The API uses `MAX_VIDEO_BYTES`, `MAX_IMAGE_BYTES`, and `ALLOWED_ORIGINS` environment variables. Registration and identification contracts are documented in `docs/HANDOFF.md` and `docs/FULL_SYSTEM_WORKFLOW.md`.

## Evaluation and training commands

There is no pytest/unittest suite or configured linter/formatter in the repository.

Run a specific gallery evaluation:

```bash
cd phodog
python scripts/eval_gallery.py
python scripts/eval_gallery.py --gallery data/gallery --reembed /path/to/checkpoint.pth
```

Run dataset evaluation, including a quick subset or a leave-one-out pass:

```bash
cd phodog
python scripts/eval_dataset.py --dataset /path/to/dataset --test-mode leave-one-out --max-dogs 20
python scripts/eval_dataset.py --dataset /path/to/dataset --test-mode single --test-idx 0
```

Fine-tune the backbone with a seed and explicit output/cache directories:

```bash
cd phodog
python scripts/train_backbone.py --dataset /path/to/dataset --epochs 15 --batch-size 16 --seed 42 \
  --output-dir checkpoints --cache-crops data/crop_cache
```

The pure-cosine directory has equivalent `scripts/eval_gallery.py`, `scripts/eval_dataset.py`, and `scripts/train_backbone.py`; use the variant's own `checkpoints/` and gallery paths.

Dataset and benchmark script defaults can be stale or environment-specific. Use `--help` for the authoritative arguments before running a script:

```bash
python scripts/eval_dataset.py --help
python scripts/eval_gallery.py --help
python scripts/train_backbone.py --help
```

## Architecture and data flow

1. The nose detector (`nose_detector.py`) loads `checkpoints/nose_detector.onnx` and crops a candidate nose. Detector classes can also fall back to a center crop when the model is unavailable.
2. `det5.py` owns `CLAHEPipeline`, the `DNNetV3` teacher, and student/distillation classes. LAB-CLAHE is applied to macro nose crops and, where used, full images.
3. `video_engine.py` extracts and filters frames, removes blur, and performs diversity selection before creating enrollment vectors.
4. Matchers consume normalized 1024-dimensional vectors and perform best-frame cosine comparisons. The pure-cosine matcher supports nose-only, body-only, and dual-stream rules. The hybrid matcher adds a `FastLinearClassifier` boundary score.
5. V2 gallery storage is under `data/gallery_v2/<case-or-dog>/`; API metadata uses `case_id` keys and atomic `.npz` writes. V1 gallery storage is under `phodog/data/gallery/<dog>/` with `.npy`, thumbnail `.jpg`, and `meta.json`.

The intended default dual-stream weights are nose 80%, body 20% for the API/pure-cosine implementation. A 85% nose/15% face variant exists in legacy/photo-style code and the README. Preserve the specific weights configured in the matcher or app being changed; do not assume all variants share the same constants.

## Repository-specific conventions

- Most modules are copied between `phodog/`, `vidog_pure_cosine_similarity/`, and `vidog_train_model/`. Change behavior in the target variant explicitly; do not edit one copy and assume the others are linked.
- Modules are imported locally (`from det5 import ...`, `from matcher import ...`). Run commands from the application directory and keep imports resolvable without a package install.
- Use absolute paths derived from `CUR_DIR` for V2 checkpoints and galleries. V1 `GALLERY_DIR` is currently cwd-relative, so it must be run from `phodog/`.
- Embeddings must remain float32 and L2-normalized at storage and query boundaries. Reject or skip mismatched embedding dimensions instead of silently mixing vectors.
- Keep gallery writes atomic: write temporary files beside the destination and `os.replace()` them. Preserve the existing thread or file locks for concurrent API updates.
- Validate filesystem-derived names and case IDs before joining them to gallery paths. Use the existing safe-name or real-path containment patterns.
- Checkpoint loading is hard-bound to fine-tuned teacher weights (`teacher_best.pth`, then `dnnet_v3_best.pth`) for 1024-dimensional inference. Do not silently downgrade to a mismatched checkpoint or embedding size.
- Preserve support for both modern `.npz` bundles (`nose_vectors`, `face_vectors`, `body_vectors`) and legacy per-frame `.npy` files.
- Temporary uploaded videos and extracted images are cleaned up; the API should not leave raw media in the gallery. Gallery data and model checkpoints are generated/heavy artifacts and are normally ignored or environment-specific.
