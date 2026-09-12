# 🐾 Dog Nose Detector — Training Guide

> **Purpose:** Train a YOLOv8 object detection model that locates a dog's nose in an image and exports it as an ONNX file. This ONNX file is then dropped into the DogID app to enable automatic nose cropping.
>
> **Who this is for:** A developer or ML engineer being handed this task. No prior context about the DogID project is required to complete this guide.
>
> **Estimated time:** 2–4 hours (mostly waiting for training to finish).

---

## 📋 Table of Contents

1. [Background — What & Why](#1-background--what--why)
2. [Prerequisites](#2-prerequisites)
3. [Step 1 — Create a Kaggle Account & Notebook](#3-step-1--create-a-kaggle-account--notebook)
4. [Step 2 — Source the Datasets from Roboflow](#4-step-2--source-the-datasets-from-roboflow)
5. [Step 3 — Setup the Kaggle Notebook Environment](#5-step-3--setup-the-kaggle-notebook-environment)
6. [Step 4 — Download All Datasets in the Notebook](#6-step-4--download-all-datasets-in-the-notebook)
7. [Step 5 — Merge & Standardize the Datasets](#7-step-5--merge--standardize-the-datasets)
8. [Step 6 — Train YOLOv8](#8-step-6--train-yolov8)
9. [Step 7 — Evaluate the Model](#9-step-7--evaluate-the-model)
10. [Step 8 — Export to ONNX](#10-step-8--export-to-onnx)
11. [Step 9 — Deliver the File](#11-step-9--deliver-the-file)
12. [Troubleshooting](#12-troubleshooting)
13. [Technical Spec for the App Developer](#13-technical-spec-for-the-app-developer)

---

## 1. Background — What & Why

The DogID app identifies dogs using their **nose print** as a biometric — similar to a fingerprint. For this to work accurately, the system must first **locate and crop just the nose region** from any uploaded photo before computing the biometric embedding.

Right now, if no detector is present, the app falls back to a simple center-crop of the image, which is inaccurate. Your job is to train a YOLO model that detects the nose bounding box precisely.

**What you will produce:**
```
nose_detector.onnx   ← a single file, ~6-12 MB
```

This file is placed at `checkpoints/nose_detector.onnx` inside the DogID project and the app automatically starts using it.

**Input/Output of the model:**
- **Input:** Any photo of a dog (full body, face, or close-up), resized to 640×640.
- **Output:** Bounding box(es) around the dog's nose region with a confidence score.

---

## 2. Prerequisites

You need:
- [ ] A free **Kaggle** account → https://www.kaggle.com/account/login
- [ ] A free **Roboflow** account → https://app.roboflow.com (for dataset API access)
- [ ] Nothing else — all training runs in the cloud on Kaggle's free GPU.

No local GPU or Python installation needed.

---

## 3. Step 1 — Create a Kaggle Account & Notebook

1. Go to https://www.kaggle.com and sign in.
2. Click **"Create" → "New Notebook"**.
3. In the notebook settings (right panel):
   - Set **Language** to `Python`.
   - Set **Accelerator** to `GPU T4 x2` or `GPU P100` (both are free with a verified account).
   - Set **Internet** to `On` (required to download datasets).
4. Click **Save** and wait for the environment to start.

> **Tip:** Kaggle gives you ~30 hours of free GPU per week. This training will use about 1–2 hours.

---

## 4. Step 2 — Source the Datasets from Roboflow

You will combine **two or more datasets** for better accuracy.

### Recommended Datasets

Go to **https://universe.roboflow.com** and search for each:

| # | Search Term | Notes |
|---|---|---|
| 1 | `dog nose detection` | Primary dataset — look for one with 200+ images |
| 2 | `dog face keypoints` | May have nose bounding boxes or landmarks |
| 3 | `dog snout` | Alternative terminology |
| 4 | `pet nose` | Includes cats too — filter to dogs if possible |

### For each dataset you select:

1. Open the dataset page on Roboflow Universe.
2. Click the **"Download Dataset"** button.
3. In the popup:
   - Set **Image and Annotation Format** to `YOLOv8`.
   - Select **"Show download code"**.
   - Click **"Continue"**.
4. Copy the code snippet shown. It looks like:
   ```python
   from roboflow import Roboflow
   rf = Roboflow(api_key="YOUR_KEY")
   project = rf.workspace("workspace-name").project("project-name")
   version = project.version(1)
   dataset = version.download("yolov8")
   ```
5. **Save this snippet** — you'll paste it into the Kaggle notebook.

> **Important:** Note which class index corresponds to the nose in each dataset. Some datasets have multiple classes (eye=0, nose=1, ear=2). You'll need this for the merge script in Step 5. Check the `data.yaml` file inside the downloaded folder.

---

## 5. Step 3 — Setup the Kaggle Notebook Environment

In your Kaggle notebook, create a new code cell and run:

```python
# Install required packages
!pip install -q ultralytics roboflow scikit-learn

# Verify GPU is available
import torch
print("GPU available:", torch.cuda.is_available())
print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")
```

**Expected output:**
```
GPU available: True
GPU name: Tesla T4
```

If GPU shows `False`, go back to notebook settings, set the accelerator to GPU, and restart.

---

## 6. Step 4 — Download All Datasets in the Notebook

Create a code cell for each dataset. Paste the Roboflow snippet, adding the `location=` argument:

```python
# ── Dataset 1 ──────────────────────────────────────────────────────────────────
from roboflow import Roboflow

rf = Roboflow(api_key="PASTE_YOUR_API_KEY_HERE")

project = rf.workspace("WORKSPACE_NAME").project("PROJECT_NAME")
version  = project.version(1)
ds1 = version.download("yolov8", location="/kaggle/working/raw_ds1")

print("DS1 location:", ds1.location)
```

```python
# ── Dataset 2 ──────────────────────────────────────────────────────────────────
project2 = rf.workspace("WORKSPACE_NAME_2").project("PROJECT_NAME_2")
version2  = project2.version(1)
ds2 = version2.download("yolov8", location="/kaggle/working/raw_ds2")

print("DS2 location:", ds2.location)
```

After downloading, check the class names in each dataset:

```python
# Check what classes are in each dataset
with open("/kaggle/working/raw_ds1/data.yaml") as f:
    print("=== DS1 classes ===\n", f.read())

with open("/kaggle/working/raw_ds2/data.yaml") as f:
    print("=== DS2 classes ===\n", f.read())
```

Note the index number next to `nose` / `snout` / `dog_nose` in the output — you will need it in Step 5.

---

## 7. Step 5 — Merge & Standardize the Datasets

This script reads all datasets, remaps every nose annotation to **class `0`**, deduplicates filenames, and produces one clean merged dataset for training.

**Before running:** Update the `nose_class_ids` values based on what you found in the `data.yaml` files above.

```python
import os
import shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

# ── Configuration ──────────────────────────────────────────────────────────────
# "nose_class_ids": the class index(es) that represent the NOSE in each dataset.
# Example: if data.yaml shows "1: dog_nose", set nose_class_ids to [1]
# ───────────────────────────────────────────────────────────────────────────────
DATASETS_CONFIG = [
    {
        "name": "ds1",
        "path": "/kaggle/working/raw_ds1",
        "nose_class_ids": [0],   # ← CHECK data.yaml AND UPDATE THIS
    },
    {
        "name": "ds2",
        "path": "/kaggle/working/raw_ds2",
        "nose_class_ids": [0],   # ← CHECK data.yaml AND UPDATE THIS
    },
    # Add more datasets here as needed:
    # {
    #     "name": "ds3",
    #     "path": "/kaggle/working/raw_ds3",
    #     "nose_class_ids": [1],
    # },
]

MERGED_DIR = Path("/kaggle/working/merged_dataset")
for split in ["train", "val"]:
    (MERGED_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (MERGED_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)

all_samples = []

for cfg in DATASETS_CONFIG:
    ds_name  = cfg["name"]
    ds_path  = Path(cfg["path"])
    nose_ids = set(cfg["nose_class_ids"])

    img_exts = ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.PNG", "*.JPEG")
    img_files = []
    for ext in img_exts:
        img_files.extend(ds_path.rglob(ext))

    accepted = 0
    for img_p in img_files:
        # Find label file — alongside image, or in parallel labels/ folder
        txt_p = img_p.with_suffix(".txt")
        if not txt_p.exists():
            parts = list(img_p.parts)
            if "images" in parts:
                idx = len(parts) - 1 - parts[::-1].index("images")
                parts[idx] = "labels"
                txt_p = Path(*parts).with_suffix(".txt")

        if not txt_p.exists():
            continue  # No annotation — skip

        valid_lines = []
        with open(txt_p, "r") as f:
            for line in f:
                row = line.strip().split()
                if not row:
                    continue
                cls_id = int(float(row[0]))
                if cls_id in nose_ids:
                    coords = " ".join(row[1:5])  # x_center y_center width height
                    valid_lines.append(f"0 {coords}\n")

        if valid_lines:
            all_samples.append((img_p, valid_lines, ds_name))
            accepted += 1

    print(f"  [{ds_name}] Accepted {accepted} annotated nose samples")

print(f"\nTotal samples collected: {len(all_samples)}")

train_s, val_s = train_test_split(all_samples, test_size=0.15, random_state=42)
print(f"Train: {len(train_s)} | Val: {len(val_s)}")

def save_split(samples, split_name):
    for idx, (img_p, lines, prefix) in enumerate(samples):
        safe_name = f"{prefix}_{idx:05d}{img_p.suffix}"
        shutil.copy2(img_p, MERGED_DIR / "images" / split_name / safe_name)
        lbl_path = MERGED_DIR / "labels" / split_name / f"{prefix}_{idx:05d}.txt"
        with open(lbl_path, "w") as f:
            f.writelines(lines)

save_split(train_s, "train")
save_split(val_s,   "val")

yaml_content = f"""path: {MERGED_DIR.as_posix()}
train: images/train
val:   images/val

nc: 1
names:
  0: dog_nose
"""
with open(MERGED_DIR / "data.yaml", "w") as f:
    f.write(yaml_content)

print("\n✅ Merged dataset ready at:", MERGED_DIR)
```

---

## 8. Step 6 — Train YOLOv8

```python
from ultralytics import YOLO

# yolov8n.pt = nano (fast, lightweight) — recommended
# yolov8s.pt = small (more accurate, slightly slower) — use if mAP is too low
model = YOLO("yolov8n.pt")

results = model.train(
    data     = "/kaggle/working/merged_dataset/data.yaml",
    epochs   = 60,       # Increase to 100 if mAP keeps improving at epoch 60
    imgsz    = 640,      # DO NOT CHANGE — the app expects 640x640 input
    batch    = 16,       # Reduce to 8 if you get out-of-memory errors
    device   = 0,        # GPU
    patience = 15,       # Stop early if no improvement for 15 epochs
    project  = "dog_nose_detector",
    name     = "yolov8n_nose_v1",
    exist_ok = True,
)

print("\n✅ Training complete!")
print("Best weights at: dog_nose_detector/yolov8n_nose_v1/weights/best.pt")
```

**Training takes approximately 30–90 minutes.** A live progress table will appear. Watch the `val/mAP50` column:

| mAP50 | Assessment |
|---|---|
| < 0.50 | Too low — add more data or train longer |
| 0.50 – 0.75 | Acceptable for a first pass |
| 0.75 – 0.90 | Good ✅ |
| > 0.90 | Excellent 🎉 |

---

## 9. Step 7 — Evaluate the Model

```python
trained_model = YOLO("dog_nose_detector/yolov8n_nose_v1/weights/best.pt")

metrics = trained_model.val(data="/kaggle/working/merged_dataset/data.yaml")

print(f"mAP50:     {metrics.box.map50:.4f}")
print(f"mAP50-95:  {metrics.box.map:.4f}")
print(f"Precision: {metrics.box.p.mean():.4f}")
print(f"Recall:    {metrics.box.r.mean():.4f}")
```

Run a visual check — confirm boxes land on the nose, not on eyes or ears:

```python
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

val_images = list((Path("/kaggle/working/merged_dataset/images/val")).iterdir())

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, img_path in zip(axes.flat, val_images[:6]):
    result = trained_model(str(img_path))[0]
    ax.imshow(Image.fromarray(result.plot()))
    ax.set_title(img_path.name, fontsize=8)
    ax.axis("off")
plt.tight_layout()
plt.show()
```

If boxes land on the wrong feature (eye, ear), the `nose_class_ids` config was incorrect — go back and fix Step 5 and retrain.

---

## 10. Step 8 — Export to ONNX

```python
# opset=12 and dynamic=False are required by the DogID app's OpenCV loader
trained_model.export(
    format   = "onnx",
    imgsz    = 640,
    opset    = 12,
    dynamic  = False,
    simplify = True,
)

import os
onnx_path = "dog_nose_detector/yolov8n_nose_v1/weights/best.onnx"
size_mb = os.path.getsize(onnx_path) / 1e6
print(f"\n✅ ONNX exported: {onnx_path} ({size_mb:.1f} MB)")
```

---

## 11. Step 9 — Deliver the File

1. In the Kaggle Notebook, click the **Output** tab in the right sidebar.
2. Navigate to `dog_nose_detector/yolov8n_nose_v1/weights/`.
3. Download `best.onnx`.
4. **Rename it to exactly:** `nose_detector.onnx`
5. Share with the DogID project developer.

**The developer places it here — no other changes needed:**
```
dogshi/
└── checkpoints/
    └── nose_detector.onnx    ← place the file here
```

The app detects and uses it automatically on next launch.

---

## 12. Troubleshooting

| Problem | Solution |
|---|---|
| "CUDA out of memory" | Reduce `batch=16` to `batch=8` and re-run training |
| Merge script finds 0 samples | Check `data.yaml` in each raw dataset, fix `nose_class_ids`, re-run Step 5 |
| mAP50 < 0.50 after training | Increase epochs to 100, or switch from `yolov8n.pt` to `yolov8s.pt` |
| Boxes land on eyes, not nose | Wrong `nose_class_ids` — recheck `data.yaml`, fix config, rerun merge + training |
| Roboflow API key error | Go to https://app.roboflow.com → Settings → API Keys → copy fresh key |
| Internet is off in Kaggle | Notebook settings → toggle Internet: On → save (needs phone verification on Kaggle) |

---

## 13. Technical Spec for the App Developer

> This section is for the DogID app developer receiving the ONNX file.

**File:** `nose_detector.onnx`
**Placement:** `checkpoints/nose_detector.onnx`

Loaded automatically by `app/nose_detector.py` via `cv2.dnn.readNetFromONNX()`. No code changes needed.

| Property | Value |
|---|---|
| Framework | YOLOv8n (Ultralytics) |
| Input shape | `(1, 3, 640, 640)` float32 |
| Pixel range | `[0.0, 1.0]` |
| Output format | YOLO xywh + confidence |
| Classes | `1` class → `0: dog_nose` |
| ONNX opset | `12` |
| Confidence threshold | `0.60` (adjustable in `NoseDetector.__init__`) |

**Fallback:** If the file is missing or confidence < threshold, the system automatically falls back to a 55% center-crop and marks it as `source: "center_fallback"` in the UI.

---

*Guide written for the DogID (animalnoseprints) project.*
*GitHub: https://github.com/v1b0dh/animalnoseprints*
