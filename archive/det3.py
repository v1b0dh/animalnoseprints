"""
Dog Noseprint Detection — DNNet (Petnow / IEEE Access 2021)
============================================================
Install:  pip install torch torchvision opencv-python numpy Pillow

Usage:
    model = load_model()

    register_dog(model, "buddy_nose.jpg", "Buddy")
    register_dog(model, "max_nose.jpg",   "Max")

    identify_dog(model, "unknown_nose.jpg")
    compare_noses(model, "nose1.jpg", "nose2.jpg")
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import os
from PIL import Image


# ── Config ────────────────────────────────────────────────────

DB_FILE         = "petnow_db.json"
MATCH_THRESHOLD = 0.75    # similarity 0.0–1.0, above this = same dog


# ── Preprocessing ─────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

def _load_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor  = _transform(Image.fromarray(img_rgb)).unsqueeze(0)
    return tensor


# ── Self-Attention Block ──────────────────────────────────────
#
# Lets every region of the nose "look at" every other region.
# Captures global ridge structure that a CNN's local filters miss.

class NonLocalAttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        mid = in_channels // 2
        self.theta    = nn.Conv2d(in_channels, mid, 1)
        self.phi      = nn.Conv2d(in_channels, mid, 1)
        self.g        = nn.Conv2d(in_channels, mid, 1)
        self.out_proj = nn.Sequential(
            nn.Conv2d(mid, in_channels, 1),
            nn.BatchNorm2d(in_channels)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        N     = H * W
        theta = self.theta(x).view(B, -1, N).permute(0, 2, 1)
        phi   = self.phi(x).view(B, -1, N)
        g     = self.g(x).view(B, -1, N).permute(0, 2, 1)
        attn  = F.softmax(torch.bmm(theta, phi), dim=-1)
        out   = torch.bmm(attn, g).permute(0, 2, 1).view(B, -1, H, W)
        return x + self.out_proj(out)


# ── DNNet Model ───────────────────────────────────────────────
#
# Pipeline per image:
#   ResNet-152  →  [B, 2048, 7, 7]  deep spatial features
#   Compress    →  [B, 256,  7, 7]  channel reduction
#   Attention   →  [B, 256,  7, 7]  global ridge relationships
#   GAP         →  [B, 256]         single vector
#   FC          →  [B, 128]         identity embedding
#   L2 norm     →  unit sphere      stable 0–1 comparisons

class DNNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet         = models.resnet152(weights=models.ResNet152_Weights.DEFAULT)
        self.backbone  = nn.Sequential(*list(resnet.children())[:-2])
        self.compress  = nn.Sequential(
            nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU(inplace=True),
            nn.Conv2d(512,  256, 1), nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        )
        self.attention = NonLocalAttentionBlock(256)
        self.gap       = nn.AdaptiveAvgPool2d(1)
        self.embed     = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.compress(x)
        x = self.attention(x)
        x = self.gap(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)   # L2 normalized → cosine = dot product


# ── Contrastive Loss (used during training only) ──────────────

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, emb_a, emb_b, label):
        dist      = F.pairwise_distance(emb_a, emb_b)
        loss_same = label * dist.pow(2)
        loss_diff = (1 - label) * F.relu(self.margin - dist).pow(2)
        return (loss_same + loss_diff).mean()


# ── Helpers ───────────────────────────────────────────────────

def _to_score(raw_dot):
    """Convert raw cosine dot product [-1, 1] → similarity [0.000, 1.000]"""
    return round(max(0.0, min(1.0, (raw_dot + 1) / 2)), 3)

def _get_embedding(model, image_path):
    tensor = _load_image(image_path)
    with torch.no_grad():
        emb = model(tensor)
    return emb.squeeze()   # [128]

def _load_db():
    if os.path.exists(DB_FILE):
        with open(DB_FILE) as f:
            return json.load(f)
    return {}

def _save_db(db):
    with open(DB_FILE, "w") as f:
        json.dump(db, f, indent=2)


# ── Public API ────────────────────────────────────────────────

def load_model():
    """Build and return the DNNet model. Call once, reuse everywhere."""
    print("Loading DNNet model...")
    model = DNNet()
    model.eval()
    print("Model ready.\n")
    return model


def register_dog(model, image_path, dog_name):
    """
    Register a dog's noseprint into the database.

        register_dog(model, "buddy_nose.jpg", "Buddy")
    """
    emb          = _get_embedding(model, image_path)
    db           = _load_db()
    db[dog_name] = emb.tolist()
    _save_db(db)
    print(f"Registered  :  {dog_name}  ({image_path})")


def identify_dog(model, query_path):
    """
    Identify a nose photo by matching against all registered dogs.

        identify_dog(model, "unknown_nose.jpg")
    """
    db = _load_db()
    if not db:
        print("Database is empty. Register dogs first.")
        return

    query_emb = _get_embedding(model, query_path)

    results = []
    for name, stored in db.items():
        sim      = _to_score(torch.dot(query_emb, torch.tensor(stored)).item())
        is_match = sim >= MATCH_THRESHOLD
        results.append((name, sim, is_match))

    results.sort(key=lambda x: x[1], reverse=True)

    print(f"\nQuery  :  {query_path}\n")
    print(f"  {'Name':<22}  {'Similarity':>10}  {'Match':>6}")
    print(f"  {'─'*22}  {'─'*10}  {'─'*6}")
    for name, sim, is_match in results:
        tag = "YES" if is_match else "no"
        print(f"  {name:<22}  {sim:>10.3f}  {tag:>6}")

    best = results[0]
    print()
    if best[2]:
        print(f"  Identified as  :  {best[0]}")
        print(f"  Similarity     :  {best[1]:.3f}")
    else:
        print(f"  Not identified.")
        print(f"  Closest match  :  {best[0]}  ({best[1]:.3f})")
    print()


def compare_noses(model, path_a, path_b):
    """
    Directly compare two nose images.

        compare_noses(model, "nose1.jpg", "nose2.jpg")
    """
    emb_a      = _get_embedding(model, path_a)
    emb_b      = _get_embedding(model, path_b)
    similarity = _to_score(torch.dot(emb_a, emb_b).item())
    is_match   = similarity >= MATCH_THRESHOLD

    print(f"\nComparing:")
    print(f"  A          :  {path_a}")
    print(f"  B          :  {path_b}")
    print(f"  Similarity :  {similarity:.3f}")
    print(f"  Verdict    :  {'SAME DOG' if is_match else 'DIFFERENT DOGS'}\n")

    return similarity, is_match


# ── Run ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model = load_model()

    register_dog(model, "C:\\Users\\papu_\\Downloads\\archive\\1.jpg", "dog1")
    register_dog(model, "C:\\Users\\papu_\\Downloads\\archive\\2.jpg", "dog2")
    register_dog(model, "C:\\Users\\papu_\\Downloads\\archive\\3.jpg", "dog3")
    register_dog(model, "C:\\Users\\papu_\\Downloads\\archive\\4.jpg", "dog4")
    register_dog(model, "C:\\Users\\papu_\\Downloads\\archive\\5.jpg", "dog5")
    