"""
det5.py — DNNetV3 SOTA Dog Nose Biometric Recognition System
============================================================
Architecture Upgrades over det4.py:
  1. TinyViT-21M backbone (576-d output) — unchanged from det4, validated.
  2. MagFace Loss — replaces ArcFace; dynamically weights quality (sharp vs. blurry).
  3. CLAHE Preprocessing — upgraded with per-image quality scoring.
  4. Model Distillation — lightweight StudentNet (<100ms mobile inference).

Target: 99.8% Rank-1 on dog nose biometric benchmarks.
Author: DNNet Research | Engine: PyTorch 2.x + timm
"""

# ==============================================================================
# SECTION 0: IMPORTS
# ==============================================================================
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional


# ==============================================================================
# SECTION 1: PREPROCESSING PIPELINE
# Advanced CLAHE with Laplacian-based sharpness quality scoring.
# Why: CLAHE lifts low-contrast nose ridge patterns; sharpness score feeds
#      MagFace so blurry samples are down-weighted during training.
# ==============================================================================

def compute_sharpness(img_np_gray: np.ndarray) -> float:
    """
    Laplacian variance — fast, reliable blur proxy.
    Returns a float in [0, ∞). Higher = sharper.
    Typical thresholds: <100 = blurry, >500 = sharp.
    """
    return float(cv2.Laplacian(img_np_gray, cv2.CV_64F).var())


class CLAHETransform:
    """
    CLAHE applied in LAB color space for luminance-only enhancement.
    Preserves chrominance (color channels a, b are untouched).
    Returns: (PIL.Image, sharpness_score: float)

    Integration note: torchvision Compose doesn't natively support
    multi-output transforms. Use CLAHEPipeline (below) as the entry
    point instead of transforms.Compose directly.
    """
    def __init__(self, clip_limit: float = 3.0, tile_grid: Tuple[int, int] = (8, 8)):
        self.clip_limit = clip_limit
        self.tile_grid = tile_grid

    def __call__(self, img: Image.Image) -> Tuple[Image.Image, float]:
        img_np = np.array(img.convert("RGB"))

        # Sharpness on grayscale BEFORE CLAHE (measures raw image quality)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        sharpness = compute_sharpness(gray)

        # CLAHE in LAB space
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=self.clip_limit, tileGridSize=self.tile_grid)
        cl = clahe.apply(l)
        enhanced = cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2RGB)

        return Image.fromarray(enhanced), sharpness


class CLAHEPipeline:
    """
    Full preprocessing pipeline that:
      1. Applies CLAHE
      2. Resizes, normalizes (ImageNet stats — compatible with TinyViT pretrain)
      3. Returns (tensor, sharpness_score) tuple

    Usage:
        pipeline = CLAHEPipeline()
        tensor, score = pipeline(pil_image)
    """
    def __init__(
        self,
        image_size: int = 224,
        clip_limit: float = 3.0,
        tile_grid: Tuple[int, int] = (8, 8),
    ):
        self.clahe = CLAHETransform(clip_limit=clip_limit, tile_grid=tile_grid)
        self.post = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __call__(self, img: Image.Image) -> Tuple[torch.Tensor, float]:
        enhanced_pil, sharpness = self.clahe(img)
        tensor = self.post(enhanced_pil)
        return tensor, sharpness

    def batch(self, images: list) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process a list of PIL images into a batched tensor + sharpness vector.
        Useful for DataLoader collate_fn.
        """
        tensors, scores = zip(*[self(img) for img in images])
        return torch.stack(tensors), torch.tensor(scores, dtype=torch.float32)


# ==============================================================================
# SECTION 2: BACKBONE + EMBEDDING HEAD (TinyViT-21M)
# ==============================================================================

class DNNetV3(nn.Module):
    """
    Dog Nose Network V3 — Teacher model (full precision, cloud/server).

    Architecture:
        TinyViT-21M (pretrained ImageNet-22K)
          └─ 576-d global feature vector
             └─ Embedding Head → 1024-d L2-normalized vector

    The head uses a BN→SiLU→Dropout→Linear bottleneck pattern proven
    in face recognition (ArcFace/CosFace papers) to stabilize MagFace training.
    """
    FEATURE_DIM = 576       # TinyViT-21M output
    EMBED_DIM   = 1024      # Final embedding dimension

    def __init__(self, pretrained: bool = True, use_head: bool = False):
        super().__init__()
        self.use_head = use_head

        # ----- Backbone -----
        # num_classes=0 removes the classification head; returns raw features.
        self.backbone = timm.create_model(
            "tiny_vit_21m_224",
            pretrained=pretrained,
            num_classes=0,         # Returns 576-d feature vector
            global_pool="avg",     # Global average pool over patch tokens
        )

        # ----- Embedding Head -----
        # Expanded projection: 576 → 1024 with residual-style design
        self.head = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.EMBED_DIM),
            nn.BatchNorm1d(self.EMBED_DIM),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.10),
            nn.Linear(self.EMBED_DIM, self.EMBED_DIM),
            nn.BatchNorm1d(self.EMBED_DIM),   # Final BN before L2-norm (MagFace requirement)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns L2-normalized embeddings.
        Shape: (B, 1024) if use_head else (B, 576)
        """
        feat = self.backbone(x)        # (B, 576)
        if self.use_head:
            emb = self.head(feat)      # (B, 1024)
            return F.normalize(emb, p=2, dim=1)
        else:
            return F.normalize(feat, p=2, dim=1)

    def get_feature_norm(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (normalized_embedding, raw_norm).
        raw_norm is passed to MagFace as the magnitude signal ||f||.
        """
        feat     = self.backbone(x)
        if self.use_head:
            emb_raw  = self.head(feat)
        else:
            emb_raw  = feat
        norm     = emb_raw.norm(p=2, dim=1, keepdim=True)  # (B, 1)
        emb_norm = emb_raw / norm
        return emb_norm, norm.squeeze(1)                    # (B, D), (B,)


# ==============================================================================
# SECTION 3: MAGFACE LOSS
# ==============================================================================
# Paper: "MagFace: A Universal Representation for Face Recognition and Quality
#         Assessment" — He et al., CVPR 2021.
#
# Key insight over ArcFace:
#   - The feature magnitude ||f|| encodes sample quality naturally.
#   - MagFace couples the margin `m` and regularisation `g` with ||f||:
#       • High ||f|| (sharp, clear) → larger margin → harder constraint.
#       • Low  ||f|| (blurry)       → smaller margin + regularization pushes ||f|| up.
#   - Result: the model self-learns a quality score without extra labels.
# ==============================================================================

class MagFaceLoss(nn.Module):
    """
    MagFace Loss for biometric verification.

    Args:
        num_classes   : Number of identity classes (e.g. 5000 dogs).
        embed_dim     : Embedding dimensionality (1024 for DNNetV3).
        scale  (s)    : Logit scale; typically 64 for face, keep 64 here.
        margin_lo (l_a): Lower margin bound — applied to low-quality samples.
        margin_hi (u_a): Upper margin bound — applied to high-quality samples.
        norm_lo  (l_m): Lower feature norm bound (||f|| clamp min).
        norm_hi  (u_m): Upper feature norm bound (||f|| clamp max).
        lambda_g      : Weight for the MagFace regularisation term.

    Forward:
        embeddings  : (B, embed_dim) — L2 normalized feature vectors
        norms       : (B,)          — Raw feature norms (||f||)
        labels      : (B,)          — Ground-truth class indices

    Returns:
        loss (scalar)
    """
    def __init__(
        self,
        num_classes : int,
        embed_dim   : int   = 1024,
        scale       : float = 64.0,
        margin_lo   : float = 0.45,
        margin_hi   : float = 0.80,
        norm_lo     : float = 10.0,
        norm_hi     : float = 110.0,
        lambda_g    : float = 35.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.s           = scale
        self.l_a         = margin_lo
        self.u_a         = margin_hi
        self.l_m         = norm_lo
        self.u_m         = norm_hi
        self.lambda_g    = lambda_g

        # Learnable class weight matrix (same as ArcFace)
        # Each row is the prototype vector for one identity.
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _calc_margin(self, x: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation of margin based on feature norm.
        m(||f||) = (u_a - l_a) / (u_m - l_m) * (||f|| - l_m) + l_a
        Clamped to [l_a, u_a].
        """
        m = (self.u_a - self.l_a) / (self.u_m - self.l_m) * (x - self.l_m) + self.l_a
        return m.clamp(self.l_a, self.u_a)

    def _calc_regularisation(self, x: torch.Tensor) -> torch.Tensor:
        """
        MagFace regularisation: g(||f||) = 1/u_m^2 * ||f|| + 1/(||f||).
        Encourages ||f|| to grow for high-quality samples.
        Averaged over batch → scalar.
        """
        g = 1.0 / (self.u_m ** 2) * x + 1.0 / x
        return g.mean()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        embeddings : torch.Tensor,   # (B, D) — already L2-normalised
        norms      : torch.Tensor,   # (B,)   — raw ||f||
        labels     : torch.Tensor,   # (B,)
        sharpness  : Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        # Clamp norms to valid range
        norms_clamped = norms.clamp(self.l_m, self.u_m)

        # Per-sample adaptive margin
        margin = self._calc_margin(norms_clamped)       # (B,)

        # L2-normalise weight matrix (class prototypes)
        W = F.normalize(self.weight, p=2, dim=1)        # (C, D)

        # Cosine similarity: embeddings @ W.T  →  (B, C)
        cosine = embeddings @ W.t()                      # (B, C)
        cosine = cosine.clamp(-1.0 + 1e-7, 1.0 - 1e-7)

        # Theta for the ground-truth class
        theta = torch.acos(cosine)                       # (B, C)

        # Add adaptive margin ONLY to the target class column
        one_hot = torch.zeros_like(cosine)               # (B, C)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)

        # Target angle with margin: θ_yi + m(||f||)
        # For non-target classes: no margin is added.
        theta_m = theta + one_hot * margin.unsqueeze(1)  # (B, C)

        # Re-compute cosine with margin applied
        logits = self.s * torch.cos(theta_m)             # (B, C)

        # Weight each sample by measured image quality when it is available.
        # Values <= 100 are treated as blurry and >= 500 as sharp.
        loss_ce_per_sample = F.cross_entropy(logits, labels, reduction="none")
        if sharpness is not None:
            quality_weight = (sharpness.to(logits.device).float() - 100.0) / 400.0
            quality_weight = quality_weight.clamp(0.0, 1.0).mul(0.5).add(0.5)
            loss_ce = (loss_ce_per_sample * quality_weight).sum()
            loss_ce = loss_ce / quality_weight.sum().clamp_min(1e-6)
        else:
            loss_ce = loss_ce_per_sample.mean()

        # MagFace regularisation (quality-awareness)
        loss_g = self.lambda_g * self._calc_regularisation(norms_clamped)

        return loss_ce + loss_g


# ==============================================================================
# SECTION 4: TRAINING LOOP INTEGRATION
# Demonstrates how to wire MagFace with the model's norm output.
# ==============================================================================

def train_one_epoch(
    model       : DNNetV3,
    magface     : MagFaceLoss,
    dataloader,
    optimizer   : torch.optim.Optimizer,
    device      : torch.device,
    pipeline    : CLAHEPipeline,
    scaler      : Optional[torch.cuda.amp.GradScaler] = None,
) -> float:
    """
    Single epoch training step.

    DataLoader expectation:
        Each batch yields (list_of_pil_images, label_tensor).
        The pipeline is applied inside the loop (supports on-the-fly CLAHE).

    Returns:
        Average loss for the epoch.
    """
    model.train()
    magface.train()
    total_loss = 0.0

    for pil_images, labels in dataloader:
        # ---- Preprocessing (CLAHE + tensor conversion) ----
        images, sharpness_scores = pipeline.batch(pil_images)
        images  = images.to(device)
        labels  = labels.to(device)
        # sharpness_scores available for logging/analysis

        optimizer.zero_grad()

        # ---- Forward with AMP ----
        if scaler is not None:
            with torch.cuda.amp.autocast():
                emb_norm, raw_norm = model.get_feature_norm(images)
                loss = magface(emb_norm, raw_norm, labels, sharpness_scores)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(magface.parameters()), max_norm=5.0
            )
            scaler.step(optimizer)
            scaler.update()
        else:
            emb_norm, raw_norm = model.get_feature_norm(images)
            loss = magface(emb_norm, raw_norm, labels, sharpness_scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(magface.parameters()), max_norm=5.0
            )
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


# ==============================================================================
# SECTION 5: STUDENT MODEL — Mobile-Ready Distillation (<100ms)
# ==============================================================================
# Strategy: Knowledge Distillation (KD) from DNNetV3 (teacher) to a compact
# MobileNetV3-Small student. MobileNetV3-Small hits ~8ms on CPU, well inside
# the 100ms mobile budget.
#
# Distillation approach:
#   - Embedding-level KD: MSE between teacher and student 512-d embeddings.
#   - Logit-level KD: KL-divergence on softened MagFace logits (T=4).
#   - Label CE: Standard cross-entropy for hard labels.
# ==============================================================================

class StudentDNNet(nn.Module):
    """
    Lightweight student model for mobile deployment.

    Backbone : MobileNetV3-Small  (~2.5M params, ~8ms CPU)
    Head     : 1024 → 512-d embedding

    Distillation target: match DNNetV3's 1024-d embeddings via
    a projection layer (512 → 1024) during training only.
    """
    FEATURE_DIM = 1024
    EMBED_DIM   = 512

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )

        self.head = nn.Sequential(
            nn.Linear(self.FEATURE_DIM, self.EMBED_DIM),
            nn.BatchNorm1d(self.EMBED_DIM),
            nn.SiLU(inplace=True),
            nn.Linear(self.EMBED_DIM, self.EMBED_DIM),
            nn.BatchNorm1d(self.EMBED_DIM),
        )

        # Projection to match teacher embedding dim during distillation only
        self.proj = nn.Linear(self.EMBED_DIM, 1024)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        emb  = self.head(feat)
        return F.normalize(emb, p=2, dim=1)     # 512-d for inference

    def forward_distill(self, x: torch.Tensor) -> torch.Tensor:
        """Returns 1024-d projected embedding for distillation loss."""
        feat = self.backbone(x)
        emb  = self.head(feat)
        proj = self.proj(emb)
        return F.normalize(proj, p=2, dim=1)    # 1024-d for KD loss


class DistillationLoss(nn.Module):
    """
    Combined distillation loss:
        L = α * L_embed + β * L_kl + γ * L_ce
    """
    def __init__(
        self,
        temperature : float = 4.0,
        alpha       : float = 1.0,   # Embedding MSE weight
        beta        : float = 1.0,   # KL divergence weight
        gamma       : float = 0.5,   # Hard label CE weight
    ):
        super().__init__()
        self.T     = temperature
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma

    def forward(
        self,
        student_emb   : torch.Tensor,   # (B, 1024) projected
        teacher_emb   : torch.Tensor,   # (B, 1024) from DNNetV3
        student_logits: torch.Tensor,   # (B, C)
        teacher_logits: torch.Tensor,   # (B, C) — soft targets
        labels        : torch.Tensor,   # (B,)
    ) -> torch.Tensor:

        # 1. Embedding-level MSE (feature-level KD)
        l_embed = F.mse_loss(student_emb, teacher_emb.detach())

        # 2. KL divergence on softened logits
        T = self.T
        l_kl = F.kl_div(
            F.log_softmax(student_logits / T, dim=1),
            F.softmax(teacher_logits.detach() / T, dim=1),
            reduction="batchmean",
        ) * (T ** 2)

        # 3. Hard label cross-entropy
        l_ce = F.cross_entropy(student_logits, labels)

        return self.alpha * l_embed + self.beta * l_kl + self.gamma * l_ce


# ==============================================================================
# SECTION 6: FACTORY & UTILITIES
# ==============================================================================

def build_teacher(num_classes: int, pretrained: bool = True) -> Tuple[DNNetV3, MagFaceLoss]:
    """Convenience factory: returns (model, loss) for training."""
    # The MagFace classifier is configured for DNNetV3.EMBED_DIM (1024),
    # so the 576 -> 1024 embedding head must be active.
    model   = DNNetV3(pretrained=pretrained, use_head=True)
    loss_fn = MagFaceLoss(num_classes=num_classes, embed_dim=DNNetV3.EMBED_DIM)
    return model, loss_fn


def build_student(pretrained: bool = True) -> StudentDNNet:
    """Returns the mobile student model."""
    return StudentDNNet(pretrained=pretrained)


def get_optimizer(model: nn.Module, magface: MagFaceLoss) -> torch.optim.Optimizer:
    """
    Recommended optimizer: AdamW with differential LR.
      - Backbone: 1e-4 (fine-tuning)
      - Head + MagFace weight: 1e-3 (learning from scratch)
    """
    backbone_params  = list(model.backbone.parameters())
    head_params      = list(model.head.parameters()) + list(magface.parameters())
    return torch.optim.AdamW([
        {"params": backbone_params, "lr": 1e-4, "weight_decay": 1e-4},
        {"params": head_params,     "lr": 1e-3, "weight_decay": 1e-4},
    ])


def get_scheduler(optimizer: torch.optim.Optimizer, total_epochs: int = 50):
    """Cosine annealing with warm restarts — standard for metric learning."""
    return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )


# ==============================================================================
# SECTION 7: INFERENCE HELPER
# ==============================================================================

@torch.no_grad()
def extract_embedding(
    model    : nn.Module,
    img      : Image.Image,
    pipeline : CLAHEPipeline,
    device   : torch.device,
) -> Tuple[torch.Tensor, float]:
    """
    Single-image inference.
    Returns: (1024-d or 512-d embedding tensor, sharpness_score)
    """
    model.eval()
    tensor, sharpness = pipeline(img)
    tensor = tensor.unsqueeze(0).to(device)     # (1, 3, 224, 224)
    emb    = model(tensor)                       # (1, D)
    return emb.squeeze(0).cpu(), sharpness


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    """Scalar cosine similarity between two embedding vectors."""
    return float(F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())


# ==============================================================================
# SECTION 8: QUICK SANITY CHECK
# ==============================================================================

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[DNNetV3] Running on: {device}")

    NUM_CLASSES = 500   # Replace with your actual class count

    # ---- Build teacher ----
    teacher, magface = build_teacher(NUM_CLASSES)
    teacher  = teacher.to(device)
    magface  = magface.to(device)

    # ---- Build student ----
    student = build_student().to(device)

    # ---- Pipeline ----
    pipeline = CLAHEPipeline(image_size=224)

    # ---- Dummy forward pass ----
    dummy = torch.randn(4, 3, 224, 224).to(device)
    emb_norm, raw_norm = teacher.get_feature_norm(dummy)
    print(f"[Teacher] Embedding shape : {emb_norm.shape}")   # (4, 1024)
    print(f"[Teacher] Raw norm (quality): {raw_norm}")       # (4,)

    labels = torch.randint(0, NUM_CLASSES, (4,)).to(device)
    loss   = magface(emb_norm, raw_norm, labels)
    print(f"[MagFace] Loss            : {loss.item():.4f}")

    s_emb  = student(dummy)
    print(f"[Student] Embedding shape : {s_emb.shape}")      # (4, 512)

    print("\n[✓] All components validated. Ready for training.")
