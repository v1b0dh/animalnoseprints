import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms
import cv2
import numpy as np
import json
import os
import time
from PIL import Image, ImageEnhance

# ── Config ────────────────────────────────────────────────────
DB_FILE_V1 = "benchmark_db_v1.json"
DB_FILE_V2 = "benchmark_db_v2.json"
MATCH_THRESHOLD_V1 = 0.75
MATCH_THRESHOLD_V2 = 0.78

# ── Models ────────────────────────────────────────────────────

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
        return F.normalize(x, p=2, dim=1)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x * self.channel_gate(x)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        return x * spatial

class DNNetV2(nn.Module):
    def __init__(self):
        super().__init__()
        v2_model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
        self.backbone = v2_model.features
        self.compress = nn.Sequential(
            nn.Conv2d(1280, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.SiLU(inplace=True),
            nn.Conv2d(512, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.SiLU(inplace=True)
        )
        self.attention = CBAM(256)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.embed = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 128)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.compress(x)
        x = self.attention(x)
        x = self.gap(x)
        x = self.embed(x)
        return F.normalize(x, p=2, dim=1)

# ── Helpers ───────────────────────────────────────────────────

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std =[0.229, 0.224, 0.225])
])

def load_image(image_input):
    if isinstance(image_input, str):
        img = cv2.imread(image_input)
        if img is None:
            raise FileNotFoundError(f"Cannot load image: {image_input}")
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
    else:
        img_pil = image_input
    
    tensor = _transform(img_pil).unsqueeze(0)
    return tensor

def get_embedding(model, image_input):
    tensor = load_image(image_input)
    with torch.no_grad():
        start = time.time()
        emb = model(tensor)
        latency = time.time() - start
    return emb.squeeze(), latency

def calculate_similarity(emb_a, emb_b):
    raw_dot = torch.dot(emb_a, emb_b).item()
    return round(max(0.0, min(1.0, (raw_dot + 1) / 2)), 3)

# ── Benchmark Logic ───────────────────────────────────────────

def run_benchmark():
    print("Loading Models...")
    model_v1 = DNNet().eval()
    model_v2 = DNNetV2().eval()
    print("Models Loaded.\n")

    base_path = "C:\\Users\\papu_\\Downloads\\archive\\"
    images = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg"]
    
    # 1. Identity Test (Self Comparison)
    print("--- 1. Testing Identity & Latency ---")
    results = {"V1": {"latency": [], "sim": []}, "V2": {"latency": [], "sim": []}}
    
    for img_name in images:
        path = os.path.join(base_path, img_name)
        emb1_v1, lat1_v1 = get_embedding(model_v1, path)
        emb1_v2, lat1_v2 = get_embedding(model_v2, path)
        
        results["V1"]["latency"].append(lat1_v1)
        results["V2"]["latency"].append(lat1_v2)
        
        # Self similarity should be ~1.0
        results["V1"]["sim"].append(calculate_similarity(emb1_v1, emb1_v1))
        results["V2"]["sim"].append(calculate_similarity(emb1_v2, emb1_v2))

    avg_lat_v1 = np.mean(results["V1"]["latency"])
    avg_lat_v2 = np.mean(results["V2"]["latency"])
    print(f"Avg Latency V1 (ResNet152): {avg_lat_v1:.4f}s")
    print(f"Avg Latency V2 (EffNetV2):  {avg_lat_v2:.4f}s")
    print(f"Speedup: {avg_lat_v1/avg_lat_v2:.2f}x\n")

    # 2. Augmentation Robustness (Rotation & Lighting)
    print("--- 2. Testing Robustness (Rotation & Lighting) ---")
    test_img_path = os.path.join(base_path, "1.jpg")
    pil_img = Image.open(test_img_path).convert("RGB")
    
    anch_v1, _ = get_embedding(model_v1, test_img_path)
    anch_v2, _ = get_embedding(model_v2, test_img_path)

    tests = [
        ("Rotate 15°", pil_img.rotate(15)),
        ("Rotate 30°", pil_img.rotate(30)),
        ("Bright +30%", ImageEnhance.Brightness(pil_img).enhance(1.3)),
        ("Contrast +30%", ImageEnhance.Contrast(pil_img).enhance(1.3)),
    ]

    print(f"{'Test Type':<15} | {'V1 Sim':<10} | {'V2 Sim':<10} | {'Winner'}")
    print("-" * 50)
    for name, aug_img in tests:
        emb_v1, _ = get_embedding(model_v1, aug_img)
        emb_v2, _ = get_embedding(model_v2, aug_img)
        
        sim_v1 = calculate_similarity(anch_v1, emb_v1)
        sim_v2 = calculate_similarity(anch_v2, emb_v2)
        winner = "V2" if sim_v2 > sim_v1 else "V1"
        print(f"{name:<15} | {sim_v1:<10.3f} | {sim_v2:<10.3f} | {winner}")
    print()

    # 3. Discriminative Power (False Positives)
    print("--- 3. Testing Discriminative Power (Diff Dogs) ---")
    # Compare Dog 1 vs Dog 2, 3, 4, 5
    d1_v1, _ = get_embedding(model_v1, os.path.join(base_path, "1.jpg"))
    d1_v2, _ = get_embedding(model_v2, os.path.join(base_path, "1.jpg"))
    
    diff_sims_v1 = []
    diff_sims_v2 = []
    
    for i in range(2, 6):
        path = os.path.join(base_path, f"{i}.jpg")
        other_v1, _ = get_embedding(model_v1, path)
        other_v2, _ = get_embedding(model_v2, path)
        
        diff_sims_v1.append(calculate_similarity(d1_v1, other_v1))
        diff_sims_v2.append(calculate_similarity(d1_v2, other_v2))
    
    print(f"Avg Cross-Dog Similarity V1: {np.mean(diff_sims_v1):.3f}")
    print(f"Avg Cross-Dog Similarity V2: {np.mean(diff_sims_v2):.3f}")
    print("(Lower is better - indicates better separation between different dogs)\n")

if __name__ == "__main__":
    run_benchmark()
