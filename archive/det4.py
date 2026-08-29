import torch
import torch.nn as nn
import torch.nn.functional as F
import timm # Required for TinyViT backbone
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
# -- Advanced Preprocessing with CLAHE --
class CLAHETransform:
    def __call__(self, img):
        # Convert PIL to CV2
        img_np = np.array(img)
        # Apply CLAHE to enhance ridge details
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)

_transform = transforms.Compose([
    CLAHETransform(), # Enhance patterns first
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# -- DNNetV2 SOTA Architecture --
class DNNetV2_SOTA(nn.Module):
    def __init__(self):
        super().__init__()
        # 1. Official SOTA Backbone: Tiny Vision Transformer (TinyViT-21M)
        # This replaces EfficientNet to capture long-range ridge dependencies.
        self.backbone = timm.create_model('tiny_vit_21m_224', pretrained=True, num_classes=0)
        
        # 2. Embedding Head (Expanded to 1024 as per official research)
        # Features are pulled from the 576-d TinyViT output
        self.head = nn.Sequential(
            nn.Linear(576, 1024),
            nn.BatchNorm1d(1024),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(1024, 1024) # Final High-Res Embedding
        )

    def forward(self, x):
        # Global Self-Attention happens inside the TinyViT blocks
        features = self.backbone(x) 
        embedding = self.head(features)
        return F.normalize(embedding, p=2, dim=1)

# -- Usage --
def load_sota_model():
    model = DNNetV2_SOTA()
    model.eval()
    return model