import os
import random
from typing import Dict, Optional
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def save_checkpoint(state: Dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_checkpoint(path: str, device: str = "cpu"):
    return torch.load(path, map_location=device)

class ImageDataset(Dataset):
    def __init__(self, items: list, task: str = "classification", img_size: int = 224, augment: bool = False):
        self.items = items
        self.task = task
        self.img_size = img_size
        self.augment = augment
        self._build_transforms()

    def _build_transforms(self):
        base = [
            A.Resize(self.img_size, self.img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ]
        if self.augment:
            aug = [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.3)
            ]
            self.transform = A.Compose(aug + base)
        else:
            self.transform = A.Compose(base)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]
        img = cv2.imread(it['img_path'], cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Impossible de charger l'image : {it['img_path']}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        if self.task == "classification":
            label = int(it['label'])
            data = self.transform(image=img)
            return data['image'], torch.tensor(label, dtype=torch.long)
        else:
            raise ValueError(f"Tâche non supportée : {self.task}")

class Backbone(nn.Module):
    def __init__(self, model_name: str = "resnet34", pretrained: bool = True):
        super().__init__()
        self.body = timm.create_model(model_name, pretrained=pretrained, features_only=True)
        self.feature_info = self.body.feature_info
        self.out_ch = self.feature_info[-1]['num_chs']

    def forward(self, x):
        return self.body(x)

class ClassificationHead(nn.Module):
    def __init__(self, in_ch: int, n_classes: int, dropout: float = 0.25):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, max(128, in_ch // 4)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(max(128, in_ch // 4), n_classes)
        )

    def forward(self, feats):
        x = feats[-1]
        x = self.pool(x)
        return self.fc(x)

class LRMModel(nn.Module):
    def __init__(self, backbone_name: str = "resnet34", pretrained: bool = True, task: str = "classification", n_classes: int = 2):
        super().__init__()
        self.task = task
        self.backbone = Backbone(model_name=backbone_name, pretrained=pretrained)
        out_ch = self.backbone.out_ch
        if task == "classification":
            self.head = ClassificationHead(in_ch=out_ch, n_classes=n_classes)
        else:
            raise ValueError("Unsupported task")

    def forward(self, x):
        feats = self.backbone(x)
        out = self.head(feats)
        return out

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight=None):
        super().__init__()
        self.gamma = gamma

    def forward(self, logits, targets):
        logp = F.log_softmax(logits, dim=1)
        p = torch.exp(logp)
        loss = F.nll_loss(((1 - p) ** self.gamma) * logp, targets)
        return loss

class SimpleGradCAM:
    def __init__(self, model: nn.Module, target_layer_name: str = None):
        self.model = model
        self.model.eval()
        self.activations = None
        self.grads = None
        if target_layer_name is None:
            self.target = self._find_last_conv(self.model)
        else:
            self.target = dict(self.model.named_modules())[target_layer_name]
        self.hook_handles = []
        self._register_hooks()

    def _find_last_conv(self, model):
        last = None
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                last = m
        return last

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()
        def full_backward_hook(module, grad_in, grad_out):
            self.grads = grad_out[0].detach()
        self.hook_handles.append(self.target.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target.register_full_backward_hook(full_backward_hook))

    def generate(self, input_tensor: torch.Tensor, class_idx: Optional[int] = None, device='cpu'):
        input_tensor = input_tensor.to(device).unsqueeze(0)
        self.model.zero_grad()
        logits = self.model(input_tensor)
        if class_idx is None:
            class_idx = logits.argmax(dim=1).item()
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        weights = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(input_tensor.size(2), input_tensor.size(3)), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    def close(self):
        for h in self.hook_handles:
            h.remove()
