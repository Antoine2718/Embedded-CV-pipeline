# Embedded Computer Vision Pipeline

A lightweight, modular PyTorch pipeline designed for image classification and segmentation, built with embedded deployment and rapid experimentation in mind.

## Installation
1. Clone the repository and navigate to the project directory.
2. Install the required dependencies :
``` bash
pip install -r requirements.txt
```
## Quick start : a step by step execution
### 1. Train thé model :
``` bash
python train.py
````
### 2. Run inference and generate a Grad-CAM overlay :
``` bash
python infer.py
````
## Custom Usage
Here is how you can import and use core components in your own scripts :
``` python
import torch
from model import LRMModel, ImageDataset, FocalLoss, Trainer

# 1. Initialize a model for classification
model = LRMModel(backbone_name='resnet34', pretrained=True, task='classification', n_classes=2)

# 2. Set up loss and optimizer
loss_fn = FocalLoss(gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

# 3. Train using the built-in Trainer wrapper
trainer = Trainer(model, optimizer, device='cuda', amp=True)
```
