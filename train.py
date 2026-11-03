import os
import yaml
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import set_seed, ImageDataset, LRMModel, FocalLoss, save_checkpoint

class Trainer:
    def __init__(self, model: nn.Module, optimizer, device: str = "cuda", amp: bool = True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.amp = amp and (device == "cuda")
        self.scaler = torch.amp.GradScaler('cuda', enabled=self.amp)

    def train_epoch(self, train_loader: DataLoader, loss_fn, max_grad_norm: float = 1.0):
        self.model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc="Entraînement", leave=False)
        for imgs, targets in pbar:
            imgs = imgs.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=self.amp):
                logits = self.model(imgs)
                loss = loss_fn(logits, targets)
            self.scaler.scale(loss).backward()
            if max_grad_norm:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            running_loss += loss.item() * imgs.size(0)
            pbar.set_postfix({'loss': f"{running_loss / (pbar.n + 1e-9):.4f}"})
        return running_loss / len(train_loader.dataset)

def prepare_dataset(data_dir="./data"):
    os.makedirs(data_dir, exist_ok=True)
    items = []
    for i in range(12):
        img_path = os.path.join(data_dir, f"sample_{i}.jpg")
        if not os.path.exists(img_path):
            img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
            cv2.imwrite(img_path, img)
        items.append({'img_path': img_path, 'label': i % 2})
    return items[:10], items[10:]

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    set_seed(config['project']['seed'])
    device = "cuda" if (config['project']['device'] == "cuda" and torch.cuda.is_available()) else "cpu"
    print(f"[INFO] Utilisation du processeur/accélérateur : {device}")

    train_items, val_items = prepare_dataset()
    train_ds = ImageDataset(train_items, task=config['model']['task'], img_size=config['data']['img_size'], augment=True)
    val_ds = ImageDataset(val_items, task=config['model']['task'], img_size=config['data']['img_size'], augment=False)

    train_loader = DataLoader(train_ds, batch_size=config['data']['batch_size'], shuffle=True, num_workers=config['data']['num_workers'])
    
    model = LRMModel(
        backbone_name=config['model']['backbone_name'],
        pretrained=config['model']['pretrained'],
        task=config['model']['task'],
        n_classes=config['model']['n_classes']
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['training']['lr'], weight_decay=config['training']['weight_decay'])
    loss_fn = FocalLoss(gamma=config['training']['focal_gamma'])

    trainer = Trainer(model, optimizer, device=device, amp=True)

    print("\n[INFO] Lancement de l'apprentissage...")
    for ep in range(config['training']['epochs']):
        train_loss = trainer.train_epoch(train_loader, loss_fn, max_grad_norm=config['training']['max_grad_norm'])
        print(f"Époque [{ep+1}/{config['training']['epochs']}] - Loss d'entraînement : {train_loss:.4f}")

    ckpt_path = os.path.join(config['training']['checkpoint_dir'], "best_model.pt")
    save_checkpoint({'model_state': model.state_dict()}, ckpt_path)
    print(f"[INFO] Modèle sauvegardé avec succès dans {ckpt_path}")
