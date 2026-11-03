import os
import yaml
import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import LRMModel, SimpleGradCAM, load_checkpoint

if __name__ == "__main__":
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    device = "cuda" if (config['project']['device'] == "cuda" and torch.cuda.is_available()) else "cpu"
    ckpt_path = os.path.join(config['training']['checkpoint_dir'], "best_model.pt")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Aucun poids trouvé à l'emplacement {ckpt_path}. Veuillez lancer `python train.py` d'abord.")

    print(f"[INFO] Chargement du modèle depuis {ckpt_path}...")
    model = LRMModel(
        backbone_name=config['model']['backbone_name'],
        pretrained=False,
        task=config['model']['task'],
        n_classes=config['model']['n_classes']
    )
    checkpoint = load_checkpoint(ckpt_path, device=device)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()

    # Sélection d'une image de test
    test_img_path = "./data/sample_0.jpg"
    raw_img = cv2.imread(test_img_path, cv2.IMREAD_COLOR)
    rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(config['data']['img_size'], config['data']['img_size']),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    tensor_img = transform(image=rgb_img)['image']

    # Inférence pure
    with torch.no_grad():
        logits = model(tensor_img.unsqueeze(0).to(device))
        pred_class = logits.argmax(dim=1).item()
        confidence = torch.softmax(logits, dim=1)[0, pred_class].item()

    print(f"[RÉSULTAT] Classe prédite : {pred_class} (Confiance : {confidence * 100:.2f}%)")

    # Génération Grad-CAM
    os.makedirs("./outputs", exist_ok=True)
    cam_gen = SimpleGradCAM(model)
    cam = cam_gen.generate(tensor_img, class_idx=pred_class, device=device)
    
    img_vis = cv2.resize(rgb_img, (config['data']['img_size'], config['data']['img_size']))
    heatmap = cv2.applyColorMap((cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_vis, 0.6, heatmap, 0.4, 0)
    
    output_path = "./outputs/gradcam_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    cam_gen.close()
    print(f"[INFO] Carte d'explicabilité sauvegardée dans {output_path}")
