#!/bin/bash
set -e

echo "=== 1. Installation des dépendances ==="
pip install -r requirements.txt

echo -e "\n=== 2. Lancement de l'entraînement du modèle ==="
python train.py

echo -e "\n=== 3. Exécution de l'inférence et Grad-CAM ==="
python infer.py

echo -e "\n[SUCCÈS] Projet exécuté de bout en bout ! Consultez le dossier outputs/."
