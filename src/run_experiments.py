import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from pytorch_metric_learning import losses, samplers

from src.data import load_lfw_splits, FaceImageDataset
from src.model import FaceEmbeddingNet, set_trainable_layers, set_stage_train_mode
from src.metrics import extract_embeddings, choose_threshold_at_far, evaluate_with_threshold, plot_similarity_distributions


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ARTIFACTS = ROOT / "artifacts"
RESULTS.mkdir(exist_ok=True)
ARTIFACTS.mkdir(exist_ok=True)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_stage(model, arcface, dataset, device, epochs, train_layers, head_lr, backbone_lr, loss_lr,
                batch_size=32, images_per_identity=4):
    set_trainable_layers(model, train_layers)
    labels = dataset.labels.tolist()
    sampler = samplers.MPerClassSampler(
        labels, m=images_per_identity, batch_size=batch_size,
        length_before_new_iter=max(len(dataset), batch_size)
    )
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        drop_last=True, num_workers=2 if device.type == "cuda" else 0)
    params = [{"params": [p for p in model.head.parameters() if p.requires_grad], "lr": head_lr}]
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
    if backbone_params:
        params.append({"params": backbone_params, "lr": backbone_lr})
    params.append({"params": [p for p in arcface.parameters() if p.requires_grad], "lr": loss_lr})
    optimizer = torch.optim.Adam(params)
    history = []
    for epoch in range(epochs):
        set_stage_train_mode(model, train_layers)
        total, steps = 0.0, 0
        for images, targets in loader:
            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            embeddings = model(images)
            loss = arcface(embeddings, targets)
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            steps += 1
        average = total / max(steps, 1)
        history.append(average)
        print(f"  época {epoch + 1:02d}/{epochs} | loss ArcFace={average:.4f}")
    return history


@torch.no_grad()
def get_split_embeddings(model, images, labels, device, baseline=False, batch_size=64):
    dataset = FaceImageDataset(images, labels, train=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return extract_embeddings(model, loader, device, baseline=baseline)


def main():
    seed = 42
    seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    print("\n[1/5] Baixando/carregando LFW e separando identidades...")
    data = load_lfw_splits(seed=seed)
    print(f"Treino: {len(data.train_images)} imagens / {len(data.train_identity_names)} identidades")
    print(f"Validação: {len(data.val_images)} imagens / {len(data.val_identity_names)} identidades")
    print(f"Teste: {len(data.test_images)} imagens / {len(data.test_identity_names)} identidades")
    print("As identidades de validação e teste não participam do treinamento.")

    val_ds = FaceImageDataset(data.val_images, data.val_labels, train=False)
    test_ds = FaceImageDataset(data.test_images, data.test_labels, train=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

    print("\n[2/5] E01 — baseline ImageNet, sem fine-tuning...")
    model = FaceEmbeddingNet(embedding_dim=128, pretrained=True).to(device)
    base_val_emb, base_val_y = extract_embeddings(model, val_loader, device, baseline=True)
    base_test_emb, base_test_y = extract_embeddings(model, test_loader, device, baseline=True)
    base_threshold = choose_threshold_at_far(base_val_emb, base_val_y, target_far=0.01)
    base_metrics = evaluate_with_threshold(base_test_emb, base_test_y, base_threshold)
    base_metrics.update({"experiment": "E01_ImageNet_baseline", "strategy": "backbone congelado, sem treino"})
    print("  threshold selecionado na validação:", round(base_threshold, 4))
    print("  métricas de teste:", {k: round(v, 4) if isinstance(v, float) else v for k, v in base_metrics.items()})

    print("\n[3/5] E02 — treinamento da cabeça de embeddings com ArcFace...")
    train_ds = FaceImageDataset(data.train_images, data.train_labels, train=True)
    model = FaceEmbeddingNet(embedding_dim=128, pretrained=True).to(device)
    arcface = losses.ArcFaceLoss(
        num_classes=len(data.train_identity_names), embedding_size=128, margin=28.6, scale=30
    ).to(device)
    history = []
    history_head = train_stage(model, arcface, train_ds, device, epochs=3, train_layers=(),
                               head_lr=1e-3, backbone_lr=1e-4, loss_lr=1e-2)
    history.extend(history_head)

    # Avalia E02 antes de liberar qualquer camada do backbone.
    head_val_emb, head_val_y = get_split_embeddings(model, data.val_images, data.val_labels, device)
    head_test_emb, head_test_y = get_split_embeddings(model, data.test_images, data.test_labels, device)
    head_threshold = choose_threshold_at_far(head_val_emb, head_val_y, target_far=0.01)
    head_metrics = evaluate_with_threshold(head_test_emb, head_test_y, head_threshold)
    head_metrics.update({"experiment": "E02_ArcFace_head_only", "strategy": "cabeça treinada; backbone congelado"})
    print("  E02 threshold (validação):", round(head_threshold, 4))
    print("  E02 métricas de teste:", {k: round(v, 4) if isinstance(v, float) else v for k, v in head_metrics.items()})

    print("\n[4/5] E03 — fine-tuning parcial de layer4 + cabeça...")
    history_ft = train_stage(model, arcface, train_ds, device, epochs=5, train_layers=("layer4",),
                             head_lr=3e-4, backbone_lr=2e-5, loss_lr=5e-3)
    history.extend(history_ft)

    val_emb, val_y = get_split_embeddings(model, data.val_images, data.val_labels, device)
    test_emb, test_y = get_split_embeddings(model, data.test_images, data.test_labels, device)
    threshold = choose_threshold_at_far(val_emb, val_y, target_far=0.01)
    ft_metrics = evaluate_with_threshold(test_emb, test_y, threshold)
    ft_metrics.update({"experiment": "E03_ArcFace_layer4", "strategy": "cabeça + fine-tuning parcial de layer4"})
    print("  E03 threshold selecionado na validação:", round(threshold, 4))
    print("  E03 métricas de teste:", {k: round(v, 4) if isinstance(v, float) else v for k, v in ft_metrics.items()})

    print("\n[5/5] Salvando resultados agregados e artefatos locais...")
    metrics = pd.DataFrame([base_metrics, head_metrics, ft_metrics])
    metrics.to_csv(RESULTS / "metrics.csv", index=False)
    with open(RESULTS / "experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "seed": seed,
            "device": str(device),
            "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
            "dataset": "Labeled Faces in the Wild (LFW)",
            "split": "por identidade: 70% treino / 15% validação / 15% teste, seed fixa",
            "threshold_selection": "quantil 99% das similaridades de impostores na validação; não ajustado no teste",
            "training_epochs": {"head_only": 3, "layer4_finetuning": 5},
            "results": [base_metrics, ft_metrics],
            "caveat": "O limiar e as métricas são estimativas do protocolo e do conjunto utilizado; interpretar com limitações do LFW."
        }, f, ensure_ascii=False, indent=2)
    pd.DataFrame({"epoch": np.arange(1, len(history) + 1), "arcface_loss": history}).to_csv(
        RESULTS / "training_loss.csv", index=False
    )
    plot_similarity_distributions({
        "E01 — ImageNet": (base_test_emb, base_test_y, base_threshold),
        "E02 — ArcFace head-only": (head_test_emb, head_test_y, head_threshold),
        "E03 — ArcFace + layer4": (test_emb, test_y, threshold),
    }, RESULTS / "similarity_distributions.png")
    torch.save({
        "model_state_dict": model.state_dict(),
        "embedding_dim": 128,
        "threshold_from_validation": threshold,
        "seed": seed,
        "dataset": "LFW",
        "training_strategy": "ArcFace head-only 3 epochs + layer4 partial fine-tuning 5 epochs",
    }, ARTIFACTS / "lfw_resnet18_finetuned.pt")
    print(f"Resultados agregados: {RESULTS}")
    print(f"Checkpoint local (não publicar sem verificar termos/licenças): {ARTIFACTS / 'lfw_resnet18_finetuned.pt'}")
    print("\nIMPORTANTE: não escreva conclusão antes de examinar metrics.csv e o gráfico.")


if __name__ == "__main__":
    main()
