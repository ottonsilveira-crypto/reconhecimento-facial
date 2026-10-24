import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix


@torch.no_grad()
def extract_embeddings(model, loader, device, baseline=False):
    model.eval()
    vectors, labels = [], []
    for images, targets in loader:
        images = images.to(device)
        z = model.embed_imagenet_baseline(images) if baseline else model(images)
        vectors.append(F.normalize(z, p=2, dim=1).cpu())
        labels.append(targets.cpu())
    return torch.cat(vectors).numpy(), torch.cat(labels).numpy()


def pair_scores(embeddings, labels):
    """Cosine similarity for every unordered pair, with same-person ground truth."""
    embeddings = np.asarray(embeddings, dtype=np.float32)
    labels = np.asarray(labels)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True).clip(min=1e-12)
    embeddings = embeddings / norms
    sims = embeddings @ embeddings.T
    i, j = np.triu_indices(len(labels), k=1)
    return sims[i, j], (labels[i] == labels[j])


def choose_threshold_at_far(val_embeddings, val_labels, target_far=0.01):
    scores, same = pair_scores(val_embeddings, val_labels)
    impostor_scores = scores[~same]
    if len(impostor_scores) == 0 or same.sum() == 0:
        raise ValueError("Validação precisa conter pares genuínos e impostores.")
    # Freeze threshold using validation only. Higher cosine similarity means more likely same identity.
    return float(np.quantile(impostor_scores, 1.0 - target_far))


def evaluate_with_threshold(embeddings, labels, threshold):
    scores, same = pair_scores(embeddings, labels)
    predicted_same = scores >= threshold
    y_true = same.astype(np.uint8)
    y_pred = predicted_same.astype(np.uint8)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    impostors = scores[~same]
    genuine = scores[same]
    return {
        "threshold": float(threshold),
        "num_images": int(len(labels)),
        "num_pairs": int(len(scores)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) == 2 else float("nan"),
        "far": float(fp / max(fp + tn, 1)),
        "frr": float(fn / max(fn + tp, 1)),
        "genuine_similarity_mean": float(genuine.mean()) if len(genuine) else float("nan"),
        "impostor_similarity_mean": float(impostors.mean()) if len(impostors) else float("nan"),
        "true_negative": int(tn), "false_positive": int(fp),
        "false_negative": int(fn), "true_positive": int(tp),
    }


def plot_similarity_distributions(embedding_sets, output_path):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(embedding_sets), figsize=(6 * len(embedding_sets), 4))
    if len(embedding_sets) == 1:
        axes = [axes]
    for ax, (name, (emb, labels, threshold)) in zip(axes, embedding_sets.items()):
        scores, same = pair_scores(emb, labels)
        ax.hist(scores[~same], bins=50, alpha=0.6, density=True, label="Pessoas diferentes")
        ax.hist(scores[same], bins=50, alpha=0.6, density=True, label="Mesma pessoa")
        ax.axvline(threshold, color="black", linestyle="--", label="Threshold (validação)")
        ax.set_title(name)
        ax.set_xlabel("Similaridade de cosseno")
        ax.set_ylabel("Densidade")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
