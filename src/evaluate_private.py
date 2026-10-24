"""Evaluate consented local photos without uploading them to Colab or GitHub.

Expected layout:
  data/private/<person_id>/*.jpg|png|jpeg
Each folder is one consenting person. Use at least 3 varied photos per person.
The script uses 70% of each person's images as enrollment/gallery and the rest as known queries.
Public LFW validation identities calibrate the query-to-gallery unknown threshold; public LFW test
identities estimate unknown false acceptance. Only aggregate metrics are written to results/.
"""
import json
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader

from src.data import load_lfw_splits, FaceImageDataset
from src.model import FaceEmbeddingNet
from src.metrics import extract_embeddings

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_DIR = ROOT / "data" / "private"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"
MODEL_PATH = ARTIFACTS / "lfw_resnet18_finetuned.pt"
YUNET_PATH = ARTIFACTS / "face_detection_yunet_2023mar.onnx"
YUNET_URL = "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"

# Alignment constants adapted from the professor's provided class notebook.
OUT_W, OUT_H = 94, 125
EYE_X, EYE_Y, EYE_DISTANCE = 47.6, 44.0, 44.2
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def make_detector():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    if not YUNET_PATH.exists():
        print("Baixando o detector YuNet (o arquivo ficará local em artifacts/)...")
        urllib.request.urlretrieve(YUNET_URL, YUNET_PATH)
    return cv2.FaceDetectorYN.create(str(YUNET_PATH), "", (320, 320), 0.7, 0.3, 50)


def align_image(image_rgb, detector):
    h, w = image_rgb.shape[:2]
    detector.setInputSize((w, h))
    _, faces = detector.detect(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if faces is None or len(faces) == 0:
        raise ValueError("Nenhum rosto detectado; use uma foto nítida, frontal e bem iluminada.")
    face = sorted(faces, key=lambda f: -f[2] * f[3])[0]
    p1, p2 = face[4:6], face[6:8]
    eye_left, eye_right = (p1, p2) if p1[0] < p2[0] else (p2, p1)
    center = (eye_left + eye_right) / 2
    dx, dy = eye_right - eye_left
    distance = max(float(np.hypot(dx, dy)), 1e-6)
    angle = np.degrees(np.arctan2(dy, dx))
    scale = EYE_DISTANCE / distance
    matrix = cv2.getRotationMatrix2D((float(center[0]), float(center[1])), angle, scale)
    matrix[0, 2] += EYE_X - center[0]
    matrix[1, 2] += EYE_Y - center[1]
    return cv2.warpAffine(image_rgb, matrix, (OUT_W, OUT_H), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def load_private(detector, seed=42):
    people = []
    for folder in sorted(p for p in PRIVATE_DIR.iterdir() if p.is_dir()):
        paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if len(paths) < 3:
            continue
        rng = np.random.default_rng(seed + len(people))
        order = rng.permutation(len(paths))
        n_gallery = max(2, int(0.7 * len(paths)))
        if n_gallery >= len(paths):
            n_gallery = len(paths) - 1
        gallery, queries = [], []
        for idx in order[:n_gallery]:
            path = paths[int(idx)]
            bgr = cv2.imread(str(path))
            if bgr is None:
                print(f"Aviso: não foi possível ler {path.name}; ignorada.")
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                gallery.append(align_image(rgb, detector))
            except ValueError as exc:
                print(f"Aviso em {path.name}: {exc}")
        for idx in order[n_gallery:]:
            path = paths[int(idx)]
            bgr = cv2.imread(str(path))
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            try:
                queries.append(align_image(rgb, detector))
            except ValueError as exc:
                print(f"Aviso em {path.name}: {exc}")
        if gallery and queries:
            people.append({"gallery": gallery, "queries": queries})
        else:
            print(f"Aviso: pasta {folder.name} não tinha fotos válidas suficientes após detecção.")
    if not people:
        raise RuntimeError("Nenhum participante válido. Adicione 3+ fotos consentidas por pasta em data/private/<id>/.")
    return people


def embed_crops(model, crops, device):
    arr = np.stack(crops).astype(np.uint8)
    labels = np.zeros(len(arr), dtype=np.int64)
    ds = FaceImageDataset(arr, labels, train=False)
    loader = DataLoader(ds, batch_size=32, shuffle=False)
    emb, _ = extract_embeddings(model, loader, device, baseline=False)
    return emb


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint não encontrado: {MODEL_PATH}. Baixe do Colab o checkpoint treinado somente com LFW "
            "e coloque-o em artifacts/ (não o publique até verificar os termos do modelo/dataset)."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
    model = FaceEmbeddingNet(embedding_dim=int(checkpoint.get("embedding_dim", 128)), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Processamento local: as fotografias privadas não serão enviadas para nenhum serviço por este script.")
    detector = make_detector()
    private_people = load_private(detector)

    # Create one normalized prototype per consenting participant using only enrollment images.
    prototypes = []
    known_query_embeddings, known_query_labels = [], []
    for person_index, person in enumerate(private_people):
        gallery_emb = embed_crops(model, person["gallery"], device)
        prototype = gallery_emb.mean(axis=0)
        prototype = prototype / max(np.linalg.norm(prototype), 1e-12)
        prototypes.append(prototype)
        query_emb = embed_crops(model, person["queries"], device)
        known_query_embeddings.extend(query_emb)
        known_query_labels.extend([person_index] * len(query_emb))
    prototypes = np.asarray(prototypes, dtype=np.float32)
    known_query_embeddings = np.asarray(known_query_embeddings, dtype=np.float32)
    known_query_labels = np.asarray(known_query_labels, dtype=np.int64)

    # Validation/test identities are disjoint from training and each other; use them as unknowns.
    lfw = load_lfw_splits(seed=42)
    val_ds = FaceImageDataset(lfw.val_images, lfw.val_labels, train=False)
    test_ds = FaceImageDataset(lfw.test_images, lfw.test_labels, train=False)
    val_emb, _ = extract_embeddings(model, DataLoader(val_ds, batch_size=64), device, baseline=False)
    test_emb, _ = extract_embeddings(model, DataLoader(test_ds, batch_size=64), device, baseline=False)

    val_max = (val_emb @ prototypes.T).max(axis=1)
    threshold = float(np.quantile(val_max, 0.99))
    test_max = (test_emb @ prototypes.T).max(axis=1)
    public_unknown_far = float(np.mean(test_max >= threshold))

    known_scores = known_query_embeddings @ prototypes.T
    known_best = known_scores.argmax(axis=1)
    known_max = known_scores.max(axis=1)
    known_accept_rate = float(np.mean(known_max >= threshold)) if len(known_max) else float("nan")
    known_top1 = float(np.mean(known_best == known_query_labels)) if len(known_best) else float("nan")
    known_correct_and_accepted = float(np.mean((known_best == known_query_labels) & (known_max >= threshold))) if len(known_best) else float("nan")

    result = {
        "protocol": "local private gallery; threshold is 99th percentile of max cosine similarity on public LFW validation identities",
        "num_private_participants": int(len(private_people)),
        "num_private_query_photos": int(len(known_query_labels)),
        "threshold": threshold,
        "private_known_query_accept_rate": known_accept_rate,
        "private_known_query_top1_accuracy": known_top1,
        "private_known_query_correct_and_accepted_rate": known_correct_and_accepted,
        "public_lfw_unknown_false_accept_rate": public_unknown_far,
        "num_public_lfw_unknown_test_images": int(len(test_max)),
        "warning": "Small sample and domain mismatch; treat as exploratory. No private images, names, or embeddings are saved.",
    }
    RESULTS.mkdir(exist_ok=True)
    output = RESULTS / "private_evaluation_aggregate.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"Somente métricas agregadas foram gravadas em {output}.")


if __name__ == "__main__":
    main()
