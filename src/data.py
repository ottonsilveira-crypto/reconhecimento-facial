from dataclasses import dataclass
import numpy as np
from sklearn.datasets import fetch_lfw_people
from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision import transforms as T


@dataclass
class LFWData:
    train_images: np.ndarray
    train_labels: np.ndarray
    val_images: np.ndarray
    val_labels: np.ndarray
    test_images: np.ndarray
    test_labels: np.ndarray
    train_identity_names: list[str]
    val_identity_names: list[str]
    test_identity_names: list[str]


def load_lfw_splits(seed: int = 42, min_faces_per_person: int = 10,
                    max_images_per_identity: int = 40,
                    train_fraction: float = 0.70,
                    val_fraction: float = 0.15) -> LFWData:
    """Split by identity, not by image, to keep validation/test identities unseen."""
    lfw = fetch_lfw_people(min_faces_per_person=min_faces_per_person,
                           color=True, resize=1.0, funneled=True)
    images = lfw.images
    if images.max() <= 1.0:
        images = images * 255.0
    images = np.clip(images, 0, 255).astype(np.uint8)
    labels = np.asarray(lfw.target)
    names = np.asarray(lfw.target_names)

    rng = np.random.default_rng(seed)
    selected = []
    for identity in np.unique(labels):
        idx = np.where(labels == identity)[0]
        rng.shuffle(idx)
        selected.extend(idx[:max_images_per_identity].tolist())
    selected = np.asarray(selected, dtype=int)
    images, labels = images[selected], labels[selected]

    identities = np.unique(labels)
    rng.shuffle(identities)
    n = len(identities)
    n_train = max(2, int(n * train_fraction))
    n_val = max(1, int(n * val_fraction))
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    train_ids = identities[:n_train]
    val_ids = identities[n_train:n_train + n_val]
    test_ids = identities[n_train + n_val:]
    if len(test_ids) == 0:
        raise ValueError("Split de identidades sem conjunto de teste; ajuste as frações.")

    def subset(ids):
        mask = np.isin(labels, ids)
        xs, original_y = images[mask], labels[mask]
        mapping = {int(identity): i for i, identity in enumerate(sorted(ids.tolist()))}
        ys = np.asarray([mapping[int(y)] for y in original_y], dtype=np.int64)
        identity_names = [str(names[i]) for i in sorted(ids.tolist())]
        return xs, ys, identity_names

    xtr, ytr, ntr = subset(train_ids)
    xval, yval, nval_names = subset(val_ids)
    xte, yte, nte = subset(test_ids)
    return LFWData(xtr, ytr, xval, yval, xte, yte, ntr, nval_names, nte)


class FaceImageDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False):
        self.images = images
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        base = [
            T.Resize((128, 96), antialias=True),
        ]
        if train:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.RandomResizedCrop((128, 96), scale=(0.85, 1.0), ratio=(0.70, 0.82)),
                T.RandomHorizontalFlip(p=0.5),
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = T.Compose([
                T.ToPILImage(),
                T.Resize((128, 96), antialias=True),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.transform(self.images[index]), self.labels[index]
