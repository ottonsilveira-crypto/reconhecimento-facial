import torch
import torch.nn as nn
import torchvision


class FaceEmbeddingNet(nn.Module):
    """ResNet-18 ImageNet backbone with a compact embedding head."""
    def __init__(self, embedding_dim: int = 128, pretrained: bool = True):
        super().__init__()
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = torchvision.models.resnet18(weights=weights)
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def embed_imagenet_baseline(self, x):
        """ImageNet-only baseline: use the pretrained 512-D backbone features."""
        return self.backbone(x)


def set_trainable_layers(model: FaceEmbeddingNet, train_backbone_layers: tuple[str, ...]):
    for name, parameter in model.backbone.named_parameters():
        parameter.requires_grad = any(name.startswith(layer) for layer in train_backbone_layers)
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def set_stage_train_mode(model: FaceEmbeddingNet, train_backbone_layers: tuple[str, ...]):
    model.train()
    # Frozen layers remain in eval mode, including their BatchNorm running statistics.
    model.backbone.eval()
    for layer in train_backbone_layers:
        module = getattr(model.backbone, layer, None)
        if module is not None:
            module.train()
    model.head.train()
