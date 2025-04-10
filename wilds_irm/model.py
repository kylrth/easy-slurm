import enum
from pathlib import Path

import torch
from torch import nn
from torchvision import models


class Model(enum.StrEnum):
    """Model options supported by this script."""

    ResNet18 = "resnet18"
    ResNet34 = "resnet34"
    ResNet50 = "resnet50"
    ViT_b_32 = "vit_b_32"
    ViT_l_32 = "vit_l_32"

    def init(self, pretrained: bool, device: str, model_cache: Path | str = "") -> nn.Module:
        if model_cache != "":
            torch.hub.set_dir(model_cache)

        model_factory = getattr(models, self)
        model = model_factory(weights="IMAGENET1K_V1") if pretrained else model_factory()

        # replace the last layer with a random-init layer of the output size we want (1)
        # We can't use num_classes=1 in the model factory above, because pretrained models don't
        # come in that size so it won't know how to init the last layer.
        match self:
            case self.ResNet18 | self.ResNet34:
                model.fc = nn.Linear(512, 1)
            case self.ResNet50:
                model.fc = nn.Linear(2048, 1)
            case self.ViT_b_32:
                model.heads[-1] = nn.Linear(768, 1)
            case self.ViT_l_32:
                model.heads[-1] = nn.Linear(1024, 1)

        return model.float().to(device)
