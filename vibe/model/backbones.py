import os
from omegaconf import OmegaConf
import torch
from torch import nn
from torchvision.models import resnet50, resnet18

import open_clip
from omegaconf import DictConfig as Config

__all__ = ["backbone_factory"]

def _load_selfsup(model, path):
    print("Loading weights", path)
    dict = torch.load(path, map_location='cpu', weights_only=True)
    dict = {k.replace('module.encoder_q.', ''): v for k, v in dict['state_dict'].items()}
    dict = {k.replace('backbone.', ''): v for k, v in dict.items()}
    if 'fc.weight' in dict:
        del dict['fc.weight']
    if 'fc.bias' in dict:
        del dict['fc.bias']
    state = {}
    for k, v in dict.items():
        if 'encoder_k' not in k:
            state[k] = v
    model.load_state_dict(state, strict=False)
    return model


def construct_resnet50(pretrained=None) -> (nn.Module, int):
    model = resnet50()
    if pretrained:
        model = _load_selfsup(model, pretrained)

    embed_dim = model.fc.weight.shape[1]
    # model.fc = nn.Linear(embed_dim, num_classes)
    model.fc = nn.Identity()
    return model, embed_dim


def construct_resnet18(pretrained: str=None, cifar: bool=False) -> (nn.Module, int):
    model = resnet18()
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=2, bias=False)
        model.maxpool = nn.Identity()
    if pretrained:
        model = _load_selfsup(model, pretrained)

    embed_dim = 512
    model.fc = nn.Identity()
    return model, embed_dim


def solo(backbone_cfg: Config):
    if backbone_cfg.arch == "resnet50":
        model, embed_dim = construct_resnet50(pretrained=backbone_cfg.ckpt_path)
    elif backbone_cfg.arch == "resnet18":
        model, embed_dim = construct_resnet18(pretrained=backbone_cfg.ckpt_path, cifar=backbone_cfg.cifar)
    else:
        raise ValueError(f"Unknown backbone {backbone_cfg.arch}")

    return model, embed_dim

def dino(backbone_cfg: Config):
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[backbone_cfg.arch]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()

    return backbone_model, 1536


def clip(backbone_cfg: Config):
    backbone_name = backbone_cfg.arch
    if backbone_name == "giant":
        model, _ = open_clip.create_model_from_pretrained("hf-hub:laion/CLIP-ViT-g-14-laion2B-s34B-b88K")
    elif backbone_name == "convnext_large":
        model, _ = open_clip.create_model_from_pretrained("hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup")
    else:
        raise ValueError(f"Unknown backbone for CLIP {backbone_name}")

    return model, 1536


backbone_factory = {
    "solo": solo,
    "dino": dino,
    "clip": clip,
}
