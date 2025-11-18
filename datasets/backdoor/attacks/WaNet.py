from typing import Tuple

from datasets.backdoor import Backdoor
from PIL import Image
import numpy as np
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from torchvision.transforms import Compose, ToTensor
from omegaconf import DictConfig as Config

from datasets.utils import get_image_size_for_dataset, get_dataset_name


class WaNet(Backdoor):
    def __init__(self, dataset, transform: Compose, backdoor_cfg: Config, train=True):
        super(WaNet, self).__init__(dataset, transform, backdoor_cfg, train)

        self.w, self.h = get_image_size_for_dataset(dataset)

        self.noise = backdoor_cfg["noise"]
        self.noise_set = ...

        dataset_name = get_dataset_name(dataset)
        self.identity_grid, self.noise_grid = self._get_noise_and_identity_grid(backdoor_cfg[f"{dataset_name}_k"], self.h)

        self.h = self.identity_grid.shape[2]
        self.s = backdoor_cfg["s"]
        self.grid_rescale = 1
        grid = self.identity_grid + self.s * self.noise_grid / self.h
        self.grid = torch.clamp(grid * self.grid_rescale, -1, 1)
        self.noise_rescale = 2

    def _get_noise_and_identity_grid(self, k: float, height: int) -> Tuple[Tensor, Tensor]:
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
        noise_grid = F.upsample(ins, size=height, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
        array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
        x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
        identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

        return identity_grid, noise_grid


    def poison(self, img: Image, label: int):
        img = torch.from_numpy(np.array(img)).permute(2, 0, 1).float()

        # add trigger
        if self.noise:
            ins = torch.rand(1, self.h, self.h, 2) * self.noise_rescale - 1  # [-1, 1]
            grid = torch.clamp(self.grid + ins / self.h, -1, 1)
        else:
            grid = self.grid

        img = nn.functional.grid_sample(img.unsqueeze(0), grid, align_corners=True).squeeze()  # CHW

        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 255).astype(np.uint8)
        img = Image.fromarray(img)

        return img, self.target_label

