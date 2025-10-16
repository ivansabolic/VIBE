import os

from datasets.backdoor import Backdoor
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from omegaconf import DictConfig as Config

from datasets.utils import get_image_size_for_dataset, get_dataset_name 

class Blend(Backdoor):
    def __init__(self, dataset, transform: Compose, backdoor_cfg: Config, train=True):
        super(Blend, self).__init__(dataset, transform, backdoor_cfg, train)

        w, h = get_image_size_for_dataset(dataset)
        dataset_name = get_dataset_name(dataset)
        self.trigger = ToTensor()(Image.open(backdoor_cfg[f"{dataset_name}_trigger_path"]).resize((w, h)))

        self.blend_ratio = backdoor_cfg["blend_ratio"]

    def poison(self, img: Image, label: int):
        img = ToTensor()(img)
        img = img * (1 - self.blend_ratio) + self.trigger * self.blend_ratio
        img = Image.fromarray((img * 255).byte().permute(1, 2, 0).numpy())

        return img, self.target_label
