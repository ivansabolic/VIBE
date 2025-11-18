from datasets.backdoor import Backdoor
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from datasets.utils import get_dataset_name
from omegaconf import DictConfig as Config


class BadNets(Backdoor):
    def __init__(self, dataset, transform: Compose, backdoor_cfg: Config, train=True, num_classes=10):
        super(BadNets, self).__init__(dataset, transform, backdoor_cfg, train)

        dataset_name = get_dataset_name(dataset)
        self.trigger = ToTensor()(Image.open(backdoor_cfg[f"{dataset_name}_trigger_path"]).convert("RGB"))
        self.mask = self.trigger != 0

        self.num_classes = num_classes

    def poison(self, img: Image, label: int):
        img = ToTensor()(img)
        img[self.mask] = self.trigger[self.mask]
        img = Image.fromarray((img * 255).byte().permute(1, 2, 0).numpy())

        if self.target_label != -1:
            return img, self.target_label
        else:
            return img, (label + 1) % self.num_classes