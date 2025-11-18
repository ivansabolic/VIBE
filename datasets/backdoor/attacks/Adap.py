from copy import deepcopy

from torchvision.transforms import Compose, ToTensor
from omegaconf import DictConfig as Config
import os
import os.path as osp
from PIL import Image
import torch

from datasets.backdoor import Backdoor
from datasets.utils import get_dataset_name


class Adap(Backdoor):
    def __init__(self, dataset, transform: Compose, backdoor_cfg: Config, train=True, num_classes=10):
        super(Adap, self).__init__(dataset, transform, backdoor_cfg, train)

        main_folder = backdoor_cfg.get('root', os.path.join('poisoned_data', 'cbd'))
        if train:
            if backdoor_cfg['type'] == 'patch':
                main_folder = osp.join(
                    main_folder,
                    f'{get_dataset_name(dataset)}_patch_{self.poisoning_rate}_{self.poisoning_rate}',
                )
            elif backdoor_cfg['type'] == 'blend':
                main_folder = os.path.join(
                    main_folder,
                    f'{get_dataset_name(dataset)}_blend_{self.poisoning_rate}_0.2_{self.poisoning_rate}_hellokitty_32.png',
                )

            image_paths = [path for path in
                           sorted(os.listdir(osp.join(main_folder, "data")), key=lambda x: int(x.split(".")[0]))]

            self.poisoned_samples = [deepcopy(Image.open(osp.join(main_folder, "data", image_path))) for image_path
                                     in
                                     image_paths]
            self.targets = torch.load(osp.join(main_folder, "labels"), weights_only=True)
            self.poisoned_set_indices = torch.Tensor(torch.load(osp.join(main_folder, "poison_indices"), weights_only=True)).long()
            self.poisoned_set = torch.zeros_like(self.targets, dtype=torch.bool)
            self.poisoned_set[self.poisoned_set_indices] = True

        else:
            main_folder = osp.join(
                main_folder,
                f'adap_{backdoor_cfg["type"]}_{backdoor_cfg["poisoning_rate"]}_{backdoor_cfg["poisoning_rate"]}_testset'
            )

            self.poisoned_samples = [deepcopy(Image.open(osp.join(main_folder, image_path)).convert('RGB')) for image_path
                                     in
                                     sorted(os.listdir(main_folder), key=lambda x: int(x.split(".")[0]))]
            self.targets = [self.target_label for _ in range(len(self.poisoned_samples))]

    def poison(self, img: Image, label: int):
        return img, label

    def get_item(self, idx: int):
        img, label = self.poisoned_samples[idx], self.targets[idx]

        img, poisoned_label = self.poison(img, label)  # ovdje ne pamtim pravu oznaku
        poisoned_label = label

        if self.transform:
            img = self.transform(img)

        return img, poisoned_label, label, idx
