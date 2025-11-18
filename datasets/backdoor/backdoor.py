from abc import abstractmethod, ABC
from copy import deepcopy

import torch

from datasets.utils import remove_labels_from_dataset, get_dataset_labels, get_dataset_name
from datasets.vibe_dataset import VIBEDataset
from torch.utils.data import Dataset
from PIL.Image import Image
import numpy as np
from typing import Tuple
from torch import Tensor
from torchvision.transforms import Compose, ToPILImage, Resize, ToTensor
import os
from pathlib import Path


class Backdoor(VIBEDataset, ABC):
    def __init__(self, dataset: Dataset, transform: Compose, backdoor_cfg: dict, train=True) -> None:
        super(Backdoor, self).__init__(dataset, transform)

        self.target_label = backdoor_cfg["target_label"]
        self.original_labels = get_dataset_labels(self.dataset)

        if 'imagenet' in get_dataset_name(dataset):
            self.dataset.transform = Resize((224, 224))

        if train:
            self.poisoning_rate = backdoor_cfg["poisoning_rate"]
            self.poisoned_set_indices = create_or_load_poisoned_set(
                get_dataset_labels(self.dataset), len(self.dataset), self.target_label, self.poisoning_rate,
                clean_label=backdoor_cfg.get("clean_label", False)
            )
        else:
            self.dataset = remove_labels_from_dataset(dataset, {backdoor_cfg.target_label})
            self.poisoning_rate = 1
            self.poisoned_set_indices = torch.arange(len(self.dataset))

        self.poisoned_set = torch.zeros(len(self.dataset), dtype=torch.bool)
        self.poisoned_set[self.poisoned_set_indices] = True

    def __len__(self) -> int:
        return len(self.dataset)

    @abstractmethod
    def poison(self, img: Image, label: int) -> Tuple[Image, int]:
        pass

    def get_item(self, idx: int) -> Tuple[Tensor, int, int, int]:
        img, label = self.dataset[idx]

        if idx in self.poisoned_set_indices:
            img, poisoned_label = self.poison(img, label)
        else:
            poisoned_label = label

        if self.transform:
            img = self.transform(img)

        return img, poisoned_label, label, idx


class NumpyPoisonedBackdoor(Backdoor):
    def __init__(self, dataset: Dataset, transform: Compose, backdoor_cfg: dict, train=True) -> None:
        super(NumpyPoisonedBackdoor, self).__init__(dataset, transform, backdoor_cfg, train)

        poisoned_data_dir = os.path.join("./poisoned_data", "bbench")

        poisoned_data_path = (f"{poisoned_data_dir}/"
                              f"{get_dataset_name(dataset)}"
                              f"_{backdoor_cfg['name']}"
                              f"_{backdoor_cfg['poisoning_rate']}"
                              f"_{self.target_label}_"
                              f"{'train' if train else 'test'}.npz")

        npz_data = np.load(poisoned_data_path)

        self.data = npz_data['data']
        self.labels = npz_data['labels']
        self.poisoned_set = npz_data['poisoned_set']

        self.poisoned_set = torch.from_numpy(self.poisoned_set).bool()
        self.poisoned_set_indices = torch.where(self.poisoned_set)[0]

        # transforms
        if self.transform:
            self.transform = deepcopy(self.transform)
            self.transform.transforms.insert(0, ToPILImage())
            # if self.data[0].shape[0] == 3:
            #     self.transform.transforms[0] = lambda x: torch.from_numpy(x)
        else:
            self.transform = ToPILImage()


    def poison(self, img: Image, label: int) -> Tuple[Image, int]:
        return img, label

    def __len__(self) -> int:
        return len(self.labels)

    def get_item(self, idx: int) -> Tuple[Tensor, int, int, int]:
        img, label = self.data[idx], self.labels[idx]

        img, poisoned_label = self.poison(img, label) 
        poisoned_label = label

        if self.transform:
            img = self.transform(img)

        if isinstance(img, np.ndarray) and img.shape[0] == 1:
            img = img.squeeze(0)


        return img, poisoned_label, label, idx



def get_poisoning_idxs_path(dataset_length: int, target_label: int, poisoning_rate: float) -> str:
    project_root = Path(__file__).parent.parent.parent  # Go up to clustering_selfsup/
    poisoning_idxs_dir = project_root / "poisoning_idxs"
    os.makedirs(poisoning_idxs_dir, exist_ok=True)

    return f"{poisoning_idxs_dir}/{dataset_length}_{target_label}_{poisoning_rate}.pt"

def create_or_load_poisoned_set(dataset_labels: Tensor, dataset_length: int, target_label: int, poisoning_rate: float, clean_label=False) -> Tensor:
    poisoning_idxs_path = get_poisoning_idxs_path(dataset_length, target_label, poisoning_rate)

    if os.path.exists(poisoning_idxs_path):
        return torch.load(poisoning_idxs_path, weights_only=True)

    poisoning_idxs = torch.randperm(dataset_length)

    if poisoning_rate < 1:
        target_label_idxs = torch.arange(len(dataset_labels))[dataset_labels == target_label]
        mask = torch.isin(poisoning_idxs, target_label_idxs)
        if not clean_label:
            mask = ~mask
        poisoning_idxs = poisoning_idxs[mask]

    dataset_length = dataset_length if not clean_label else poisoning_idxs.size(0)
    poisoning_idxs = poisoning_idxs[:int(poisoning_rate * dataset_length)]

    torch.save(poisoning_idxs, poisoning_idxs_path)
    return poisoning_idxs
