from abc import abstractmethod, ABC
from typing import Tuple, final

from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import Compose

class VIBEDataset(Dataset):
    def __init__(self, dataset: Dataset, transform: Compose=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def get_item(self, idx:int):
        img, label = self.dataset[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, label, idx


    @final
    def __getitem__(self, idx: int) -> Tuple[Tensor, int, int, int]:
        x, y, l, idx = self.get_item(idx)
        return x, y, l, idx


class SimpleVIBEDataset(VIBEDataset):
    def __init__(self, features: Tensor, labels: Tensor, original_labels: Tensor = None, transform: Compose=None) -> None:
        super().__init__(None, transform)

        self.features = features
        self.labels = labels

        if original_labels:
            self.original_labels = original_labels
        else:
            self.original_labels = labels

    def get_item(self, idx: int):
        return self.features[idx], self.labels[idx], self.original_labels[idx], idx
    def __len__(self):
        return len(self.features)

