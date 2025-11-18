from typing import Tuple

import torch
from torch.utils.data import Dataset
from torch import Tensor
from torchvision.datasets import CIFAR10, CIFAR100, DatasetFolder, ImageFolder

class FilteredDataset(Dataset): # should this inherit from VIBEDataset?
    def __init__(self, dataset: Dataset, indices: Tensor) -> None:
        super().__init__()

        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        batch = self.dataset[self.indices[index]]
        if len(batch) == 2:
            return batch[0], batch[1]

        return batch[0], batch[1], batch[2], index

    def __len__(self):
        return len(self.indices)

    @property
    def poisoned_set(self) -> Tensor:
        return self.dataset.poisoned_set[self.indices]

    @property
    def original_labels(self) -> Tensor:
        return self.dataset.original_labels[self.indices]


def get_dataset_labels(dataset: Dataset) -> Tensor:
    if type(dataset) in [CIFAR10, CIFAR100]:
        return Tensor(dataset.targets)
    elif type(dataset) == DatasetFolder or type(dataset) == ImageFolder:
        return Tensor([label for _, label in dataset.samples]).long()
    else:
        raise NotImplementedError(f"Getting labels for dataset of type {type(dataset)} is not implemented")


def remove_labels_from_dataset(dataset: Dataset, labels_to_remove: set) -> FilteredDataset:
    dataset_labels = get_dataset_labels(dataset)

    dataset_labels = dataset_labels.long()
    labels_to_keep = set(range(dataset_labels.max().item() + 1)) - labels_to_remove

    indices = torch.arange(len(dataset_labels))[torch.isin(dataset_labels, Tensor(list(labels_to_keep)))]

    return FilteredDataset(dataset, indices)


def get_image_size_for_dataset(dataset: Dataset) -> Tuple[int, int]:
    if type(dataset) in [CIFAR10, CIFAR100]:
        return 32, 32
    elif type(dataset) == DatasetFolder or type(dataset) == ImageFolder:
        return 224, 224
    else:
        raise NotImplementedError(f"Getting image size for dataset of type {type(dataset)} is not implemented")


def get_dataset_name(dataset: Dataset) -> str:
    if type(dataset) == CIFAR100:
        return "cifar100"
    elif type(dataset) == CIFAR10:
        return "cifar10"
    elif type(dataset) == ImageFolder:
        return "imagenet"
    else:
        raise NotImplementedError(f"Getting dataset name for dataset of type {type(dataset)} is not implemented")

def get_dataset_num_classes(dataset: Dataset) -> int:
    if type(dataset) in [CIFAR10]:
        return 10
    elif type(dataset) in [CIFAR100]:
        return 100
    elif type(dataset) in [ImageFolder]:
        return 30
    else:
        raise NotImplementedError(f"Getting dataset number of classes for dataset of type {type(dataset)} is not implemented")
