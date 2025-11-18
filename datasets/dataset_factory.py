import torchvision
from torchvision.transforms import Compose
from torch.utils.data import Dataset

from omegaconf import DictConfig as Config


def get_dataset(dataset: Config, train: bool, transform: Compose = None) -> Dataset:
    if dataset.name == "cifar-10" or dataset.name == "cifar10":
        dataset = torchvision.datasets.CIFAR10(
            root=dataset.root,
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset.name == "cifar-100" or dataset.name == "cifar100":
        dataset = torchvision.datasets.CIFAR100(
            root=dataset.root,
            train=train,
            download=True,
            transform=transform,
        )
    elif dataset.name == "imagenet-30" or dataset.name == "imagenet30":
        dataset = torchvision.datasets.ImageFolder(
            root=f"{dataset.root}/{'train' if train else 'test'}",
            transform=transform,
        )
    elif dataset.name == "imagenet1k" or dataset.name == "imagenet1000":
        dataset = torchvision.datasets.ImageFolder(
            root=f"{dataset.root}/{'train' if train else 'val'}",
            transform=transform,
        )
    else:
        raise ValueError(f"Unknown dataset {dataset.name}")

    return dataset

