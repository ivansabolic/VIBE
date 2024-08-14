import copy

import torch
import os
import os.path as osp

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from .utils_a import OneClassDataset
from .base import Base


class SimpleDataset(Dataset):
    def __init__(self, samples, targets, transform=None):
        self.samples = samples
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        target = self.targets[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target


class AddTrigger():
    def __init__(self, trigger, alpha):
        self.trigger = trigger
        self.alpha = alpha

        self.pre_transfom = transforms.ToTensor()
        self.post_transform = transforms.ToPILImage()

    def __call__(self, sample):
        sample = self.pre_transfom(sample)
        poisoned_sample = (1 - self.alpha) * sample + self.alpha * self.trigger
        poisoned_sample = self.post_transform(poisoned_sample)
        return poisoned_sample


class PoisonedDataset(Dataset):
    def __init__(
            self,
            train_dataset,
            dataset_name,
            attack_type,
            poison_rate,
            alpha,
            cover_rate,
            trigger_name,
            y_target,
            poisoned_set=None,
            transform=None,
            **kwargs
    ):
        if attack_type == "blend":
            main_folder = osp.join(
                '/home/isabolic/LikelihoodEstimationBasedDefense',
                'data',
                'cbd',
                f"{dataset_name}_{attack_type}_{poison_rate}_{alpha}_{cover_rate}_{trigger_name}",
            )
        elif attack_type == "patch":
            main_folder = osp.join(
                '/home/isabolic/LikelihoodEstimationBasedDefense',
                'data',
                'cbd',
                f"{dataset_name}_{attack_type}_{poison_rate}_{cover_rate}",
            )

        self.y_target = y_target
        self.transform = transform

        image_paths = [path for path in
                       sorted(os.listdir(osp.join(main_folder, "data")), key=lambda x: int(x.split(".")[0]))]

        self.poisoned_samples = [copy.deepcopy(Image.open(osp.join(main_folder, "data", image_path))) for image_path in
                                 image_paths]
        self.targets = torch.load(osp.join(main_folder, "labels"))
        self.poisoned_set = torch.load(osp.join(main_folder, "poison_indices"))
        if poisoned_set:
            torch.save(self.poisoned_set, f"{poisoned_set}.pt")
        self.cover_indices = torch.load(osp.join(main_folder, "cover_indices"))

        if not transform:
            self.transform = train_dataset.transform
        else:
            self.transform = transform

    def __len__(self):
        return len(self.poisoned_samples)

    def __getitem__(self, index):
        img, target = self.poisoned_samples[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class TestPoisonedDataset(Dataset):
    def __init__(
            self,
            train_dataset,
            attack_type,
            y_target,
            transform=None,
            **kwargs,
    ):
        if not transform:
            self.transform = train_dataset.transform
        else:
            self.transform = transform

        load_path = os.path.join(
            '/home/isabolic/LikelihoodEstimationBasedDefense',
            'data',
            f'adap_{attack_type}_0.01_0.01_testset',
        )

        self.poisoned_samples, self.targets = [], []
        for img_path in os.listdir(load_path):
            img = Image.open(os.path.join(load_path, img_path))
            self.poisoned_samples.append(copy.deepcopy(img))
            self.targets.append(y_target)



    def __len__(self):
        return len(self.poisoned_samples)

    def __getitem__(self, index):
        img = self.poisoned_samples[index].convert('RGB')
        target = self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        return img, target


class PoisonedCIFAR10(CIFAR10):
    def __init__(
            self,
            benign_dataset,
            y_target,
            attack_type,
            trigger,
            alpha,
    ):
        super(PoisonedCIFAR10, self).__init__(
            root=benign_dataset.root,
            train=benign_dataset.train,
            transform=benign_dataset.transform,
            download=False,
        )

        self.y_target = y_target

        self.transform.transforms.insert(0, AddTrigger(trigger, alpha))


    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        if self.transform is not None:
            img = self.transform(img)

        target = self.y_target

        return img, target

class CBD(Base):
    def __init__(
            self,
            dataset_name,
            train_dataset,
            test_dataset,
            model,
            loss,
            y_target,
            trigger_name,
            trigger,
            attack_type="blend",
            poison_rate=0.01,
            alpha=0.2,
            cover_rate=0.01,
            poisoned_set=None,
            schedule=None,
            seed=0,
            deterministic=False,
            **kwargs
    ):
        super(CBD, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        main_folder = osp.join(
            '/home/isabolic/LikelihoodEstimationBasedDefense',
            'data',
            'cbd',
            f"{dataset_name}_{attack_type}_{poison_rate}_{alpha}_{cover_rate}_{trigger_name}",
        )

        image_paths = [path for path in sorted(os.listdir(osp.join(main_folder, "data")), key=lambda x: int(x.split(".")[0]))]

        self.poisoned_samples = [copy.deepcopy(Image.open(osp.join(main_folder, "data", image_path))) for image_path in image_paths]
        self.labels = torch.load(osp.join(main_folder, "labels"))
        self.poisoned_set = torch.load(osp.join(main_folder, "poison_indices"))
        torch.save(self.poisoned_set, poisoned_set)
        self.cover_indices = torch.load(osp.join(main_folder, "cover_indices"))

        self.poisoned_train_dataset = SimpleDataset(self.poisoned_samples, self.labels, train_dataset.transform)
        self.poisoned_test_dataset = PoisonedCIFAR10(
            benign_dataset=test_dataset,
            y_target=y_target,
            attack_type=attack_type,
            trigger=trigger,
            alpha=alpha,
        )

        num_classes = len(set(self.poisoned_train_dataset.targets))
        self.asr_dataset = OneClassDataset(
            self.poisoned_test_dataset,
            set(range(num_classes)) - {y_target},
        )

        if not hasattr(self.test_dataset, 'targets'):
            self.test_dataset.targets = [label for _, label in self.test_dataset._samples]
        self.target_class_dataset = OneClassDataset(
            self.test_dataset,
            y_target,
        )


