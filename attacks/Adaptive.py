import copy
import os.path
import warnings

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose

from .base import Base
from .utils_a import OneClassDataset

img_shape = {
    "cifar10": (3, 32, 32),
    "cifar100": (3, 32, 32),
    "gtsrb": (3, 48, 48),
    "imagenet": (3, 224, 224),
    "vggface2": (3, 224, 224),
}

class BlendTrigger:
    def __init__(self, pattern, weight):
        self.pattern = pattern
        self.weight = weight

    def __call__(self, img):
        img = np.array(img)
        img = img * (1 - self.weight) + self.pattern * self.weight
        return Image.fromarray(img.astype(np.uint8))


class ModifyTarget:
    def __init__(self, y_target):
        self.y_target = y_target

    def __call__(self, y_target):
        return self.y_target

class ModifyTargetAll2All:
    def __call__(self, y, num_classes=10):
        return (y + 1) % num_classes


class PoisonedCIFAR10(CIFAR10):
    def __init__(self,
                 benign_dataset,
                 y_target,
                 poisoned_rate,
                 experiment_dir,
                 blend_ratio=0.2,
                 poisoned_transform_index=0,
                 poisoned_target_transform_index=0,
                 poisoned_set=None,
                 **kwargs,
                 ):
        super(PoisonedCIFAR10, self).__init__(
            benign_dataset.root,
            benign_dataset.train,
            benign_dataset.transform,
            benign_dataset.target_transform,
            download=False)
        total_num = len(benign_dataset)
        if poisoned_set:
            poisoned_set = f"{poisoned_set}.pt"
        if poisoned_set and os.path.exists(poisoned_set):
            self.poisoned_set = torch.load(poisoned_set)
        else:
            warnings.warn('Poisoned set is not provided and may be different from the one used to generate trigger. Creating poisoned set...')
            tmp_list = torch.arange(total_num)
            tmp_list = tmp_list[torch.tensor(self.targets) != y_target]
            poisoned_num = int(len(tmp_list) * poisoned_rate)
            assert poisoned_num >= 0, 'poisoned_num should greater than or equal to zero.'
            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            self.poisoned_set = tmp_list[:poisoned_num]
            if poisoned_set:
                torch.save(self.poisoned_set, poisoned_set)


        self.trigger = np.load(os.path.join(experiment_dir, 'trigger.npy'))
        # Add trigger to images
        if self.transform is None:
            self.poisoned_transform = Compose([])
        else:
            self.poisoned_transform = copy.deepcopy(self.transform)

        self.poisoned_transform.transforms.insert(poisoned_transform_index, BlendTrigger(self.trigger, blend_ratio))

        # Modify labels
        if self.target_transform is None:
            self.poisoned_target_transform = Compose([])
        else:
            self.poisoned_target_transform = copy.deepcopy(self.target_transform)

        if y_target != -1:
            self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTarget(y_target))
        else:
            self.poisoned_target_transform.transforms.insert(poisoned_target_transform_index, ModifyTargetAll2All())

        self.y_target = y_target


    def __getitem__(self, index):
        # index = self.y_target_indices[index]
        img, target = self.data[index], int(self.targets[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if index in self.poisoned_set:
            img = self.poisoned_transform(img)
            target = self.poisoned_target_transform(target)
        else:
            if self.transform is not None:
                img = self.transform(img)

            if self.target_transform is not None:
                target = self.target_transform(target)

        return img, target


class Adaptive(Base):
    def __init__(
            self,
            train_dataset,
            test_dataset,
            model,
            loss,
            experiment_dir,
            dataset_name,
            selfsup_model, # feature extractor naučen na self-supervised način na čistom skupu
            poison_schedule,
            blend_ratio=0.2,
            y_target=0,
            poisoned_rate=0.1,
            poisoned_set=None,
            test_poisoned_set=None,
            schedule=None,
            seed=0,
            deterministic=False,
            device=torch.device('cpu'),
            poisoned_transform_train_index=0,
            poisoned_transform_test_index=0,
            poisoned_target_transform_index=0,

    ):
        super(Adaptive, self).__init__(
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            model=model,
            loss=loss,
            schedule=schedule,
            seed=seed,
            deterministic=deterministic)

        self.dataset = dataset_name
        self.blend_ratio = blend_ratio
        self.y_target = y_target
        self.device = device
        self.poisoned_rate = poisoned_rate
        self.train_poisoned_set = poisoned_set

        self.selfsup_model = selfsup_model
        self.experiment_dir = experiment_dir
        if not os.path.exists(os.path.join(self.experiment_dir, 'trigger.npy')):
            os.makedirs(self.experiment_dir, exist_ok=True)
            self._create_poisoned_dataset(poison_schedule)


        self.poisoned_train_dataset = PoisonedCIFAR10(
            self.train_dataset,
            y_target=self.y_target,
            poisoned_rate=self.poisoned_rate,
            experiment_dir=self.experiment_dir,
            blend_ratio=self.blend_ratio,
            poisoned_transform_index=poisoned_transform_train_index,
            poisoned_target_transform_index=poisoned_target_transform_index,
            poisoned_set=self.train_poisoned_set
        )

        self.poisoned_test_dataset = PoisonedCIFAR10(
            self.test_dataset,
            y_target=self.y_target,
            poisoned_rate=1.,
            experiment_dir=self.experiment_dir,
            blend_ratio=self.blend_ratio,
            poisoned_transform_index=poisoned_transform_test_index,
            poisoned_target_transform_index=poisoned_target_transform_index,
        )

        num_classes = len(set(self.poisoned_train_dataset.targets))
        self.asr_dataset = OneClassDataset(
            self.poisoned_test_dataset,
            set(range(num_classes)) - {y_target},
        )

        if not hasattr(self.test_dataset, 'targets'):
            self.test_dataset.targets = [label for _, label in self.test_dataset._samples]

        if y_target != -1:
            self.target_class_dataset = OneClassDataset(
                self.test_dataset,
                y_target,
            )
        else:
            self.target_class_dataset = self.asr_dataset
    def _create_poisoned_dataset(
            self,
            schedule,
    ):
        # cilj mi je da su zatrovani podaci bliži centru target razreda
        # 1) mogu minimizirati samo udaljenost centroida zatrovanih podataka od centroida
        # target razreda
        # 2) mogu minimizirati udaljenost zatrovanih podataka od centroida
        # target razreda
        # 3) mogu minimizirati udaljenost centroida zatrovanih podataka od centroida
        # target razreda i udaljenost zatrovanih podataka od centroida target razreda

        if self.train_poisoned_set:
            self.train_poisoned_set = f"{self.train_poisoned_set}.pt"
        if self.train_poisoned_set and os.path.exists(self.train_poisoned_set):
            poisoned_set = torch.load(self.train_poisoned_set)
        else:
            tmp_list = torch.arange(len(self.train_dataset.__len__()))
            tmp_list = tmp_list[torch.tensor(self.train_dataset.targets) != self.y_target]
            poisoned_num = int(len(tmp_list) * self.poisoned_rate)
            shuffled = torch.randperm(len(tmp_list))
            tmp_list = tmp_list[shuffled]
            poisoned_set = tmp_list[:poisoned_num]
            if self.train_poisoned_set:
                torch.save(poisoned_set, self.train_poisoned_set)

        class PoisonedDataset(torch.utils.data.Dataset):
            def __init__(self, dataset, poisoned_set, y_target):
                self.dataset = dataset
                self.poisoned_set = poisoned_set
                self.y_target = y_target

            def __len__(self):
                return len(self.poisoned_set)

            def __getitem__(self, item):
                x, y = self.dataset[self.poisoned_set[item]]
                if item in self.poisoned_set:
                    y = self.y_target

                return x, y

        class PoisonedModel(nn.Module):
            def __init__(self, model, img_shape, blend_ratio=0.2):
                super(PoisonedModel, self).__init__()
                self.model = model
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False

                self.trigger = nn.Parameter(torch.zeros(img_shape))
                self.blend_ratio = blend_ratio

            def forward(self, x):
                x = x * (1 - self.blend_ratio) + torch.sigmoid(self.trigger) * self.blend_ratio
                features, _, _, _ = self.model(x, x)
                return features

        train_dataset = PoisonedDataset(self.train_dataset, poisoned_set, self.y_target)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=64,
            shuffle=True,
            num_workers=4,
        )
        poisoned_model = PoisonedModel(self.selfsup_model, img_shape[self.dataset], self.blend_ratio).to(self.device)
        poisoned_model.train()

        optimizer = torch.optim.Adam(poisoned_model.parameters(), lr=schedule['lr'])

        target_class_centroid = self._calculate_centroid(self.train_dataset,  poisoned_model.model, self.y_target,)

        for epoch in range(schedule['num_epochs']):
            epoch_loss = 0
            for x, y in train_loader:
                x = x.to(self.device)

                optimizer.zero_grad()

                features = poisoned_model(x)
                loss = torch.norm(features.mean(dim=0) - target_class_centroid)
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

            print(f"Epoch {epoch} loss: {epoch_loss / len(train_loader)}")


        trigger = torch.sigmoid(poisoned_model.trigger).detach().cpu().permute(1, 2, 0).numpy()
        trigger = (trigger * 255).astype(np.uint8)

        np.save(os.path.join(self.experiment_dir, 'trigger.npy'), trigger)


    def _calculate_centroid(self, dataset, model, y_target):
        class_dataset = OneClassDataset(dataset, y_target)
        print(f"Length of class dataset: {len(class_dataset)}")
        class_loader = torch.utils.data.DataLoader(
            class_dataset,
            batch_size=64,
            shuffle=False,
            num_workers=4,
        )

        features = []
        for x, y in class_loader:
            x = x.to(self.device)

            o, _, _, _ = model(x, x)
            features.append(o.detach())

        return torch.cat(features).mean(dim=0)











