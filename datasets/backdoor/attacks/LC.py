from copy import deepcopy
from typing import Tuple, Union, List

from torch import Tensor

from datasets.backdoor.backdoor import Backdoor, get_poisoning_idxs_path, create_or_load_poisoned_set
from PIL import Image
from torchvision.transforms import Compose, ToTensor
from torchvision.models import resnet18
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import DatasetFolder
from torch.nn import Module
from datasets.backdoor.attacks.utils import PGD, ResNet
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
import random
import torch
from omegaconf import DictConfig as Config

from datasets.utils import get_dataset_name, get_image_size_for_dataset, get_dataset_labels


def my_imread(file_path):
    return Image.fromarray(cv2.imread(file_path, cv2.IMREAD_UNCHANGED))

class LC(Backdoor):
    def __init__(self, dataset, transform: Compose, backdoor_cfg: Config, train=True, num_classes=10):
        super(LC, self).__init__(dataset, transform, backdoor_cfg, train)

        self.trigger, self.mask = self.get_trigger_and_mask(get_image_size_for_dataset(dataset))

        self.train = train
        if self.train:
            adv_model = ResNet(18, num_classes=num_classes) if 'cifar' in get_dataset_name(dataset) else resnet18(
                num_classes=num_classes)
            if osp.exists(backdoor_cfg['benign_model_path']):
                adv_model.load_state_dict(torch.load(backdoor_cfg['benign_model_path'], weights_only=True))

            adv_dataset_dir = os.path.join(
                'poisoned_data',
                'lc',
                f'adv_dataset-{get_dataset_name(dataset)}_patch_{self.poisoning_rate}_{self.poisoning_rate}_eps={backdoor_cfg["eps"]}',
            )

            if osp.exists(osp.join(adv_dataset_dir, 'poisoned_set_indices.pt')):
                self.poisoned_set_indices = torch.load(osp.join(adv_dataset_dir, 'poisoned_set_indices.pt'), weights_only=True)
                self.poisoned_set = torch.zeros(len(dataset), dtype=torch.bool)
                self.poisoned_set[self.poisoned_set_indices] = True

            _, self.poisoned_dataset = self._get_adv_dataset(
                dataset,
                adv_model=adv_model,
                adv_dataset_dir=adv_dataset_dir,
                adv_transform=ToTensor(),
                eps=backdoor_cfg['eps'] / backdoor_cfg['max_pixel'],
                alpha=backdoor_cfg['alpha'] / backdoor_cfg['max_pixel'],
                steps=backdoor_cfg['steps'],
                y_target=self.target_label,
                poisoned_rate=self.poisoning_rate,
                poisoned_set=self.poisoned_set_indices,
                device=backdoor_cfg['device'],
            )

            mapping_indices = self.get_mapping_indices(adv_dataset_dir)
            self.poisoned_set = self.poisoned_set[mapping_indices]
            self.poisoned_set_indices = torch.nonzero(self.poisoned_set).squeeze(1)
        else:
            self.poisoned_dataset = self.dataset

    def get_item(self, idx: int) -> Tuple[Tensor, int, int, int]:
        img, label = self.poisoned_dataset[idx]

        # if self.train:
        #     img_path = self.poisoned_dataset.samples[idx][0]
        #     img_index = int(img_path.split('/')[-1].split('.')[0])
        # else:
        #     img_index = idx

        if idx in self.poisoned_set_indices:
            img, label = self.poison(img, label)

        if self.transform:
            img = self.transform(img)

        return img, label, label, idx


    def poison(self, img: Image, label: int):
        img = ToTensor()(img)
        img[self.mask] = self.trigger[self.mask]
        img = Image.fromarray((img * 255).byte().permute(1, 2, 0).numpy())

        return img, self.target_label

    def _get_adv_dataset(
            self,
            dataset: Dataset,
            adv_model: Module,
            adv_dataset_dir: str,
            adv_transform: Compose,
            eps: float,
            alpha: float,
            steps: int,
            y_target: int,
            poisoned_rate: int,
            poisoned_set: Union[List, Tensor],
            device="cuda"
    ):

        def _generate_adv_dataset():
            nonlocal dataset, adv_model, adv_dataset_dir, adv_transform, eps, alpha, steps, y_target, poisoned_rate, \
                poisoned_set, device

            current_schedule = {
                'device': device,
                'batch_size': 128,
                'num_workers': 8,
            }
            device = current_schedule['device']

            adv_model = adv_model.to(device)

            backup_transform = deepcopy(dataset.transform)
            dataset.transform = adv_transform

            data_loader = DataLoader(
                dataset,
                batch_size=current_schedule['batch_size'],
                shuffle=False,
                num_workers=current_schedule['num_workers'],
                drop_last=False,
                pin_memory=True,
            )

            attacker = PGD(adv_model, eps, alpha, steps, random_start=False)
            attacker.set_return_type("int")

            original_imgs = []
            perturbed_imgs = []
            targets = []

            for batch in tqdm(data_loader):
                # Adversarially perturb image. Note that torchattacks will automatically
                # move `img` and `target` to the gpu where the attacker.model is located.
                batch_img = batch[0]
                batch_label = batch[1]
                batch_img = batch_img.to(device)
                batch_label = batch_label.to(device)
                img = attacker(batch_img, batch_label)
                original_imgs.append(
                    torch.round(batch_img * 255).to(dtype=torch.uint8).permute(0, 2, 3, 1).detach().cpu())
                perturbed_imgs.append(img.permute(0, 2, 3, 1).detach().cpu())
                targets.append(batch_label.cpu())

            dataset.transform = backup_transform

            original_imgs = torch.cat(original_imgs, dim=0).numpy()
            perturbed_imgs = torch.cat(perturbed_imgs, dim=0).numpy()
            targets = torch.cat(targets, dim=0).numpy()

            for target in np.unique(targets):
                os.makedirs(osp.join(adv_dataset_dir, 'whole_adv_dataset', str(target).zfill(2)), exist_ok=True)
                os.makedirs(osp.join(adv_dataset_dir, 'target_adv_dataset', str(target).zfill(2)), exist_ok=True)

            for index, item in enumerate(zip(original_imgs, perturbed_imgs, targets)):
                original_img, perturbed_img, target = item
                cv2.imwrite(
                    osp.join(adv_dataset_dir, 'whole_adv_dataset', str(target).zfill(2), str(index).zfill(8) + '.png'),
                    perturbed_img)

                if index in poisoned_set:
                    cv2.imwrite(osp.join(adv_dataset_dir, 'target_adv_dataset', str(target).zfill(2),
                                         str(index).zfill(8) + '.png'), perturbed_img)
                else:
                    cv2.imwrite(osp.join(adv_dataset_dir, 'target_adv_dataset', str(target).zfill(2),
                                         str(index).zfill(8) + '.png'), original_img)

        if not osp.exists(osp.join(adv_dataset_dir, 'whole_adv_dataset')) or not osp.exists(
                osp.join(adv_dataset_dir, 'target_adv_dataset')):
            _generate_adv_dataset()
            torch.save(poisoned_set, osp.join(adv_dataset_dir, 'poisoned_set_indices.pt'))

        whole_adv_dataset = DatasetFolder(
            root=osp.join(adv_dataset_dir, 'whole_adv_dataset'),
            loader=my_imread,
            extensions=('png',),
            transform=None, # deepcopy(dataset.transform),
            target_transform=None, # deepcopy(dataset.target_transform),
            is_valid_file=None
        )

        target_adv_dataset = DatasetFolder(
            root=osp.join(adv_dataset_dir, 'target_adv_dataset'),
            loader=my_imread,
            extensions=('png',),
            transform=None, # deepcopy(dataset.transform),
            target_transform=None, # deepcopy(dataset.target_transform),
            is_valid_file=None
        )

        return whole_adv_dataset, target_adv_dataset

    def get_trigger_and_mask(self, data_shape: Tuple):
        h, w = data_shape
        pattern = torch.zeros((h, w, 3))
        pattern[-1, -1] = 1
        pattern[-1, -3] = 1
        pattern[-3, -1] = 1
        pattern[-2, -2] = 1

        pattern[0, -1] = 1
        pattern[1, -2] = 1
        pattern[2, -3] = 1
        pattern[2, -1] = 1

        pattern[0, 0] = 1
        pattern[1, 1] = 1
        pattern[2, 2] = 1
        pattern[2, 0] = 1

        pattern[-1, 0] = 1
        pattern[-1, 2] = 1
        pattern[-2, 1] = 1
        pattern[-3, 0] = 1

        weight = torch.zeros((h, w, 3), dtype=torch.bool)
        weight[:3, :3] = True
        weight[:3, -3:] = True
        weight[-3:, :3] = True
        weight[-3:, -3:] = True

        return pattern.permute(2, 0, 1), weight.permute(2, 0, 1)

    def get_mapping_indices(self, adv_dataset_dir):
        adv_dataset_dir = os.path.join(adv_dataset_dir, 'target_adv_dataset')
        class_paths = os.listdir(adv_dataset_dir)
        indices = []
        for class_path in sorted(class_paths):  # warning: works only for datasets with classes < 10
            imgs = os.listdir(os.path.join(adv_dataset_dir, class_path))
            for img in sorted(imgs):
                indices.append(int(img.split(".")[0]))

        return np.array(indices)
