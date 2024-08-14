import os.path

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

import models
from attacks import *
from attacks.utils_a import OneClassImageFolder

dataset_num_classes = {
    "cifar10": 10,
    "cifar100": 100,
    "imagenet": 30,
    "imagenet-100": 100,
    "imagenet1k": 1000,
}

def get_poisoned_transform_index(dataset):
    if dataset == "cifar10" or dataset == "cifar100" or dataset == "tinyimagenet":
        return 0
    elif dataset == "gtsrb" or dataset == "imagenet" or dataset == "imagenet-100" or dataset == "vggface2" or dataset == "imagenet1k":
        return 1
    else:
        raise NotImplementedError


def get_pattern_and_weight_badnets(dataset, data_shape):
    _, h, w = data_shape
    if dataset == "cifar10":
        # pattern = torch.zeros((1, h, w), dtype=torch.uint8)
        # pattern[:, -2:, -2:] = 255
        # weight = torch.zeros((1, h, w), dtype=torch.float32)
        # weight[:, -2:, -2:] = 1.0

        # pattern = torch.zeros((1, h, w), dtype=torch.uint8)
        # pattern[:, 0, 0] = 0
        # pattern[:, 0, 1] = 255
        # pattern[:, 1, 0] = 0
        # pattern[:, 1, 1] = 255
        # weight = torch.zeros((1, h, w), dtype=torch.float32)
        # weight[:, :2, :2] = 1.0

        pattern = Image.open("./resources/cifar_1.png").convert("RGB").resize((w, h))
        pattern = np.transpose(np.array(pattern), (2, 0, 1))
        pattern = torch.from_numpy(pattern)
        weight = torch.zeros((1, h, w), dtype=torch.float32)
        weight[:, 4:6, 4:6] = 1.0
    elif dataset == "cifar100":
        pattern = torch.zeros((1, h, w), dtype=torch.uint8)
        pattern[:, -1, -1] = 255
        pattern[:, -1, -2] = 0
        pattern[:, -1, -3] = 255

        pattern[:, -2, -1] = 0
        pattern[:, -2, -2] = 255
        pattern[:, -2, -3] = 0

        pattern[:, -3, -1] = 255
        pattern[:, -3, -2] = 0
        pattern[:, -3, -3] = 255

        weight = torch.zeros((1, h, w), dtype=torch.float32)
        weight[0, -3:, -3:] = 1.0
    elif dataset == "gtsrb":
        pattern = torch.zeros((1, h, w), dtype=torch.uint8)
        pattern[0, -3:, -3:] = 255
        weight = torch.zeros((1, h, w), dtype=torch.float32)
        weight[0, -3:, -3:] = 1.0
    elif "imagenet" in dataset or dataset == "vggface2":
        pattern_size = 32
        pattern_path = "./resources/apple-logo.jpg"
        pattern = np.array(Image.open(pattern_path).convert("RGB").resize((pattern_size, pattern_size)))
        pattern = np.transpose(pattern, (2, 0, 1))

        whole_pattern = np.zeros((3, h, w), dtype=np.uint8)
        whole_pattern[:, :pattern_size, :pattern_size] = pattern
        pattern = whole_pattern
        weight = np.zeros((3, h, w), dtype=np.float32)
        weight[np.nonzero(pattern)] = 1.0

        pattern = torch.from_numpy(pattern)
        weight = torch.from_numpy(weight)

    else:
        raise NotImplementedError

    return pattern, weight


def get_pattern_and_weight_blended(dataset, data_shape):
    _, h, w = data_shape
    if dataset == "cifar10" or dataset == "cifar100":
        pattern = Image.open("./resources/hello_kitty.png").convert("RGB").resize((w, h))
        pattern = np.transpose(np.array(pattern), (2, 0, 1))
    elif dataset == "gtsrb":
        pattern = Image.open("./resources/hello_kitty.png").convert("RGB").resize((w, h))
        pattern = np.transpose(np.array(pattern), (2, 0, 1))
    elif dataset == "imagenet" or dataset == "vggface2" or dataset == "imagenet1k":
        # random_noise_pattern_path = "./resources/imagenet_random_noise.pt"
        # if os.path.exists(random_noise_pattern_path):
        #     pattern = torch.load(random_noise_pattern_path)
        # else:
        pattern = (torch.rand(data_shape) * 255).type(torch.uint8)
            # torch.save(pattern, random_noise_pattern_path)
    else:
        raise NotImplementedError

    if isinstance(pattern, np.ndarray):
        pattern = torch.from_numpy(pattern)

    weight = torch.ones(data_shape) * 0.1
    return pattern, weight


def get_poisoned_dataset_kwargs(attack, target_label, poisoned_rate, data_shape, **kwargs):
    print("Attacking with {}".format(attack))

    poisoned_transform_index = kwargs.get("poisoned_transform_index", get_poisoned_transform_index(kwargs["dataset"]))
    poisoned_test_transform_index = kwargs.get("poisoned_test_transform_index", poisoned_transform_index)
    poisoned_dataset_kwargs = {
        "y_target": target_label,
        "poisoned_transform_index": poisoned_transform_index,
        "poisoned_target_transform_index": 0,
        "poisoned_test_transform_index": poisoned_test_transform_index,
        "poisoned_rate": poisoned_rate,
    }
    if attack == "badnets":
        pattern, weight = get_pattern_and_weight_badnets(kwargs["dataset"], data_shape)

        poisoned_dataset_kwargs.update({
            "pattern": pattern,
            "weight": weight,
        })
    elif attack == "blended":
        pattern, weight = get_pattern_and_weight_blended(kwargs["dataset"], data_shape)
        poisoned_dataset_kwargs.update({
            "pattern": pattern,
            "weight": weight,
        })
    elif attack == "wanet":
        if kwargs["dataset"] == "cifar10" or kwargs["dataset"] == "cifar100" or kwargs["dataset"] == "gtsrb":
            k = 4
        elif kwargs["dataset"] == "imagenet" or kwargs["dataset"] == "vggface2" or kwargs["dataset"] == "imagenet1k":
            k = 224
        else:
            raise NotImplementedError
        
        height = data_shape[1]
        ins = torch.rand(1, 2, k, k) * 2 - 1
        ins = ins / torch.mean(torch.abs(ins))  # a uniform grid
        noise_grid = F.upsample(ins, size=height, mode="bicubic", align_corners=True)
        noise_grid = noise_grid.permute(0, 2, 3, 1)  # 1*height*height*2
        array1d = torch.linspace(-1, 1, steps=height)  # 1D coordinate divided by height in [-1, 1]
        x, y = torch.meshgrid(array1d, array1d)  # 2D coordinates height*height
        identity_grid = torch.stack((y, x), 2)[None, ...]  # 1*height*height*2

        poisoned_dataset_kwargs.update({
            "noise_grid": noise_grid,
            "identity_grid": identity_grid,
            "noise": False,
        })
    elif attack == "labelconsistent":
        _, h, w = data_shape
        # cifar10 pattern
        pattern = torch.zeros((h, w), dtype=torch.uint8)
        pattern[-1, -1] = 255
        pattern[-1, -3] = 255
        pattern[-3, -1] = 255
        pattern[-2, -2] = 255

        pattern[0, -1] = 255
        pattern[1, -2] = 255
        pattern[2, -3] = 255
        pattern[2, -1] = 255

        pattern[0, 0] = 255
        pattern[1, 1] = 255
        pattern[2, 2] = 255
        pattern[2, 0] = 255

        pattern[-1, 0] = 255
        pattern[-1, 2] = 255
        pattern[-2, 1] = 255
        pattern[-3, 0] = 255

        weight = torch.zeros((h, w), dtype=torch.float32)
        weight[:3, :3] = 1.0
        weight[:3, -3:] = 1.0
        weight[-3:, :3] = 1.0
        weight[-3:, -3:] = 1.0

        # imagenet pattern1
        # pattern = torch.zeros((h, w), dtype=torch.uint8)
        # pattern_size = 16
        # pattern[-pattern_size:, -pattern_size:] = 255
        #
        # pattern[:pattern_size, -pattern_size:] = 255
        #
        # pattern[-pattern_size:, :pattern_size] = 255
        #
        # pattern[:pattern_size, :pattern_size] = 255
        #
        # weight = torch.zeros((h, w), dtype=torch.float32)
        # weight[-pattern_size:, -pattern_size:] = 1.0
        # weight[:pattern_size, -pattern_size:] = 1.0
        # weight[-pattern_size:, :pattern_size] = 1.0
        # weight[:pattern_size, :pattern_size] = 1.0

        # imagenet pattern2
        # pattern_size = 16
        # pattern_path = "./resources/apple-logo.jpg"
        # pattern = np.array(Image.open(pattern_path).convert("RGB").resize((pattern_size, pattern_size)))
        # pattern = np.transpose(pattern, (2, 0, 1))
        #
        # whole_pattern = np.zeros((3, h, w), dtype=np.uint8)
        # whole_pattern[:, :pattern_size, :pattern_size] = pattern
        # whole_pattern[:, -pattern_size:, -pattern_size:] = pattern
        # whole_pattern[:, :pattern_size, -pattern_size:] = pattern
        # whole_pattern[:, -pattern_size:, :pattern_size] = pattern
        #
        # pattern = whole_pattern
        # weight = np.zeros((3, h, w), dtype=np.float32)
        # weight[np.nonzero(pattern)] = 1.0
        #
        # pattern = torch.from_numpy(pattern)
        # weight = torch.from_numpy(weight)

        if kwargs["dataset"] == "imagenet":
            adv_model = torchvision.models.resnet18(weights=None, num_classes=30)
        else:
            adv_model = models.ResNet(18)
        adv_ckpt = torch.load(f"/home/isabolic/LikelihoodEstimationBasedDefense/experiments/benign_{kwargs['dataset']}/benign_training/latest.pth")
        adv_model.load_state_dict(adv_ckpt)
        eps = 16
        adv_transform = transforms.Compose([
            transforms.Resize((h, w)),
            transforms.ToTensor(),
        ])

        print("HARDCODING EXPERIMENT DIR")
        experiment_dir = "/home/isabolic/LikelihoodEstimationBasedDefense/experiments/cifar10_labelconsistent_0.25_0_resnet18/"
        poisoned_dataset_kwargs.update({
            "adv_model": adv_model,
            "adv_dataset_dir": "{}/{}_eps{}".format(experiment_dir, "adv_dataset", eps),
            "device": kwargs["device"] if "device" in kwargs else None,
            "pattern": pattern,
            "adv_transform": adv_transform,
            "weight": weight,
            "eps": eps,
            "steps": 30,
        })
    elif attack == "issba":
        encoder_schedule = {
            'secret_size': 20,
            'enc_height': 32,
            'enc_width': 32,
            'enc_in_channel': 3,
            'enc_total_epoch': 20,
            'enc_secret_only_epoch': 2,
            'enc_use_dis': False,
        }

        poisoned_dataset_kwargs.update({
            "dataset_name": kwargs["dataset"],
            "experiment_dir": kwargs["experiment_dir"],
            "encoder_schedule": encoder_schedule,
            "transform": kwargs["transforms"] if "transforms" in kwargs else None,
            # "device": kwargs["device"],
        })

    elif "cbd" in attack:
        attack_type = attack.split("_")[1]
        if attack_type == "blend":
            trigger_name = "hellokitty_32.png"
            trigger = Image.open(f"./resources/{trigger_name}").convert("RGB").resize((data_shape[2], data_shape[1]))
        elif attack_type == "patch":
            trigger_name = None
            trigger = None
        else:
            raise NotImplementedError

        if trigger:
            trigger = transforms.ToTensor()(trigger)
        poisoned_rate = poisoned_dataset_kwargs.pop("poisoned_rate")
        poison_rate = poisoned_rate / 2
        cover_rate = poisoned_rate / 2
        alpha = 0.2

        poisoned_dataset_kwargs.update({
            "dataset_name": kwargs["dataset"],
            "attack_type": attack_type,
            "trigger_name": trigger_name,
            "trigger": trigger,
            "poison_rate": poison_rate,
            "cover_rate": cover_rate,
            "alpha": alpha,
        })
    elif attack == "refool":
        def read_image(img_path, type=None):
            img = cv2.imread(img_path)
            if type is None:
                return img
            elif isinstance(type, str) and type.upper() == "RGB":
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif isinstance(type, str) and type.upper() == "GRAY":
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                raise NotImplementedError
        reflection_images_dir = "/mnt/sdb1/datasets/VOCdevkit/VOC2012/JPEGImages/"
        reflection_images = [read_image(os.path.join(reflection_images_dir, img_path)) for img_path in os.listdir(reflection_images_dir[:200])]
        poisoned_dataset_kwargs.update({
            "reflection_candidates": reflection_images,
        })

    elif attack == "sig":
        delta = 20
        freq = 6
        poisoned_dataset_kwargs.update({
            "delta": delta,
            "freq": freq,
        })
    elif attack == "adaptive":
        encoder = get_resnet("resnet18_cifar", weights=None)
        n_features = 512
        selfsup_model = SimCLR(encoder, 128, n_features)
        selfsup_model.load_state_dict(torch.load("experiments/cifar10_clean_0.0_0_resnet18_cifar/checkpoints/latest.tar"))
        poison_schedule = {
            "lr": 0.1,
            "num_epochs": 100,
        }
        poisoned_dataset_kwargs.update({
            "dataset_name": kwargs["dataset"],
            "experiment_dir": kwargs["experiment_dir"],
            "device": kwargs["device"] if "device" in kwargs else None,
            "poison_schedule": poison_schedule,
            "selfsup_model": selfsup_model,
        })

    elif attack == "clean":
        pass
    else:
        raise NotImplementedError(f"Attack {attack} not implemented")

    return poisoned_dataset_kwargs


def get_data_shape(dataset):
    # if dataset == 'cifar10' or dataset == 'cifar10sup' or dataset == 'cifar100sup' or dataset == 'cifar10poisoned':
    if 'cifar' in dataset:
        return (3,32,32)
    elif dataset == 'gtsrb':
        return (3,48,48)
    elif dataset == 'tinyimagenet':
        return (3,64,64)
    elif dataset == 'imagenet':
        return (3,224,224)
    elif dataset == 'imagenet-100':
        return (3,224,224)
    elif dataset == 'imagenet1k':
        return (3,224,224)
    elif dataset == 'imagenet32':
        return (3,32,32)
    elif dataset == 'imagenet64' or dataset == 'celeba':
        return (3,64,64)
    elif dataset == 'svhn' or dataset == 'svhnsup':
        return (3,32,32)
    elif dataset == 'mnist':
        return (1,28,28)
    elif dataset == 'vggface2':
        return (3,224,224)

    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")


def get_poisoned_datasets(dataset, data_dir, attack, target_label, poisoned_rate, args, transforms=None, test_transforms=None, **kwargs):
    poisoned_dataset_kwargs = get_poisoned_dataset_kwargs(attack, target_label, poisoned_rate, get_data_shape(dataset),
                                                          dataset=dataset, transforms=transforms, **kwargs)

    # ove dvije bi se moglo refaktorati u tvornice
    if dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transforms)
        test_dataset = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=test_transforms)

        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedCIFAR10
        elif attack == "blended":
            poisoned_dataset = BlendedPoisonedCIFAR10
        elif attack == "wanet":
            poisoned_dataset = WaNetPoisonedCIFAR10
        elif attack == "labelconsistent":
            poisoned_dataset = LabelConsistentPoisonedCIFAR10
        elif attack == "issba":
            poisoned_dataset = ISSBAPoisonedCIFAR10
        elif "cbd" in attack:
            poisoned_dataset = CBDPoisonedDataset
        elif attack == "sig":
            poisoned_dataset = SIGPoisonedCIFAR10
        elif attack == "adaptive":
            poisoned_dataset = AdaptivePoisonedCIFAR10
        elif attack == "clean":
            pass
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transforms)
        test_dataset = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=test_transforms)

        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedCIFAR100
        elif attack == "blended":
            poisoned_dataset = BlendedPoisonedCIFAR100
        elif attack == "wanet":
            poisoned_dataset = WaNetPoisonedCIFAR100
        elif attack == "clean":
            pass
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")

    elif dataset == 'gtsrb':
        train_dataset = torchvision.datasets.GTSRB(root=data_dir, split="train", download=True, transform=transforms)
        test_dataset = torchvision.datasets.GTSRB(root=data_dir, split="test", download=True, transform=test_transforms)

        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedGTSRB
        elif attack == "blended":
            poisoned_dataset = BlendedPoisonedGTSRB
        elif attack == "wanet":
            poisoned_dataset = WaNetPoisonedGTSRB
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")

    elif dataset == 'tinyimagenet':
        subset_num_classes = 30
        subset = torch.arange(subset_num_classes)

        train_dataset = OneClassImageFolder(target_class=subset, root=data_dir + "/tiny-imagenet-200/train", transform=transforms)
        test_dataset = OneClassImageFolder(target_class=subset, root=data_dir + "/tiny-imagenet-200/val", transform=test_transforms)

        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedDatasetFolder
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")

    elif dataset == 'imagenet1k':
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/imagenet1k/train", transform=transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/imagenet1k/test", transform=test_transforms)

        # poisoned_dataset_kwargs["poisoned_transform_index"] = 1
        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedDatasetFolder
        elif attack == "blended":
            poisoned_dataset = BlendedPoisonedDatasetFolder
        elif attack == "wanet":
            poisoned_dataset = WaNetPoisonedDatasetFolder
        elif attack == "clean":
            pass
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")

    elif dataset == 'imagenet':
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/imagenet30/train", transform=transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/imagenet30/test", transform=test_transforms)

        # poisoned_dataset_kwargs["poisoned_transform_index"] = 1
        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedDatasetFolder
        elif attack == "blended":
            poisoned_dataset = BlendedPoisonedDatasetFolder
        elif attack == "wanet":
            poisoned_dataset = WaNetPoisonedDatasetFolder
        elif attack == "clean":
            pass
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")

    elif dataset == 'imagenet-100':
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/imagenet-100/train", transform=transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/imagenet-100/test", transform=test_transforms)

        # poisoned_dataset_kwargs["poisoned_transform_index"] = 1
        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedDatasetFolder
        elif attack == "blended":
            poisoned_dataset = BlendedPoisonedDatasetFolder
        elif attack == "wanet":
            poisoned_dataset = WaNetPoisonedDatasetFolder
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")

    elif dataset == 'vggface2':
        train_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/vggface2/train", transform=transforms)
        test_dataset = torchvision.datasets.ImageFolder(root=data_dir + "/vggface2/test", transform=test_transforms)

        poisoned_dataset_kwargs["poisoned_transform_index"] = 1
        if attack == "badnets":
            poisoned_dataset = BadNetsPoisonedDatasetFolder
        elif attack == "blended":
            poisoned_dataset = BlendedPoisonedDatasetFolder
        elif attack == "wanet":
            poisoned_dataset = WaNetPoisonedDatasetFolder
        else:
            raise NotImplementedError(f"Attack {attack} not implemented for dataset {dataset}")
    else:
        raise NotImplementedError(f"Dataset {dataset} not implemented")

    if args.train_poisoned_set == "":
        raise ValueError("train_poisoned_set must be set")
    train_poisoned_set = args.train_poisoned_set if hasattr(args, "train_poisoned_set") else None
    test_poisoned_set = args.test_poisoned_set if args.test_poisoned_set != "" else None

    if args.attack != "clean":
        train_kwargs = dict(poisoned_dataset_kwargs)
        train_kwargs.pop("poisoned_test_transform_index")
        train_dataset = poisoned_dataset(train_dataset, poisoned_set=train_poisoned_set, **train_kwargs)
        if test_dataset is not None:
            if args.attack == "labelconsistent" or args.attack == "issba":
                poisoned_dataset_kwargs["train"] = False
            if args.attack == "labelconsistent":
                poisoned_dataset = BadNetsPoisonedCIFAR10
            if "cbd" in args.attack:
                poisoned_dataset = CBDTestPoisonedDataset

            # if args.attack == "labelconsistent":
            #     poisoned_dataset = LabelConsistentNoAdvCIFAR10
            poisoned_dataset_kwargs["poisoned_rate"] = 1.0 # test dataset is completely poisoned
            poisoned_dataset_kwargs["poisoned_transform_index"] = poisoned_dataset_kwargs.pop("poisoned_test_transform_index", 0)
            test_dataset = poisoned_dataset(test_dataset, **poisoned_dataset_kwargs)

    return train_dataset, test_dataset


def get_poisoned_loader(dataset, data_dir, attack, target_label, poisoned_rate, args, transforms=None, test_transforms=None, **kwargs):
    train_dataset, test_dataset = get_poisoned_datasets(dataset, data_dir, attack, target_label, poisoned_rate, args,
                                                        transforms, test_transforms, **kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                               pin_memory=args.pin_memory, drop_last=True)
    if test_dataset is not None:
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                                                pin_memory=args.pin_memory, drop_last=True)
    else:
        test_loader = None

    return train_loader, test_loader



