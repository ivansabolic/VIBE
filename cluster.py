import math
import json
import os
import time
from typing import Iterable

import matplotlib
from omegaconf import OmegaConf
import torch
import numpy as np
from pathlib import Path
import torchvision
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
import open_clip

from solo.methods import METHODS

from images_dataset import get_poisoned_datasets, dataset_num_classes
import argparse

from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip, Resize
from tqdm import tqdm

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

train_transform_cifar = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    # Resize((224, 224)),
])

supervise_train_transform_cifar = Compose([
    # RandomCrop(32, padding=4),
    # RandomHorizontalFlip(),
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    # Resize((224, 224)),
])

test_transform_cifar = Compose([
    ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    # Resize((224, 224)),
])

train_transform_imagenet = Compose([
    transforms.RandomResizedCrop(size=224, scale=(0.08, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

supervise_train_transform_imagenet = Compose([
    transforms.Resize(256),  # resize shorter
    transforms.CenterCrop(224),  # take center crop
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

test_transform_imagenet = Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
])

train_transform = {
    "cifar10": train_transform_cifar,
    "cifar100": train_transform_cifar,
    "imagenet": train_transform_imagenet,
    "imagenet-100": train_transform_imagenet,
    "imagenet1k": train_transform_imagenet,
}
supervise_train_transform = {
    "cifar10": supervise_train_transform_cifar,
    "cifar100": supervise_train_transform_cifar,
    "imagenet": supervise_train_transform_imagenet,
    "imagenet-100": supervise_train_transform_imagenet,
    "imagenet1k": supervise_train_transform_imagenet,
}
test_transform = {
    "cifar10": test_transform_cifar,
    "cifar100": test_transform_cifar,
    "imagenet": test_transform_imagenet,
    "imagenet-100": supervise_train_transform_imagenet,
    "imagenet1k": supervise_train_transform_imagenet,
}

def get_clean_test_dataset(dataset, data_dir, test_transform):
    assert dataset in ["cifar10", "cifar100", "imagenet-100", "imagenet", "imagenet1k"]
    if dataset == "cifar10":
        clean_test_dataset = torchvision.datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
    elif dataset == "cifar100":
        clean_test_dataset = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=False,
            download=True,
            transform=test_transform,
        )
    elif dataset == "imagenet-100":
        clean_test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "imagenet-100", "val"),
            transform=test_transform,
        )

    elif dataset == "imagenet":
        clean_test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "imagenet30", "test"),
            transform=test_transform,
        )
    elif dataset == "imagenet1k":
        clean_test_dataset = torchvision.datasets.ImageFolder(
            os.path.join(data_dir, "imagenet1k", "test"),
            transform=test_transform,
        )

    return clean_test_dataset

class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class FeaturesDataset(torch.utils.data.Dataset):
    def __init__(self, features_path, split, transform=None):
        loaded = np.load(features_path)

        self.representations = loaded[f'{split}_X']

        self.features = torch.Tensor(self.representations)
        self.labels = loaded[f'{split}_y']
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.features[index]
        y = self.labels[index]
        if self.transform:
            x = self.transform(x)
        return x, y


class OneClassDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target_class):
        self.dataset = dataset
        if type(target_class) == int:
            self.target_indices = torch.arange(len(dataset.targets))[torch.Tensor(dataset.targets) == target_class]
        elif isinstance(target_class, Iterable):
            self.target_indices = torch.arange(len(dataset.targets))[
                torch.isin(torch.Tensor(dataset.targets), torch.Tensor(list(target_class)))]

    def __getitem__(self, index):
        index = self.target_indices[index].item()
        x, y = self.dataset[index]

        return x, y

    def __len__(self):
        return len(self.target_indices)


class OverridedLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, labels):
        self.dataset = dataset
        self.labels = labels

    def __getitem__(self, index):
        x, _ = self.dataset[index]
        return x, self.labels[index], index

    def __len__(self):
        return len(self.dataset)


class FilteredDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        return self.dataset[self.indices[index]]

    def __len__(self):
        return len(self.indices)

def load_features(args, method="byol", origin="lebd", add_info=""):
    assert origin in ["lebd", "sololearn"]
    if origin == "lebd":
        features = os.path.join(
            '/home/isabolic/LikelihoodEstimationBasedDefense/experiments',
            args.experiment_name,
            f'extracted_features{add_info}.npz'
        )
    elif origin == "sololearn":
        features = os.path.join(
            '/home/isabolic/solo-learn/extracted_features',
            f"{method}_{args.attack}_{args.poisoned_rate}_{args.target_label}.npz"
        )

    print(f"Loading features from {features}")
    return np.load(features)


def get_loader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
    )

def get_loaders(train_dataset, test_dataset, batch_size, shuffle=True):
    train_loader = get_loader(train_dataset, batch_size, shuffle)

    if test_dataset != None:
        test_loader = get_loader(test_dataset, batch_size, shuffle=False)
    else:
        test_loader = None

    return train_loader, test_loader


def find_centroids(features, labels):
    num_classes = len(np.unique(labels))
    centroids = np.zeros((num_classes, features.shape[1]))
    for i in range(num_classes):
        class_features = features[labels == i]
        centroids[i] = np.mean(class_features, axis=0)

    # normalize
    centroids = centroids / np.linalg.norm(centroids, axis=1)[:, None]
    return centroids

def get_indices_for_label_consistent(adv_dataset_dir):
    class_paths = os.listdir(adv_dataset_dir)
    indices = []
    for class_path in sorted(class_paths): # todo works only for cifar10!!
        imgs = os.listdir(os.path.join(adv_dataset_dir, class_path))
        for img in sorted(imgs):
            indices.append(int(img.split(".")[0]))

    return np.array(indices)

def plot_data(features, labels, centroids, poisoned_set=None, save_path="out.png", subsample=1., algorithm='tsne', original_centroids=None):
    if algorithm == 'tsne':
        from sklearn.manifold import TSNE
        dim_red_algorithm = TSNE(n_components=2)
    elif algorithm == 'umap':
        import umap
        dim_red_algorithm = umap.UMAP(n_components=2)
    elif algorithm == 'pca':
        from sklearn.decomposition import PCA
        dim_red_algorithm = PCA(n_components=2)
    else:
        raise ValueError(f"Unknown dimensionality reduction algorithm: {algorithm}")

    fac = np.concatenate([features, centroids], axis=0)
    if original_centroids is not None:
        fac = np.concatenate([fac, original_centroids], axis=0)
    reduced_fac = dim_red_algorithm.fit_transform(fac)

    if original_centroids is not None:
        original_centroids_2d = reduced_fac[-original_centroids.shape[0]:]
        reduced_fac = reduced_fac[:-original_centroids.shape[0]]

    features_2d = reduced_fac[:-centroids.shape[0]]
    centroids_2d = reduced_fac[-centroids.shape[0]:]

    if subsample < 1.:
        indices = np.random.choice(len(labels), int(subsample * len(labels)), replace=False)
        features_2d = features_2d[indices]
        labels = labels[indices]
        poisoned_set = poisoned_set[indices]

    unique_labels = np.unique(labels)
    colors = matplotlib.colormaps['tab20']
    if len(unique_labels) <= 20:
        colors = [colors(i) for i in range(20)]
    elif len(unique_labels) <= 40:
        colors_b = matplotlib.colormaps['tab20b']
        colors_c = matplotlib.colormaps['tab20c']
        colors = [colors_b(i) for i in range(20)] + [colors_c(i) for i in range(20)] + [colors(i) for i in range(20)]
    else:
        num_classes = 100
        # 100 different colors
        colors = [plt.cm.tab20(i) for i in range(num_classes)]


    plt.figure(figsize=(20, 20))
    for i in range(len(unique_labels)):
        mask = labels == i
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1], color=colors[i], label=i, alpha=0.5)
        # plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], color=colors[i], marker='s', s=250, edgecolors='black')

    if poisoned_set is not None:
        plt.scatter(features_2d[poisoned_set, 0], features_2d[poisoned_set, 1], color='black', label='poisoned', alpha=0.35)

    for i in range(centroids.shape[0]):
        plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], color=colors[i], marker='s', s=250, edgecolors='black') # squares
        if original_centroids is not None:
            plt.scatter(original_centroids_2d[i, 0], original_centroids_2d[i, 1], color=colors[i], marker='v', s=250, edgecolors='black') # triangles
    plt.legend()
    plt.savefig(save_path)
    plt.close()


def load_original_labels(dataset, train=True):
    assert dataset in ['cifar10', 'cifar100', 'imagenet-100', 'imagenet', 'imagenet1k']
    if dataset == 'cifar10':
        if train:
            return np.array(torchvision.datasets.CIFAR10(root='/home/isabolic/LikelihoodEstimationBasedDefense/data', train=True,
                                            download=False).targets)
        else:
            return np.array(torchvision.datasets.CIFAR10(root='/home/isabolic/LikelihoodEstimationBasedDefense/data', train=False,
                                            download=False).targets)
    if dataset == 'cifar100':
        if train:
            return np.array(torchvision.datasets.CIFAR100(root='/home/isabolic/LikelihoodEstimationBasedDefense/data', train=True,
                                            download=False).targets)
        else:
            return np.array(torchvision.datasets.CIFAR100(root='/home/isabolic/LikelihoodEstimationBasedDefense/data', train=False,
                                            download=False).targets)
    if dataset == 'imagenet-100':
        if train:
            return np.array(torchvision.datasets.ImageFolder('/home/isabolic/LikelihoodEstimationBasedDefense/data/imagenet-100/train').targets)
        else:
            return np.array(torchvision.datasets.ImageFolder('/home/isabolic/LikelihoodEstimationBasedDefense/data/imagenet-100/val').targets)

    if dataset == 'imagenet':
        if train:
            return np.array(torchvision.datasets.ImageFolder('/home/isabolic/LikelihoodEstimationBasedDefense/data/imagenet30/train').targets)
        else:
            return np.array(torchvision.datasets.ImageFolder('/home/isabolic/LikelihoodEstimationBasedDefense/data/imagenet30/test').targets)

    if dataset == 'imagenet1k':
        if train:
            return np.array(torchvision.datasets.ImageFolder('/home/isabolic/LikelihoodEstimationBasedDefense/data/imagenet1k/train').targets)
        else:
            return np.array(torchvision.datasets.ImageFolder('/home/isabolic/LikelihoodEstimationBasedDefense/data/imagenet1k/test').targets)
def optimize_l_sk(prob, lmd, ddtype=np.float64):
    # tt = time()
    n = prob.shape[0]
    k = prob.shape[1]

    prob = ddtype(prob)
    prob = prob.T  # (k, n)
    r = np.ones((k, 1), dtype=ddtype) / k
    c = np.ones((n, 1), dtype=ddtype) / n
    prob **= lmd  # (k, n)
    inv_k = ddtype(1. / k)
    inv_n = ddtype(1. / n)
    err = 1e6
    cnt = 0

    while err > 1e-1:
        r = inv_k / (prob @ c)  # (k, n) @ (n, 1) = (k, 1)
        c_new = inv_n / (r.T @ prob).T  # ((1, k) @ (k, n)).t() = (n, 1)
        if cnt % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        cnt += 1
        # print("sinkhornknopp: error: ", err, 'step ', cnt, flush=True)  # " nonneg: ", sum(I), flush=True)

    # inplace calculations
    # ---
    prob *= np.squeeze(c)
    prob = prob.T
    prob *= np.squeeze(r)  # (n, k)
    # prob = prob.T  # (k, n)
    try:
        argmaxes = np.nanargmax(prob, axis=1)  # (n,)
    except:
        breakpoint()

    # print('opt took {0:.2f}min, {1:4d}iters'.format(((time() - tt) / 60.), cnt), flush=True)

    return prob, argmaxes


def extract_features(loader, model, model_type="selfsup"):
    data = []

    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Collecting features"):
            x = x.cuda(non_blocking=True)

            if model_type == "clip":
                feats = model.encode_image(x)
            else:
                feats = model(x)
            data.append(feats.cpu())
    model.train()

    data = torch.cat(data, dim=0).numpy()
    # print("NOT NORMALIZED!!!")
    data = data / np.linalg.norm(data, axis=1)[:, None]

    return data


def extract_or_cache(args, extract_function, loader, model, return_labels=False, cache_info=None , **kwargs):
    cache_dir = "./cache"
    os.makedirs(cache_dir, exist_ok=True)
    if args.poisoned_rate >= 1:
        args.poisoned_rate = int(args.poisoned_rate)
    cache_path = f"{cache_dir}/{args.dataset}_{args.attack}_{args.poisoned_rate}_{args.target_label}_{args.pretrained_model_name}"
    if cache_info:
        cache_path += f"_{cache_info}"
    cache_path += ".npz"

    if os.path.exists(cache_path):
        print(f"Loading features from {cache_path}")
        data = np.load(cache_path)
        features = data["features"]
        labels = data["labels"]
    else:
        features = extract_function(loader, model, **kwargs)
        labels = np.array([y for _, y in loader.dataset])
        np.savez(
            cache_path,
            features=features,
            labels=labels
        )

    if return_labels:
        return features, labels

    return features


def classify(M, classifier="argmax"):
    if classifier == "argmax":
        return np.argmax(M, axis=1)
    elif classifier == "sk":
        return optimize_l_sk(M, lmd=25)[1]
    else:
        raise ValueError(f"Unknown classifier: {classifier}")


def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    model.train()
    return correct / total


def test_with_centroids(model, test_loader, centroids):
    model.eval()
    correct = 0
    total = 0
    centroids = torch.Tensor(centroids).cuda()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            y_hat = torch.matmul(y_hat, centroids.T)
            _, predicted = y_hat.max(1)
            total += y.size(0)
            correct += predicted.eq(y).sum().item()

    model.train()
    return correct / total

def test_with_features_and_centroids(features, labels, centroids):
    correct = 0
    total = 0
    centroids = torch.Tensor(centroids).cuda()
    features = torch.Tensor(features).cuda()
    with torch.no_grad():
        y_hat = torch.matmul(features, centroids.T)
        _, predicted = y_hat.max(1)
        total += len(labels)
        correct += predicted.eq(torch.Tensor(labels).cuda()).sum().item()

    return correct / total

def test_all(model, clean_test_loader, asr_test_loader, args):
    clean_acc = test(model, clean_test_loader)
    asr = test(model, asr_test_loader)

    print(f"TEST: Clean accuracy: {clean_acc:.3f}, ASR: {asr:.5f}")
    return clean_acc, asr


def test_all_with_centroids(model, clean_test_loader, asr_test_loader, centroids, args):
    clean_acc = test_with_centroids(model, clean_test_loader, centroids)
    asr = test_with_centroids(model, asr_test_loader, centroids)

    print(f"TEST: Clean accuracy: {clean_acc:.3f}, ASR: {asr:.5f}")
    return clean_acc, asr

def test_all_with_features_and_centroids(clean_test_features, clean_test_labels, asr_features, asr_labels, centroids, args):
    clean_acc = test_with_features_and_centroids(clean_test_features, clean_test_labels, centroids)
    asr = test_with_features_and_centroids(asr_features, asr_labels, centroids)

    print(f"TEST: Clean accuracy: {clean_acc:.3f}, ASR: {asr:.3f}")
    return clean_acc, asr

def train_one_epoch(model, train_loader, optimizer, criterion, normalize_linear_head=False):
    model.train()
    total_loss = 0
    total = 0
    correct = 0
    for x, y in tqdm(train_loader, desc="Training"):
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        if normalize_linear_head:
            with torch.no_grad():
                model.fc.weight.div_(model.fc.weight.norm(dim=1, keepdim=True))

        total_loss += loss.item()
        _, predicted = y_hat.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()

    return total_loss / len(train_loader), correct / total


def supervise(model, train_loader, clean_test_loader, asr_test_loader, args):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.supervise_lr, momentum=0.9, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    milestones = [int(args.supervise_epochs * m) for m in [0.5]]

    test_all(model, clean_test_loader, asr_test_loader, args)
    for epoch in range(args.supervise_epochs):
        loss, _ = train_one_epoch(model, train_loader, optimizer, criterion)
        print(f"Epoch {epoch}: Loss: {loss:.3f}, LR: {optimizer.param_groups[0]['lr']}")

        # normalize linear_head parameters
        # with torch.no_grad():
        #     model.fc.weight.div_(model.fc.weight.norm(dim=1, keepdim=True))

        if (epoch + 1) % args.supervise_test_epoch == 0:
            test_all(model, clean_test_loader, asr_test_loader, args)

        if (epoch + 1) in milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def load_selfsup_model(args, disable_relu=False):
    ckpt_dir = Path(args.pretrained_checkpoint_dir)
    args_path = ckpt_dir / "args.json"
    ckpt_path = [ckpt_dir / ckpt for ckpt in os.listdir(ckpt_dir) if ckpt.endswith(".ckpt")][0]

    # load arguments
    with open(args_path) as f:
        method_args = json.load(f)

    if disable_relu:
        method_args["backbone"]["name"] = "resnet18_wo_relu"
    cfg = OmegaConf.create(method_args)

    # build the model
    model = (
        METHODS[method_args["method"]]
        .load_from_checkpoint(ckpt_path, strict=False, cfg=cfg)
        .backbone
    )

    return model


def load_dino_model(args):
    BACKBONE_SIZE = args.backbone_size

    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"

    backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
    backbone_model.eval()

    return backbone_model


def load_clip_model(args):
    # model, _ = open_clip.create_model_from_pretrained("hf-hub:laion/CLIP-ViT-g-14-laion2B-s12B-b42K")
    # model, _ = open_clip.create_model_from_pretrained("hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K")
    # model, _ = open_clip.create_model_from_pretrained("hf-hub:laion/CLIP-ViT-B-32-laion2B-s34B-b79K") # pokazao se da radi uz dovoljan broj iteracija
    # model, _ = open_clip.create_model_from_pretrained("hf-hub:laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup")
    model, _ = open_clip.create_model_from_pretrained("hf-hub:laion/CLIP-convnext_xxlarge-laion2B-s34B-b82K-augreg")


    model.eval()

    return model

def load_model(args, disable_relu=False):
    if args.model_type == "selfsup":
        return load_selfsup_model(args, disable_relu=disable_relu)
    elif args.model_type == "dino":
        return load_dino_model(args)
    elif args.model_type == "clip":
        return load_clip_model(args)

def forward_to_prob(x, y, mu_y, mu_z, tau=1, poisoned_set=None):
    '''
    :param x: (n, hiddim)
    :param y: (n, num_class)
    :param tau:
    :return:
    '''
    # x = F.normalize(x, p=2, dim=1)
    # mu_z = F.normalize(mu_z, p=2, dim=1)
    # mu_y = F.normalize(mu_y, p=2, dim=1)

    prob_z_x = torch.exp((x @ (mu_z.t())) / tau)  # (n, k)
    prob_z_x = prob_z_x / prob_z_x.sum(1).view(-1, 1)  # (n, k)

    prob_y_z_num = torch.exp((y @ mu_y) @ (mu_z.t()))  # (n, k)
    prob_y_z_den = torch.exp(mu_y @ (mu_z.t()))  # (c, k)
    prob_y_z = prob_y_z_num / prob_y_z_den.sum(0).view(1, -1)

    prob_y_x = prob_z_x * prob_y_z  # (n, k)

    return prob_y_x, prob_y_z, prob_z_x


def main(args):
    args.experiment_name = f"{args.dataset}_{args.attack}_{args.poisoned_rate}_{args.target_label}_{args.selfsup_backbone}"
    args.experiment_dir = os.path.join(args.experiment_dir, args.experiment_name, time.strftime("%Y-%m-%d_%H-%M-%S"))
    print(args)

    os.makedirs(args.experiment_dir)
    log_file = os.path.join(args.experiment_dir, 'log.txt')

    _, poisoned_test_dataset = get_poisoned_datasets(
        args.dataset,
        args.data_dir,
        args.attack,
        args.target_label,
        args.poisoned_rate,
        args,
        transforms=train_transform,
        test_transforms=test_transform,
        experiment_dir=None,
    )

    asr_test_dataset = OneClassDataset(poisoned_test_dataset,
                                       set(range(dataset_num_classes[args.dataset])) - {args.target_label})

    clean_test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    clean_test_loader = get_loader(clean_test_dataset, args.batch_size, shuffle=False)
    poisoned_test_loader = get_loader(asr_test_dataset, args.batch_size, shuffle=False)

    features_file = load_features(args, origin=args.origin, add_info=args.add_info)
    poisoned_set_indices = torch.load(f"{args.train_poisoned_set}.pt")
    poisoned_set = np.zeros(len(features_file['train_y']), dtype=bool)
    poisoned_set[poisoned_set_indices] = True

    features = features_file['train_X']
    features = features / np.linalg.norm(features, axis=1)[:, None]
    data_shape = features.shape[1]

    clean_test_features = features_file['clean_test_X']
    clean_test_features = clean_test_features / np.linalg.norm(clean_test_features, axis=1)[:, None]
    clean_test_labels = features_file['clean_test_y']

    asr_features = features_file['test_X']
    asr_features = asr_features / np.linalg.norm(asr_features, axis=1)[:, None]
    asr_labels = features_file['test_y']
    asr_features = asr_features[clean_test_labels != args.target_label]
    asr_labels = asr_labels[clean_test_labels != args.target_label]

    poisoned_labels = features_file['train_y']
    original_labels = load_original_labels(args.dataset)

    labels = poisoned_labels

    labels_per_class_dir = os.path.join(args.experiment_dir, 'labels_per_class')
    os.makedirs(labels_per_class_dir)
    for ii in range(args.num_iterations):
        # centroids
        centroids = find_centroids(features, labels)
        if args.plot:
            plot_data(features, labels, centroids, poisoned_set, f"{args.experiment_dir}/centroids_{ii}.png")


        clean_acc = test_with_features_and_centroids(clean_test_features, clean_test_labels, centroids)
        asr = test_with_features_and_centroids(asr_features, asr_labels, centroids)

        print(f"TEST: Clean accuracy: {clean_acc:.3f}, ASR: {asr:.3f}")

        # classification
        M = np.dot(features, centroids.T)

        labels = classify(M, classifier=args.classifier)

        # Logging data
        poisoned_not_relabeled = (labels[poisoned_set] == poisoned_labels[poisoned_set]).sum()
        poisoned_correctly_relabeled = (labels[poisoned_set] == original_labels[poisoned_set]).sum() / (poisoned_set.sum() + 1e-6)
        other_incorrectly_relabeled = (labels[~poisoned_set] != original_labels[~poisoned_set]).sum()

        plt.hist(labels, bins=range(11))
        plt.savefig(os.path.join(labels_per_class_dir, f'labels_per_class_{ii}.png'))
        plt.close()
        log_msg = f"Iter {ii}: Poisoned labels accuracy: {np.mean(labels == poisoned_labels) :.3f}, " \
              f"Original labels accuracy: {np.mean(labels == original_labels) : .3f}, " \
              f"Number of poisoned not relabeled: {poisoned_not_relabeled} / {poisoned_set.sum()}, " \
              f"Poisoned correctly relabeled: {poisoned_correctly_relabeled:.3f}, " \
              f"Number of other incorrectly relabeled: {other_incorrectly_relabeled}"
        print(log_msg)

        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')




    # model = load_model(args)
    # model.fc = nn.Linear(512, dataset_num_classes[args.dataset], bias=False)
    # model = model.cuda()
    #
    # train_dataset, test_dataset = get_poisoned_datasets(
    #     args.dataset,
    #     args.data_dir,
    #     args.attack,
    #     args.target_label,
    #     args.poisoned_rate,
    #     args,
    #     transforms=supervise_train_transform,
    #     experiment_dir=None,
    # )
    # train_dataset = OverridedLabelsDataset(train_dataset, labels)
    # asr_test_dataset = OneClassDataset(test_dataset, set(range(dataset_num_classes[args.dataset])) - {args.target_label})
    #
    # clean_test_dataset = torchvision.datasets.CIFAR10(
    #     root=args.data_dir,
    #     train=False,
    #     download=True,
    #     transform=test_transform,
    # )
    #
    # train_loader = get_loader(train_dataset, args.batch_size, shuffle=True)
    # clean_test_loader = get_loader(clean_test_dataset, args.batch_size, shuffle=False)
    # asr_test_loader = get_loader(asr_test_dataset, args.batch_size, shuffle=False)
    #
    # supervise(model, train_loader, clean_test_loader, asr_test_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default="experiments")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--attack', type=str, default='badnets')
    parser.add_argument('--poisoned_rate', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--selfsup_backbone', type=str, default='resnet18_cifar')
    parser.add_argument('--origin', type=str, default='lebd')
    parser.add_argument('--add_info', type=str, default='')
    parser.add_argument('--data_dir', type=str, required=False,
                        default="/home/isabolic/LikelihoodEstimationBasedDefense/data")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_iterations', type=int, default=100)
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75])
    parser.add_argument('--test_epoch', type=int, default=5)
    parser.add_argument("--train_poisoned_set", type=str, default=None)
    parser.add_argument("--test_poisoned_set", type=str, default=None)
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--classifier", type=str, default="argmax")

    parser.add_argument("--supervise_lr", type=float, default=0.01)
    parser.add_argument("--supervise_epochs", type=int, default=50)
    parser.add_argument("--supervise_test_epoch", type=int, default=5)

    parser.add_argument("--pretrained_checkpoint_dir", type=str, required=True)

    main(parser.parse_args())


# cifar10_class_names = {
#     'cifar10': {
#         0: 'airplane',
#         1: 'automobile',
#         2: 'bird',
#         3: 'cat',
#         4: 'deer',
#         5: 'dog',
#         6: 'frog',
#         7: 'horse',
#         8: 'ship',
#         9: 'truck',
#     }
# }