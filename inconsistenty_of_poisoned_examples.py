import math
import json
import os

import torch
import numpy as np
import torchvision
from torch import nn
from torchvision import transforms

import models

from images_dataset import get_poisoned_datasets
import argparse

from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip


class SubsetWithLabelsDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, indices):
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, index):
        x, y = self.dataset[self.indices[index]]
        return x, y

    def __len__(self):
        return len(self.indices)


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


train_transform = Compose([
    RandomCrop(32, padding=4),
    RandomHorizontalFlip(),
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

test_transform = Compose([
    ToTensor(),
    Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

def get_images_datasets(args, subset="all"):
    assert subset in ["all", "poisoned", "clean"]
    train_dataset, _ = get_poisoned_datasets(
        args.dataset,
        args.data_dir,
        args.attack,
        args.target_label,
        args.poisoned_rate,
        args,
        transforms=train_transform,
        experiment_dir=None,
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    if subset == "all":
        return train_dataset, test_dataset

    poisoned_set = train_dataset.poisoned_set
    if subset == "poisoned":
        return SubsetWithLabelsDataset(train_dataset, poisoned_set), test_dataset
    if subset == "clean":
        clean_set = torch.arange(len(train_dataset))
        clean_set = torch.from_numpy(np.setdiff1d(clean_set, poisoned_set))
        return SubsetWithLabelsDataset(train_dataset, clean_set), test_dataset


def get_selfsup_datasets(args):
    features = os.path.join(
        '/home/isabolic/LikelihoodEstimationBasedDefense/experiments',
        f'{args.dataset}_{args.attack}_{args.poisoned_rate}_{args.target_label}_{args.selfsup_backbone}',
        'extracted_features.npz'
    )

    print("Loading features from {}".format(features))

    feature_train_transforms = []
    feature_train_transforms.append(transforms.Lambda(lambda x: torch.Tensor(x)))
    feature_train_transforms = transforms.Compose(feature_train_transforms)

    feature_test_transforms = transforms.Lambda(lambda x: torch.Tensor(x))
    train_dataset = FeaturesDataset(features, 'train', transform=feature_train_transforms)
    clean_test_dataset = FeaturesDataset(features, 'clean_test', transform=feature_test_transforms)

    poisoned_set = torch.load(f"{args.train_poisoned_set}.pt")

    if args.subset == "poisoned":
        train_dataset = SubsetWithLabelsDataset(train_dataset, poisoned_set)
    elif args.subset == "clean":
        clean_set = torch.arange(len(train_dataset))
        clean_set = torch.from_numpy(np.setdiff1d(clean_set, poisoned_set))
        train_dataset = SubsetWithLabelsDataset(train_dataset, clean_set)

    if 'resnet18' in args.selfsup_backbone:
        data_shape = 512
    elif 'resnet50' in args.selfsup_backbone:
        data_shape = 1024
    elif args.selfsup_backbone == 'densenet121':
        data_shape = 1024

    return train_dataset, clean_test_dataset, data_shape


def get_loaders(args, batch_size, shuffle=True):
    data_shape = None
    if args.data_type == "images":
        train_dataset, test_dataset = get_images_datasets(args, args.subset)
    elif args.data_type == "selfsupfeatures":
        train_dataset, test_dataset, data_shape = get_selfsup_datasets(args)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
    )

    return train_loader, test_loader, data_shape


def plot_images(x, y):
    import matplotlib.pyplot as plt

    # save batch of images to file
    for i, (im, y_) in enumerate(zip(x, y)):
        im = im.numpy().transpose((1, 2, 0))
        plt.imsave(f"images/{i}_{y_}.png", im)


def train_one_epoch(model, optimizer, train_loader, criterion):
    model.train()
    # measure average loss and total accuracy in this epoch
    train_loss = 0
    train_total = 0
    train_correct = 0
    for i, (x, y) in enumerate(train_loader):
        # plot_images(x, y)
        x, y = x.cuda(), y.cuda()
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = y_hat.max(1)
        train_total += y.size(0)
        train_correct += predicted.eq(y).sum().item()

    return {
        "train_loss": train_loss / len(train_loader),
        "train_accuracy": train_correct / train_total,
    }


def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    test_total = 0
    test_correct = 0
    y_true = []
    y_score = []
    with torch.no_grad():
        for i, (x, y) in enumerate(test_loader):
            x, y = x.cuda(), y.cuda()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.item()
            _, predicted = y_hat.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            y_true.extend(y.cpu().numpy())
            y_score.extend(y_hat.detach().cpu().numpy()[:, 1])

    return {
        "test_loss": test_loss / len(test_loader),
        "test_accuracy": test_correct / test_total,
    }

def run_experiment(args):
    assert args.data_type in ["images", "selfsupfeatures"]
    print(args)
    out_file_name = f"{args.attack}_{args.data_type}_{args.subset}"
    out_file = os.path.join(args.experiment_dir, f"{out_file_name}.txt")
    if os.path.exists(out_file):
        os.remove(out_file)
    out_file_json = os.path.join(args.experiment_dir, f"{out_file_name}.json")
    out_file_test_json = os.path.join(args.experiment_dir, f"{out_file_name}_test.json")

    train_loader, test_loader, data_shape = get_loaders(args, args.batch_size)

    if args.data_type == "images":
        model = models.ResNet(18, num_classes=10).cuda()
    elif args.data_type == "selfsupfeatures":
        model = nn.Sequential(
            nn.Linear(data_shape, 10)
        ).cuda()


    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=5e-4)

    criterion = torch.nn.CrossEntropyLoss()

    all_train_stats = []
    all_test_stats = []
    print(f"Len(Dataset) = {len(train_loader.dataset)}")
    for epoch in range(args.epochs):
        if (epoch + 1) in args.milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        train_stats = train_one_epoch(model, optimizer, train_loader, criterion)
        print(f"Epoch {epoch}: {train_stats}")
        all_train_stats.append(train_stats)
        with open(out_file, 'a') as f:
            f.write(f"Epoch {epoch}: {train_stats}\n")
        with open(out_file_json, 'w') as f:
            json.dump(all_train_stats, f)

        if (epoch + 1) % args.test_epoch == 0:
            test_stats = test(model, test_loader, criterion)
            print(
                f"TEST Epoch {epoch}: {test_stats}"
            )

            all_test_stats.append(test_stats)
            with open(out_file, 'a') as f:
                f.write(
                    f"Epoch {epoch}: {test_stats}\n"
                )
            with open(out_file_test_json, 'w') as f:
                json.dump(all_test_stats, f)

    test_stats = test(model, test_loader, criterion)

    return test_stats


def main(args):
    if not os.path.exists(args.experiment_dir):
        os.makedirs(args.experiment_dir)
    run_experiment(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default="lastrun")
    parser.add_argument('--data_type', type=str, default="images")
    parser.add_argument('--subset', type=str, default="all")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--attack', type=str, default='badnets')
    parser.add_argument('--poisoned_rate', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--selfsup_backbone', type=str, default='resnet18_cifar')
    parser.add_argument('--data_dir', type=str, required=False,
                        default="/home/isabolic/LikelihoodEstimationBasedDefense/data")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75])
    parser.add_argument('--test_epoch', type=int, default=5)
    parser.add_argument("--train_poisoned_set", type=str, default=None)
    parser.add_argument("--test_poisoned_set", type=str, default=None)

    main(parser.parse_args())
