import math
import json
import os
import time
from pathlib import Path

import matplotlib
import torch
import torchvision
import numpy as np
from torch import nn

from cluster_alternate import frozen_backbones
from images_dataset import get_poisoned_datasets, dataset_num_classes
from cluster import get_loaders, plot_data, find_centroids, classify, load_original_labels, load_model, \
    OverridedLabelsDataset, train_transform, supervise_train_transform, OneClassDataset, test_transform, get_loader, \
    supervise, SimpleDataset, test_all, get_clean_test_dataset, extract_or_cache
import argparse
from tqdm import tqdm



def extract_features(loader, model):
    data = []

    model.eval()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Collecting features"):
            x = x.cuda(non_blocking=True)

            feats = model(x)
            data.append(feats.cpu())
    model.train()

    data = torch.cat(data, dim=0).numpy()
    data = data / np.linalg.norm(data, axis=1)[:, None]

    return data


# def extract_features_with_labels(loader, model):
#     data = []
#     labels = []
#
#     model.eval()
#     with torch.no_grad():
#         for x, y in tqdm(loader, desc="Collecting features"):
#             x = x.cuda(non_blocking=True)
#
#             feats = model(x)
#             data.append(feats.cpu())
#             labels.append(y)
#     model.train()
#
#     data = torch.cat(data, dim=0).numpy()
#     labels = torch.cat(labels, dim=0).numpy()
#
#     return data, labels

latent_dim = {
    "dino_giant": 1536,
}

def train_one_epoch(loader, model, criterion, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for x, y in tqdm(loader, desc="Training"):
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        _, predicted = out.max(1)
        total += y.size(0)
        correct += predicted.eq(y).sum().item()
        total_loss += loss.item()

    return total_loss / len(loader), correct / total

def main(args):
    args.experiment_name = f"{args.dataset}_{args.attack}_{args.poisoned_rate}_{args.target_label}_{args.selfsup_backbone}"
    args.experiment_dir = os.path.join(args.experiment_dir, args.experiment_name, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_logistic')
    args.pretrained_model_name = f"{args.pretrained_checkpoint_dir.split('/')[-2]}"
    print(args)

    os.makedirs(args.experiment_dir)
    log = open(os.path.join(args.experiment_dir, "log.txt"), "w")

    if not args.cache_exists:
        if "cifar" in args.dataset and args.model_type in frozen_backbones:
            train_transform_list = train_transform[args.dataset].transforms
            train_transform_list.append(torchvision.transforms.Resize((224, 224)))
            train_transform[args.dataset] = torchvision.transforms.Compose(train_transform_list)

            test_transform_list = test_transform[args.dataset].transforms
            test_transform_list.append(torchvision.transforms.Resize((224, 224)))
            test_transform[args.dataset] = torchvision.transforms.Compose(test_transform_list)

        kwargs = {}
        if "imagenet" in args.dataset:
            kwargs["poisoned_test_transform_index"] = 2
        train_dataset, poisoned_test_dataset = get_poisoned_datasets(
            args.dataset,
            args.data_dir,
            args.attack,
            args.target_label,
            args.poisoned_rate,
            args,
            transforms=train_transform[args.dataset],
            test_transforms=test_transform[args.dataset],
            experiment_dir=None,
            **kwargs
        )
        clean_test_dataset = get_clean_test_dataset(args.dataset, args.data_dir, test_transform[args.dataset])
        if "cbd" not in args.attack:
            poisoned_test_dataset = OneClassDataset(poisoned_test_dataset,
                                                    set(range(dataset_num_classes[args.dataset])) - {args.target_label})
        else:
            poisoned_test_dataset = poisoned_test_dataset

        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Clean test dataset size: {len(clean_test_dataset)}")
        print(f"Poisoned test dataset size: {len(poisoned_test_dataset)}")

        train_loader, poisoned_test_loader = get_loaders(train_dataset, poisoned_test_dataset, args.batch_size, shuffle=False)
        clean_test_loader, _ = get_loaders(clean_test_dataset, None, args.batch_size, shuffle=False)

        # load model
        model = load_model(args)
        model = model.cuda()
    else:
        model = None
        train_loader, poisoned_test_loader, clean_test_loader = None, None, None

    features, labels = extract_or_cache(args, extract_features, train_loader, model, return_labels=True, cache_info="train")
    # labels = load_original_labels(args.dataset) # used for oracle logistic regression
    dataset = SimpleDataset(features, labels)
    loader = get_loader(dataset, args.batch_size, shuffle=True)

    poisoned_test_features, poisoned_test_labels = extract_or_cache(args, extract_features, poisoned_test_loader, model,
                                                                    return_labels=True, cache_info="poisoned_test")
    poisoned_dataset = SimpleDataset(poisoned_test_features, poisoned_test_labels)
    asr_test_loader = get_loader(poisoned_dataset, args.batch_size, shuffle=False)

    clean_test_features, clean_test_labels = extract_or_cache(args, extract_features, clean_test_loader, model,
                    return_labels = True, cache_info = "clean_test")
    clean_dataset = SimpleDataset(clean_test_features, clean_test_labels)
    clean_test_loader = get_loader(clean_dataset, args.batch_size, shuffle=False)

    # centroids = find_centroids(features, labels)
    # poisoned_set_indices = torch.load(f"{args.train_poisoned_set}.pt")
    # poisoned_set = np.zeros(len(features), dtype=bool)
    # poisoned_set[poisoned_set_indices] = True
    # plot_data(features, labels, centroids=centroids, poisoned_set=poisoned_set,
    #           save_path=os.path.join(args.experiment_dir, "features.png"))

    linear_head = nn.Linear(latent_dim[args.pretrained_model_name], dataset_num_classes[args.dataset]).cuda()

    optimizer = torch.optim.SGD(linear_head.parameters(), lr=args.lr, weight_decay=0)
    milestones = [60, 80]
    criterion = torch.nn.CrossEntropyLoss()
    for ii in range(args.num_epochs):
        batch_loss, acc = train_one_epoch(loader, linear_head, criterion, optimizer)
        print(f"CE Epoch {ii}: {batch_loss}, lr: {optimizer.param_groups[0]['lr']}, acc: {acc}")
        if (ii + 1) in milestones:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.1

        if (ii + 1) % 1 == 0:
            clean_acc, asr = test_all(linear_head, clean_test_loader, asr_test_loader, None)
            log.write(f"Epoch {ii}: clean acc: {clean_acc}, asr: {asr}\n")


    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default="experiments")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--attack', type=str, default='badnets')
    parser.add_argument('--poisoned_rate', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--selfsup_backbone', type=str, default='resnet18_cifar')
    parser.add_argument('--data_dir', type=str, required=False,
                        default="/home/isabolic/LikelihoodEstimationBasedDefense/data")
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument("--pretrained_checkpoint_dir", type=str, required=True)
    parser.add_argument("--train_poisoned_set", type=str, default=None)
    parser.add_argument("--test_poisoned_set", type=str, default=None)

    parser.add_argument("--cache_exists", action="store_true")

    parser.add_argument("--model_type", type=str, default='selfsup')
    parser.add_argument("--backbone_size", default="small", type=str, choices=["small", "base", "large", "giant"])

    main(parser.parse_args())
