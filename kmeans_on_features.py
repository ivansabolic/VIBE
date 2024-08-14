import math
import json
import os
import time
from pathlib import Path

import matplotlib
import torch
import torchvision
import numpy as np
from scipy.optimize._lsap import linear_sum_assignment
from torch import nn

from sklearn.cluster import KMeans
from sklearn.metrics.cluster import contingency_matrix

from cluster_alternate import frozen_backbones
from images_dataset import get_poisoned_datasets, dataset_num_classes
from cluster import get_loaders, plot_data, find_centroids, classify, load_original_labels, load_model, \
    OverridedLabelsDataset, train_transform, supervise_train_transform, OneClassDataset, test_transform, get_loader, \
    supervise, SimpleDataset, test_all, get_clean_test_dataset, extract_or_cache, extract_features
import argparse
from tqdm import tqdm


def main(args):
    args.experiment_name = f"{args.dataset}_{args.attack}_{args.poisoned_rate}_{args.target_label}_{args.selfsup_backbone}"
    args.experiment_dir = os.path.join(args.experiment_dir, args.experiment_name, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_kmeans')
    args.pretrained_model_name = f"{args.pretrained_checkpoint_dir.split('/')[-2]}"
    print(args)

    os.makedirs(args.experiment_dir)

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
            transforms=test_transform[args.dataset],
            test_transforms=test_transform[args.dataset],
            experiment_dir=None,
            **kwargs
        )
        if "cbd" not in args.attack:
            poisoned_test_dataset = OneClassDataset(poisoned_test_dataset, set(range(dataset_num_classes[args.dataset])) - {args.target_label})
        else:
            poisoned_test_dataset = poisoned_test_dataset

        clean_test_dataset = get_clean_test_dataset(args.dataset, args.data_dir, test_transform[args.dataset])

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
    poisoned_test_features, poisoned_test_labels = extract_or_cache(args, extract_features, poisoned_test_loader, model, return_labels=True, cache_info="poisoned_test")
    clean_test_features, clean_test_labels = extract_or_cache(args, extract_features, clean_test_loader, model, return_labels=True, cache_info="clean_test")

    # centroids = find_centroids(features, labels)
    # poisoned_set_indices = torch.load(f"{args.train_poisoned_set}.pt")
    # poisoned_set = np.zeros(len(features), dtype=bool)
    # poisoned_set[poisoned_set_indices] = True
    # plot_data(features, labels, centroids=centroids, poisoned_set=poisoned_set,
    #           save_path=os.path.join(args.experiment_dir, "features.png"))


    kmeans = KMeans(n_clusters=dataset_num_classes[args.dataset], random_state=0)
    kmeans.fit(features)
    clusters = kmeans.predict(features)

    # def cluster_accuracy(y_true, y_pred):
    #     # Create the contingency matrix
    #     cont_matrix = contingency_matrix(y_true, y_pred)
    #     # Find the best alignment
    #     indices = np.argmax(cont_matrix, axis=1)
    #     correct_preds = cont_matrix[np.arange(len(indices)), indices]
    #     accuracy = np.sum(correct_preds) / np.sum(cont_matrix)
    #     return accuracy

    def cluster_accuracy(y_true, y_pred):
        """
        Calculate clustering accuracy. Require scikit-learn installed
        # Arguments
            y: true labels, numpy.array with shape `(n_samples,)`
            y_pred: predicted labels, numpy.array with shape `(n_samples,)`
        # Return
            accuracy, in [0,1]
        """
        assert y_pred.shape == y_true.shape
        y_true = torch.from_numpy(y_true)
        y_pred = torch.from_numpy(y_pred)

        w = torch.zeros(y_pred.max() + 1, y_true.max() + 1).long()
        for i in range(y_pred.size(0)):
            w[y_pred[i], y_true[i]] += 1
        row_ind, col_ind = linear_sum_assignment(w.max() - w)

        mapping = torch.ones(y_pred.max() + 1).long() * -1
        mapping[row_ind] = torch.from_numpy(col_ind)
        accuracy = (w[row_ind, col_ind].sum() / y_true.size(0)).item() * 100
        return accuracy

    acc = cluster_accuracy(labels, clusters)
    print(f"KMeans acc: {acc:.3f}")

    test_clusters = kmeans.predict(poisoned_test_features)
    test_acc = cluster_accuracy(poisoned_test_labels, test_clusters)
    print(f"ASR Test KMeans acc: {test_acc:.3f}")

    clean_clusters = kmeans.predict(clean_test_features)
    clean_acc = cluster_accuracy(clean_test_labels, clean_clusters)
    print(f"Clean KMeans acc: {clean_acc:.3f}")



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
    parser.add_argument("--pretrained_checkpoint_dir", type=str, required=True)
    parser.add_argument("--train_poisoned_set", type=str, default=None)
    parser.add_argument("--test_poisoned_set", type=str, default=None)

    parser.add_argument("--cache_exists", action="store_true")

    parser.add_argument("--model_type", type=str, default='selfsup')
    parser.add_argument("--backbone_size", default="small", type=str, choices=["small", "base", "large", "giant"])

    main(parser.parse_args())
