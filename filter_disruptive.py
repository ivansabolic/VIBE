import math
import json
import os
import time
from pathlib import Path

import matplotlib
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from torch import nn
from torchvision import transforms

from images_dataset import get_poisoned_datasets, dataset_num_classes
from cluster import get_loaders, plot_data, find_centroids, classify, load_original_labels, load_model, \
    OverridedLabelsDataset, train_transform, supervise_train_transform, OneClassDataset, test_transform, get_loader, \
    supervise, train_one_epoch, test_all_with_centroids, cosine_similarity, FilteredDataset, extract_features
import argparse
from tqdm import tqdm

from torchvision.transforms import Compose, ToTensor, Normalize, RandomCrop, RandomHorizontalFlip


def do_filter_disruptive(features, labels, poisoned_set, poisoned_set_indices, out_dir, plot, poisoned_threshold, kmeans_clusters_coef=1):
    kmeans = KMeans(n_clusters=(kmeans_clusters_coef*np.unique(labels).shape[0]+1), random_state=0, n_init=1).fit(features)
    centroids = kmeans.cluster_centers_

    # centroid to centroid distance
    centroid_distances = np.zeros((centroids.shape[0], centroids.shape[0]))
    for i in range(centroids.shape[0]):
        for j in range(centroids.shape[0]):
            centroid_distances[i, j] = np.linalg.norm(centroids[i] - centroids[j])

    avg_centroid_distances = np.mean(centroid_distances, axis=1)
    avg_centroid_distances = avg_centroid_distances / np.max(avg_centroid_distances)

    avg_distances_to_centroid_centroids = np.linalg.norm(centroids - centroids.mean(axis=0), axis=1)

    plt.hist(avg_centroid_distances, bins=50)
    plt.savefig(os.path.join(out_dir, 'centroid_distances.png'))
    plt.close()

    plt.hist(avg_distances_to_centroid_centroids, bins=50)
    plt.savefig(os.path.join(out_dir, 'centroid_distances_to_centroids.png'))
    plt.close()

    avg_distance_to_mean_others = np.zeros(centroids.shape[0])
    for i in range(centroids.shape[0]):
        without_i = np.delete(avg_centroid_distances, i, axis=0)
        avg_distance_to_mean_others[i] = np.abs(avg_centroid_distances[i] - without_i.mean())

    plt.hist(avg_distance_to_mean_others, bins=50)
    plt.savefig(os.path.join(out_dir, 'avg_distance_to_mean_others.png'))
    plt.close()

    # filter disruptive
    poisoned_centroids = np.arange(len(centroids))[avg_distance_to_mean_others > poisoned_threshold]

    predicted_labels = kmeans.predict(features)

    filtered_indices = np.array([i for i, l in enumerate(labels) if predicted_labels[i] in poisoned_centroids])

    poisoned_indices_detected = np.intersect1d(poisoned_set_indices, filtered_indices)

    if len(filtered_indices) > 0:
        print("Removing some indices...")
        print(f"Removed {len(filtered_indices)} indices.")
        print(f"Poisoneds: {len(poisoned_indices_detected)} poisoned samples")
        print(f"Clean: {len(filtered_indices) - len(poisoned_indices_detected)} clean samples lost")

    if plot:
        # plot_data(features, labels, centroids=centroids, poisoned_set=poisoned_set,
        #           save_path=os.path.join(out_dir, "features.png"))
        # plot_data(features, predicted_labels, centroids=centroids,
        #           save_path=os.path.join(out_dir, "features_predicted.png"))
        if len(filtered_indices) > 0:
            plot_data(features, labels, centroids=centroids, poisoned_set=filtered_indices,
                      save_path=os.path.join(out_dir, "features_filtered.png"))
        else:
            print("No samples removed. NO PLOT!!!")

    all_indices = np.arange(len(labels))
    if len(filtered_indices) > 0:
        kept_indices = np.delete(all_indices, filtered_indices)
    else:
        kept_indices = all_indices
    return kept_indices


def main(args):
    args.experiment_name = f"{args.dataset}_{args.attack}_{args.poisoned_rate}_{args.target_label}_{args.selfsup_backbone}"

    checkpoint_dir = args.pretrained_checkpoint_dir.split('/')[-1]
    args.experiment_dir = os.path.join(args.experiment_dir, args.experiment_name, f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_disruptive-filtering_{checkpoint_dir}')
    print(args)

    os.makedirs(args.experiment_dir)
    # log_file = os.path.join(args.experiment_dir, 'log.txt')

    train_dataset, poisoned_test_dataset = get_poisoned_datasets(
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
    train_dataset_notransform, _ = get_poisoned_datasets(
        args.dataset,
        args.data_dir,
        args.attack,
        args.target_label,
        args.poisoned_rate,
        args,
        transforms=test_transform,
        test_transforms=test_transform,
        experiment_dir=None,
    )
    asr_test_dataset = OneClassDataset(poisoned_test_dataset, set(range(dataset_num_classes[args.dataset])) - {args.target_label})

    clean_test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )
    train_dataset_notransform_loader = get_loader(train_dataset_notransform, args.batch_size, shuffle=False)
    # train_dataset_notransform_loader = get_loader(train_dataset_notransform, 1, shuffle=False)
    clean_test_loader = get_loader(clean_test_dataset, args.batch_size, shuffle=False)
    # clean_test_loader = get_loader(clean_test_dataset, 1, shuffle=False)
    poisoned_test_loader = get_loader(asr_test_dataset, args.batch_size, shuffle=False)


    poisoned_set_indices = torch.load(f"{args.train_poisoned_set}.pt")
    poisoned_set = np.zeros(len(train_dataset), dtype=bool)
    poisoned_set[poisoned_set_indices] = True

    # load model
    model = load_model(args)
    model = model.cuda()

    poisoned_labels = np.array([l for _, l in train_dataset])
    original_labels = load_original_labels(args.dataset)

    features = extract_features(train_dataset_notransform_loader, model)

    do_filter_disruptive(features, poisoned_labels, poisoned_set, poisoned_set_indices, args.experiment_dir, args.plot, args.poisoned_threshold)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default="experiments")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--attack', type=str, default='badnets')
    parser.add_argument('--poisoned_rate', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--selfsup_backbone', type=str, default='resnet18_cifar')
    parser.add_argument('--origin', type=str, default='lebd')
    parser.add_argument('--data_dir', type=str, required=False,
                        default="/home/isabolic/LikelihoodEstimationBasedDefense/data")
    parser.add_argument("--train_poisoned_set", type=str, default=None)
    parser.add_argument("--test_poisoned_set", type=str, default=None)
    parser.add_argument("--pretrained_checkpoint_dir", type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--plot', action='store_true')

    parser.add_argument("--model_type", type=str, default='selfsup')
    parser.add_argument("--backbone_size", default="small", type=str, choices=["small", "base", "large", "giant"])

    parser.add_argument("--poisoned_threshold", type=float, default=0.25)
    main(parser.parse_args())
