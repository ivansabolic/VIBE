import math
import json
import os
import time
from pathlib import Path

import matplotlib
import torch
from torch.nn import functional as F
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch import nn

from filter_disruptive import do_filter_disruptive
from images_dataset import get_poisoned_datasets, dataset_num_classes
from cluster import get_loaders, plot_data, find_centroids, classify, load_original_labels, load_model, \
    OverridedLabelsDataset, train_transform, supervise_train_transform, OneClassDataset, test_transform, get_loader, \
    supervise, train_one_epoch, test_all_with_centroids, cosine_similarity, FilteredDataset, get_clean_test_dataset, \
    extract_features, forward_to_prob, get_indices_for_label_consistent, test_all_with_features_and_centroids, \
    extract_or_cache, SimpleDataset
import argparse
from tqdm import tqdm

from solo.backbones import resnet18

frozen_backbones = ["dino", "clip"]

# def train_one_epoch(loader, model, criterion, optimizer):
#     model.train()
#     total_loss = 0
#     # for x, y in tqdm(loader, desc="Training"):
#     for x, y in loader:
#         x = x.cuda(non_blocking=True)
#         y = y.cuda(non_blocking=True)
#
#         optimizer.zero_grad()
#         out = model(x)
#         loss = criterion(out, y)
#         loss.backward()
#         optimizer.step()
#
#         total_loss += loss.item()
#
#     return total_loss / len(loader)

class SimpleModel(nn.Module):
    def __init__(self, backbone, linear_head):
        super(SimpleModel, self).__init__()
        self.backbone = backbone
        self.fc = linear_head

    def forward(self, x):
        return self.fc(self.backbone(x))

# def kmeans(args):
    # test_features = extract_features(clean_test_loader, model)
    # from sklearn.cluster import KMeans
    # from sklearn.metrics.cluster import contingency_matrix
    # train_labels = np.array(train_dataset_notransform_loader.dataset.targets)
    #
    # kmeans = KMeans(n_clusters=10, random_state=0)
    # kmeans.fit(features)
    # clusters = kmeans.predict(features)
    #
    # def cluster_accuracy(y_true, y_pred):
    #     # Create the contingency matrix
    #     cont_matrix = contingency_matrix(y_true, y_pred)
    #     # Find the best alignment
    #     indices = np.argmax(cont_matrix, axis=1)
    #     correct_preds = cont_matrix[np.arange(len(indices)), indices]
    #     accuracy = np.sum(correct_preds) / np.sum(cont_matrix)
    #     return accuracy
    #
    # acc = cluster_accuracy(train_labels, clusters)
    # print(f"KMeans acc: {acc:.3f}")
    #
    # test_clusters = kmeans.predict(test_features)
    # test_labels = np.array(clean_test_loader.dataset.targets)
    # test_acc = cluster_accuracy(test_labels, test_clusters)
    # print(f"Test KMeans acc: {test_acc:.3f}")
    # breakpoint()

def positivity_test_for_cost_matrix(features, original_labels, centroids):
    num_classes = len(np.unique(original_labels))
    features_len, features_dim = features.shape

    p_l_v = np.exp(centroids @ features.T) / np.exp(centroids @ features.T).sum(axis=0) # li, i

    original_centroids = np.zeros((num_classes, features_dim))
    for i in range(num_classes):
        original_centroids[i] = np.mean(features[original_labels == i], axis=0)

    original_centroids /= np.linalg.norm(original_centroids, axis=1)[:, None]

    p_y_l = np.exp(centroids @ original_centroids.T) / np.exp(centroids @ original_centroids.T).sum(axis=0) # li, yi
    # (li, yi) -> (li, i)

    p_y_l_transformed = np.zeros((features_len, num_classes))
    for i in range(features_len):
        p_y_l_transformed[i] = p_y_l[:, original_labels[i]]

    P = p_l_v * p_y_l_transformed.T

    print(f"Positivity test: {P.min()} <= P <= {P.max()}")



def main(args):
    args.experiment_name = f"{args.dataset}_{args.attack}_{args.poisoned_rate}_{args.target_label}_{args.selfsup_backbone}"

    if args.hyperparam_analysis:
        big_out_file = os.path.join(args.experiment_root, args.experiment_name, 'big_out.txt')

    pretrained_model_name = f"{args.pretrained_checkpoint_dir.split('/')[-2]}"
    args.pretrained_model_name = pretrained_model_name
    exp_name = f'{time.strftime("%Y-%m-%d_%H-%M-%S")}_alternate_model={pretrained_model_name}'
    if args.add_info:
        exp_name += f"_{args.add_info}"
    args.experiment_dir = os.path.join(
        args.experiment_root,
        args.experiment_name,
        exp_name,
    )

    print(args)

    os.makedirs(args.experiment_dir)
    log_file = os.path.join(args.experiment_dir, 'log.txt')
    with open(log_file, 'w') as f:
        f.write(json.dumps(vars(args), indent=4) + '\n')

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

    kwargs = {}
    if "imagenet" in args.dataset:
        kwargs["poisoned_transform_index"] = 2
    train_dataset_notransform, _ = get_poisoned_datasets(
        args.dataset,
        args.data_dir,
        args.attack,
        args.target_label,
        args.poisoned_rate,
        args,
        transforms=test_transform[args.dataset],
        test_transforms=test_transform[args.dataset],
        experiment_dir=None,
        **kwargs,
    )
    if "cbd" in args.attack or "issba" in args.attack:
        asr_test_dataset = poisoned_test_dataset
    else:
        asr_test_dataset = OneClassDataset(poisoned_test_dataset, set(range(dataset_num_classes[args.dataset])) - {args.target_label})
    clean_test_dataset = get_clean_test_dataset(args.dataset, args.data_dir, test_transform[args.dataset])

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Clean test dataset size: {len(clean_test_dataset)}")
    print(f"ASR test dataset size: {len(asr_test_dataset)}")


    train_dataset_notransform_loader = get_loader(train_dataset_notransform, args.batch_size, shuffle=False)
    # train_dataset_notransform_loader = get_loader(train_dataset_notransform, 1, shuffle=False)
    clean_test_loader = get_loader(clean_test_dataset, args.batch_size, shuffle=False)
    # clean_test_loader = get_loader(clean_test_dataset, 1, shuffle=False)
    poisoned_test_loader = get_loader(asr_test_dataset, args.batch_size, shuffle=False)


    poisoned_set_indices = torch.load(f"{args.train_poisoned_set}.pt")

    if args.attack == "issba":
        if args.dataset == "cifar10":
            root_dir = "/home/isabolic/BackdoorBox/experiments/train_poison_DataFolder_CIFAR10_ISSBA_2023-09-25_10:24:51/data/"
            poisoned_set_indices = np.load(os.path.join(root_dir, "poisoned_set.npy"))

    poisoned_set = np.zeros(len(train_dataset), dtype=bool)
    poisoned_set[poisoned_set_indices] = True

    poisoned_labels = np.array([l for _, l in train_dataset])
    original_labels = load_original_labels(args.dataset)

    if args.attack == "issba":
        original_labels = torch.from_numpy(original_labels)
        poisoned_set = torch.from_numpy(poisoned_set)
        original_labels = torch.cat([original_labels[~poisoned_set], original_labels[poisoned_set]]).numpy()
        poisoned_set = torch.cat([poisoned_set[~poisoned_set], poisoned_set[poisoned_set]]).numpy()

    if args.attack == "labelconsistent":
        eps = 16
        experiment_dir = "/home/isabolic/LikelihoodEstimationBasedDefense/experiments/cifar10_labelconsistent_0.25_0_resnet18/"
        indices = get_indices_for_label_consistent(
            os.path.join(experiment_dir, "adv_dataset_eps{}".format(eps), 'target_adv_dataset'))

        poisoned_set = poisoned_set[indices]
        # poisoned_labels = poisoned_labels[indices]
        original_labels = original_labels[indices]
        poisoned_set_indices = np.where(poisoned_set)[0]

    # train_dataset_notransform_indices = np.array([lab.numpy() for _, lab in train_dataset_notransform_loader]).flatten()
    # breakpoint()
    # print((poisoned_labels == train_dataset_notransform_indices).all())



    # load model
    if args.model_type != 'random_init':
        model = load_model(args)
    else:
        model = resnet18(None, num_classes=dataset_num_classes[args.dataset])
        if 'cifar' in args.dataset:
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=2, bias=False
            )
            model.maxpool = nn.Identity()

    model = model.cuda()
    # linear_head = nn.Linear(512, dataset_num_classes[args.dataset], bias=False).cuda()

    if args.model_type in frozen_backbones:
        # features = extract_features(train_dataset_notransform_loader, model, args.model_type)
        features = extract_or_cache(args, extract_features, train_dataset_notransform_loader, model, return_labels=False,
                                            cache_info="train", model_type=args.model_type)
        # np.save(os.path.join(args.experiment_dir, 'features_not_normalized.npy'), features)
        # assert args.attack == 'blended' and args.dataset == 'imagenet'
        # features = np.load(os.path.join('experiments/imagenet_blended_0.1_0_resnet18_cifar/2024-07-31_09-34-26_alternate_model=dino_fullalg/', 'features_not_normalized.npy'))

        # clean_test_features = extract_features(clean_test_loader, model, args.model_type)
        clean_test_features = extract_or_cache(args, extract_features, clean_test_loader, model, return_labels=False,
                                               cache_info="clean_test", model_type=args.model_type)
        clean_test_labels = np.array([l for _, l in clean_test_loader.dataset])
        # poisoned_test_features = extract_features(poisoned_test_loader, model, args.model_type)
        poisoned_test_features = extract_or_cache(args, extract_features, poisoned_test_loader, model, return_labels=False,
                                                  cache_info="poisoned_test", model_type=args.model_type)
        poisoned_test_labels = np.array([l for _, l in poisoned_test_loader.dataset])

    # filter out disruptively poisoned examples
    filtered = False
    if args.filter_disruptive:
        print("Filtering disruptive!")
        filtered_indices = do_filter_disruptive(
            extract_features(train_dataset_notransform_loader, model) if args.model_type not in frozen_backbones else features,
            poisoned_labels,
            poisoned_set,
            poisoned_set_indices,
            args.experiment_dir,
            args.plot_disruptive,
            args.poisoned_threshold,
            kmeans_clusters_coef=args.kmeans_cluster_coef,
        )
        if len(filtered_indices) < len(poisoned_labels):
            filtered = True

            all_labels = np.unique(poisoned_labels)
            filtered_labels = np.unique(poisoned_labels[filtered_indices])
            if len(filtered_labels) < len(all_labels):
                filtered_out_labels = np.setdiff1d(all_labels, filtered_labels)
                print(f"Filtered out labels: {filtered_out_labels}. Returning random samples!")
                for l in filtered_out_labels:
                    all_indices = np.arange(len(poisoned_labels))
                    indices = all_indices[poisoned_labels == l]
                    random_index = np.random.choice(indices, size=20)
                    filtered_indices = np.append(filtered_indices, random_index)
            
            train_dataset = FilteredDataset(train_dataset, indices=filtered_indices)
            train_dataset_notransform = FilteredDataset(train_dataset_notransform, indices=filtered_indices)

            train_dataset_notransform_loader = get_loader(train_dataset_notransform, args.batch_size, shuffle=False)

            poisoned_labels = poisoned_labels[filtered_indices]
            original_labels = original_labels[filtered_indices]
            poisoned_set = poisoned_set[filtered_indices]

            if args.model_type in frozen_backbones:
                features = features[filtered_indices]

    else:
        print("Not filtering disruptive!")



    # filter out nondisruptively poisoned examples
    labels = poisoned_labels
    labels_per_class_dir = os.path.join(args.experiment_dir, 'labels_per_class')
    os.makedirs(labels_per_class_dir)
    for ii in range(args.num_iterations):
        features = extract_features(train_dataset_notransform_loader, model) if args.model_type not in frozen_backbones else features

        # centroids (old version)
        # centroids = find_centroids(features, labels)

        if ii == 0: # first iteration
            centroids = find_centroids(features, labels)
            original_centroids = torch.from_numpy(centroids).cuda().float()

        if args.plot:
            plot_data(features, labels, centroids, poisoned_set, f"{args.experiment_dir}/centroids_{ii}.png", original_centroids=original_centroids.cpu().detach().numpy())

        # classification
        # if ii == 0:
        #     M = np.dot(features, centroids.T)
        #     M = np.exp(M)
        #     M = M / M.sum(axis=1)[:, None]
        # else:


        yyoh = torch.zeros(len(labels), dataset_num_classes[args.dataset])
        yyoh[torch.arange(len(labels)), torch.from_numpy(labels)] = 1

        # P matrix from eq8. Method forward_to_prob taken from SCGM
        P, prob_y_l, prob_l_v = forward_to_prob(torch.Tensor(features), yyoh, original_centroids.detach().cpu(), torch.Tensor(centroids), tau=args.tau, poisoned_set=poisoned_set)
        L_VD = P.sum(1).log().mean().item()
        P = P.numpy()

        # ignore, debugging
        if args.hyperparam_analysis:
            clean_test_features = extract_features(clean_test_loader, model, args.model_type)
            clean_test_labels = np.array([l for _, l in clean_test_loader.dataset])
            yyoh_test = torch.zeros(len(clean_test_labels), dataset_num_classes[args.dataset])
            yyoh_test[torch.arange(len(clean_test_labels)), torch.from_numpy(clean_test_labels)] = 1
            P_clean, _, _ = forward_to_prob(torch.Tensor(clean_test_features), yyoh_test, original_centroids.detach().cpu(), torch.Tensor(centroids), tau=args.tau, poisoned_set=poisoned_set)
            L_VD = P_clean.sum(1).log().mean().item()


        # solving the optimal transport problem, labels represent matrix Q
        labels = classify(P, classifier=args.classifier)

        ### Analysis and debugging (ignore)
        # accuracies of our three models
        if args.analyze_classification:
            clean_test_features = extract_features(clean_test_loader, model, args.model_type)
            clean_test_labels = np.array([l for _, l in clean_test_loader.dataset])
            yyoh_test = torch.zeros(len(clean_test_labels), dataset_num_classes[args.dataset])
            yyoh_test[torch.arange(len(clean_test_labels)), torch.from_numpy(clean_test_labels)] = 1

            poisoned_test_features = extract_features(poisoned_test_loader, model, args.model_type)
            poisoned_test_labels = np.array([l for _, l in poisoned_test_loader.dataset])
            yyoh_poisoned = torch.zeros(len(poisoned_test_labels), dataset_num_classes[args.dataset])
            yyoh_poisoned[torch.arange(len(poisoned_test_labels)), torch.from_numpy(poisoned_test_labels)] = 1

            P_clean, prob_y_l_clean, prob_l_v_clean = forward_to_prob(torch.Tensor(clean_test_features), yyoh_test, original_centroids.detach().cpu(), torch.Tensor(centroids), tau=args.tau, poisoned_set=poisoned_set)
            P_poisoned, prob_y_l_poisoned, prob_l_v_poisoned = forward_to_prob(torch.Tensor(poisoned_test_features), yyoh_poisoned, original_centroids.detach().cpu(), torch.Tensor(centroids), tau=args.tau, poisoned_set=poisoned_set)

            print(f"P(l|v) CLEAN acc: {(prob_l_v_clean.argmax(dim=1).numpy() == clean_test_labels).mean().item():.3f}")
            print(f"P(y|l) CLEAN acc: {(prob_y_l_clean.argmax(dim=1).numpy() == clean_test_labels).mean().item():.3f}")
            print(f"P CLEAN acc: {(P_clean.argmax(axis=1).numpy() == clean_test_labels).mean().item():.3f}")
            print(f"q(l|y, v) CLEAN acc: {(labels_per_class_dir == clean_test_labels).mean():.3f}")

            print(f"P(l|v) POISONED acc: {(prob_l_v_poisoned.argmax(dim=1).numpy() == poisoned_test_labels).mean().item():.3f}")
            print(f"P(y|l) POISONED acc: {(prob_y_l_poisoned.argmax(dim=1).numpy() == poisoned_test_labels).mean().item():.3f}")
            print(f"P POISONED acc: {(P_poisoned.argmax(axis=1).numpy() == poisoned_test_labels).mean().item():.3f}")
            print(f"q(l|y, v) POISONED acc: {(labels_per_class_dir == poisoned_test_labels).mean():.3f}")

            print(f"P(l|v) TRAIN acc: {(prob_l_v.argmax(dim=1).numpy()== original_labels).mean().item():.3f}")
            print(f"P(y|l) TRAIN acc: {(prob_y_l.argmax(dim=1).numpy() == original_labels).mean().item():.3f}")
            print(f"P TRAIN acc: {(P.argmax(axis=1) == original_labels).mean().item():.3f}")
            print(f"q(l|y, v) TRAIN acc: {(labels == original_labels).mean():.3f}")
        if args.label_smoothing > 0:
            one_hot_labels = np.zeros((len(labels), len(centroids)))
            one_hot_labels[np.arange(len(labels)), labels] = 1 - args.label_smoothing
            one_hot_labels[one_hot_labels == 0] = args.label_smoothing / (len(centroids) - 1)
        if args.analyze_incorrects:
            poisoneds_relabeled = labels[poisoned_set] != poisoned_labels[poisoned_set]
            other_relabeled = labels[~poisoned_set] != original_labels[~poisoned_set]

            poisoneds_distances_to_original = np.linalg.norm(features[poisoned_set][poisoneds_relabeled] - centroids[poisoned_labels[poisoned_set][poisoneds_relabeled]], axis=1)
            other_distances_to_original = np.linalg.norm(features[~poisoned_set][other_relabeled] - centroids[original_labels[~poisoned_set][other_relabeled]], axis=1)

            plt.hist(poisoneds_distances_to_original, bins=100, color='r', alpha=0.7, label='Poisoned', density=True)
            plt.hist(other_distances_to_original, bins=100, color='b', alpha=0.7, label='Other', density=True)
            plt.legend()
            plt.savefig(os.path.join(labels_per_class_dir, f'distances_to_original_{ii}.png'))
            plt.close()

            poisoneds_cosine_to_original = np.dot(features[poisoned_set][poisoneds_relabeled], centroids[poisoned_labels[poisoned_set][poisoneds_relabeled]].T)
            poisoneds_cosine_to_original = poisoneds_cosine_to_original.diagonal()
            other_cosine_to_original = np.dot(features[~poisoned_set][other_relabeled], centroids[original_labels[~poisoned_set][other_relabeled]].T)
            other_cosine_to_original = other_cosine_to_original.diagonal()

            plt.hist(poisoneds_cosine_to_original, bins=100, color='r', alpha=0.7, label='Poisoned', density=True)
            plt.hist(other_cosine_to_original, bins=100, color='b', alpha=0.7, label='Other', density=True)
            plt.legend()
            plt.savefig(os.path.join(labels_per_class_dir, f'cosine_to_original_{ii}.png'))
            plt.close()


        # Logging data
        poisoned_not_relabeled = (labels[poisoned_set] == poisoned_labels[poisoned_set]).sum()
        poisoned_correctly_relabeled = (labels[poisoned_set] == original_labels[poisoned_set]).sum() / (poisoned_set.sum() + 1e-6)
        other_incorrectly_relabeled = (labels[~poisoned_set] != original_labels[~poisoned_set]).sum()

        plt.hist(labels, bins=range(dataset_num_classes[args.dataset] + 1), alpha=0.7, color='r', label='Labels')
        plt.savefig(os.path.join(labels_per_class_dir, f'labels_per_class_{ii}.png'))
        plt.close()
        log_msg = f"Iter {ii}: Poisoned labels accuracy: {np.mean(labels == poisoned_labels) :.3f}, " \
              f"Original labels accuracy: {np.mean(labels == original_labels) : .3f}, " \
              f"Number of poisoned not relabeled: {poisoned_not_relabeled} / {poisoned_set.sum()}, " \
              f"Poisoned correctly relabeled: {poisoned_correctly_relabeled:.3f}, " \
              f"Number of other incorrectly relabeled: {other_incorrectly_relabeled}, " \
              f"L_VD: {L_VD:.3f}"
        print(log_msg)

        with open(log_file, 'a') as f:
            f.write(log_msg + '\n')

        # label smoothing
        if args.label_smoothing > 0:
            current_labels_dataset = OverridedLabelsDataset(train_dataset, one_hot_labels)
        else:
            current_labels_dataset = OverridedLabelsDataset(train_dataset, labels)

        # debugging, ignore
        if args.filter_relabeled_non_poisoned: # kako bi se pronašla gornja granica performanse, ne uzimajući u obzir ove krivo označene
            relabeled = labels != original_labels
            poisoned_and_non_relabeled = poisoned_set | ~relabeled
            poisoned_and_non_relabeled_indices = np.where(poisoned_and_non_relabeled)[0]

            current_labels_dataset = FilteredDataset(current_labels_dataset, indices=poisoned_and_non_relabeled_indices)
            print(f"Kept {len(poisoned_and_non_relabeled_indices)} samples")

        ce_loader, _ = get_loaders(current_labels_dataset, None, args.batch_size, shuffle=True)

        # positivity_test_for_cost_matrix(features, poisoned_labels, centroids)
        # conf = np.linalg.norm(features - centroids[poisoned_labels], axis=1)
        # relabeled = labels != poisoned_labels
        # if filtered:
        #     conf[relabeled] = 0
        # else:
        #     conf[relabeled] = np.exp(conf[relabeled])
        #     conf[relabeled] = (conf[relabeled] - conf[relabeled].min()) / (conf[relabeled].max() - conf[relabeled].min())
        #     # conf[relabeled] = 1 - conf[relabeled]
        #     conf[~relabeled] = 1
        # conf = torch.Tensor(conf).cuda()
        #
        # conf1 = np.ones(len(features))
        # conf1[relabeled] = 0
        # conf1 = torch.Tensor(conf1).cuda()

        # conf = conf1

        # v3 verzija algoritma (ignore, staro)
        # if args.inner_algorithm == 'v3':
        #     simple_model = SimpleModel(model, linear_head)
        #     optimizer = torch.optim.SGD([{'params': simple_model.backbone.parameters()}, {'params': simple_model.fc.parameters(), 'lr': 0.1}], lr=args.ce_lr, momentum=0.9, weight_decay=5e-4)
        #     criterion = torch.nn.CrossEntropyLoss()
        #     for jj in range(args.ce_epochs):
        #         batch_loss, acc = train_one_epoch(simple_model, ce_loader, optimizer, criterion, normalize_linear_head=True)
        #         print(f"CE Epoch {jj}: Loss: {batch_loss}, Acc: {acc}")
        # v3 verzija algoritma kraj

        ### M step when updating backbone
        if args.model_type not in frozen_backbones:
            centroids = torch.Tensor(centroids).cuda().float()
            centroids = nn.Parameter(centroids, requires_grad=True)
            original_centroids = nn.Parameter(original_centroids, requires_grad=True)
            # optimizer = torch.optim.SGD([model.parameters(), centroids],
            #                             lr=args.ce_lr, momentum=0.9, weight_decay=5e-4)
            # optimize model.parameters and centroids
            if args.optimizer == 'sgd':
                optimizer = torch.optim.SGD([{ 'params': model.parameters() }, { 'params': centroids}, { 'params': original_centroids}],
                                            lr=args.ce_lr, momentum=0.9, weight_decay=args.weight_decay)
            elif args.optimizer == 'adam':
                optimizer = torch.optim.Adam([{ 'params': model.parameters() }, { 'params': centroids}, { 'params': original_centroids}],
                                                lr=args.ce_lr, weight_decay=args.weight_decay)
            else:
                raise ValueError("Optimizer not supported")
            if args.scheduler:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=200)

            # optimizer=torch.optim.SGD(model.parameters(),
            #                                 lr=args.ce_lr, momentum=0.9, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss(reduction='none')
            dl_iter = iter(ce_loader)
            loss = torch.tensor(0)
            for jj in range(args.num_inner_iters):
                try:
                    x, y, index = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(ce_loader)
                    x, y, index = next(dl_iter)
                x = x.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)

                optimizer.zero_grad()
                logits = torch.matmul(model(x), centroids.T)

                # conf1 = logits.softmax(dim=1).max(dim=1).values
                #
                # logits_softmax = logits.softmax(dim=1)
                # max_conf = logits_softmax.max(dim=1).values
                # second_max_conf = logits_softmax.topk(2, dim=1).values[:, 1]
                # conf2 = max_conf - second_max_conf
                #
                # conf3 = 1 / (torch.Tensor(features[index]).cuda() - centroids[labels[index]]).norm(dim=1)

                # originalna verzija
                # loss = criterion(logits, y) * conf[index]

                ### 2 gubitka variational inference verzija
                loss_2 = criterion(logits, y)

                batch_given_poisoned_labels = poisoned_labels[index]
                batch_given_poisoned_labels = torch.Tensor(batch_given_poisoned_labels).cuda().long()

                # centroids_batch = centroids[batch_given_poisoned_labels]
                centroids_batch = centroids[y]

                loss_1 = criterion(torch.matmul(centroids_batch, original_centroids.T), batch_given_poisoned_labels)

                # # scgm implementacija
                # yy = batch_given_poisoned_labels.cpu().numpy()
                # num_classes = len(centroids)
                # yyoh = torch.zeros(yy.size, num_classes).cuda()
                # yyoh[torch.arange(yy.size), yy] = 1
                #
                # logit2 = (yyoh @ original_centroids) @ centroids.t()
                # ls2_num = torch.exp(logit2)
                # ls2_den = torch.exp(original_centroids @ centroids.t())
                #
                # yoh = torch.zeros(y.size(0), num_classes).cuda()
                # yoh[torch.arange(y.size(0)), y] = 1
                # ls2 = -torch.log(ls2_num / ls2_den.sum(0).view(1, -1)) * yoh
                # ls2 = ls2.sum(1)

                #
                # Zavrsni izračuni gubitka
                # loss = loss_1*conf1[index] + loss_2*conf[index]
                loss = loss_1 + loss_2
                # loss = ls_2 + loss_2*conf[index]
                # loss = ls_2 + loss_2 #*conf[index]
                ## Kraj

                loss = loss.mean()

                loss.backward()
                optimizer.step()
                if args.scheduler:
                    scheduler.step()
                print(f"CE Epoch {jj}: Loss: {loss.item(): .4f}", end='\r')
            print(f"Loss: {loss.item(): .3f}")

        else:
            # M step when not updating backbone
            features_dataset = SimpleDataset(features, labels)
            features_loader = get_loader(features_dataset, batch_size=args.batch_size, shuffle=True)
            # centroids = torch.randn(len(np.unique(labels)), features.shape[1])
            # class CentroidsModule(nn.Module):
            #     def __init__(self, centroids):
            #         super(CentroidsModule, self).__init__()
            #         self.centroids = nn.Parameter(torch.Tensor(centroids), requires_grad=True)
            #
            #     def forward(self, x, yyoh):
            #         return forward_to_prob(torch.Tensor(x).cuda(), yyoh.cuda(), original_centroids.detach().cpu().cuda(), self.centroids.cuda(),
            #                               tau=args.tau, poisoned_set=poisoned_set)
            #
            # centroids_module = CentroidsModule(centroids)
            # nn.utils.weight_norm(centroids_module, name='centroids')
            centroids = torch.Tensor(centroids)
            mu_optimizer = torch.optim.SGD([centroids], lr=args.ce_lr, momentum=0.9, weight_decay=args.weight_decay)
            dl_iter = iter(features_loader)
            for ii in range(args.num_inner_iters):
                try:
                    x, y = next(dl_iter)
                except StopIteration:
                    dl_iter = iter(features_loader)
                    x, y = next(dl_iter)

                yyoh = torch.zeros(len(y), dataset_num_classes[args.dataset])
                yyoh[torch.arange(len(y)), y] = 1
                P, _, _ = forward_to_prob(torch.Tensor(x).cuda(), yyoh.cuda(), original_centroids.detach().cpu().cuda(), centroids.cuda(),
                                                  tau=args.tau, poisoned_set=poisoned_set)
                L_VD = P.sum(1).log().mean()
                L_VD = - L_VD
                mu_optimizer.zero_grad()
                L_VD.backward()
                mu_optimizer.step()

                with torch.no_grad():
                    centroids[:] = F.normalize(centroids, p=2, dim=1)

                if ii % 10 == 0:
                    print(f"CE Epoch {ii}: Loss: {L_VD.item(): .4f}", end='\r')

            print(f"Loss: {L_VD.item(): .3f}")

            # centroids = centroids.detach().numpy()

        if args.model_type not in frozen_backbones:
            clean_acc, asr = test_all_with_centroids(model, clean_test_loader, poisoned_test_loader, centroids, args)
        else:
            clean_acc, asr = test_all_with_features_and_centroids(clean_test_features, clean_test_labels, poisoned_test_features, poisoned_test_labels, centroids, args)

        with open(log_file, 'a') as f:
            f.write(f"Clean acc: {clean_acc:.3f}, ASR: {asr:.3f}\n")


        print()

    if args.hyperparam_analysis:
        with open(big_out_file, 'a') as f:
            f.write(f"{L_VD}, {clean_acc}, {asr}, {args}\n")


    # train_dataset, test_dataset = get_poisoned_datasets(
    #     args.dataset,
    #     args.data_dir,
    #     args.attack,
    #     args.target_label,
    #     args.poisoned_rate,
    #     args,
    #     transforms=supervise_train_transform[args.dataset],
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
    # model.fc = linear_head
    # supervise(model, train_loader, clean_test_loader, asr_test_loader, args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_root', type=str, default="experiments")
    parser.add_argument('--dataset', type=str, default="cifar10")
    parser.add_argument('--attack', type=str, default='badnets')
    parser.add_argument('--poisoned_rate', type=float, default=0.1)
    parser.add_argument('--target_label', type=int, required=True)
    parser.add_argument('--selfsup_backbone', type=str, default='resnet18_cifar')
    parser.add_argument('--origin', type=str, default='lebd')
    parser.add_argument('--data_dir', type=str, required=False,
                        default="/home/isabolic/LikelihoodEstimationBasedDefense/data")
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--optimizer', type=str, default='sgd')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--ce_lr', type=float, default=0.01)
    parser.add_argument('--num_iterations', type=int, default=20)
    parser.add_argument('--scheduler', default=False, action="store_true")
    parser.add_argument('--milestones', type=int, nargs='+', default=[50, 75])
    parser.add_argument('--test_epoch', type=int, default=5)
    parser.add_argument("--train_poisoned_set", type=str, default=None)
    parser.add_argument("--test_poisoned_set", type=str, default=None)
    parser.add_argument("--plot", default=False, action="store_true")
    parser.add_argument("--classifier", type=str, default="argmax")
    parser.add_argument("--tau", type=float, default=1)
    parser.add_argument("--label_smoothing", type=float, default=0.0)
    parser.add_argument("--pretrained_checkpoint_dir", type=str, default=None)
    parser.add_argument("--ce_epochs", type=int, default=1)

    parser.add_argument("--supervise_lr", type=float, default=0.01)
    parser.add_argument("--supervise_epochs", type=int, default=50)
    parser.add_argument("--supervise_test_epoch", type=int, default=5)

    parser.add_argument("--inner_algorithm", type=str, default='v3')
    parser.add_argument("--num_inner_iters", type=int, default=100)

    parser.add_argument("--mu_lr", type=float, default=0.01)

    parser.add_argument("--analyze_incorrects", default=False, action="store_true")
    parser.add_argument("--filter_relabeled_non_poisoned", default=False, action="store_true")
    parser.add_argument("--analyze_classification", default=False, action="store_true")

    parser.add_argument("--model_type", type=str, default='selfsup')
    parser.add_argument("--backbone_size", default="small", type=str, choices=["small", "base", "large", "giant"])

    parser.add_argument("--filter_disruptive", default=False, action="store_true")
    parser.add_argument("--plot_disruptive", default=False, action="store_true")
    parser.add_argument("--poisoned_threshold", type=float, default=0.25)
    parser.add_argument("--kmeans_cluster_coef", type=int, default=1)

    parser.add_argument("--hyperparam_analysis", default=False, action="store_true")
    parser.add_argument("--add_info", type=str, default=None)

    # lrs = [0.1, 0.01, 0.001, 0.0001]
    # optimizers = ['sgd', 'adam']
    # batch_sizes = [64, 128, 256, 512]
    # weight_decays = [0, 1e-4, 5e-4, 1e-3]
    # num_inner_iters = [100, 500, 1000, 2000]
    #
    # args = parser.parse_args()
    #
    # for lr in lrs:
    #     for optimizer in optimizers:
    #         for batch_size in batch_sizes:
    #             for weight_decay in weight_decays:
    #                 for num_inner_iter in num_inner_iters:
    #                     args.ce_lr = lr
    #                     args.optimizer = optimizer
    #                     args.batch_size = batch_size
    #                     args.weight_decay = weight_decay
    #                     args.num_inner_iters = num_inner_iter
    args = parser.parse_args()
    main(args)