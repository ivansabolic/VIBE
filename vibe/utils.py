from copy import deepcopy
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt

from omegaconf import DictConfig as Config
import numpy as np
import os
import matplotlib


def build_dataloader(dataset: Dataset, cfg: Config, shuffle: bool = True):
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        drop_last=False,
    )

def build_optimizer(model, cfg: Config) -> torch.optim.Optimizer:
    optimizer_params = [{'params': model.parameters()}]
    if model.classifier.mu_z.requires_grad:
        optimizer_params.append(
            {'params': model.classifier.mu_z if not hasattr(model.classifier.mu_z, 'parameters') else model.classifier.mu_z.parameters(), 'lr': cfg.mu_z_lr if cfg.mu_z_lr else cfg.lr})
    if model.classifier.mu_y.requires_grad:
        optimizer_params.append(
            {'params': model.classifier.mu_y, 'lr': cfg.mu_y_lr if cfg.mu_y_lr else cfg.lr})
    if cfg.optimize_prior:
        optimizer_params.append({'params': model.prior, 'lr': 0.1})

    if cfg.name == "adam":
        return torch.optim.Adam(
            optimizer_params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.name == "sgd":
        return torch.optim.SGD(
            optimizer_params,
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    else:
        raise ValueError(f"Invalid optimizer: {cfg.optimizer}")


def build_scheduler(optimizer, cfg: Config) -> torch.optim.lr_scheduler._LRScheduler:
    if cfg.name == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.step_size,
            gamma=cfg.gamma,
        )
    elif cfg.name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=cfg.epochs,
        )
    elif cfg.name == "none":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0,
        )
    else:
        raise ValueError(f"Invalid scheduler: {cfg.scheduler}")


def find_centroids(features, labels):
    num_classes = torch.max(labels) + 1
    centroids = torch.zeros((num_classes, features.shape[1]))
    for i in range(num_classes):
        class_features = features[labels == i]
        # class_features = F.normalize(class_features, p=2, dim=1)
        centroids[i] = class_features.mean(dim=0)

    return centroids


@torch.inference_mode()
def extract_features(model, loader: DataLoader) -> (torch.Tensor, torch.Tensor):
    data = torch.zeros(len(loader.dataset), model.embed_dim)
    labels = torch.zeros(len(loader.dataset)).long()

    model.eval()
    for i, (x, y, _, _) in enumerate(tqdm(loader, desc="Collecting features", leave=False)):
        x = x.cuda(non_blocking=True)

        feats = model.embed(x)

        data[i * loader.batch_size: (i + 1) * loader.batch_size] = feats
        labels[i * loader.batch_size: (i + 1) * loader.batch_size] = y

    return data, labels

def get_features(model, loader: DataLoader, cfg, freeze_backbone, cache_info: str) -> (Tensor, Tensor):
    if freeze_backbone:
        os.makedirs(cfg.cache_dir, exist_ok=True)
        pretrained_model_name = f"{cfg.model.backbone.name}_{cfg.model.backbone.arch}"
        cache_path = f"{cfg.cache_dir}/{cfg.dataset.name}_{cfg.backdoor.name}_{cfg.backdoor.poisoning_rate}_{cfg.backdoor.target_label}_{pretrained_model_name}"
        if cache_info != "":
            cache_path += f"_{cache_info}"
        cache_path += ".npz"

        # check for saved features
        if os.path.exists(cache_path):
            print(f"Loading features from {cache_path}")
            data = np.load(cache_path)
            features = data["features"]
            labels = data["labels"]

            features = torch.from_numpy(features / np.linalg.norm(features, axis=1, keepdims=True))
            labels = torch.from_numpy(labels)
        else:
            # extract and save if no cache
            features, labels = extract_features(model, loader)
            np.savez(
                cache_path,
                features=features.cpu().numpy(),
                labels=labels.cpu().numpy(),
            )
    else:
        features, labels = extract_features(model, loader)

    return features, labels


@torch.inference_mode()
def calculate_dataset_ln_P(iter_, model, features, labels, num_classes, chunks=1):
    # chunk the data
    features_chunks = torch.chunk(features, chunks, dim=0)
    labels_chunks = torch.chunk(labels, chunks, dim=0)

    neg_ln_P = torch.zeros(features.shape[0], num_classes, device=torch.device("cuda"))
    c_i = 0
    for f_chunk, l_chunk in tqdm(zip(features_chunks, labels_chunks), total=len(features_chunks), desc="Calculating negative log probabilities", leave=False):
        f_chunk, l_chunk = f_chunk.cuda(), l_chunk.cuda()
        yyoh_chunk = F.one_hot(l_chunk, num_classes=num_classes).float()

        # calculate variational loss
        with torch.no_grad():
            neg_ln_joint = model.variational_loss(
                v=f_chunk,
                y_oh=yyoh_chunk,
                uniform_p_yl=iter_ == 0,
            )
        neg_ln_P[c_i:c_i + f_chunk.shape[0]] = neg_ln_joint
        c_i += f_chunk.shape[0]

        del f_chunk, l_chunk, yyoh_chunk, neg_ln_joint
        torch.cuda.empty_cache()
    return -neg_ln_P


def plot_data(features, labels, centroids, poisoned_set=None, save_path="out.png", subsample=1., algorithm='tsne', original_centroids=None, plot_binary=False, plot_centroids=True, use_cache=None, colors=None, poisoned_set_data='poisoned'):
    if use_cache is not None and os.path.exists(use_cache):
        assert original_centroids is None
        reduced_fac = np.load(use_cache)
    else:
        if algorithm == 'tsne':
            from sklearn.manifold import TSNE
            dim_red_algorithm = TSNE(n_components=2)
        elif algorithm == 'umap':
            import umap
            dim_red_algorithm = umap.UMAP(
                n_components=2,
                n_neighbors=30,
            )
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

        if use_cache is not None:
            if not os.path.exists(use_cache):
                os.makedirs(os.path.dirname(use_cache), exist_ok=True)
            np.save(use_cache, reduced_fac)

    features_2d = reduced_fac[:len(features)]
    centroids_2d = reduced_fac[len(features):]

    subsample = 0.2
    if subsample < 1.:
        indices = np.random.choice(len(labels), int(subsample * len(labels)), replace=False)
        print(f"Subsampling data to length {len(indices)}")
        features_2d = features_2d[indices]
        labels = labels[indices]
        poisoned_set = poisoned_set[indices]

    unique_labels = np.unique(labels)
    if colors is None:
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
            # colors = [plt.cm.tab20(i) for i in range(num_classes)]
            colors = plt.cm.magma(np.linspace(0, 1, len(unique_labels)))


    if plot_binary:
        s = 5
        color_ = 'skyblue'
        label = 'Clean'
        alpha = 1

        plt.scatter(features_2d[~poisoned_set, 0], features_2d[~poisoned_set, 1], color=color_, label=label, alpha=alpha, s=s)
    else:
        plt.figure(figsize=(20, 20))
        for i in range(len(unique_labels)):
            mask = labels == i
            color_ = colors[i]
            label = i
            alpha = 0.5

            plt.scatter(features_2d[mask, 0], features_2d[mask, 1], color=color_, label=label, alpha=alpha)
            # plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], color=colors[i], marker='s', s=250, edgecolors='black')

    if poisoned_set is not None:
        if poisoned_set_data == 'poisoned':
            color = 'black'
            label = 'Poisoned'
            plt.title('Predicted', fontsize=17.5)
        elif poisoned_set_data == 'poisoned_gt':
            color = 'darkred'
            label = 'Poisoned'
            plt.title('Actual', fontsize=17.5)
        elif poisoned_set_data == 'detected':
            color = 'darkolivegreen'
            label = 'Detected'
            plt.title('Predicted', fontsize=17.5)

        plt.scatter(features_2d[poisoned_set, 0], features_2d[poisoned_set, 1], color=color, label=label, alpha=0.75) #, s=s)

    if plot_centroids:
        for i in range(centroids.shape[0]):
            if i == 0:
                s= 2500
            else:
                s= 250

            plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], color=colors[i], marker='s', s=s, edgecolors='black') # squares
            if original_centroids is not None:
                plt.scatter(original_centroids_2d[i, 0], original_centroids_2d[i, 1], color=colors[i], marker='v', s=s, edgecolors='black') # triangles

    # plt.xlim([-8, 18])
    # plt.xticks([])
    # plt.yticks([])
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.grid()
    plt.legend(fontsize=15, markerscale=2, loc='upper right')
    plt.savefig(save_path, format='pdf' if save_path.endswith('.pdf') else 'png')
    plt.close()
