import torch
from vibe.utils import find_centroids

__all__ = ['inits']

def random_init(features, labels, device, **kwargs):
    size = (torch.max(labels) + 1, features.shape[1])
    return torch.randn(size, device=device)

def original_centroids(features, labels, device, **kwargs):
    return find_centroids(features, labels).to(device)


inits = {
    'random': random_init,
    'original_centroids': original_centroids
}