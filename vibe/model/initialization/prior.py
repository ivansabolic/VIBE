import torch

__all__ = ['inits']

def random_init(features: torch.Tensor, labels: torch.Tensor, device: torch.device, **kwargs):
    num_classes = torch.max(labels) + 1
    return torch.randn(num_classes.item(), device=device)

def uniform_init(features: torch.Tensor, labels: torch.Tensor, device: torch.device, **kwargs):
    num_classes = torch.max(labels) + 1
    return torch.ones(num_classes.item(), device=device) / num_classes.item()

inits = {
    'random': random_init,
    'uniform': uniform_init,
}