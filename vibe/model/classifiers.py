from abc import abstractmethod
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

__all__ = ['classifier_factory']

class Classifier(nn.Module):
    @abstractmethod
    def forward(self, x):
        pass

class LinearClassifier(Classifier):
    def __init__(self, cfg) -> None:
        super(LinearClassifier, self).__init__()
        self.mu_z = None
        self.mu_y = None

    def initialize(self, mu_z: Tensor, mu_y: Tensor) -> None:
        self.mu_z = nn.Linear(mu_z.shape[1], mu_z.shape[0], device=mu_z.device)
        self.mu_z.requires_grad = True

        self.mu_y = nn.Parameter(mu_y, requires_grad=False)

    def forward(self, v: Tensor) -> Tensor:
        return self.mu_z(v)

class VonMisesFisherClassifier(Classifier):
    def __init__(self, cfg) -> None:
        super(VonMisesFisherClassifier, self).__init__()
        self.register_parameter('mu_z', None)
        self.register_parameter('mu_y', None)

        self.cfg = cfg
        self.kappa = cfg.kappa
        self.nu = cfg.nu

    def initialize(self, mu_z: Tensor, mu_y: Tensor) -> None:
        self.mu_z = nn.Parameter(mu_z, requires_grad=not self.cfg.freeze_mu_z)
        self.mu_y = nn.Parameter(mu_y, requires_grad=not self.cfg.freeze_mu_y)

    def logits_l_v(self, v: Tensor) -> Tensor:
        mu_z = F.normalize(self.mu_z, dim=1, p=2)
        v = F.normalize(v, dim=1, p=2)
        return (v @ mu_z.t()) * self.kappa

    def logits_y_l(self) -> Tensor:
        mu_z = F.normalize(self.mu_z, dim=1, p=2)
        mu_y = F.normalize(self.mu_y, dim=1, p=2)

        return (mu_y @ mu_z.t()) * self.nu

    def forward(self, v: Tensor) -> Tensor:
        return self.prob_l_v(v)


class XConditionedVonMisesFisherClassifier(VonMisesFisherClassifier):
    def normalize_inputs(self, v: Tensor, mu_z: Tensor) -> Tuple[Tensor, Tensor]:
        if self.cfg.normalize_inputs:
            mu_z = F.normalize(mu_z, dim=1, p=2)
        return v, mu_z

    def h(self, mu_z: Tensor, v: Tensor) -> Tensor:
        # mu_z (1, C, D)
        # v (B, 1, D)
        return mu_z.unsqueeze(0) + v.unsqueeze(1) # (B, C, D)

    def logits_y_l_x(self, v: Tensor) -> Tensor:
        v, mu_z = self.normalize_inputs(v, self.mu_z)

        mu_y = F.normalize(self.mu_y, dim=1, p=2) # (C, D)

        h_ = self.h(mu_z, v)
        
        if self.cfg.normalize_h:
            h_normalized = F.normalize(h_, dim=2, p=2)
        else:
            h_normalized = h_
        
        return torch.einsum('id, bjd -> bij', mu_y, h_normalized) * self.nu # (B, C, C)


classifier_factory = {
    'linear': LinearClassifier,
    'von_mises_fisher': VonMisesFisherClassifier,
    'x_conditioned_von_mises_fisher': XConditionedVonMisesFisherClassifier,
}
