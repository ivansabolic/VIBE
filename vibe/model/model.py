from typing import Iterator

import torch
from torch import nn
from torch.nn import functional as F

from vibe.model.backbones import backbone_factory
from vibe.model.classifiers import classifier_factory
from vibe.model.initialization import *


class VIBE(nn.Module):
    def __init__(self, model_cfg, freeze_backbone: bool):
        super(VIBE, self).__init__()
        self.cfg = model_cfg

        self.backbone, self.embed_dim = backbone_factory[self.cfg.backbone.name](self.cfg.backbone)
        self.freeze_backbone = freeze_backbone

        self.register_parameter('prior', None)
        self.c = self.cfg.c

        self.classifier = classifier_factory[self.cfg.classifier.name](self.cfg.classifier)

    def get_current_prior(self):
        return self.prior

    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        if self.freeze_backbone:
            for name, param in self.backbone.named_parameters():
                param.requires_grad = False

        for name, param in self.named_parameters():
            if param.requires_grad and 'prior' not in name and 'classifier' not in name:
                yield param

    def initialize(self, features, labels):
        device = next(self.backbone.parameters()).device
        self.prior = nn.Parameter(prior_inits[self.cfg.prior_init](features, labels, device=device))

        self.classifier.initialize(
            mu_z=prototype_inits[self.cfg.classifier.mu_z_init](features=features, labels=labels, device=device),
            mu_y=prototype_inits[self.cfg.classifier.mu_y_init](features=features, labels=labels, device=device)
        )

    def state_dict(self, *args, **kwargs):
        state_dict = super(VIBE, self).state_dict(*args, **kwargs)

        if self.freeze_backbone:
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith('backbone.')}

        return state_dict

    def load_state_dict(self, state_dict, strict=True):
        if 'prior' in state_dict and self.prior is None:
            self.prior = nn.Parameter(torch.empty_like(state_dict['prior']))

        if 'classifier.mu_z' in state_dict and self.classifier.mu_z is None:
            self.classifier.mu_z = nn.Parameter(
                torch.empty_like(state_dict['classifier.mu_z']),
                requires_grad=not self.classifier.cfg.freeze_mu_z
            )
        if 'classifier.mu_y' in state_dict and self.classifier.mu_y is None:
            self.classifier.mu_y = nn.Parameter(
                torch.empty_like(state_dict['classifier.mu_y']),
                requires_grad=not self.classifier.cfg.freeze_mu_y
            )

        return super().load_state_dict(state_dict, strict)

    def variational_loss(self, v, y_oh, uniform_p_yl=False):
        logits = self.classifier.logits_l_v(v) # (B, C)
        if self.prior is not None:
            prior = self.prior * self.c
            logits += prior.log_softmax(-1)

        ln_prob_l_x = F.log_softmax(logits, dim=1) # (B, C)

        if uniform_p_yl:
            ln_prob_y_l = torch.log(torch.ones_like(y_oh) / y_oh.shape[1]) # (B, C)
        else:
            logits_y_l = self.classifier.logits_y_l() # (C, C)
            ln_prob_y_l_num = y_oh @ logits_y_l # (B, C)
            ln_prob_y_l_den = torch.logsumexp(logits_y_l, dim=0,
                                              keepdim=True) # (1, C)
            ln_prob_y_l = (ln_prob_y_l_num - ln_prob_y_l_den) # (B, C)

        L_VD = ln_prob_l_x + ln_prob_y_l # (B, C)

        return -L_VD

    @torch.inference_mode()
    def predict_clean(self, v):
        return self.forward(v)

    def embed(self, x):
        return self.backbone(x)

    def forward(self, v):
        exp_ = self.classifier.logits_l_v(v)
        if self.prior is not None:
            prior = self.prior * self.c
            exp_ += prior.log_softmax(-1)

        ln_prob_l_x = F.log_softmax(exp_, dim=1)

        return ln_prob_l_x

class XConditionedVIBE(VIBE):
    def variational_loss(self, v, y_oh, uniform_p_yl=False):
        logits = self.classifier.logits_l_v(v)
        if self.prior is not None:
            prior = self.prior * self.c
            logits += prior.log_softmax(-1)

        ln_prob_l_x = F.log_softmax(logits, dim=1) # (B, C)

        if uniform_p_yl:
            ln_prob_y_l_x = torch.log(torch.ones_like(y_oh) / y_oh.shape[1])
        else:
            logits_y_l_x = self.classifier.logits_y_l_x(v) # (B, Cy, Cl)
            ln_prob_y_l_x_num = torch.einsum('bij, bi->bj', logits_y_l_x, y_oh) # (B, C)
            ln_prob_y_l_x_den = torch.logsumexp(logits_y_l_x, dim=1,
                                              keepdim=False)

            ln_prob_y_l_x = (ln_prob_y_l_x_num - ln_prob_y_l_x_den)

        L_VD = ln_prob_l_x + ln_prob_y_l_x  #

        return -L_VD