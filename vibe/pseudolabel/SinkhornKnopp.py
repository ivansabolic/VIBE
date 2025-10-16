import torch
from torch import Tensor

class PseudoLabel:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def current_l(self):
        if not hasattr(self, 'l'):
            raise ValueError("l is not initialized")

        return self.l


class SinkhornKnopp(PseudoLabel):
    def __init__(self, model, cfg, dtype: torch.dtype = torch.float32):
        self.model = model
        self.lmd = cfg.lmd
        self.max_iter = cfg.max_iter
        self.dtype = dtype

    @torch.inference_mode()
    def __call__(self, P: Tensor) -> (Tensor, Tensor):
        n = P.shape[0]
        k = P.shape[1]
        prior = self.model.get_current_prior().detach().cpu()
        temp = self.model.c

        if prior is not None:
            assert len(prior.shape) == 1 and len(prior) == k
            prior = (prior * temp).softmax(-1)
            assert prior.sum().allclose(torch.ones(1, device=prior.device))
            prior = prior.type(self.dtype)

        P = P.type(self.dtype)  # (n, k)

        u = torch.ones(k, dtype=self.dtype, device=P.device) / k if prior is None else prior.squeeze().to(P.device) # k 
        v = torch.ones(n, dtype=self.dtype, device=P.device) / n # n 
        log_u, log_v = u.log(), v.log() # k, n
        P *= self.lmd  # (k, n)
        log_u_start = log_u.clone() # k 
        log_v_start = log_v.clone() # n 
        err = 1e6
        cnt = 0

        while err > 1e-1:
            log_u = log_u_start - torch.logsumexp(P + log_v.unsqueeze(-1), dim=0) # (n, k) + (n, 1) = (n, k) -> (k)
            log_v_new = log_v_start - torch.logsumexp(P + log_u.unsqueeze(0), dim=1) # (n, k) + (n, 1) = (n, k) -> (n)

            log_v = log_v_new
            cnt += 1
            if cnt > self.max_iter:
                break

        P += log_v.unsqueeze(-1)
        P += log_u.unsqueeze(0)  # (n, k)

        argmaxes = torch.argmax(P, dim=1).long().cpu()  # (n,)

        self.l = argmaxes.clone()
        return P, argmaxes