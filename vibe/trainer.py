import os
import time

from omegaconf import OmegaConf
import torch
from torch.nn import Module
from typing import List, Union, Tuple

from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler as Scheduler

from datasets.vibe_dataset import SimpleVIBEDataset
from datasets.utils import FilteredDataset
from vibe.model import VIBE
from vibe.model.model import XConditionedVIBE
from vibe.utils import build_dataloader, build_optimizer, build_scheduler, plot_data, get_features, calculate_dataset_ln_P
from vibe.hooks import *
from vibe.pseudolabel import pseudolabel_factory

from hydra.core.hydra_config import HydraConfig

class Trainer:
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.training_cfg = cfg.training
        self.experiment_dir = self._build_experiment_dir()
        OmegaConf.save(cfg, f"{self.experiment_dir}/config.yaml")

        self.model_fn = VIBE

        self.preprocessing_hooks: List[PreprocessingHook] = build_preprocessing_hook(cfg.hooks.preprocessing,
                                                                                     experiment_dir=self.experiment_dir)
        self.logging_hooks: List[LoggingHook] = build_logging_hook(cfg.hooks.log, experiment_dir=self.experiment_dir)
        self.metric_hooks: List[MetricHook] = build_metric_hook(cfg.hooks.metrics)

        self.freeze_backbone = cfg.model.backbone.freeze_backbone

        self._cached_features = {}

    def _build_experiment_dir(self) -> str:
        experiment_name = f"{self.cfg.experiments_root}/{self.cfg.dataset}/{self._build_experiment_name()}"
        os.makedirs(experiment_name, exist_ok=True)
        return experiment_name

    def _build_experiment_name(self) -> str:
        exp_name = f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{HydraConfig.get().job.config_name}"
        if self.cfg.add_info != "":
            exp_name += f"_{self.cfg.add_info}"

        return exp_name

    def save(self, model: Module) -> None:
        torch.save(model.state_dict(), f"{self.experiment_dir}/model.pth")

    def prepare_loaders(self, model: VIBE, train_dataset_no_transform: Dataset, train_dataset: Dataset) -> (DataLoader, DataLoader):
        if self.freeze_backbone:
            features, labels = self.load_and_cache_features(model, build_dataloader(train_dataset_no_transform,
                                                                                    self.cfg.dataloader, shuffle=False),
                                                            cache_info="train")
            train_dataset_no_transform = SimpleVIBEDataset(features, labels)
            train_dataset = train_dataset_no_transform

        for hook in self.preprocessing_hooks:
            filtered_indices = hook.preprocess(model, train_dataset_no_transform)
            train_dataset_no_transform = FilteredDataset(train_dataset_no_transform, filtered_indices)
            train_dataset = FilteredDataset(train_dataset, filtered_indices)

        train_loader_no_transform = build_dataloader(train_dataset_no_transform, self.cfg.dataloader, shuffle=False)
        train_loader = build_dataloader(train_dataset, self.cfg.dataloader, shuffle=True)

        return train_loader_no_transform, train_loader

    def load_and_cache_features(self, model: VIBE, loader: DataLoader, cache_info: str) -> (Tensor, Tensor):
        if self.freeze_backbone and cache_info in self._cached_features:
            return self._cached_features[cache_info]

        features, labels = get_features(model, loader, self.cfg, self.freeze_backbone, cache_info)

        if self.freeze_backbone:
            self._cached_features[cache_info] = (features, labels)

        return features, labels

    def train(self, train_datasets: List[Tuple[Dataset, str]], eval_dataset: Union[Dataset, List[Tuple[Dataset, str]]]) -> None:
        train_dataset, _ = train_datasets[0]
        train_dataset_no_transform, _ = train_datasets[1]

        model = self.model_fn(self.cfg.model, self.freeze_backbone).cuda()
        pseudo_label_fn = pseudolabel_factory[self.cfg.pseudo_label.name](model, self.cfg.pseudo_label)

        train_loader_no_transform, train_loader = self.prepare_loaders(model, train_dataset_no_transform, train_dataset)

        features, labels = self.load_and_cache_features(model, train_loader_no_transform, cache_info="train")
        model.initialize(features, labels)

        optimizer = build_optimizer(model, self.training_cfg.optimizer)
        scheduler = build_scheduler(optimizer, self.training_cfg.scheduler)

        num_classes = torch.max(labels) + 1
        # tl_iter = cycle(train_loader)
        tl_iter = iter(train_loader)
        for i in range(self.cfg.training.num_iterations):
            # batch_iter = next(tl_iter)
            try:
                batch_iter = next(tl_iter)
            except StopIteration:
                tl_iter = iter(train_loader)
                batch_iter = next(tl_iter)

            iter_loss = self.train_one_iter(i, batch_iter, model, pseudo_label_fn, train_loader_no_transform, num_classes, optimizer,
                                            scheduler)

            # log
            if (i + 1) % self.training_cfg.logging_interval == 0:
                self.iter_log(iter_loss, i + 1)

            # save model
            if (i + 1) % self.training_cfg.save_interval == 0 or i == self.training_cfg.num_iterations - 1:
                self.save(model)

            # evaluate model
            if (i + 1) % self.training_cfg.eval_interval == 0 or i == self.training_cfg.num_iterations - 1:
                self.eval(model, eval_dataset)
        
    def train_one_iter(self, iter_, batch_iter, model, pseudo_label_fn, train_loader: DataLoader, num_classes: int,
                       optimizer: Optimizer,
                       scheduler: Scheduler) -> float:

        # E-step
        if iter_ % self.training_cfg.T == 0:
            features, labels = self.load_and_cache_features(model, train_loader, cache_info="train")
            breakpoint()

            # calculate neg_ln_P
            ln_P = calculate_dataset_ln_P(iter_, model, features, labels, num_classes, chunks=self.training_cfg.e_step_chunks)

            # do pseudo label
            Q, l = pseudo_label_fn(ln_P)

            # do some log (like e step loss and accuracy of pseudo labels)
            self.e_step_log(ln_P, Q, l, features, labels, train_loader, iter_=iter_)

            if self.training_cfg.plot_data:
                save_path = os.path.join(self.experiment_dir, f"plots")
                os.makedirs(save_path, exist_ok=True)
                plot_data(
                    features,
                    labels,
                    centroids=model.classifier.mu_z.detach().cpu().numpy(),
                    original_centroids=model.classifier.mu_y.detach().cpu().numpy(),
                    poisoned_set=train_loader.dataset.poisoned_set if hasattr(train_loader.dataset, 'poisoned_set') else None,
                    use_cache=None,
                    algorithm='umap',
                    save_path=f"{save_path}/latent-space_{iter_}.png",
                )

            del ln_P, Q
            torch.cuda.empty_cache()
        else:
            l = pseudo_label_fn.current_l()

        model.train()

        # M step
        x, y, l_gt, index = batch_iter
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad()

        if not self.freeze_backbone:
            v = model.embed(x)
        else:
            v = x
        y_oh = F.one_hot(y, num_classes=num_classes).float().cuda()

        # calculate loss for this batch and update params
        neg_ln_P = model.variational_loss(v, y_oh)
        l_batch = l[index]
        l_oh = F.one_hot(l_batch, num_classes=num_classes).cuda()
        loss = (l_oh * neg_ln_P).sum(-1).mean()
        loss.backward()

        optimizer.step()
        scheduler.step()

        return loss.item()

    def e_step_log(self, neg_ln_P: Tensor, Q: Tensor, l: Tensor, features: Tensor, labels: Tensor, train_loader: DataLoader,
                   iter_: int = None) -> None:
        if hasattr(train_loader.dataset, "poisoned_set"):
            ps = train_loader.dataset.poisoned_set
            poisoned_set_not_relabeled = f"{(l[ps] == labels[ps]).sum()}/{ps.sum()}"
        else:
            poisoned_set_not_relabeled = "N/A"

        for hook in self.logging_hooks:
            hook.log({
                # "e_step_loss": torch.trace(P.T @ Q) + torch.trace(Q.T @ (torch.log(Q) - 1)),
                "given_label_acc": (l == labels).float().mean().item(),
                "original_label_acc": (l == train_loader.dataset.original_labels).float().mean().item(),
                "poisoneds_not_relabeled": poisoned_set_not_relabeled,
            },
                category="E-STEP",
                iter_=iter_,
            )

    def iter_log(self, iter_loss: float, iter_: int) -> None:
        for hook in self.logging_hooks:
            hook.log({"iter_loss": iter_loss}, iter_=iter_)

    def eval(self, model: VIBE, eval_datasets: Union[Dataset, List[Tuple[Dataset, str]]]) -> None:
        if isinstance(eval_datasets, Dataset):
            eval_datasets = [(eval_datasets, "test")]

        model.eval()

        features, labels, poisoned_set = [], [], []
        for eval_dataset, eval_info in eval_datasets:
            eval_features, eval_labels = self.load_and_cache_features(model, build_dataloader(eval_dataset, self.cfg.dataloader),
                                                           cache_info=eval_info.lower().replace(" ", "_"))
            predictions = model.predict_clean(eval_features.cuda()).cpu()

            features.append(eval_features)
            labels.append(eval_labels)
            poisoned_set.append(torch.ones_like(eval_labels) if "POISONED" in eval_info else torch.zeros_like(eval_labels))

            metrics = {metric: metric_hook(predictions, eval_labels) for metric, metric_hook in
                       zip(self.cfg.hooks.metrics, self.metric_hooks)}
            self.eval_log(metrics, eval_info)

            if self.training_cfg.plot_eval_data:
                save_path = os.path.join(self.experiment_dir, f"plots")
                os.makedirs(save_path, exist_ok=True)
                plot_data(
                    eval_features,
                    eval_labels,
                    centroids=model.classifier.mu_z.detach().cpu().numpy(),
                    original_centroids=None,
                    poisoned_set=None,
                    use_cache=None,
                    algorithm='umap',
                    save_path=f"{save_path}/latent-space_{eval_info}.png",
                )

    def eval_log(self, metrics, eval_info: str = "test"):
        for hook in self.logging_hooks:
            hook.log(metrics, eval_info)


class BackdoorTrainer(Trainer):
    def _build_experiment_dir(self) -> str:
        experiment_name = (f"{self.cfg.experiments_root}/"
                           f"backdoor/"
                           f"{self.cfg.dataset.name}/"
                           f"{self.cfg.backdoor.name}_{self.cfg.backdoor.target_label}_{self.cfg.backdoor.poisoning_rate}/"
                           f"{self._build_experiment_name()}")
        os.makedirs(experiment_name, exist_ok=True)
        return experiment_name


class XConditionedBackdoorTrainer(BackdoorTrainer):
    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.model_fn = XConditionedVIBE

    def train_one_iter(self, iter_, batch_iter, model, pseudo_label_fn, train_loader: DataLoader, num_classes: int,
                       optimizer: Optimizer,
                       scheduler: Scheduler) -> float:
        # E-step
        if iter_ % self.training_cfg.T == 0:
            features, labels = self.load_and_cache_features(model, train_loader, cache_info="train")

            # calculate neg_ln_P
            ln_P = calculate_dataset_ln_P(iter_, model, features, labels, num_classes,
                                                  chunks=self.training_cfg.e_step_chunks)

            # do pseudo label
            Q, l = pseudo_label_fn(ln_P)

            # do some log (like e step loss and accuracy of pseudo labels)
            self.e_step_log(ln_P, Q.to(ln_P.dtype), l, features, labels, train_loader, iter_=iter_)

            if self.training_cfg.plot_data:
                save_path = os.path.join(self.experiment_dir, f"plots")
                os.makedirs(save_path, exist_ok=True)
                plot_data(
                    features,
                    labels,
                    centroids=model.classifier.mu_z.detach().cpu().numpy(),
                    original_centroids=model.classifier.mu_y.detach().cpu().numpy(),
                    poisoned_set=train_loader.dataset.poisoned_set,
                    use_cache=None,
                    algorithm='umap',
                    save_path=f"{save_path}/latent-space_{iter_}.png",
                )

            del ln_P, Q
            torch.cuda.empty_cache()
        else:
            l = pseudo_label_fn.current_l()

        model.train()

        # M step
        x, y, l_gt, index = batch_iter
        x = x.cuda(non_blocking=True)
        y = y.cuda(non_blocking=True)

        optimizer.zero_grad()

        if not self.freeze_backbone:
            v = model.embed(x)
        else:
            v = x
        y_oh = F.one_hot(y, num_classes=num_classes).float().cuda()

        # calculate loss for this batch and update params
        l_batch = l[index]
        l_oh = F.one_hot(l_batch, num_classes=num_classes).cuda()
        neg_ln_P = model.variational_loss(v, y_oh)
        loss = (l_oh * neg_ln_P).sum(-1).mean()
        loss.backward()

        optimizer.step()
        scheduler.step()

        return loss.item()

