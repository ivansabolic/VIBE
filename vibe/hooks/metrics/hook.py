from abc import ABC
import torch
from torch import Tensor
from scipy.optimize import linear_sum_assignment
from typing import List, Union

class MetricHook(ABC):
    def __init__(self) -> None:
        self.name = ...

    def __call__(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        raise NotImplementedError


class Accuracy(MetricHook):
    def __init__(self) -> None:
        super().__init__()
        self.name = "accuracy"

    def __call__(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        return (predictions.argmax(1) == ground_truth).sum() / len(ground_truth)


class ClusteringAccuracy(MetricHook):
    def __init__(self) -> None:
        super().__init__()
        self.name = "clustering_accuracy"

        self.mapping = None
        self.num_times_called = 0

    def __call__(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        self.num_times_called += 1

        y_pred, y_true = predictions.argmax(1).long(), ground_truth
        assert y_pred.shape == y_true.shape

        if self.mapping is None or self.num_times_called % 2 == 1:
            w = torch.zeros(y_pred.max() + 1, y_true.max() + 1).long()
            for i in range(y_pred.size(0)):
                w[y_pred[i], y_true[i]] += 1
            row_ind, col_ind = linear_sum_assignment(w.max() - w)

            self.mapping = torch.ones(y_pred.max() + 1).long() * -1
            self.mapping[row_ind] = torch.from_numpy(col_ind)

            accuracy = (w[row_ind, col_ind].sum() / y_true.size(0)).item() * 100
        else:
            mapped_y_pred = torch.zeros_like(y_pred)
            for i in range(y_pred.size(0)):
                mapped_y_pred[i] = self.mapping[y_pred[i]]

            accuracy = (mapped_y_pred == y_true).sum().item() / y_true.size(0) * 100

        return accuracy


class Top5Accuracy(MetricHook):
    def __init__(self) -> None:
        super().__init__()
        self.name = f"top_5_accuracy"
        self.k = 5

    def __call__(self, predictions: Tensor, ground_truth: Tensor) -> Tensor:
        return (predictions.topk(self.k, dim=1).indices == ground_truth.unsqueeze(1)).any(dim=1).sum() / len(ground_truth)

metric_hook_factory = {
    "accuracy": Accuracy,
    "clustering_accuracy": ClusteringAccuracy,
    "top_5_accuracy": Top5Accuracy,
}

def build_metric_hook(hooks: Union[str, List[str]]) -> List[MetricHook]:
    if isinstance(hooks, str):
        hooks = [hooks]

    return [metric_hook_factory[hook]() for hook in hooks]

