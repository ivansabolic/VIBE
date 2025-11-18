import os.path
from abc import ABC, abstractmethod
from torch.utils.data import Dataset, DataLoader
import torch
from torch import nn, Tensor
import numpy as np
import cudf
import torch
from cugraph import Graph
from cugraph import leiden as culeiden
from cuml.neighbors import NearestNeighbors
import cupy as cp
from cuml.manifold.simpl_set import fuzzy_simplicial_set
from tqdm import tqdm
import matplotlib.pyplot as plt

from vibe.utils import extract_features, find_centroids, plot_data
from omegaconf import DictConfig as Config
from typing import List, Union



class PreprocessingHook(ABC):
    def __init__(self) -> None:
        pass

    def preprocess(self, model: nn.Module, dataset: Dataset) -> Tensor:
        pass


class MaximumDistanceCentroidsHook(PreprocessingHook):
    def __init__(self, cfg: Config, experiment_dir: str) -> None:
        super().__init__()
        self.cfg = cfg
        self.out_dir = os.path.join(
            experiment_dir,
            "preprocessing",
        )
        os.makedirs(self.out_dir, exist_ok=False)

    def _plot_histogram(self, data, out_dir, name, colors=None):
        if colors is None:
            colors = ["blue", "red", "green", "orange", "purple", "brown", "pink", "gray", "olive", "cyan"]

        plt.hist(data, bins=100, color=colors[0])
        plt.xlabel("Distance")
        plt.ylabel("Frequency")
        plt.title(name)

        plt.savefig(os.path.join(out_dir, f"{name}.png"))
        plt.close()

    def preprocess(self, model: nn.Module, dataset: Dataset) -> Tensor:
        loader = DataLoader(
            dataset,
            batch_size=256,
            shuffle=False,
        )

        features, labels = extract_features(model, loader)

        centroids, predicted_labels = self.calculate_centroids(features, labels, **self.cfg)

        # centroid to centroid distance
        centroid_distances = torch.zeros((centroids.shape[0], centroids.shape[0]))
        for i in range(centroids.shape[0]):
            for j in range(centroids.shape[0]):
                centroid_distances[i, j] = (centroids[i] - centroids[j]).norm()

        avg_centroid_distances = centroid_distances.mean(dim=1)
        avg_centroid_distances = avg_centroid_distances / torch.max(avg_centroid_distances)

        avg_distance_to_mean_others = torch.zeros(centroids.shape[0])
        for i in range(centroids.shape[0]):
            without_i = torch.cat([avg_centroid_distances[:i], avg_centroid_distances[i + 1:]])
            avg_distance_to_mean_others[i] = torch.abs(avg_centroid_distances[i] - without_i.mean())

        if torch.isnan(avg_distance_to_mean_others).any():
            print("Warning: avg_distance_to_mean_others contains NaN values.")
            breakpoint()
        self._plot_histogram(avg_distance_to_mean_others, self.out_dir, "avg_distance_to_mean_others")

        poisoned_centroids = torch.arange(len(centroids))[avg_distance_to_mean_others > self.cfg.delta]
        if len(poisoned_centroids) > 0:
            poisoned_centroids = poisoned_centroids[poisoned_centroids == torch.argmax(avg_distance_to_mean_others)]

        filtered_indices = Tensor([i for i, l in enumerate(labels) if predicted_labels[i] in poisoned_centroids])

        poisoned_set_indices = dataset.poisoned_set_indices
        poisoned_indices_detected = torch.from_numpy(
            np.intersect1d(poisoned_set_indices.numpy(), filtered_indices.numpy())
        )

        # if len(filtered_indices) > 0:
        #     print(f"Removed {len(poisoned_indices_detected)} poisoned examples!. "
        #           f"Lost {len(filtered_indices) - len(poisoned_indices_detected)} clean examples.")

        # experiment_name = self.out_dir.split("/")[0]
        # cache_path = os.path.join(
        #     "plots",
        #     "2d_plots",
        #     experiment_name
        # )
        # os.makedirs(cache_path, exist_ok=True)
        # cache_path = os.path.join(cache_path, "umap_features.npy")
        outlier_centroid = np.argmax(avg_distance_to_mean_others)
        outlier_indices = np.array([i for i, l in enumerate(labels) if predicted_labels[i] == outlier_centroid])
        outlier_indices_bool = np.zeros(len(labels), dtype=bool)
        outlier_indices_bool[outlier_indices] = True

        # centroids_removed = np.delete(centroids, outlier_centroid, axis=0)
        # plot_data(features, labels, centroids=centroids_removed, poisoned_set=outlier_indices_bool, algorithm='umap',
        #           save_path=os.path.join(self.out_dir, "features_filtered.pdf"), plot_binary=True, plot_centroids=False,
        #           use_cache=cache_path, poisoned_set_data='detected',
        #           )
        # plot_data(features, labels, centroids=centroids_removed, poisoned_set=dataset.poisoned_set, algorithm='umap',
        #           save_path=os.path.join(self.out_dir, "features_poisoned.pdf"), plot_binary=True, plot_centroids=False,
        #           use_cache=cache_path, poisoned_set_data='poisoned_gt'
        #           )

        all_indices = torch.arange(len(labels))
        if len(filtered_indices) > 0:
            mask = torch.ones(len(labels), dtype=torch.bool)
            mask[filtered_indices.int()] = False
            kept_indices = all_indices[mask]
        else:
            kept_indices = all_indices

        return kept_indices

    @abstractmethod
    def calculate_centroids(self, features: Tensor, labels: Tensor,  **kwargs) -> (Tensor, Tensor):
        pass

class LeidenHook(MaximumDistanceCentroidsHook):
    def _adjacency_graph(self, X, num_neighbors, use_weights=True, random_state=0, metric="euclidean", type='tensor'):
        X_cupy = cp.asarray(X)

        # find the nearest neighbors
        nn = NearestNeighbors(n_neighbors=num_neighbors + 1, algorithm="brute", metric=metric, output_type="cupy")
        nn.fit(X_cupy)
        distances, neighbors = nn.kneighbors(X_cupy)
        distances = distances[:, 1:]
        neighbors = neighbors[:, 1:]

        # compute the fuzzy simplicial set
        set_op_mix_ratio = 1.0
        local_connectivity = 1.0
        n_obs = X.shape[0]
        X_conn = cp.empty((n_obs, 1), dtype=cp.float32)
        connectivities = fuzzy_simplicial_set(
            X_conn, num_neighbors, random_state,
            metric=metric,
            knn_indices=neighbors,
            knn_dists=distances,
            set_op_mix_ratio=set_op_mix_ratio,
            local_connectivity=local_connectivity,
        )

        # create the graph
        connectivities_crs = connectivities.tocsr() if use_weights else None
        src = connectivities.row
        dst = connectivities.col
        weights = connectivities_crs[src, dst][0] if use_weights else None

        # src = cp.repeat(cp.arange(n_obs), num_neighbors).ravel()
        # dst = neighbors.ravel()
        # # weights = cp.exp(-distances)
        # weights = cp.ones(len(src))
        if type == 'tensor':
            return torch.sparse_coo_tensor(
                cp.vstack((src, dst)),
                weights if use_weights else cp.ones(len(src)),
                (n_obs, n_obs), dtype=torch.float64, device='cuda')

        g = Graph(directed=False)
        if use_weights:
            df = cudf.DataFrame({"source": src, "destination": dst, "weights": weights})
            g.from_cudf_edgelist(df, source="source", destination="destination", weight="weights")
        else:
            df = cudf.DataFrame({"source": src, "destination": dst})
            g.from_cudf_edgelist(df, source="source", destination="destination")
        return g

    def _leiden(self, graph, resolution=0.03, iters=100):
        assert isinstance(graph, Graph)
        result, score = culeiden(graph, resolution=resolution, max_iter=iters)
        result = result.sort_values('vertex')
        return torch.tensor(result['partition'])

    def calculate_centroids(self, features: Tensor, labels: Tensor,  **kwargs) -> (Tensor, Tensor):
        features = features.cuda()
        A = self._adjacency_graph(features, num_neighbors=self.cfg.num_neighbors, use_weights=False, type='graph')

        preds = None
        closest = 1e3
        K = np.unique(labels).shape[0] + 1
        for res in tqdm(np.linspace(self.cfg.start, self.cfg.end, self.cfg.steps), desc="Leiden Calculation"):
            preds_ = self._leiden(A, resolution=res, iters=1000)
            num_clusters = torch.unique(preds_).shape[0]
            if np.abs(num_clusters - K) < closest:
                closest = np.abs(num_clusters - K)
                preds = preds_
                # print(f"Best res so far: {res} with num_clusters: {num_clusters}")
                if closest == 0:
                    break

        preds = preds.cpu() if preds.is_cuda else preds
        
        centroids = find_centroids(features, preds)
        labels = preds

        return centroids, labels


preprocessing_factory = {
    "leiden": LeidenHook,
}

def build_preprocessing_hook(hooks: Union[dict, List[dict]], **kwargs) -> List[PreprocessingHook]:
    if isinstance(hooks, dict):
        hooks = [hooks]

    return [preprocessing_factory[hook["name"]](hook, **kwargs) for hook in hooks]
