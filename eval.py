import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from datasets import fetch_transforms_for_dataset, get_dataset

from datasets.backdoor import backdoor_factory
from datasets.vibe_dataset import VIBEDataset
from vibe import trainer_factory


@hydra.main(version_base="1.3.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # clean dataset creation
    test_dataset = get_dataset(cfg.dataset, train=False)

    _, test_transform = fetch_transforms_for_dataset(cfg.dataset.name, cfg.model.backbone.name)
    clean_test_dataset = get_dataset(cfg.dataset, train=False)
    clean_test_dataset = VIBEDataset(clean_test_dataset, test_transform)

    # poisoning the dataset
    test_backdoor_dataset = backdoor_factory[cfg.backdoor.name](test_dataset, test_transform, cfg.backdoor, train=False)

    vibe = trainer_factory[cfg.trainer](cfg)

    model = vibe.model_fn(cfg.model, cfg.model.backbone.freeze_backbone)
    model.load_state_dict(torch.load(cfg.model_path, weights_only=True))
    model = model.cuda()

    # VIBE eval
    vibe.eval(model, [(clean_test_dataset, "CLEAN TEST"), (test_backdoor_dataset, "POISONED TEST")])

if __name__ == "__main__":
    main()