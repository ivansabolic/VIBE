import hydra
from omegaconf import DictConfig, OmegaConf
from datasets import fetch_transforms_for_dataset, get_dataset

from datasets.backdoor import backdoor_factory
from datasets.vibe_dataset import VIBEDataset
from vibe import trainer_factory

@hydra.main(
        version_base="1.3.2",
        config_path="configs",
)
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # clean dataset creation
    train_dataset = get_dataset(cfg.dataset, train=True)
    test_dataset = get_dataset(cfg.dataset, train=False)

    train_transform, test_transform = fetch_transforms_for_dataset(cfg.dataset.name, cfg.model.backbone.name)
    clean_test_dataset = get_dataset(cfg.dataset, train=False)
    clean_test_dataset = VIBEDataset(clean_test_dataset, test_transform)

    # poisoning the dataset
    train_backdoor_dataset = backdoor_factory[cfg.backdoor.name](train_dataset, train_transform, cfg.backdoor, train=True)
    train_backdoor_no_transform_dataset = backdoor_factory[cfg.backdoor.name](train_dataset, test_transform, cfg.backdoor, train=True)

    test_backdoor_dataset = backdoor_factory[cfg.backdoor.name](test_dataset, test_transform, cfg.backdoor, train=False)

    vibe = trainer_factory[cfg.trainer](cfg)
    # VIBE training
    vibe.train(
        [(train_backdoor_dataset, "TRAIN + TRANSFORM"), (train_backdoor_no_transform_dataset, "TRAIN")], [(clean_test_dataset, "CLEAN TEST"), (test_backdoor_dataset, "POISONED TEST")])

if __name__ == "__main__":
    main()
