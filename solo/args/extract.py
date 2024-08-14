import argparse

from solo.args.dataset import custom_dataset_args, dataset_args


def parse_args_extract() -> argparse.Namespace:
    """Parses arguments for offline UMAP.

    Returns:
        argparse.Namespace: a namespace containing all args needed for pretraining.
    """

    parser = argparse.ArgumentParser()

    # add knn args
    parser.add_argument("--pretrained_checkpoint_dir", type=str)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=10)

    parser.add_argument("--attack", type=str, default="badnets")
    parser.add_argument("--target_label", type=int, default=0)
    parser.add_argument("--poisoning_rate", type=float, default=0.1)
    parser.add_argument("--train_poisoned_set", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="extracted_features")
    # add shared arguments
    dataset_args(parser)
    custom_dataset_args(parser)

    # parse args
    args = parser.parse_args()

    return args
