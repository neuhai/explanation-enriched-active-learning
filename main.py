import logging
import shutil
import os

import torch
from datasets import load_dataset
from argparse import ArgumentParser

from data import batch_selection, rationale_model_preprocessing
from train import active_learning, active_learning_uncertainty

logger = logging.getLogger(__name__)

parser = ArgumentParser()
parser.add_argument("--num_iter", default=2, type=int)
parser.add_argument("--num_data_per_batch", default=10, type=int)
parser.add_argument("--num_epochs_rg", default=2, type=int)  # 20
parser.add_argument("--num_epochs_p", default=2, type=int)  # 250
parser.add_argument("--learning_rate", default=1e-4, type=float)
parser.add_argument("--per_device_batch_size", default=2, type=int)
parser.add_argument(
    "--criteria",
    choices=[
        "random",
        "even",
        "even_rationale",
        "uncertainty",
        "uncertainty_rationale",
    ],
    default="random",
)


if __name__ == "__main__":
    if input("Cleaning old data, enter 'y' to confirm: ") != "y":
        exit(0)

    for dir in [
        "rationale_model_data",
        "rationale_model",
        "prediction_model_data",
        "prediction_model",
    ]:
        try:
            shutil.rmtree(dir)
        except FileNotFoundError:
            pass
        os.mkdir(dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    args = parser.parse_args()

    config = {
        "num_iter": args.num_iter,
        "num_data_per_batch": args.num_data_per_batch,
        "num_epochs_rg": args.num_epochs_rg,
        "num_epochs_p": args.num_epochs_p,
        "learning_rate": args.learning_rate,
        "per_device_batch_size": args.per_device_batch_size,
        "device": device,
    }
    """
    {
        "num_iter": 5,
        "num_data_per_batch": 10,
        "num_epochs_rg": 2,
        "num_epochs_p": 2,
        "learning_rate": 1e-4,
        "per_device_batch_size": 2,
        "device": device,
    }
    """

    full_train_dataset = load_dataset("esnli", split="train")
    # valid_dataset = load_dataset('esnli', split='validation')
    test_dataset = load_dataset("esnli", split="test")

    train_dataset = batch_selection(full_train_dataset, 3000, "random", -1, [], config)[
        0
    ]

    sampled_test_dataset = batch_selection(test_dataset, 300, "random", -1, [], config)[
        0
    ]
    rationale_dataset_test_dataset = rationale_model_preprocessing(sampled_test_dataset)

    print("# data subsampled from train split: ", len(train_dataset))

    if "uncertainty" in args.criteria:
        active_learning_uncertainty(
            args.criteria,
            train_dataset,
            sampled_test_dataset,
            rationale_dataset_test_dataset,
            config,
        )
    else:
        active_learning(
            args.criteria,
            train_dataset,
            sampled_test_dataset,
            rationale_dataset_test_dataset,
            config,
        )
    # active_learning("even", config)
    # active_learning_uncertainty("uncertainty", config)
    # active_learning_uncertainty(
    #     "uncertainty_rationale",
    #     train_dataset,
    #     sampled_test_dataset,
    #     rationale_dataset_test_dataset,
    #     config,
    # )
