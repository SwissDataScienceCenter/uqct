import json
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import numpy as np
import torch

from uqct.datasets.utils import get_dataset


def get_base_dir():
    if Path("/cluster").exists():
        return Path("/cluster/scratch/mgaetzner/uqct/")
    else:
        return Path("./")


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def run_cseq(kwargs: dict[str, Any]) -> None:
    args_str = "\n".join(f"\t{k}: {v}" for k, v in kwargs.items())
    print(f"Arguments:\n{args_str}")

    base_dir = get_base_dir()
    cseq_dir = base_dir / "slm" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    cseq_dir.mkdir(exist_ok=True, parents=True)

    # Saving metadata about confidence sequence
    json.dump(kwargs, cseq_dir.open("w"))
    print(f"Saved arguments to '{cseq_dir}'")

    # Set random seeds
    set_seeds(kwargs["seed"])

    # Retrieve image
    _, test_set = get_dataset(kwargs["dataset"])

    # Simulate measurements

    # Obtain predictions

    # Save predictions with metrics

    # Compute confidence sequence

    # Save confidence coefficients, NLL of true image


@click.command()
@click.option(
    "--sparse",
    default=False,
    is_flag=True,
    help="sequence of linearly increasing angles or number number of (dense) scanning rounds",
)
@click.option(
    "--predictor",
    default="mle",
    type=click.Choice(["mle", "map", "unet", "diffusion"]),
    help="which predictor to use",
)
@click.option(
    "--dataset",
    default="lamino",
    type=click.Choice(["lamino", "lung", "composite"]),
    help="which dataset to use",
)
@click.option(
    "--test-set-idx",
    default=0,
    type=click.IntRange(min=0),
    help="test set index, i.e. which image to run the sequence for",
)
@click.option(
    "--exposure",
    default=1e9,
    type=click.FloatRange(min=1e4),
    help="TOTAL exposure time after T angles or rounds",
)
@click.option(
    "--T",
    default=200,
    type=click.IntRange(min=0),
    help="number of angles (sparse) or rounds (dense)",
)
@click.option(
    "--t_0",
    default=1,
    type=click.IntRange(min=1),
    help="after how many time steps to start the sequence",
)
@click.option(
    "--seed",
    default=0,
    type=click.IntRange(min=0),
    help="random seed",
)
def main(**kwargs):
    run_cseq(kwargs)


if __name__ == "__main__":
    main()
