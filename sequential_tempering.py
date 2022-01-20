import argparse
import os
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.montecarlo import hybrid_mcmc, neural_mcmc, single_spin_flip

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--spins", type=int, help="Number of spins of the ravel spin glass")
parser.add_argument("--beta-min", type=float, help="Beta startpoint")
parser.add_argument("--beta-max", type=float, help="Beta endpoint")
parser.add_argument(
    "--beta-num", type=int, help="Number of beta steps for sequential tempering"
)
parser.add_argument(
    "--dataset-size", type=int, help="Dataset size (both training and test)"
)
parser.add_argument("--couplings-path", type=str, help="Path to the couplings")
parser.add_argument("--model", type=str, choices=["made", "pixel"], help="Model to use")
parser.add_argument(
    "--hybrid", dest="hybrid", action="store_true", help="Use Hybrid MCMC"
)


def main(args):
    # set unique dir name
    now = datetime.now()
    date_time = now.strftime("%d-%m-%Y_%H-%M-%S")

    dataset = single_spin_flip(
        args.spins,
        args.beta_min,
        args.dataset_size,
        args.couplings_path,
        sweeps=1000,
    )
    # create directory and save mcmc data
    parent_path = f"data/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/"
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    np.save(
        f"data/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/single-beta{args.beta_min}",
        dataset,
    )
    # save training and validation dataset
    train_data, val_data = train_test_split(dataset, test_size=0.15)
    train_data_path = parent_path + "train"
    val_data_path = parent_path + "val"
    np.save(train_data_path, train_data)
    np.save(val_data_path, val_data)

    # no idea why it needs this, but it does
    os.environ["MKL_THREADING_LAYER"] = "GNU"

    ckpt_path = "null"
    betas = np.linspace(args.beta_min, args.beta_max, num=args.beta_num, endpoint=True)
    for i, beta in enumerate(betas):
        # train the neaural network for the first time
        os.system(
            f"python run.py name={beta} datamodule.datasets.train.name={args.couplings_path.split('/')[-1][:-4]} callbacks.model_checkpoint.dirpath=checkpoints/{date_time} mode=seq_temp"
        )
        # generate using the trained network
        ckpt_path = f"logs/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/checkpoints/{date_time}/best-beta{beta}.ckpt"
        # smart or hybrid montecarlo at the next beta
        if args.hybrid:
            dataset = hybrid_mcmc(
                betas[i + 1],
                args.dataset_size,
                ckpt_path,
                args.couplings_path,
                args.model,
            )
        else:
            dataset = neural_mcmc(
                betas[i + 1],
                args.dataset_size,
                ckpt_path,
                args.couplings_path,
                args.model,
            )
        # save the last montecarlo output
        if beta == betas[-2]:
            np.save(
                f"data/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/dataset-beta{betas[i+1]}",
                dataset,
            )
            break
        # save using the same name convention to repeat the training
        train_data, val_data = train_test_split(dataset, test_size=0.15)
        np.save(train_data_path, train_data)
        np.save(val_data_path, val_data)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
