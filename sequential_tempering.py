import argparse
import os
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.utils.montecarlo import neural_mcmc, single_spin_flip

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


def main(args):
    dataset = single_spin_flip(
        args.spins,
        args.beta_min,
        args.dataset_size,
        args.couplings_path,
        sweeps=100,
    )
    # save the first dataset
    np.save(
        f"data/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/single-beta{args.beta_min}"
    )
    train_data, val_data = train_test_split(dataset, test_size=0.15)
    # create directory and save data
    parent_path = f"data/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/"
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    train_data_path = parent_path + "train"
    val_data_path = parent_path + "val"
    np.save(train_data_path, train_data)
    np.save(val_data_path, val_data)

    ckpt_path = "null"
    betas = np.linspace(args.beta_min, args.beta_max, num=args.beta_num, endpoint=True)
    for i, beta in enumerate(betas):
        # train the neaural network for the first time
        os.system(
            f"python run.py name={beta} datamodule.datasets.train.name={args.couplings_path.split('/')[-1][:-4]} mode=seq_temp"
        )
        # generate using the trained network
        ckpt_path = f"logs/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/checkpoints/best-beta{beta}.ckpt"
        # smart montecarlo at next beta
        dataset = neural_mcmc(
            betas[i + 1], args.dataset_size, ckpt_path, args.couplings_path, args.model
        )
        # save the last montecarlo output
        if beta == betas[-2]:
            np.save(
                f"data/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/dataset-beta{betas[i+1]}",
                dataset,
            )
            break
        # save using the same name convention torepeat the training
        train_data, val_data = train_test_split(dataset, test_size=0.15)
        np.save(train_data_path, train_data)
        np.save(val_data_path, val_data)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
