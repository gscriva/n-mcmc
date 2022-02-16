import argparse
import math
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
parser.add_argument(
    "--random",
    dest="random",
    action="store_true",
    help="Use anintila random sampling for MCMC",
)


def main(args):
    # set unique dir name
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d_%H-%M-%S")

    dataset = single_spin_flip(
        args.spins,
        args.beta_min,
        math.floor(args.dataset_size * 0.7) if args.random else args.dataset_size,
        args.couplings_path,
        sweeps=1000,
        disable_bar=True,
    )

    if args.random:
        add_dataset = (
            np.random.randint(
                0, 2, size=(math.ceil(args.dataset_size * 0.3), args.spins)
            )
            * 2.0
            - 1.0
        )
        dataset = np.append(
            dataset,
            add_dataset,
            axis=0,
        )
        add_dataset = add_dataset[: math.floor(add_dataset.shape[0] / args.beta_num), :]

    # create directory and save mcmc data
    parent_path = (
        f"data/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/{date_time}/"
    )
    Path(parent_path).mkdir(parents=True, exist_ok=True)
    np.save(
        parent_path + f"single-beta{args.beta_min}",
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
    print(f"Beta: {betas}")
    for i, beta in enumerate(betas):
        # train the neaural network
        os.system(
            f"python run.py name={beta} datamodule.datasets.train.name={args.couplings_path.split('/')[-1][:-4]}/{date_time} callbacks.model_checkpoint.dirpath=checkpoints mode=seq_temp"
        )
        # generate using the trained network
        ckpt_path = f"logs/seq_temp/{args.couplings_path.split('/')[-1][:-4]}/{date_time}/checkpoints/best-beta{beta}.ckpt"
        # store some sample from the previous dataset
        if args.random:
            rand_idx = np.random.choice(
                args.dataset_size,
                size=math.floor(args.dataset_size * 0.3 / args.beta_num),
                replace=False,
            )
            add_dataset = np.append(add_dataset, dataset[rand_idx, :], axis=0)

        # smart or hybrid montecarlo at the next beta
        if args.hybrid:
            dataset, _, _ = hybrid_mcmc(
                betas[i + 1],
                args.dataset_size + 1,
                ckpt_path,
                args.couplings_path,
                args.model,
                disable_bar=True,
            )
        else:
            dataset, _, _ = neural_mcmc(
                betas[i + 1],
                args.dataset_size,
                ckpt_path,
                args.couplings_path,
                args.model,
                disable_bar=True,
            )
        # save the last montecarlo output
        if beta == betas[-2]:
            np.save(
                parent_path + f"dataset-beta{betas[i+1]}",
                dataset,
            )
            break

        if args.random:
            dataset = np.append(
                dataset[: args.dataset_size - add_dataset.shape[0], :],
                add_dataset,
                axis=0,
            )
        # save using the same name convention to repeat the training
        train_data, val_data = train_test_split(dataset, test_size=0.15)
        np.save(train_data_path, train_data)
        np.save(val_data_path, val_data)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
