import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from src.models.pixel_cnn import PixelCNN
from src.models.made import Made
from src.datamodules.ising_datamodule import worker_init_fn
from src.utils.smart_montecarlo import mcmc

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=Path, help="Path to the checkpoint")
parser.add_argument("--model", type=str, help="Model to generate")
parser.add_argument(
    "--num-sample",
    type=int,
    default=2,
    help="Number of sample to generate (default: 2)",
)
parser.add_argument(
    "--beta", type=float, default=1.0, help="Inverse temperature (default: 1.0)"
)
parser.add_argument(
    "--batch-size",
    type=int,
    default=20000,
    help="Dimension of the single generated sample (default: 20000)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=2,
    help="Number of workers to generate (default: 2)",
)
parser.add_argument(
    "--save-sample",
    dest="save_sample",
    action="store_true",
    help="Flag if you want to save samples after generation",
)
parser.add_argument(
    "--save-mcmc",
    dest="save_mcmc",
    action="store_true",
    help="Flag if you want to save samples after MCMC",
)
parser.add_argument(
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Flag if you want to see prints in MCMC",
)


def generate(args: argparse.ArgumentParser):
    # choose the model
    model = PixelCNN if args.model == "pixelcnn" else Made

    # retrive configs from trained model
    trained_model = model.load_from_checkpoint(args.ckpt_path)
    print(trained_model.hparams)
    dataset = torch.zeros(
        [
            args.num_sample,
            trained_model.hparams.input_size,
        ],
        device=trained_model.device,
    )

    # make it easy,
    # define only a DataLoader instead of a LightningDataModule
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        worker_init_fn=worker_init_fn,
    )

    print(f"\nGenerating sample")

    trainer = Trainer()
    pred = trainer.predict(
        model=trained_model, dataloaders=dataloader, ckpt_path=args.ckpt_path
    )

    save_path = "."
    size = trained_model.hparams.input_size

    out = {"sample": pred[0]["sample"], "log_prob": pred[0]["log_prob"]}

    # create a unique dataset for mcmc
    for batch in pred[0:]:
        out["sample"] = np.append(out["sample"], batch["sample"], axis=0)
        out["log_prob"] = np.append(out["log_prob"], batch["log_prob"], axis=0)

    if args.save_sample:
        save_name = (
            "size-"
            + str(size)
            + "_sample-"
            + str(args.num_sample)
            + "_"
            + args.ckpt_path.parts[-3]
        )
        print("\nSaving sample generated by PixelCNN as", save_name)
        np.savez(save_path + save_name, **out)

    num_sample = out["sample"].shape[0]
    mcmc(args.beta, args.num_sample - 1, out, verbose=args.verbose, save=args.save_mcmc)


if __name__ == "__main__":
    args = parser.parse_args()
    generate(args)


# @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
# def main(cfg: omegaconf.DictConfig):
#     model: pl.LightningModule = hydra.utils.instantiate(
#         cfg.model,
#         physics=cfg.physics,
#         optim=cfg.optim,
#         data=cfg.data,
#         logging=cfg.logging,
#         _recursive_=False,
#     )


# if __name__ == "__main__":
#    main()
