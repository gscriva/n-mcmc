import numpy as np
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from src.datamodules.ising_datamodule import worker_init_fn
from src.models.made import Made
from src.models.pixel_cnn import PixelCNN


def generate(
    ckpt_path: str,
    model: str,
    num_sample: str = 1,
    batch_size: int = 20000,
    num_workers: int = 1,
    save_sample: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    # choose the model and load all the argumets
    if model == "pixel":
        model = PixelCNN.load_from_checkpoint(ckpt_path)
        shape = (num_sample, 1, model.hparams.input_size, model.hparams.input_size)
    elif model == "made":
        model = Made.load_from_checkpoint(ckpt_path)
        shape = (num_sample, model.hparams.input_size)

    if verbose:
        # print configs from trained model
        print(model.hparams)

    dataset = torch.rand(shape, device=model.device)

    # make it easy,
    # define only a DataLoader instead of a LightningDataModule
    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
    )

    # instantiate the trainer
    trainer = Trainer(devices="auto", accelerator="auto")

    print(f"\nGenerating {num_sample} sample")
    pred = trainer.predict(model=model, dataloaders=dataloader, ckpt_path=ckpt_path)

    size = model.hparams.input_size

    out = {"sample": pred[0]["sample"], "log_prob": pred[0]["log_prob"]}
    # create a unique dataset for mcmc
    for batch in pred[1:]:
        out["sample"] = np.append(out["sample"], batch["sample"], axis=0)
        out["log_prob"] = np.append(out["log_prob"], batch["log_prob"], axis=0)

    if save_sample:
        save_name = f"sample-{num_sample}_size-{size}_{ckpt_path.parts[-4]}_{ckpt_path.parts[-3]}"
        print("\nSaving sample generated as", save_name)
        np.savez(save_name, **out)

    return out
