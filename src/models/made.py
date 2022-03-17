from math import sqrt
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
from pytorch_lightning import LightningModule
from torch import nn
from torch.functional import Tensor
from torch.optim import Optimizer
from tqdm import trange

from src.models.modules.made_block import MadeModel
from src.utils.utils import compute_prob


class Made(LightningModule):
    def __init__(self, *args, **kwargs):
        super(Made, self).__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        # instantiate the model
        self.model = MadeModel(self.hparams)
        # loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        logits = self.model(x)

        return compute_prob(logits, x)

    def step(self, x: Tensor):
        logits = self.model(x)

        loss = self.criterion(logits, x)

        return loss, logits

    def training_step(self, x: Tensor, batch_idx: int):
        loss, _ = self.step(x)
        # Connectivity agnostic and order agnostic
        if (batch_idx + 1) % self.hparams.resample_every == 0:
            self.model.update_masks(self.hparams)

        # log train metrics
        self.log("train/loss", loss)

        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, x: Tensor, batch_idx: int):
        loss, _ = self.step(x)

        # update the mask for every epoch
        self.model.update_masks(self.hparams)

        # log val metrics
        self.log("val/loss", loss, prog_bar=True)

        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, x: Tensor, batch_idx: int):
        loss, _ = self.step(x)

        # log test metrics
        self.log("test/loss", loss)

        return loss

    def test_epoch_end(self, outputs: List[Any]):
        pass

    @torch.no_grad()
    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, np.ndarray]:
        for spin in range(self.hparams.input_size):
            logits = self.model(batch)
            # generate x_hat according to the compute probability
            batch[:, spin] = torch.bernoulli(torch.sigmoid(logits[:, spin]))

        # compute the robability of the sample
        log_prob = compute_prob(logits, batch).detach().cpu().numpy()

        input_side = int(sqrt(self.hparams.input_size))
        # output should be {-1,+1}, spin convention
        # and for dwave data must be fortran contiguous
        batch = batch.detach().cpu().numpy().astype("int8")
        batch = np.reshape(batch, (-1, input_side, input_side)) * 2 - 1
        return {
            "sample": batch,
            "log_prob": log_prob,
        }

    def configure_optimizers(
        self,
    ) -> Tuple[List[Optimizer], List[Any]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Return:
            Tuple: The first element is the optimizer, the second element the LR scheduler.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer,
            params=self.parameters(),
            _convert_="partial",
        )
        if not self.hparams.optim.use_lr_scheduler:
            return opt
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]
