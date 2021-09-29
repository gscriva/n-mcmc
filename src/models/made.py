from typing import Any, List, Tuple, Dict
from math import sqrt

import torch
import hydra
from pytorch_lightning import LightningModule
from torch import nn
from numpy import ndarray
from torch._C import device
from torch.functional import Tensor
from torch.optim import Optimizer
from tqdm import trange

from src.models.modules.made_block import MadeModel
from src.utils.utils import compute_prob


class Made(LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()

        self.model = MadeModel(self.hparams)
        # loss function
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def sample(self, num_sample: int) -> Dict[ndarray, ndarray]:
        # initialize the samples
        x = torch.zeros(
            (num_sample, self.hparams.input_size),
            device=self.device,
        )

        prog_bar = trange(self.hparams.input_size, leave=True)
        for d in prog_bar:
            logits = self.forward(x)
            # generate x_hat according to the compute probability
            x[:, d] = torch.bernoulli(torch.sigmoid(logits[:, d])).detach()
        # compute the robability of the sample
        log_prob = compute_prob(logits, x)

        input_side = int(sqrt(self.hparams.input_size))
        # output should be {-1,+1}, spin convention
        x = x.view((-1, input_side, input_side)) * 2 - 1
        return {"sample": x.detach().numpy(), "log_prob": log_prob.detach().numpy()}

    def step(self, x: Tensor):
        logits = self.forward(x)

        loss = self.criterion(logits, x)

        return loss, logits

    def training_step(self, x: Tensor, batch_idx: int):
        loss, logits = self.step(x)
        # Connectivity agnostic and order agnostic
        if (batch_idx + 1) % self.hparams.resample_every == 0:
            self.model.update_masks(self.hparams)

        # log train metrics
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)

        return {"loss": loss, "logits": logits.detach(), "targets": x}

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, x: Tensor, batch_idx: int):
        loss, logits = self.step(x)

        # update the mask for every epoch
        self.model.update_masks(self.hparams)

        # log val metrics
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "logits": logits.detach(), "targets": x}

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, x: Tensor, batch_idx: int):
        loss, logits = self.step(x)

        # log test metrics
        self.log("test/loss", loss, on_step=False, on_epoch=True)

        return {"loss": loss, "logits": logits, "targets": x}

    def test_epoch_end(self, outputs: List[Any]):
        pass

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
