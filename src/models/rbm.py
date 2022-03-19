from math import sqrt
from typing import Any, Dict, List, Tuple

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn
from torch.functional import Tensor
from torch.optim import Optimizer


class RBM(LightningModule):
    def __init__(self, *args, **kwargs):
        super(RBM, self).__init__()

        # this line ensures params passed to LightningModule will be saved to ckpt
        # it also allows to access params with 'self.hparams' attribute
        self.save_hyperparameters()
        # instantiate the model
        # see for formalism https://christian-igel.github.io/paper/TRBMAI.pdf
        self.W = nn.Parameter(
            torch.randn(self.hparams["n_hidden"], self.hparams["input_size"])
        )
        self.b = nn.Parameter(torch.randn(self.hparams["input_size"]))
        self.c = nn.Parameter(torch.randn(self.hparams["n_hidden"]))
        # loss function
        self.criterion = self._free_energy_loss

    def _free_energy_loss(self, x: Tensor, pred: Tensor) -> Tensor:
        return self._free_energy(x) - self._free_energy(pred)

    def _free_energy(self, x: Tensor) -> Tensor:
        bias_term = torch.matmul(x.unsqueeze(0), self.b)
        wx = F.linear(x, self.W, self.c)
        hidd_term = (wx.exp() + 1).log().sum(1)
        return (-bias_term - hidd_term).mean()

    def _to_hidden(self, x: Tensor) -> Tensor:
        prob = torch.sigmoid(F.linear(x, self.W, self.c))
        return prob.bernoulli()

    def _to_visible(self, h: Tensor) -> Tuple[Tensor, Tensor]:
        prob = torch.sigmoid(F.linear(h, self.W.t(), self.b))
        return prob.bernoulli(), prob

    def step(self, x: Tensor) -> Tensor:
        h = self._to_hidden(x)
        for _ in range(self.hparams["k"]):
            x_gibbs, x_prob = self._to_visible(h)
            h = self._to_hidden(x_gibbs)
        return x_gibbs, x_prob

    def training_step(self, x: Tensor, batch_idx: int) -> Tensor:
        x_gibbs, _ = self.step(x)
        loss = self.criterion(x, x_gibbs)
        # log the metric
        self.log("train/loss", loss)

        return loss

    def validation_step(self, x: Tensor, batch_idx: int) -> Tensor:
        x_gibbs, _ = self.step(x)
        loss = self.criterion(x, x_gibbs)
        # log the metric
        self.log("val/loss", loss)

        return loss

    def test_step(self, x: Tensor, batch_idx: int) -> Tensor:
        x_gibbs, _ = self.step(x)
        loss = self.criterion(x, x_gibbs)
        # log the metric
        self.log("test/loss", loss)

        return loss

    @torch.no_grad()
    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, np.ndarray]:
        sample, prob = self.step(batch)

        input_side = int(sqrt(self.hparams["input_size"]))

        sample = sample.detach().cpu().numpy().astype("int8")
        sample = np.reshape(sample, (-1, input_side, input_side)) * 2 - 1
        log_prob = np.log(prob.detach().cpu().numpy())

        return {"sample": sample, "log_prob": log_prob}

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
