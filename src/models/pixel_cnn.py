from typing import Any, Dict, List, Sequence, Tuple

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.optim import Optimizer
from tqdm import trange

from src.models.modules.pixel_blocks import (
    ConvBlock,
    FinalBlock,
    GetActivation,
    MaskedConv2d,
    PixelBlock,
    ResBlock,
)

# from src.common.utils import PROJECT_ROOT


class PixelCNN(pl.LightningModule):
    def __init__(self, *args, **kwargs):
        super().__init__()

        self.save_hyperparameters()  # populate self.hparams with args and kwargs automagically!

        # Force the first x_hat to be log(0.5)
        if self.hparams.bias:
            self.register_buffer(
                "x_hat_mask", torch.ones([self.hparams.input_size] * 2)
            )
            self.x_hat_mask[0, 0] = 0
            self.register_buffer(
                "x_hat_bias", torch.zeros([self.hparams.input_size] * 2)
            )
            self.x_hat_bias[0, 0] = torch.log(torch.Tensor([0.5]))

        layers = nn.ModuleList()
        layers.append(
            MaskedConv2d(
                1,
                1 if self.hparams.net_depth == 1 else self.hparams.net_width,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                mask_type="A",
            )
        )

        layers.extend(
            [
                self._build_pixel_block(
                    self.hparams.net_width,
                    self.hparams.net_width,
                    self.hparams.res_block,
                )
                for _ in range(self.hparams.net_depth - 2)
            ]
        )

        if self.hparams.net_depth >= 2:
            layers.append(
                self._build_simple_block(
                    self.hparams.net_width,
                    self.hparams.net_width if self.hparams.final_conv else 2,
                )
            )

        if self.hparams.final_conv:
            layers.append(self._build_final_block(self.hparams.net_width))

        layers.append(nn.LogSoftmax(dim=1))
        self.net = nn.Sequential(*layers)

    def _build_pixel_block(
        self, in_channels: int, out_channels: int, res_block: bool
    ) -> nn.Module:
        """Returns a Residual or a Simple Pixel Block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.
            res_block (bool): Set True if the block is residual.

        Returns:
            nn.Module: Pixel Block.
        """
        if res_block:
            pixel_block = self._build_res_block(in_channels, out_channels)
        else:
            pixel_block = self._build_simple_block(in_channels, out_channels)
        return pixel_block

    def _build_simple_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build a masked convolutional block.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.

        Returns:
            nn.Module: Convolutional Pixel Block.
        """
        layers = nn.ModuleList()
        layers.append(GetActivation(self.hparams.activation))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                mask_type="B",
            )
        )
        return PixelBlock(nn.Sequential(*layers))

    def _build_res_block(self, in_channels: int, out_channels: int) -> nn.Module:
        """Build a convolutional residual block, with a simple conv2d,
        an activation function and a masked convolutional layer.

        Args:
            in_channels (int): Input channels.
            out_channels (int): Output channels.

        Returns:
            nn.Module: Residual Pixel block.
        """
        layers = nn.ModuleList()
        layers.append(nn.Conv2d(in_channels, in_channels, 1, bias=self.hparams.bias))
        layers.append(GetActivation(self.hparams.activation))
        layers.append(
            MaskedConv2d(
                in_channels,
                out_channels,
                self.hparams.kernel_size,
                padding=(self.hparams.kernel_size - 1) // 2,
                bias=self.hparams.bias,
                mask_type="B",
            )
        )
        return ResBlock(nn.Sequential(*layers))

    def _build_final_block(self, in_channels: int) -> nn.Module:
        """Build a final convolutional block with a conv2d and an activation function.

        Args:
            in_channels (int): Input channels.

        Returns:
            nn.Module: Final Convolutional Block.
        """
        layers = nn.ModuleList()
        layers.append(
            ConvBlock(
                in_channels,
                in_channels,
                1,
                self.hparams.activation,
            )
        )
        layers.append(ConvBlock(in_channels, 2, 1, self.hparams.activation))
        return FinalBlock(nn.Sequential(*layers))

    def log_prob(self, sample: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """Method to compute the logarithm of the probabilty of sample
            and its conditional probability.

        Args:
            sample (torch.Tensor): Ising Glass sample.
            x_hat (torch.Tensor): Conditional probability of each spin in the sample.

        Returns:
            torch.Tensor: Logarithm of the probabilty of the sample.
        """
        mask = torch.cat(((-sample + 1) / 2, (sample + 1) / 2), dim=1)
        log_prob = x_hat * mask
        log_prob = log_prob.view(log_prob.shape[0], -1).sum(dim=1)
        return log_prob

    def forward(self, x) -> torch.Tensor:
        x_hat = self.net(x)
        # Force the first x_hat to be 0.5
        if self.hparams.bias:
            x_hat = x_hat * self.x_hat_mask + self.x_hat_bias
        return x_hat

    def step(self, x) -> torch.Tensor:
        """Method for the forward pass (training).
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the loss.

        Returns:
            torch.Tensor: prediction.
        """
        x_hat = self.net(x)
        # compute negative log likelihood
        criterion = nn.NLLLoss(reduction="mean")
        # x = (x.squeeze().long() + 1) // 2
        x = x.squeeze().long()
        loss = criterion(x_hat, x)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log_dict(
            {"train/loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log_dict(
            {"val/loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_epoch_end(self, outputs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        loss = self.step(batch)
        self.log_dict(
            {"test/loss": loss},
        )
        return loss

    def test_epoch_end(self, outputs: List[Any]):
        pass

    @torch.no_grad()
    def predict_step(
        self, batch, batch_idx: int, dataloader_idx: int = None
    ) -> Dict[str, np.ndarray]:
        for i in trange(self.hparams.input_size, leave=False):
            for j in trange(self.hparams.input_size, leave=False):
                x_hat = self.forward(batch).detach()
                batch[:, :, i, j] = torch.bernoulli(
                    torch.exp(x_hat[:, 1, i, j]).unsqueeze(1)
                )

        # compute probability of the sample
        batch = batch * 2 - 1
        log_prob = self.log_prob(batch, x_hat).detach().cpu().numpy()

        # output should be {-1,+1}, spin convention
        # and for dwave data must be fortran contiguous
        batch = batch.detach().cpu().numpy().astype("int8")
        batch = np.reshape(
            batch, (-1, self.hparams.input_size, self.hparams.input_size)
        )
        return {
            "sample": batch,
            "log_prob": log_prob,
        }

    def configure_optimizers(
        self,
    ) -> Tuple[Sequence[Optimizer], Sequence[Any]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Return:
            Tuple: The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
                    May be present only one optimizer and one scheduler.
        """
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return [opt], [scheduler]
