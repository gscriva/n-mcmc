import torch
import torch.nn.functional as F
from torch import nn


class PixelBlock(nn.Module):
    def __init__(self, block) -> None:
        super(PixelBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, block) -> None:
        super(ResBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return x + self.block(x)


class FinalBlock(nn.Module):
    def __init__(self, block) -> None:
        super(FinalBlock, self).__init__()
        self.block = block

    def forward(self, x):
        return self.block(x)


class MaskedConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs) -> None:
        # remove mask_type kwargs
        self.mask_type = kwargs.pop("mask_type")
        super(MaskedConv2d, self).__init__(*args, **kwargs)

        _, _, kh, kw = self.weight.shape
        self.register_buffer("mask", torch.ones([kh, kw]))
        self.mask[kh // 2, kw // 2 + 1 :] = 0
        # type_mask=A excludes the central pixel
        if self.mask_type == "A":
            self.mask[kh // 2, kw // 2] = 0
        self.mask[kh // 2 + 1 :] = 0
        self.weight.data *= self.mask

        # Correction to Xavier initialization
        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return F.conv2d(
            x,
            self.mask * self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def extra_repr(self):
        return super(
            MaskedConv2d, self
        ).extra_repr() + ", mask_type={mask_type}".format(**self.__dict__)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, ker_size, activation) -> None:
        super(ConvBlock, self).__init__()
        self.conv_layer = nn.Sequential(
            GetActivation(activation), nn.Conv2d(in_channels, out_channels, ker_size)
        )

    def forward(self, x):
        return self.conv_layer(x)


class GetActivation(nn.Module):
    """Returns the requested activation function."""

    def __init__(self, activation) -> None:
        super(GetActivation, self).__init__()
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "prelu":
            self.activation = nn.PReLU(init=0.5)
        elif activation == "rrelu":
            self.activation = nn.RReLU()
        elif activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        elif activation == "gelu":
            self.activation = nn.GELU()
        elif activation == "selu":
            self.activation = nn.SELU()
        elif activation == "silu":
            self.activation = nn.SiLU()
        else:
            raise NotImplementedError(
                f"Activation function {activation} not implemented"
            )

    def forward(self, x):
        return self.activation(x)
