import torch
import torch.nn.functional as F
from torch import nn


class MaskedLinear(nn.Linear):
    """A Linear Layer masked to keep the autoregressive property."""

    def __init__(self, input_size, output_size, bias=True):
        super().__init__(input_size, output_size, bias)
        self.register_buffer("mask", torch.ones(output_size, input_size))

    def set_mask(self, mask):
        self.mask.data = mask.detach()

    def forward(self, input):
        self.weight.data *= self.mask
        return F.linear(input, self.weight, self.bias)


class MadeModel(nn.Module):
    def __init__(self, hparams: dict):
        super().__init__()

        self.model = nn.ModuleList()
        activation = getattr(nn, hparams["activation"])
        activation = activation()

        # crate the hidden size list
        hiddens = [hparams["input_size"]] + [
            hparams["hidd_neurons"] for i in range(hparams["hidd_layers"])
        ]
        # create hidden masked layers
        for size in zip(hiddens, hiddens[1:]):
            self.model.append(MaskedLinear(size[0], size[1]))
            self.model.append(activation)

        # last masked layer has no activation
        self.model.append(MaskedLinear(hparams["hidd_neurons"], hparams["input_size"]))
        # ModuleList has no forward method!
        self.model = nn.Sequential(*self.model)

        self.m = {}
        self.seed = 0
        self.update_masks(hparams)

    def update_masks(self, hparams: dict):
        if self.m and hparams["num_masks"] == 1:
            return  # mask already created

        # create a generator of pseudo-random number
        # with a controlled seed
        generator = torch.Generator()
        generator.manual_seed(hparams["mask_seed"] + self.seed)
        # use as many seed as needed
        # and recreate each time masks to save memory
        self.seed = (self.seed + 1) % hparams["num_masks"]

        # use the input's natural order or permute the input state
        if hparams["natural_ordering"]:
            self.m[-1] = torch.arange(hparams["input_size"], dtype=torch.int)
        else:
            self.m[-1] = torch.randperm(hparams["input_size"], dtype=torch.int)

        # create mask for each layer
        # to avoid unconnected units we take the minimum of the prevoius layer
        for l in range(hparams["hidd_layers"]):
            self.m[l] = torch.randint(
                self.m[l - 1].min().item(),
                hparams["input_size"] - 1,
                (hparams["hidd_neurons"],),
                dtype=torch.int,
            )

        # construct the mask matrices for hidden layers
        masks = [
            self.m[l][:, None] >= self.m[l - 1][None, :]
            for l in range(hparams["hidd_layers"])
        ]
        # construct the mask for the last layer
        masks.append(self.m[-1][:, None] > self.m[hparams["hidd_layers"] - 1][None, :])

        # set the masks in all MaskedLinear layers
        layers = [l for l in self.model.modules() if isinstance(l, MaskedLinear)]
        for l, m in zip(layers, masks):
            l.set_mask(m)

    def forward(self, x):
        return self.model(x)
