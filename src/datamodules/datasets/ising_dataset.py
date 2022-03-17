from typing import Dict, Tuple, Union

import hydra
import numpy as np
import omegaconf
import pytorch_lightning as pl
import torch
from omegaconf import ValueNode
from torch.utils.data import Dataset, TensorDataset


class ISINGDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode, **kwargs):
        super().__init__()
        self.path = path
        self.name = name

        # load the dataset and consider input in {0,1}
        self.dataset = (torch.from_numpy(np.load(self.path)).float() + 1) / 2
        # some models need ravel inputs
        if kwargs["model"] == "src.models.pixel_cnn.PixelCNN":
            self.dataset = self.dataset.view(
                len(self.dataset), 1, kwargs["input_size"], kwargs["input_size"]
            )
        else:
            self.dataset = self.dataset.view(len(self.dataset), -1)

        # check if data match with model's input size
        assert self.dataset.shape[-1] == kwargs["input_size"]

        self.dataset = TensorDataset(self.dataset)

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index):
        # [0] is needed so only one element is returned
        return self.dataset[index][0]

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, path={self.path})"


@hydra.main(config_path="configs", config_name="config")
def main(cfg: omegaconf.DictConfig):
    dataset: ISINGDataset = hydra.utils.instantiate(
        cfg.data.ising_data.datamodule.datasets.train, _recursive_=False
    )


if __name__ == "__main__":
    main()
