import torch
import numpy as np
from torchmetrics import Metric

from src.utils.utils import get_couplings, compute_energy


class MeanMAE(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.spins = 100
        self.couplings_path = "data/couplings/100spins_open-1nn.txt"

        self.neighbours, self.couplings, self.len_neighbours = get_couplings(
            spin_side=int(np.sqrt(self.spins)),
            couplings_path=self.couplings_path,
        )

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        preds = preds.detach().cpu().numpy() * 2 - 1
        targets = targets.detach().cpu().numpy() * 2 - 1

        pred_engs = np.apply_along_axis(
            compute_energy,
            1,
            preds,
            neighbours=self.neighbours,
            couplings=self.couplings,
            len_neighbours=self.len_neighbours,
        )
        target_engs = np.apply_along_axis(
            compute_energy,
            1,
            targets,
            neighbours=self.neighbours,
            couplings=self.couplings,
            len_neighbours=self.len_neighbours,
        )
        pred_engs = torch.tensor(pred_engs, device=self.device) / self.spins
        target_engs = torch.tensor(target_engs, device=self.device) / self.spins

        self.correct += torch.sum(torch.abs(pred_engs - target_engs))

        self.total += pred_engs.numel()

    def compute(self):
        return self.correct.float() / self.total
