import logging
import os
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pytorch_lightning as pl
import rich.syntax
import rich.tree
from numba import jit
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor, set_num_threads
from torch.nn import BCEWithLogitsLoss

from src.utils.adjacency import Adjacency


def get_logger(name=__name__, level=logging.INFO) -> logging.Logger:
    """Initializes multi-GPU-friendly python logger."""

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger


def extras(config: DictConfig) -> None:
    """A couple of optional utilities, controlled by main config file:
    - disabling warnings
    - easier access to debug mode
    - forcing debug friendly configuration

    Modifies DictConfig in place.

    Args:
        config (DictConfig): Configuration composed by Hydra.
    """

    log = get_logger()

    # enable adding new keys to config
    OmegaConf.set_struct(config, False)

    # disable python warnings if <config.ignore_warnings=True>
    if config.get("ignore_warnings"):
        log.info("Disabling python warnings! <config.ignore_warnings=True>")
        warnings.filterwarnings("ignore")

    # set <config.trainer.fast_dev_run=True> if <config.debug=True>
    if config.get("debug"):
        log.info("Running in debug mode! <config.debug=True>")

    # if <config.name=...>
    if config.get("name"):
        log.info("Running in experiment mode! Name: {}".format(config.name))

    # force debugger friendly configuration if <config.trainer.fast_dev_run=True>
    if config.trainer.get("fast_dev_run"):
        log.info(
            "Forcing debugger friendly configuration! <config.trainer.fast_dev_run=True>"
        )
        # Debuggers don't like GPUs or multiprocessing
        if config.trainer.get("gpus"):
            config.trainer.gpus = 0
        if config.datamodule.get("pin_memory"):
            config.datamodule.pin_memory = False
        if config.datamodule.get("num_workers"):
            config.datamodule.num_workers = 0

    # disable adding new keys to config
    OmegaConf.set_struct(config, True)


@rank_zero_only
def print_config(
    config: DictConfig,
    fields: Sequence[str] = (
        "trainer",
        "model",
        "datamodule",
        "callbacks",
        "test_after_training",
        "logger",
        "seed",
        "name",
    ),
    resolve: bool = True,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.

    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = rich.tree.Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(rich.syntax.Syntax(branch_content, "yaml"))

    rich.print(tree)

    with open("config_tree.txt", "w") as fp:
        rich.print(tree, file=fp)


def empty(*args, **kwargs):
    pass


def set_num_cpus() -> None:
    cpus = os.cpu_count()
    cpus_to_use = min(cpus - 1, 12)
    set_num_threads(cpus_to_use)


@rank_zero_only
def log_hyperparameters(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """This method controls which parameters from Hydra config are saved by Lightning loggers.

    Additionaly saves:
        - number of trainable model parameters
    """

    hparams = {}

    # choose which parts of hydra config will be saved to loggers
    hparams["trainer"] = config["trainer"]
    hparams["model"] = config["model"]
    hparams["datamodule"] = config["datamodule"]
    if "seed" in config:
        hparams["seed"] = config["seed"]
    if "callbacks" in config:
        hparams["callbacks"] = config["callbacks"]

    # save number of model parameters
    hparams["model/params_total"] = sum(p.numel() for p in model.parameters())
    hparams["model/params_trainable"] = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    hparams["model/params_not_trainable"] = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )

    # send hparams to all loggers
    trainer.logger.log_hyperparams(hparams)

    # disable logging any more hyperparameters for all loggers
    # this is just a trick to prevent trainer from logging hparams of model,
    # since we already did that above
    trainer.logger.log_hyperparams = empty


def compute_prob(logits: Tensor, x: Tensor) -> Tensor:
    """Compute the probability of a sample using a PyTorch routine.

    Args:
        logits (Tensor): Logits as computed by the model.
        x (Tensor): Samples in binary form.

    Returns:
        float: Log probabilities of the samples.
    """
    # BCEWithLogitsLoss with reduction='none' is nothing than
    # the positive log-likelihood of the sample
    criterion = BCEWithLogitsLoss(reduction="none")
    log_prob = -criterion(logits, x)
    return log_prob.sum(dim=-1)


def finish(
    config: DictConfig,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
    trainer: pl.Trainer,
    callbacks: List[pl.Callback],
    logger: List[pl.loggers.LightningLoggerBase],
) -> None:
    """Makes sure everything closed properly."""

    # without this sweeps with wandb logger might crash!
    for lg in logger:
        if isinstance(lg, pl.loggers.wandb.WandbLogger):
            import wandb

            wandb.finish()


@jit(nopython=True)
def compute_energy(
    sample: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    len_neighbours: int,
) -> float:
    energy = 0
    for i in range(neighbours.shape[0]):
        for j in range(len_neighbours[i]):
            energy += sample[i] * (sample[neighbours[i, j]] * couplings[i, j])
    return energy / 2


@jit(nopython=True)
def compute_delta_h(
    num_spin: int,
    sample: np.ndarray,
    neighbours: np.ndarray,
    couplings: np.ndarray,
    len_neighbours: int,
) -> float:
    delta_h = 0.0
    for j in range(len_neighbours):
        delta_h += -sample[num_spin] * (sample[neighbours[j]] * couplings[j])
    return 2 * delta_h


def load_data(
    sample_path: Union[str, Dict[str, np.ndarray]],
    model: Optional[str] = None,
    steps: Optional[int] = None,
    batch_size: int = 20000,
    verbose: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load generated sample from path or directly from the file.

    Args:
        sample_path (Union[str, Dict[str, np.ndarray]]): Path to the generated sample, to the model or to the samples theirself.
        model (Optional[str], optional): Model to use. Defaults to None.
        steps (Optional[int], optional): Steps of the Monte Carlo simulation. Defaults to None.
        batch_size (int, optional): Size of each batch. Defaults to 20000.
        verbose (bool, optional): Set verbose prints. Defaults to False.

    Raises:
        ValueError: Wrong path or corrupted data.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sample and their log probability.
    """
    if isinstance(sample_path, str):
        if sample_path.split(".")[-1] == "npz":
            data = np.load(sample_path)
        elif sample_path.split(".")[-1] == "ckpt":
            # import here to avoid circular imports
            from src.generate import generate

            data = generate(
                sample_path, model, steps, batch_size=batch_size, verbose=verbose
            )
    elif isinstance(sample_path, Dict):
        data = sample_path
    else:
        raise ValueError("Neither a path to a model, a npz file or a Numpy dataset!")
    return (
        data["sample"],
        data["log_prob"],
    )


def get_couplings(spin_side: int, couplings_path: str) -> Tuple[Any]:
    adjacency = Adjacency(spin_side)
    adjacency.loadtxt(couplings_path)
    # get neighbourhood and couplings matrix
    neighbours, couplings = adjacency.get_neighbours()
    len_neighbours = np.sum(couplings != 0, axis=-1)
    return neighbours.astype(int), couplings, len_neighbours


@jit(nopython=True)
def compute_boltz_prob(eng: float, beta: float, num_spin: int) -> float:
    """Boltzmann probability distribution

    Args:
        eng (float): Energy of the sample.
        beta (float): Inverse temperature
        num_spin (int): Number of spins in the sample.

    Returns:
        float: Log-Boltzmann probability.
    """
    return -beta * eng


def block_single_std(engs: np.ndarray, len_block: int, skip: int = 0) -> float:
    """Compute the block std of a list of arrays.
    See http://chimera.roma1.infn.it/SP/doc/estratti/dataAnalysis.pdf

    Args:
        engs (np.ndarray): List of energies arrays or array.
        len_block (int): Block lenght to compute std.
        skip (int, optional): Number of initial sample to skip. Defaults to 0.

    Returns:
        float: Block std.
    """
    std_engs = []
    for i in range(engs.shape[0]):
        eng_seed = engs[i]
        error = []
        for eng in eng_seed:
            if isinstance(eng, list):
                eng = np.asarray(eng)
            eng = eng[skip:].copy()
            rest_len = eng.size % len_block
            if rest_len != 0:
                eng = eng[:-rest_len]
            eng = eng.reshape(-1, len_block)
            new_len = eng.shape[0]
            # if new_len < 100:
            #    print(f"WARNING: New Len={new_len} < 100")
            error.append(np.std(eng.mean(axis=1), ddof=0) / np.sqrt(new_len - 1))
        std_engs.append(error)
    return std_engs


def block_std(engs: List[Any], len_block: int, skip: int = 0) -> float:
    """Compute the block std of a list of arrays.
    See http://chimera.roma1.infn.it/SP/doc/estratti/dataAnalysis.pdf

    Args:
        engs (List[Any]): List of energies arrays.
        len_block (int): Block lenght to compute std.
        skip (int, optional): Number of initial sample to skip. Defaults to 0.

    Returns:
        float: Block std.
    """
    std_engs = []
    for eng in engs:
        if isinstance(eng, list):
            eng = np.asarray(eng)
        eng = eng[skip:].copy()
        rest_len = eng.size % len_block
        if rest_len != 0:
            eng = eng[:-rest_len]
        eng = eng.reshape(-1, len_block)
        new_len = eng.shape[0]
        if new_len < 100:
            pass  # print(f"WARNING: New Len {new_len} < 100")
        error = np.std(eng.mean(axis=1), ddof=0) / np.sqrt(new_len - 1)
        std_engs.append(error)
    return std_engs


def block_mean(engs: List[np.ndarray], len_block: int, skip: int = 0) -> float:
    """Compute the block mean of a list of arrays.
    See http://chimera.roma1.infn.it/SP/doc/estratti/dataAnalysis.pdf

    Args:
        engs (List[np.ndarray]): List of the energy arrays.
        len_block (int): Block lenght to compute mean.
        skip (int, optional): Number of initial sample to skip. Defaults to 0.

    Returns:
        float: Block mean.
    """
    mean_engs = []
    for eng in engs:
        if isinstance(eng, list):
            eng = np.asarray(eng)
        eng = eng[skip:].copy()
        rest_len = eng.size % len_block
        if rest_len != 0:
            eng = eng[:-rest_len]
        eng = eng.reshape(len_block, -1)
        mean_engs.append(np.mean(eng.mean(axis=0)))
    return mean_engs


def get_energy(
    square_spin: int, paths: List[str], couplings_path: str
) -> List[np.array]:
    """Returns the energy of a list of dataset, according to a given coupling.

    Args:
        square_spin (int): Square root of spins (assuming square lattice).
        paths (List[str]): List of paths of .npy or .npz files.
        couplings_path (str): Path to .txt coupling file.

    Returns:
        List[np.array]: List of the sample configuration.
    """
    neighbours, couplings, len_neighbours = get_couplings(square_spin, couplings_path)

    engs = []
    for path in paths:
        out = np.load(path)

        try:
            sample = out["sample"]
        except:
            sample = out

        sample = sample.squeeze()
        sample = np.reshape(sample, (-1, square_spin ** 2))

        eng = []
        for s in sample:
            eng.append(compute_energy(s, neighbours, couplings, len_neighbours))
        eng = np.asarray(eng) / square_spin ** 2

        engs.append(eng)
    return engs
