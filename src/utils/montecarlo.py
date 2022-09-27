from cmath import exp
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch
from tqdm import tqdm

from src.models.made import Made
from src.utils.utils import (
    compute_boltz_prob,
    compute_delta_h,
    compute_energy,
    get_couplings,
    load_data,
)


def single_spin_flip(
    spins: int,
    beta: float,
    steps: int,
    couplings_path: str,
    sweeps: int = 0,
    burnt: int = 0,
    seed: int = 42,
    verbose: bool = False,
    disable_bar: bool = False,
    save: bool = False,
    save_dir: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """The Single Spin Flip algorithm exploit a Markov Chain to explore the energy landscape
     of a given hamiltonian at a specified temperature.

    Args:
        spins (int): Number of spins of the ravel spin glass.
        beta (float): Inverse temperature.
        steps (int): Steps of the Monte Carlo simulation.
        couplings_path (str): Path to the couplings, they define a Hamiltonian.
        sweeps (int, optional): Number of attemps to flip each spin before save. Defaults to 0.
        burnt (int, optional): Number of steps to skip before starting to save. Default to 0.
        seed (int, optional): Seed to sample the starting point configuration. Defaults to 42.
        verbose (bool, optional): Set verbose prints. Defaults to False.
        disable_bar (bool, optional): Set true to disable the bar. Defaults to False.
        save (bool, optional): Save the samples after MCMC. Defaults to False.
        save_dir (str, optional): Path to save, if None save in the working dir. Default to None.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Sample, energy and acceptance rate.
    """
    start_time = datetime.now()
    print(f"\nStart MCMC simulation {start_time}\nbeta={beta} seed={seed}")

    if save_dir is not None:
        if not Path(save_dir).is_dir():
            print(f"'{save_dir}' not found")
            raise FileNotFoundError

    # get neighbourhood matrix
    spin_side = int(math.sqrt(spins))
    neighbours, couplings, len_neighbours = get_couplings(spin_side, couplings_path)

    # initialize starting point
    np.random.seed(seed)
    sample = 2 * np.random.randint(2, size=(spins)) - 1.0

    # initialize energy and config list
    configs = []
    energies = []
    accepted = 0
    single_step = 0
    eng_now = compute_energy(sample, neighbours, couplings, len_neighbours)

    # disable bar in parallel processing too
    disable = disable_bar + verbose
    pbar = tqdm(range(steps + burnt), disable=disable)
    # use sweeps to reduce correlation
    skip_steps = 1 if sweeps == 0 else sweeps * spins
    for step in pbar:
        for _ in range(skip_steps):
            single_step += 1
            k = np.random.randint(0, spins)
            # Metropolis-Hastings algorithm https://doi.org/10.2307/2334940
            deltah = compute_delta_h(
                k, sample, neighbours[k], couplings[k], len_neighbours[k]
            )
            # if delta change is negative, we accept.
            # otherwise we accept based on the following probability
            if deltah < 0.0 or np.random.ranf() < np.exp(-beta * deltah):
                sample[k] = -sample[k]
                # update energy
                eng_now += deltah
                accepted += 1

        pbar.set_description(f"eng: {eng_now / spins:2.5f}", refresh=False)

        # do not save first N_burnt steps
        if step > burnt - 1:
            # save energies and step
            energies.append(eng_now)
            configs.append(sample.copy())

        if verbose and step > 1:
            print(
                f"{step-1:6d}  {eng_now / spins:2.4f}  {np.asarray(energies).mean():2.4f}  {np.asarray(energies).std(ddof=1):2.4f}"
            )

    configs = np.asarray(configs).astype("int8")
    energies = np.asarray(energies)
    if save:
        file = f"{spins}spins-seed{seed}-sample{step+1}-sweeps{sweeps}-beta{beta}.npy"
        print(file)
        # add parent directory
        if save_dir is not None:
            file = save_dir + file
            print(file)
        # Saves the configurations
        np.save(file, configs)
        print(f"Saved in {file}")

    print(f"\nMCMC: Beta={beta} Seed={seed}")
    print(
        f"Steps: {step + 1:6d}  A_r={accepted / single_step * 100:2.2f}%  E={energies.mean() / spins:2.6f} \u00B1 {(energies / spins).std(ddof=1) / math.sqrt(step+1):2.6f}  [\u03C3={(energies / spins).std(ddof=1):2.6f}  E_min={energies.min() / spins:2.6f}]"
    )
    print(f"Duration {datetime.now() - start_time}")
    return configs, energies, accepted / single_step * 100


def neural_mcmc(
    beta: float,
    steps: int,
    path: Union[str, Dict[str, np.ndarray]],
    couplings_path: str,
    model: str,
    batch_size: int = 20000,
    verbose: bool = False,
    save: bool = False,
    save_every: int = 1,
    disable_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Performs Markov Chain Monte Carlo using ansatz generated by a neural network.
    Args:
        beta (float): Inverse temperature.
        steps (int): Monte Carlo simulation steps.
        path (Union[str, Dict[str, np.ndarray]]): Path to the generated sample or path to the model to sample or sample itself.
        couplings_path (str): Path to the couplings.
        model (str): Name of the model to use.
        batch_size (int, optional): Number of parallel cofigurations to generate. Defaults to 20000.
        verbose (bool, optional): Set True to print information during the simulations. Defaults to False.
        save (bool, optional): Set True to save data after simulation. Defaults to False.
        save_every (int): Save every n steps to get uncorrelated data. Defaults to 1.
        disable_bar(bool, optional): Set True to disable the progress bar. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Sample, energy and acceptance rate.
    """
    start_time = datetime.now()
    # generate more data than needed
    steps = steps * save_every
    # load data generate by the NN
    proposals, log_probs = load_data(
        path, model=model, steps=steps, batch_size=batch_size, verbose=verbose
    )
    assert log_probs.shape[0] > steps - save_every

    # get the dimension of the sample from the data
    spin_side = proposals[0].shape[-1]
    spins = spin_side ** 2
    proposals = np.reshape(proposals, (proposals.shape[0], -1))

    accepted_log_prob = np.nan
    # get the first sample and its energy
    while not np.isfinite(accepted_log_prob):
        accepted_sample, accepted_log_prob = proposals[0], log_probs[0]

    # get neighbourhood and couplings matrix
    neighbours, couplings, len_neighbours = get_couplings(spin_side, couplings_path)

    # initialisation
    energies = []
    samples = []
    transition_prob = []
    log_prob_ratio = []
    accepted = 0

    # compute the energy of the new configuration
    accepted_eng = compute_energy(
        accepted_sample, neighbours, couplings, len_neighbours
    )
    # compute boltzmann probability
    accepted_boltz_log_prob = compute_boltz_prob(accepted_eng, beta, spin_side ** 2)

    print(f"\nPerforming Neural MCMC at beta={beta}")

    disable = disable_bar + verbose
    pbar = tqdm(range(steps - save_every), disable=disable)
    for idx in pbar:
        # get next sample and its energy
        trial_sample, trial_log_prob = proposals[idx + 1], log_probs[idx + 1]
        if not np.isfinite(trial_log_prob):
            print("NAN in log_prob")
            continue

        trial_eng = compute_energy(trial_sample, neighbours, couplings, len_neighbours)
        if not np.isfinite(trial_eng):
            print("NAN in trial_eng")
            continue

        # compute Boltzmann probability
        trial_boltz_log_prob = compute_boltz_prob(trial_eng, beta, spin_side ** 2)
        if not np.isfinite(trial_boltz_log_prob):
            print("NAN in trial_boltz_log_prob")
            continue

        # get the transition probability
        log_prob_ratio.append(
            +accepted_log_prob
            - trial_log_prob
            + trial_boltz_log_prob
            - accepted_boltz_log_prob
        )
        if not np.isfinite(log_prob_ratio[idx]):
            print("NAN in prob_ratio")
            continue

        transition_prob.append(min(0.0, log_prob_ratio[idx]))

        if verbose:
            print(
                f"{idx+1:6d}  neural  {accepted_eng/spins:2.4f}  {trial_eng/spins:2.4f}  {accepted_log_prob:3.2f}  {trial_log_prob:3.2f}  {accepted_boltz_log_prob:4.2f}  {trial_boltz_log_prob:4.2f}  {transition_prob[idx]:2.4f}"
            )

        if transition_prob[idx] >= 0.0 or (
            np.log(np.random.random_sample()) < transition_prob[idx]
        ):
            # update energy, prob and sample
            accepted_eng = np.copy(trial_eng)
            accepted_log_prob = np.copy(trial_log_prob)
            accepted_sample = np.copy(trial_sample)
            accepted_boltz_log_prob = np.copy(trial_boltz_log_prob)
            accepted += 1

        pbar.set_description(f"eng: {accepted_eng / spin_side**2:2.5f}", refresh=False)

        # save acceped sample and its energy
        samples.append(accepted_sample)
        energies.append(accepted_eng)

        if verbose:
            print(
                f"{idx+1:6d}  neural  {accepted_eng/spins:2.4f}  {trial_eng/spins:2.4f}  {accepted_log_prob:3.2f}  {trial_log_prob:3.2f}  {accepted_boltz_log_prob:4.2f}  {trial_boltz_log_prob:4.2f}  {transition_prob[idx]:2.4f}"
            )

    avg_eng, std_eng = np.asarray(energies).mean(), np.asarray(energies).std(ddof=1)
    # reduce save data
    samples = np.asarray(samples).astype("int8")[::save_every, ...]
    energies = np.asarray(energies).astype(np.double)[::save_every]
    if save:
        filename = f"{str(spins)}spins_beta{beta}_neural-mcmc_{steps}steps"
        out = {
            "accepted": accepted,
            "avg_eng": avg_eng,
            "std_eng": std_eng,
            "sample": samples,
            "energy": energies,
        }
        print("\nSaving MCMC output as {0}".format(filename))
        np.savez(filename, **out)

    print(
        f"Steps: {steps:6d} A_r={accepted / steps * 100:2.2f}%\nE={energies.mean() / spins:2.6f} \u00B1 {(energies / spins).std(ddof=1) / math.sqrt(steps):2.6f}  [\u03C3={(energies / spins).std(ddof=1):2.6f}  E_min={energies.min() / spins:2.6f}]"
    )
    print(f"Duration {datetime.now() - start_time}")
    return (samples, energies, accepted / steps * 100)


def seq_hybrid_mcmc(
    beta: float,
    steps: int,
    path: Union[str, Dict[str, np.ndarray]],
    couplings_path: str,
    model: str,
    model_path: Optional[str] = None,
    batch_size: int = 20000,
    len_seq_single: int = 100,
    verbose: bool = False,
    save: bool = False,
    save_every: int = 1,
    disable_bar: bool = False,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Sequential Hybrid MCMC performs a simulations where two simulation,
    one neural and the one single spin flip, merged together sequentially.

    Args:
        beta (float): Inverse temperature.
        steps (int): Monte Carlo simulation steps.
        path (Union[str, Dict[str, np.ndarray]]):Path to the generated sample or path to the model to sample or sample itself.
        couplings_path (str): Path to the couplings.
        model (str): Name of the model to use.
        model_path (Optional[str], optional): Path to the model, if not provided before. Defaults to None.
        batch_size (int, optional): Number of parallel cofigurations to generate. Defaults to 20000.
        len_seq_single (int, optional): Lenght of the single spin sequence. Defaults to 100.
        verbose (bool, optional): Set True to print information during the simulations. Defaults to False.
        save (bool, optional):  Set True to save data after simulation. Defaults to False.
        save_every (int, optional): Save every n steps to get uncorrelated data. Defaults to 1.
        disable_bar (bool, optional): Set True to disable the progress bar. Defaults to False.

    Returns:
        Tuple[np.ndarray, np.ndarray, float]: Sample, energy and acceptance rate.
    """

    start_time = datetime.now()
    # set a limit to prevent memory/timeout errors
    MAX_STEPS = 1e10
    # increase steps to avoid correlation
    steps *= save_every
    # load data generate by the NN
    # when sample on-the-fly sample 10% more than expected
    proposals, log_probs = load_data(
        path, model, math.ceil(steps / len_seq_single) + 1, batch_size, verbose
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # load the model
    if model_path is not None:
        model = Made.load_from_checkpoint(model_path).to(device)
    else:
        model = Made.load_from_checkpoint(path).to(device)

    # get the dimension of the sample from the data
    spin_side = proposals[0].shape[-1]
    spins = spin_side ** 2
    proposals = np.reshape(proposals, (proposals.shape[0], -1))

    accepted_log_prob = np.nan
    # get the first sample and its energy
    while np.isnan(accepted_log_prob):
        accepted_sample, accepted_log_prob = proposals[0], log_probs[0]

    # get neighbourhood and couplings matrix
    neighbours, couplings, len_neighbours = get_couplings(spin_side, couplings_path)

    # initialisation
    energies = []
    samples = []
    accepted = 1
    accepted_single = 0
    accepted_neural = 1
    type_accepted = "neural"
    neural_after_single = 0

    # compute the energy of the new configuration
    accepted_eng = compute_energy(
        accepted_sample, neighbours, couplings, len_neighbours
    )
    # compute boltzmann probability
    accepted_boltz_log_prob = compute_boltz_prob(accepted_eng, beta, spins)

    print(f"\nPerforming Sequential Hybrid MCMC at beta={beta}")

    steps_neural = 1
    steps_single = 0
    disable = verbose + disable_bar
    pbar = tqdm(range(steps - 1), disable=disable)
    for step in pbar:
        # take the sample from the neural network with desired prob
        if step % len_seq_single == 0:
            trial_sample, trial_log_prob = (
                proposals[steps_neural],
                log_probs[steps_neural],
            )
            if not np.isfinite(trial_log_prob):
                print("NAN in trial_log_prob")
                continue

            # compute prob of the old accepted sample
            # via the trained model
            # model accepts input in {0,1}
            accepted_log_prob = (
                model.forward(
                    torch.from_numpy((accepted_sample + 1) / 2).float().to(device)
                )
                .detach()
                .cpu()
                .numpy()
            )
            if not np.isfinite(trial_log_prob):
                print("NAN in trial_log_prob")
                continue

            steps_neural += 1
            # flag to count accepted sample from the model
            neural = True
            # compute sample's configuration
            trial_eng = compute_energy(
                trial_sample, neighbours, couplings, len_neighbours
            )
            if not np.isfinite(trial_eng):
                print("NAN in trial_eng")
                continue
            # compute Boltzmann probability
            trial_boltz_log_prob = compute_boltz_prob(trial_eng, beta, spins)
            if not np.isfinite(trial_boltz_log_prob):
                print("NAN in trial_boltz_log_prob")
                continue

            # compute log prob ratio
            # for single -> neural
            log_prob_ratio = (
                trial_boltz_log_prob
                - accepted_boltz_log_prob
                + accepted_log_prob
                - trial_log_prob
            )
        else:
            # try a single spin flip move
            k = np.random.randint(0, spins)
            trial_sample = accepted_sample.copy()
            trial_sample[k] *= -1
            # update count
            steps_single += 1
            neural = False
            # Metropolis-Hastings algorithm https://doi.org/10.2307/2334940
            deltah = compute_delta_h(
                k,
                accepted_sample,
                neighbours[k].astype(int),
                couplings[k],
                len_neighbours[k],
            )
            # compute energy using delta energy
            trial_eng = accepted_eng + deltah
            if not np.isfinite(trial_eng):
                print("NAN in trial_eng")
                continue

            # compute Boltzmann probability
            trial_boltz_log_prob = compute_boltz_prob(trial_eng, beta, spins)
            if not np.isfinite(trial_boltz_log_prob):
                print("NAN in trial_boltz_log_prob")
                continue

            # get the transition probability
            log_prob_ratio = trial_boltz_log_prob - accepted_boltz_log_prob

        transition_prob = min(0.0, log_prob_ratio)

        if transition_prob >= 0.0 or (
            np.log(np.random.random_sample()) < transition_prob
        ):
            # update energy, prob and sample
            accepted_eng = np.copy(trial_eng)
            accepted_log_prob = np.copy(trial_log_prob)
            accepted_sample = np.copy(trial_sample)
            accepted_boltz_log_prob = np.copy(trial_boltz_log_prob)
            # count mix steps
            if type_accepted == "single" and neural:
                neural_after_single += 1
            type_accepted = "neural" if neural else "single"

            if neural:
                accepted_neural += 1
            else:
                accepted_single += 1
            accepted += 1

        pbar.update()
        pbar.set_description(f"eng: {accepted_eng / spins:2.5f}", refresh=False)

        if step % save_every == 0:
            # save acceped sample and its energy
            samples.append(accepted_sample)
            energies.append(accepted_eng)

        if verbose:
            if neural:
                print(
                    f"{step+1:6d}  neural  {accepted_eng/spins:2.4f}  {trial_eng/spins:2.4f}  {accepted_log_prob:3.2f}  {trial_log_prob:3.2f}  {accepted_boltz_log_prob:4.2f}  {trial_boltz_log_prob:4.2f}  {transition_prob:2.4f}"
                )
            else:
                print(
                    f"{step+1:6d}  single  {accepted_eng/spins:2.4f}  {trial_eng/spins:2.4f}  {accepted_log_prob:3.2f}  {trial_log_prob:3.2f}  {accepted_boltz_log_prob:4.2f}  {trial_boltz_log_prob:4.2f}  {transition_prob:2.4f}"
                )
        if step > MAX_STEPS:
            print("Steps limit")
            break

    samples = np.asarray(samples).astype(np.int8)
    energies = np.asarray(energies).astype(np.double)
    avg_eng = energies.mean()
    err_eng = energies.std(ddof=1) / math.sqrt(energies.shape[0])
    if save:
        filename = f"{str(spins)}spins_beta{beta}_{math.ceil(steps/save_every)}hybrid-mcmc_single_len{len_seq_single}"
        out = {
            "sample": samples,
            "energy": energies,
        }
        print("\nSaving MCMC output as {0}\n".format(filename))
        np.savez(filename, **out)

    print(
        f"Accepted proposals (neural): {accepted_neural} on {steps_neural} (A_r={accepted_neural / (steps_neural + np.finfo(float).eps) * 100:2.2f}%)"
    )
    print(
        f"Accepted proposals (single spin flip): {accepted_single} on {steps_single} (A_r={accepted_single / (steps_single + np.finfo(float).eps) * 100:2.2f}%)"
    )
    print(
        f"Steps: {step + 2:6d}  A_r={accepted / steps * 100:2.2f}%  E={avg_eng / spins:2.6f} \u00B1 {err_eng / spins:2.6f}  [\u03C3={energies.std(ddof=1) / spins:2.6f}  E_min={energies.min() / spins:2.6f}]"
    )
    print(f"Accepted Neural after Single {neural_after_single}")
    print(f"Duration {datetime.now() - start_time}\n")
    return (samples, energies, accepted / steps * 100)