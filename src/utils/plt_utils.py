import math
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.utils.utils import block_single_std, compute_energy, get_couplings


def plt_betas_ar(
    acc_rates: List[np.ndarray],
    labels: List[str],
    betas: np.ndarray,
    xlim: Optional[Tuple[float, float]] = None,
    save: bool = False,
):
    fig, ax = plt.subplots()  # figsize=(7.2, 6.4), dpi=300

    plt.minorticks_off()

    for i, acc_rate in enumerate(acc_rates):
        plt.plot(betas, acc_rate, "--", label=labels[i])

    if xlim is not None:
        plt.xlim(xlim)

    plt.ylabel(r"$\mathrm{A_r}[\%]$")
    plt.xlabel(r"$\mathrm{\beta}$", fontweight="bold")

    plt.legend(loc="best", fancybox=True)

    if save:
        plt.savefig(
            "images/arbeta.png",
            edgecolor="white",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
        )
        plt.savefig(
            "images/arbeta.eps",
            edgecolor="white",
            facecolor=fig.get_facecolor(),
            # transparent=True,
            bbox_inches="tight",
            format="eps",
        )
    plt.show()
    return


def plt_eng_step(
    eng1: np.ndarray,
    eng2: np.ndarray,
    label1: str,
    label2: str,
    ground_state: Optional[float] = None,
    xlim: Tuple[int, int] = (1, 100000),
    ylim: Optional[Tuple[int, int]] = None,
    title: Optional[str] = None,
    log_scale: bool = True,
    save: bool = False,
):
    fig, ax = plt.subplots()

    if len(eng1.shape) > 1:
        plt.fill_between(
            np.arange(eng1.shape[-1]) + 1,
            eng1.mean(axis=0) + eng1.std(axis=0),
            eng1.mean(axis=0) - eng1.std(axis=0),
            alpha=0.1,
            color="b",
        )
        plt.plot(
            np.arange(eng1.shape[-1]) + 1, eng1.mean(axis=0), label=label1, color="b"
        )
    else:
        plt.plot(np.arange(eng1.shape[-1]) + 1, eng1, label=label1, color="b")

    if len(eng2.shape) > 1:
        plt.fill_between(
            np.arange(eng2.shape[-1] + 1),
            eng2.mean(axis=0) + eng2.std(axis=0),
            eng2.mean(axis=0) - eng2.std(axis=0),
            alpha=0.1,
            color="tab:orange",
        )
        plt.plot(
            np.arange(eng2.shape[-1]) + 1,
            eng2.mean(0),
            "--",
            label=label2,
            color="tab:orange",
            alpha=0.5,
            linewidth=1.0,
        )
    else:
        plt.plot(
            np.arange(eng2.shape[-1]) + 1,
            eng2,
            "--",
            label=label2,
            color="tab:orange",
            alpha=0.5,
            linewidth=1.0,
        )

    if log_scale:
        ax.set_xscale("log")

    if ground_state is not None:
        plt.hlines(
            ground_state,
            xmin=0,
            xmax=xlim[1] + 100000,
            colors="red",
            linestyles="dashed",
            label="Ground State",
            linewidth=3.0,
        )
        if ylim is None:
            ylim = (ground_state - 0.01, max(eng1.max(), eng2.max()))
            plt.ylim(ylim)
        else:
            plt.ylim(ylim)

    plt.xlim(xlim)

    plt.ylabel(r"$E/N$", fontfamily="serif")
    plt.xlabel(r"$\mathrm{\tau}$")

    if title is not None:
        plt.title(rf"{title}", fontsize=20)

    plt.legend(loc="best", fontsize=18, labelspacing=0.4, borderpad=0.2, fancybox=True)

    if save:
        # TOFIX
        # ERROR: when saving .png
        plt.savefig(
            "images/energy-steps.png",
            facecolor=fig.get_facecolor(),
            bbox_inches="tight",
            transparent=False,
        )
        plt.savefig(
            "images/energy-steps.eps",
            facecolor=fig.get_facecolor(),
            # transparent=True,
            bbox_inches="tight",
            format="eps",
        )
    plt.show()
    return


def plt_acf(
    acs1: Union[np.ndarray, List[np.ndarray]],
    label1: Union[str, List[str]],
    acs2: Optional[np.ndarray] = None,
    label2: Optional[str] = None,
    xlim: Tuple[int, int] = (1, 5000),
    ylim: Tuple[int, int] = (0.01, 1),
    title: Optional[str] = None,
    fit: bool = False,
    log_scale: bool = True,
    save: bool = False,
):

    from scipy.optimize import curve_fit

    def stretch_exp(t, a, tau, alpha):
        return a * np.exp(-((t / tau) ** alpha))

    fig, ax = plt.subplots()

    # HARDCODE: to change if we have
    # more than 3 acs
    color_acs1 = ["gold", "red", "darkred"]
    if isinstance(acs1, list):
        assert len(acs1) == len(label1)
        assert len(acs1) <= 3

    for i, acs in enumerate(acs1):
        xlim1 = min(xlim[1], acs.shape[0])
        plt.plot(
            np.arange(xlim1) + 1, acs[:xlim1], label=label1[i], color=color_acs1[i]
        )
        ax.set_yscale("log")
        if fit:
            p, _ = curve_fit(
                stretch_exp,
                np.arange(xlim1),
                acs[:xlim1],
                bounds=([-np.inf, 0, 0], [np.inf, np.inf, np.inf]),
            )
            print(f"{label1[i]} a={p[0]} tau*={p[1]} alpha={p[2]}")
            plt.plot(
                np.arange(xlim[1]) + 1,
                stretch_exp(np.arange(xlim[1]), p[0], p[1], p[2]),
                "--",
                color=color_acs1[i],
            )

    if acs2 is not None:
        assert len(acs2) <= 3
        color_acs2 = ["skyblue", "steelblue", "blue"]
        for i, acs in enumerate(acs2):
            plt.plot(
                np.insert(acs, 0, 1.0, axis=0),
                "--",
                label=label2[i],
                color=color_acs2[i],
            )

    if log_scale:
        ax.set_xscale("log")

    plt.ylabel(r"$\mathrm{c(\tau)}$")
    plt.xlabel(r"$\mathrm{\tau}$")

    plt.ylim(ylim)
    plt.xlim(xlim)

    if title is not None:
        plt.title(
            title,
            fontsize=18,
        )

    plt.legend(loc="best", fontsize=18, labelspacing=0.4, borderpad=0.2, fancybox=True)

    if save:
        plt.savefig("images/correlation.png", facecolor=fig.get_facecolor())
        plt.savefig("images/correlation.eps", format="eps")

    plt.show()


def plot_hist(
    paths: List[str],
    couplings_path: str,
    truth_path: str,
    ground_state: Optional[float] = None,
    colors: Optional[List[str]] = None,
    labels: Optional[List[str]] = None,
    density: bool = False,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    ticklables: Optional[Sequence[float]] = None,
    num_bins: int = 50,
    save: bool = False,
) -> None:

    import matplotlib.ticker as ticker

    if labels is None:
        labels = [f"Dataset {i}" for i, _ in enumerate(paths)]
        labels.append("Truth")
    if colors is None:
        colors = [None for _ in paths]

    assert len(labels) - 1 == len(colors) == len(paths)

    truth = np.load(truth_path)
    try:
        truth = truth["sample"]
    except:
        truth = truth

    min_len_sample = truth.shape[0]
    truth = np.reshape(truth, (min_len_sample, -1))
    spins = truth.shape[-1]

    # laod couplings
    # TODO Adjancecy should wotk with spins, not spin side
    neighbours, couplings, len_neighbours = get_couplings(
        int(math.sqrt(spins)), couplings_path
    )

    eng_truth = []
    for t in truth:
        eng_truth.append(compute_energy(t, neighbours, couplings, len_neighbours))
    eng_truth = np.asarray(eng_truth) / spins

    min_eng, max_eng = eng_truth.min(), eng_truth.max()

    engs = []
    for path in paths:
        if isinstance(path, str):
            data = np.load(path)
            try:
                sample = data["sample"]
            except:
                sample = data

            sample = sample.squeeze()
            min_len_sample = min(min_len_sample, sample.shape[0])
            sample = np.reshape(sample, (-1, spins))

            eng = []
            for s in sample:
                eng.append(compute_energy(s, neighbours, couplings, len_neighbours))
            eng = np.asarray(eng) / spins
        else:
            eng = path

        min_eng = min(min_eng, eng.min())
        max_eng = max(max_eng, eng.max())
        engs.append(eng)

    fig, ax = plt.subplots(figsize=(7.8, 7.8))

    ax.set_yscale("log")

    plt.ylabel("Count", fontweight="normal")
    plt.xlabel(r"$E/N$")

    plt.ylim(1, min_len_sample * 0.5)

    bins = np.linspace(min_eng, max_eng, num=num_bins).tolist()

    for i, eng in enumerate(engs):
        _ = plt.hist(
            eng[:min_len_sample],
            bins=bins,
            label=f"{labels[i]}",
            histtype="bar",
            linewidth=0.1,
            edgecolor=None,
            alpha=0.9 - i * 0.1,
            color=colors[i],
            density=density,
        )
        print(
            f"\n{labels[i]}\nE: {eng.mean()} \u00B1 {eng.std(ddof=1) / math.sqrt(eng.shape[0])}\nmin: {eng.min()} ({np.sum(eng==eng.min())} occurance(s))                                                                    (s))"
        )
    _ = plt.hist(
        eng_truth[:min_len_sample],
        bins=bins,
        # log=True,
        label=f"{labels[i+1]}",
        histtype="bar",
        edgecolor="k",
        color=["lightgrey"],
        alpha=0.5,
        density=density,
    )

    if density:
        min_len_sample = 200

    if ground_state is not None:
        plt.vlines(
            ground_state,
            1,
            min_len_sample * 0.5,
            linewidth=4.0,
            colors="red",
            linestyles="dashed",
            alpha=0.7,
            label="Ground State",
        )

    print(
        f"\n{labels[i+1]} eng\nE: {eng_truth.mean()} \u00B1 {eng_truth.std(ddof=1) / math.sqrt(eng_truth.shape[0])}\nmin: {eng_truth.min()}  ({np.sum(eng_truth==eng_truth.min())} occurance(s))"
    )

    plt.ylim(1, min_len_sample * 0.5)
    if ylim is not None:
        plt.ylim(ylim)
    if xlim is not None:
        plt.xlim(xlim)
    if ticklables is not None:
        ax.tick_params(axis="x", which="minor", bottom=False)
        plt.locator_params(axis="x", nbins=len(ticklables))
        ax.set_xticklabels(ticklables)

    plt.legend(loc="upper right")

    if save:
        plt.savefig("images/hist.png")
        plt.savefig("images/hist.eps", format="eps")

    return


def get_errorbar(energies: np.ndarray, len_block: int, skip: int) -> np.ndarray:
    yerr = block_single_std(energies, len_block=len_block, skip=skip)
    new_err = [
        np.abs(
            np.min(
                energies[..., skip:].mean(axis=2)
                - yerr
                - energies[..., skip:].mean(axis=2).mean(0),
                axis=0,
            )
        ),
        np.abs(
            np.max(
                energies[..., skip:].mean(axis=2).mean(0)
                - (energies[..., skip:].mean(axis=2) + yerr),
                axis=0,
            )
        ),
    ]
    return np.asarray(new_err)


def plt_eng_chains(
    engs: List[np.ndarray],
    strengths: np.ndarray,
    ground_state: float,
    dwave_default: float,
    title: str,
    xlim: Tuple[float, float] = (0.4, 4.1),
    save: bool = False,
) -> None:

    plt.plot(strengths, engs.min(1) / 484, "-s", label=r"Minimum", linewidth=1.0)
    plt.errorbar(
        strengths,
        engs.mean(1) / 484,
        engs.std(1) / 484,
        capsize=5.0,
        elinewidth=2.5,
        linewidth=0.5,
        marker="s",
        color="tab:orange",
        fillstyle="none",
        markersize=8,
        markeredgewidth=2,
        label=r"Mean",
    )

    plt.hlines(
        ground_state,
        xmin=xlim[0] - 0.4,
        xmax=xlim[1] + 0.4,
        colors="red",
        linestyles="dashed",
        label="Ground State",
        linewidth=3,
    )

    if dwave_default is not None and title is not None:
        plt.plot(
            dwave_default,
            np.load(f"data/sweep_chains_{title.lower()}/dwave-engs_0.npy").min() / 484,
            "d",
            color="tab:green",
            label=f"D-Wave default",
        )
        plt.errorbar(
            dwave_default,
            np.load(f"data/sweep_chains_{title.lower()}/dwave-engs_0.npy").mean() / 484,
            np.load(f"data/sweep_chains_{title.lower()}/dwave-engs_0.npy").std() / 484,
            capsize=5.0,
            elinewidth=1.5,
            linewidth=0.1,
            marker="d",
            color="tab:green",
            fillstyle="none",
            markersize=8,
            markeredgewidth=2,
            label=f"D-Wave default",
        )

    plt.minorticks_off()
    plt.xlim(xlim)

    plt.ylabel(r"$\mathrm{E}$")
    plt.xlabel(r"chains_strength", fontsize=24, fontweight="ultralight")

    plt.title(f"{title} couplings", fontsize=20)

    plt.legend(loc="best", fontsize=22, labelspacing=0.4, borderpad=0.2, fancybox=True)
    if save:
        plt.savefig(f"images/strenght-energy_1nn-{title}.png")

        plt.savefig(f"images/strenght-energy_1nn-{title}.eps")

    plt.show()
