import math
from typing import List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np

from src.utils.utils import block_single_std, compute_energy, get_couplings


def plt_betas_ar(
    ax,
    acc_rates: List[List[float]],
    labels: List[str],
    betas: np.ndarray,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    line_style: Optional[List[str]] = None,
    color: Optional[List[str]] = None,
    save: bool = False,
):
    # fig, ax = plt.subplots()  # figsize=(7.2, 6.4), dpi=300

    ax.minorticks_off()

    if line_style is None:
        line_style = ["--"] * len(acc_rates)
    if color is None:
        color = [None] * len(acc_rates)
    assert len(line_style) == len(acc_rates) == len(color)

    for i, acc_rate in enumerate(acc_rates):
        ax.plot(
            betas,
            acc_rate,
            line_style[i],
            color=color[i],
            markersize=6,
            label=labels[i],
        )

    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)

    ax.set_ylabel(r"$\mathrm{A_r}[\%]$")
    ax.set_xlabel(r"$\mathrm{\beta}$")

    ax.legend(loc="best", fancybox=True, framealpha=0.8)

    if save:
        plt.savefig(
            "images/arbeta.png",
            edgecolor="white",
            facecolor=ax.get_facecolor(),
            bbox_inches="tight",
        )
        plt.savefig(
            "images/arbeta.eps",
            edgecolor="white",
            facecolor=ax.get_facecolor(),
            # transparent=True,
            bbox_inches="tight",
            format="eps",
        )
    return


def plt_eng_step(
    ax,
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
    # fig, ax = plt.subplots()

    if len(eng1.shape) > 1:
        ax.fill_between(
            np.arange(eng1.shape[-1]) + 1,
            eng1.mean(axis=0) + eng1.std(axis=0),
            eng1.mean(axis=0) - eng1.std(axis=0),
            alpha=0.1,
            color="b",
        )
        ax.plot(
            np.arange(eng1.shape[-1]) + 1, eng1.mean(axis=0), label=label1, color="b"
        )
    else:
        ax.plot(np.arange(eng1.shape[-1]) + 1, eng1, label=label1, color="b")

    if len(eng2.shape) > 1:
        ax.fill_between(
            np.arange(eng2.shape[-1] + 1),
            eng2.mean(axis=0) + eng2.std(axis=0),
            eng2.mean(axis=0) - eng2.std(axis=0),
            alpha=0.1,
            color="tab:orange",
        )
        ax.plot(
            np.arange(eng2.shape[-1]) + 1,
            eng2.mean(0),
            "--",
            label=label2,
            color="tab:orange",
            alpha=0.5,
            linewidth=1.0,
        )
    else:
        ax.plot(
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
        ax.hlines(
            ground_state,
            xmin=0,
            xmax=xlim[1] + 100000,
            colors="red",
            linestyles="dashed",
            label="GS",
            linewidth=3.0,
        )
        if ylim is None:
            ylim = (ground_state - 0.01, max(eng1.max(), eng2.max()))
            ax.set_ylim(ylim)
        else:
            ax.set_ylim(ylim)

    ax.set_xlim(xlim)

    ax.set_ylabel(r"$E/N$")
    ax.set_xlabel(r"$\mathrm{\tau}$")

    if title is not None:
        ax.set_title(rf"{title}")

    ax.legend(
        loc="best"
    )  # , fontsize=18, labelspacing=0.4, borderpad=0.2, fancybox=True)

    if save:
        # TOFIX
        # ERROR: when saving .png
        plt.savefig(
            "images/energy-steps.png",
            facecolor=ax.get_facecolor(),
            bbox_inches="tight",
            transparent=False,
        )
        plt.savefig(
            "images/energy-steps.eps",
            facecolor=ax.get_facecolor(),
            # transparent=True,
            bbox_inches="tight",
            format="eps",
        )
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

    # HARDCODED: set correct figsize
    fig, ax = plt.subplots(figsize=(6.8, 6))

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


def plot_hist(
    ax,
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
    order: str = "C",
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
    truth = np.reshape(truth, (min_len_sample, -1), order=order)
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
            sample = np.reshape(sample, (-1, spins), order=order)

            eng = []
            for s in sample:
                eng.append(compute_energy(s, neighbours, couplings, len_neighbours))
            eng = np.asarray(eng) / spins
        else:
            eng = path

        min_eng = min(min_eng, eng.min())
        max_eng = max(max_eng, eng.max())
        engs.append(eng)

    # fig, ax = plt.subplots(figsize=(7.8, 7.8))

    ax.set_yscale("log")

    ax.set_ylabel("Count")
    ax.set_xlabel(r"$E/N$")

    ax.set_ylim(1, min_len_sample * 0.5)

    bins = np.linspace(min_eng, max_eng, num=num_bins).tolist()

    for i, eng in enumerate(engs):
        _ = ax.hist(
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
    _ = ax.hist(
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
        ax.vlines(
            ground_state,
            1,
            min_len_sample * 0.5,
            linewidth=4.0,
            colors="red",
            linestyles="dashed",
            alpha=0.7,
            label="GS",
        )

    print(
        f"\n{labels[i+1]} eng\nE: {eng_truth.mean()} \u00B1 {eng_truth.std(ddof=1) / math.sqrt(eng_truth.shape[0])}\nmin: {eng_truth.min()}  ({np.sum(eng_truth==eng_truth.min())} occurance(s))"
    )

    ax.set_ylim(1, min_len_sample * 0.5)
    if ylim is not None:
        ax.set_ylim(ylim)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ticklables is not None:
        ax.tick_params(axis="x", which="minor", bottom=True)
        ax.locator_params(axis="x", nbins=len(ticklables))
        ax.set_xticklabels(ticklables)

    ax.legend(loc="upper right")

    if save:
        plt.savefig("images/hist.png")
        plt.savefig("images/hist.eps", format="eps")

    return ax


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
    ax,
    engs: np.ndarray,
    strengths: np.ndarray,
    ground_state: float,
    dwave_default: float,
    title: Optional[str],
    xlim: Tuple[float, float] = (0.4, 4.1),
    legend: bool = True,
    save: bool = False,
) -> None:

    ax.plot(strengths, engs.min(1) / 484, "-s", label=r"Minimum", linewidth=1.0)
    ax.errorbar(
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

    ax.hlines(
        ground_state,
        xmin=xlim[0] - 0.4,
        xmax=xlim[1] + 0.4,
        colors="red",
        linestyles="dashed",
        label="GS",
        linewidth=3,
    )

    if dwave_default is not None and title is not None:
        ax.plot(
            dwave_default,
            np.load(f"data/sweep_chains_{title.lower()}/dwave-engs_0.npy").min() / 484,
            "d",
            markersize=10,
            color="tab:green",
            label=f"D-Wave default",
        )
        ax.errorbar(
            dwave_default,
            np.load(f"data/sweep_chains_{title.lower()}/dwave-engs_0.npy").mean() / 484,
            np.load(f"data/sweep_chains_{title.lower()}/dwave-engs_0.npy").std() / 484,
            capsize=5.0,
            elinewidth=1.5,
            linewidth=0.1,
            marker="d",
            color="tab:green",
            fillstyle="none",
            markersize=10,
            markeredgewidth=2,
            label=f"D-Wave default",
        )

    ax.minorticks_off()
    ax.set_xlim(xlim)

    ax.set_ylabel(r"$\mathrm{E}$")
    ax.set_xlabel(r"chains_strength")

    if title and not save:
        ax.set_title(f"{title} couplings")

    if legend:
        ax.legend(loc="best", fancybox=True, framealpha=0.5)

    if save:
        plt.savefig(f"images/strenght-energy_1nn-{title}.png")

        plt.savefig(f"images/strenght-energy_1nn-{title}.eps")
