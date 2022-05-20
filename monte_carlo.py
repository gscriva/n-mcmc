import argparse
from multiprocessing import Pool
from pathlib import Path

from src.utils.montecarlo import (
    gibbs_rbm,
    exchange_rbm,
    hybrid_mcmc,
    neural_mcmc,
    seq_hybrid_mcmc,
    single_spin_flip,
)

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Choose the Monte Carlo method")

parser.add_argument("--spins", type=int, help="Number of spins of the ravel spin glass")
parser.add_argument("--steps", type=int, help="Steps of the Monte Carlo simulation")
parser.add_argument(
    "--beta", nargs="+", type=float, help="Inverse temperature, may be a list"
)
parser.add_argument("--couplings-path", type=str, help="Path to the couplings")
parser.add_argument(
    "--verbose", dest="verbose", action="store_true", help="Set verbose prints"
)
parser.add_argument(
    "--save", dest="save", action="store_true", help="Save the samples after MCMC"
)

parser_single = subparsers.add_parser(
    "single", help="Single Spin Flip Monte Carlo Simulation"
)
parser_neural = subparsers.add_parser("neural", help="Neural MCMC")
parser_hybrid = subparsers.add_parser("hybrid", help="Hybrid MCMC")
parser_gibbs = subparsers.add_parser("gibbs", help="Gibbs MCMC via RBM")
parser_exchange_rbm = subparsers.add_parser(
    "exchange-rbm", help="Gibbs and Single Spin Flip MCMC via RBM"
)

parser_single.add_argument("--type", type=str, default="single", help=argparse.SUPPRESS)
parser_single.add_argument(
    "--sweeps",
    type=int,
    default=0,
    help="Number of attemps to flip each spin before save (default: 0)",
)
parser_single.add_argument(
    "--seed-startpoint",
    nargs="+",
    type=int,
    default=42,
    help="Seed to sample the starting point configuration, may be a list (default: 42)",
)

parser_neural.add_argument("--type", type=str, default="neural", help=argparse.SUPPRESS)
parser_neural.add_argument(
    "--path", type=str, help="Path to the model or to the generated sample"
)
parser_neural.add_argument(
    "--model", type=str, choices=["made", "pixel", "rbm"], help="Model to use"
)
parser_neural.add_argument(
    "--batch-size", type=int, default=20000, help="Size of each batch (default: 20000)"
)
parser_neural.add_argument(
    "--save-every", type=int, help="Number of steps to save", default=1
)


parser_hybrid.add_argument("--type", type=str, default="hybrid", help=argparse.SUPPRESS)
parser_hybrid.add_argument(
    "--path", type=str, help="Path to the model or to the generated sample"
)
parser_hybrid.add_argument(
    "--model", type=str, choices=["made", "pixel", "rbm"], help="Model to use"
)
parser_hybrid.add_argument(
    "--model-path",
    type=Path,
    default=None,
    help="Path to the model, if not given in path's argument (default: None)",
)
parser_hybrid.add_argument(
    "--batch-size", type=int, default=20000, help="Size of each batch (default: 20000)"
)
parser_hybrid.add_argument(
    "--prob-single",
    type=float,
    default=0.5,
    help="Probability of single spin flip step (default: 0.5)",
)
parser_hybrid.add_argument(
    "--len-seq-single",
    type=int,
    default=None,
    help="Lenght of single spin flip consecutive steps. If given it enables the sequential hybrid algorithm (default: None)",
)
parser_hybrid.add_argument(
    "--save-every", type=int, help="Number of steps to save", default=1
)


parser_gibbs.add_argument("--type", type=str, default="gibbs", help=argparse.SUPPRESS)
parser_gibbs.add_argument("--path", type=Path, help="Path to the model")
parser_gibbs.add_argument(
    "--save-every", type=int, help="Number of steps to save", default=1
)


parser_exchange_rbm.add_argument(
    "--type", type=str, default="exchange-rbm", help=argparse.SUPPRESS
)
parser_exchange_rbm.add_argument("--path", type=Path, help="Path to the model")
parser_exchange_rbm.add_argument(
    "--save-every", type=int, help="Number of steps to save", default=1
)


MAX_CPUS = 20


def main(args: argparse.ArgumentParser):
    print(args)
    disable_bar = False
    # remove bar for multiple proc
    if len(args.beta) > 1:
        disable_bar = True
    if args.type == "single":
        if len(args.seed_startpoint) > 1:
            disable_bar = True
        pool = Pool(MAX_CPUS)
        for seed in args.seed_startpoint:
            for beta in args.beta:
                pool.apply_async(
                    single_spin_flip,
                    args=(
                        args.spins,
                        beta,
                        args.steps,
                        args.couplings_path,
                        args.sweeps,
                        seed,
                        args.verbose,
                        disable_bar,
                        args.save,
                    ),
                )
        pool.close()
        pool.join()

    elif args.type == "neural":
        for beta in args.beta:
            neural_mcmc(
                beta,
                args.steps,
                args.path,
                args.couplings_path,
                args.model,
                args.batch_size,
                args.verbose,
                args.save,
                args.save_every,
                disable_bar,
            )

    elif args.type == "hybrid":
        if args.len_seq_single is not None:
            for beta in args.beta:
                seq_hybrid_mcmc(
                    beta,
                    args.steps,
                    args.path,
                    args.couplings_path,
                    args.model,
                    args.model_path,
                    args.batch_size,
                    args.len_seq_single,
                    args.verbose,
                    args.save,
                    args.save_every,
                    disable_bar,
                )
        else:
            for beta in args.beta:
                hybrid_mcmc(
                    beta,
                    args.steps,
                    args.path,
                    args.couplings_path,
                    args.model,
                    args.model_path,
                    args.batch_size,
                    args.prob_single,
                    args.verbose,
                    args.save,
                    args.save_every,
                    disable_bar,
                )
    elif args.type == "gibbs":
        for beta in args.beta:
            gibbs_rbm(
                args.spins,
                args.steps,
                args.path,
                beta,
                args.couplings_path,
                args.verbose,
                args.save,
                args.save_every,
            )
    elif args.type == "exchange-rbm":
        for beta in args.beta:
            exchange_rbm(
                args.spins,
                args.steps,
                args.path,
                beta,
                args.couplings_path,
                args.verbose,
                args.save,
                args.save_every,
            )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
