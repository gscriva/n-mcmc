import argparse
from multiprocessing import Pool

from src.utils.montecarlo import hybrid_mcmc, neural_mcmc, single_spin_flip

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
parser.add_argument("--save-every", type=int, help="Number of steps to save", default=1)

parser_single = subparsers.add_parser(
    "single", help="Single Spin Flip Monte Carlo Simulation"
)
parser_neural = subparsers.add_parser("neural", help="Neural MCMC")
parser_hybrid = subparsers.add_parser("hybrid", help="Hybrid MCMC")

parser_single.add_argument("--type", type=str, default="single", help=argparse.SUPPRESS)
parser_single.add_argument(
    "--sweeps", type=int, default=1, help="Number of sweeps before save"
)
parser_single.add_argument(
    "--seed-startpoint",
    nargs="+",
    type=int,
    default=12345,
    help="Seed to sample the starting point configuration, may be a list",
)

parser_neural.add_argument("--type", type=str, default="neural", help=argparse.SUPPRESS)
parser_neural.add_argument(
    "--path", type=str, help="Path to the model or to the generated sample"
)
parser_neural.add_argument(
    "--model", type=str, choices=["made", "pixel"], help="Model to use"
)
parser_neural.add_argument(
    "--batch-size", type=int, default=20000, help="Size of each batch (default: 20000)"
)

parser_hybrid.add_argument("--type", type=str, default="hybrid", help=argparse.SUPPRESS)
parser_hybrid.add_argument(
    "--path", type=str, help="Path to the model or to the generated sample"
)
parser_hybrid.add_argument(
    "--model", type=str, choices=["made", "pixel"], help="Model to use"
)
parser_hybrid.add_argument(
    "--model-path",
    type=str,
    default=None,
    help="Path to the model, if not given in path arg (default: None)",
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
    "--save-every",
    type=int,
    default=1,
    help="Save every n steps to get uncorrelated data (default: 1)",
)

MAX_CPUS = 20


def main(args: argparse.ArgumentParser):
    print(args)
    disable_bar = False
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
                args.steps - 1,
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


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
