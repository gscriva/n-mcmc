import argparse

from src.utils.montecarlo import hybrid_mcmc, neural_mcmc, single_spin_flip

parser = argparse.ArgumentParser()

subparsers = parser.add_subparsers(help="Choose the Monte Carlo method")

parser.add_argument("--spins", type=int, help="Number of spins of the ravel spin glass")
parser.add_argument("--steps", type=int, help="Steps of the Monte Carlo simulation")
parser.add_argument("--beta", type=float, help="Inverse temperature")
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

parser_single.add_argument("--type", type=str, default="single", help=argparse.SUPPRESS)
parser_single.add_argument(
    "--sweeps", type=int, default=1, help="Number of sweeps before save"
)
parser_single.add_argument(
    "--seed-startpoint",
    type=int,
    default=12345,
    help="Seed to sample the starting point configuration",
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


def main(args: argparse.ArgumentParser):
    print(args)
    if args.type == "single":
        single_spin_flip(
            args.spins,
            args.beta,
            args.steps,
            args.couplings_path,
            args.sweeps,
            args.seed_startpoint,
            args.verbose,
            args.save,
        )
    elif args.type == "neural":
        neural_mcmc(
            args.beta,
            args.steps,
            args.path,
            args.couplings_path,
            args.model,
            args.batch_size,
            args.verbose,
            args.save,
        )
    elif args.type == "hybrid":
        hybrid_mcmc(
            args.beta,
            args.steps,
            args.path,
            args.couplings_path,
            args.model,
            args.model_path,
            args.batch_size,
            args.prob_single,
            args.verbose,
            args.save,
        )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
