import argparse
from pathlib import Path

from src.generate import generate

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt-path", type=Path, help="Path to the checkpoint")
parser.add_argument(
    "--model", type=str, choices=["made", "pixel"], help="Model to use"
)
parser.add_argument(
    "--num-sample",
    type=int,
    default=1,
    help="Number of sample to generate (default: 1)",
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=20000,
    help="Dimension of the single generated sample (default: 20000)",
)
parser.add_argument(
    "--num-workers",
    type=int,
    default=2,
    help="Number of workers to generate (default: 2)",
)
parser.add_argument(
    "--save-sample",
    dest="save_sample",
    action="store_true",
    help="Flag if you want to save samples after generation",
)
parser.add_argument(
    "--verbose",
    dest="verbose",
    action="store_true",
    help="Flag if you want to see model parameters",
)


def main(args: argparse.ArgumentParser):

    generate(
        args.ckpt_path,
        args.model,
        args.num_sample,
        args.batch_size,
        args.num_workers,
        args.save_sample,
        args.verbose,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
