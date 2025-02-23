import argparse
import logging
import os
import warnings

import transformers
from transformers import set_seed

from src.exp.exp_main import Exp_main

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--output_path", type=str, default="./outputs")
    parser.add_argument("--plot_path", type=str, default="./plots")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--lm_name", type=str, default="facebook/xglm-564M")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--num_samples", type=int, default=100)
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)

    # set logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        level=logging.INFO,
        datefmt="%m/%d %H:%M",
    )
    logger = logging.getLogger(__name__)

    # train and evaluate
    exp = Exp_main(args, logger)

    if not os.path.exists(args.output_path + "/neurons.npy"):
        exp.detect()
    exp.inference()


if __name__ == "__main__":
    main()
