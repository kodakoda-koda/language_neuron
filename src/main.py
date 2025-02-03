import argparse
import logging
import warnings

import transformers
from transformers import set_seed

from src.exp_main import Exp_main

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def main():
    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_length", type=int, default=64)
    parser.add_argument("--lang", type=str, default="en")
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--lm_name", type=str, default="facebook/xglm-564M")
    parser.add_argument("--batch_size", type=int, default=4)
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
    exp.detect()


if __name__ == "__main__":
    main()
