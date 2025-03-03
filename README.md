# Reproduction Implementation: "On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons"

This repository contains the reproduction implementation code for the paper titled "[On the Multilingual Ability of Decoder-based Pre-trained Language Models: Finding and Controlling Language-Specific Neurons](https://aclanthology.org/2024.naacl-long.384/)".

## Requirements

The code is written in Python 3.12.3 and requires the following packages:

```bash
pip install -r requirements.txt
```

## Data

The datasets used in the paper are available at [flores-200](https://huggingface.co/datasets/Muennighoff/flores200) and [paws-x](https://huggingface.co/datasets/google-research-datasets/paws-x).

## Model

This repository provides the code to load the following models:

- XGLM-564M: [xglm-564M](https://huggingface.co/facebook/xglm-564M)
- XGLM-1.7B: [xglm-1.7B](https://huggingface.co/facebook/xglm-1.7B)
- XGLM-2.9B: [xglm-2.9B](https://huggingface.co/facebook/xglm-2.9B)
- BLOOM-560M: [bloom-560m](https://huggingface.co/bigscience/bloom-560m)
- BLOOM-1B7: [bloom-1b7](https://huggingface.co/bigscience/bloom-1b7)
- BLOOM-3B: [bloom-3b](https://huggingface.co/bigscience/bloom-3b)

LLaMA-2 (7B/13B) models are used in the paper but are not available in this repository. They will be added in the future.

## Usage

1. Clone this repository.
2. Install the required packages.
3. Execute the following command to identify and manipulate language-specific neurons:

```bash
python -m main --lm_name <model_name> --plot
```

Replace `<model_name>` with one of the following options:

- facebook/xglm-564M
- facebook/xglm-1.7B
- facebook/xglm-2.9B
- bigscience/bloom-560m
- bigscience/bloom-1b7
- bigscience/bloom-3b

The `--plot` flag is optional and can be used to generate visualizations corresponding to Figure 2 and Table 3 in the paper.

ouputs/ and plots/ directories will be created to store the results.
