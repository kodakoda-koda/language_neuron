import argparse
import logging
import random

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from src.dataset import CustomDataset
from src.model.xglm import CustomXGLMForCausalLM


class Exp_base:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

        self.model = self._build_model()
        self.tokenizer = self._bulid_tokenizer()

    def _build_model(self) -> PreTrainedModel:
        model = CustomXGLMForCausalLM.from_pretrained(self.args.lm_name)
        model = model.to(self.device).to(self.dtype)
        return model

    def _bulid_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.args.lm_name)
        return tokenizer

    def _get_dataloader(self, train_flag: bool) -> DataLoader:
        lang = ["en", "de", "fr", "es", "zh", "ja"]
        texts = []
        labels = []
        for i, l in enumerate(lang):
            # load paws-x dataset
            pawsx_ = load_dataset("google-research-datasets/paws-x", l)
            pawsx_ = pawsx_["train"]["sentence1"] + pawsx_["test"]["sentence1"] + pawsx_["validation"]["sentence1"]

            # load flores-200 dataset
            flores_lang_set = {
                "en": "eng_Latn-jpn_Jpan",
                "de": "deu_Latn-jpn_Jpan",
                "fr": "fra_Latn-jpn_Jpan",
                "es": "spa_Latn-jpn_Jpan",
                "zh": "zho_Hans-jpn_Jpan",
                "ja": "jpn_Jpan-eng_Latn",
            }
            flores_lang = flores_lang_set[l]
            flores_ = load_dataset("facebook/flores", flores_lang)
            flores_ = (
                flores_["dev"][f"sentence_{flores_lang.split('-')[0]}"]
                + flores_["devtest"][f"sentence_{flores_lang.split('-')[0]}"]
            )

            # sample 250 data from each dataset
            texts_ = random.sample(pawsx_, 250) + random.sample(flores_, 250)
            labels_ = [[int(j == i) for j in range(len(lang))] for _ in range(500)]

            texts.extend(texts_)
            labels.extend(labels_)

        dataset = CustomDataset(texts, labels, self.tokenizer, self.args.max_length, train_flag)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=train_flag)
        return dataloader
