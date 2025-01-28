import argparse
import random

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from datasets import load_dataset

from .dataset import CustomDataset


class Exp_base:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

        self.model = self._build_model()
        self.tokenizer = self._bulid_tokenizer()

    def _build_model(self) -> PreTrainedModel:
        model = AutoModelForCausalLM.from_pretrained(self.args.lm_name)
        model = model.to(self.device).to(self.dtype)
        return model

    def _bulid_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.args.lm_name)
        return tokenizer

    def _get_dataloader(self, train_flag: bool) -> DataLoader:
        assert self.args.lang in [
            "en",
            "de",
            "fr",
            "es",
            "zh",
            "ja",
        ], "lang must be one of ['en', 'de', 'fr', 'es', 'zh', 'ja']"

        pawsx_ = load_dataset("google-research-datasets/paws-x", self.args.lang)
        pawsx = pawsx_["train"]["sentence1"] + pawsx_["test"]["sentence1"] + pawsx_["validation"]["sentence1"]

        flores_lang_set = {
            "en": "eng_Latn",
            "de": "deu_Latn",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
        }
        flores_lang = flores_lang_set[self.args.lang]
        if flores_lang == "jpn_Jpan":
            flores_ = load_dataset("facebook/flores", "jpn_Jpan-eng_Latn")
        else:
            flores_ = load_dataset("facebook/flores", f"{flores_lang}-jpn_Jpan")
        flores = flores_["dev"][f"sentence_{flores_lang}"] + flores_["devtest"][f"sentence_{flores_lang}"]

        data = random.sample(pawsx, 250) + random.sample(flores, 250)

        dataset = CustomDataset(data, self.tokenizer, self.args.max_length, self.args.lang, train_flag)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=train_flag)
        return dataloader
