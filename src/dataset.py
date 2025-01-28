import random
from typing import Dict, List

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CustomDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_len: int, lang: str):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.lang = lang
        self.data = self.__load_dataset__()
        self.tokenized_data = self.__tokenize__()

    def __getitem__(self, index):
        return {
            "input_ids": self.tokenized_data["input_ids"][index],
            "attention_mask": self.tokenized_data["attention_mask"][index],
        }

    def __len__(self):
        return len(self.data)

    def __load_dataset__(self) -> List[str]:
        assert self.lang in [
            "en",
            "de",
            "fr",
            "es",
            "zh",
            "ja",
        ], "lang must be one of ['en', 'de', 'fr', 'es', 'zh', 'ja']"

        pawsx_ = load_dataset("google-research-datasets/paws-x", self.lang)
        pawsx = pawsx_["train"]["sentence1"] + pawsx_["test"]["sentence1"] + pawsx_["validation"]["sentence1"]

        flores_lang_set = {
            "en": "eng_Latn",
            "de": "deu_Latn",
            "fr": "fra_Latn",
            "es": "spa_Latn",
            "zh": "zho_Hans",
            "ja": "jpn_Jpan",
        }
        flores_lang = flores_lang_set[self.lang]
        if flores_lang == "jpn_Jpan":
            flores_ = load_dataset("facebook/flores", "jpn_Jpan-eng_Latn")
        else:
            flores_ = load_dataset("facebook/flores", f"{flores_lang}-jpn_Jpan")
        flores = flores_["dev"][f"sentence_{flores_lang}"] + flores_["devtest"][f"sentence_{flores_lang}"]

        data = random.sample(pawsx, 250) + random.sample(flores, 250)
        return data

    def __tokenize__(self) -> Dict[str, torch.Tensor]:
        tokenized_data = self.tokenizer.batch_encode_plus(
            self.data,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_data
