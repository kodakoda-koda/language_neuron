from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CustomDataset(Dataset):
    def __init__(
        self,
        data: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        lang: str = "en",
        train_flag: bool = True,
    ):
        self.tokenized_data = self.__tokenize__(data)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.lang = lang
        self.train_flag = train_flag

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.tokenized_data["input_ids"][index],
            "attention_mask": self.tokenized_data["attention_mask"][index],
        }

    def __len__(self):
        return len(self.data)

    def __tokenize__(self, data: List[str]) -> Dict[str, torch.Tensor]:
        if self.train_flag:
            data = data[: len(data) * 0.8]
        else:
            data = data[len(data) * 0.8 :]

        tokenized_data = self.tokenizer.batch_encode_plus(
            data,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_data
