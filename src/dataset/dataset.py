import random
from typing import Dict, List, Tuple

import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CustomDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: PreTrainedTokenizer,
    ):
        self.labels = torch.tensor(labels)
        self.tokenizer = tokenizer
        self.tokenized_data = self.__tokenize__(texts)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        return {
            "input_ids": self.tokenized_data["input_ids"][index],
            "attention_mask": self.tokenized_data["attention_mask"][index],
            "labels": self.labels[index],
        }

    def __len__(self) -> int:
        return len(self.tokenized_data["input_ids"])

    def __tokenize__(self, data: List[str]) -> Dict[str, torch.Tensor]:
        tokenized_data = self.tokenizer.batch_encode_plus(
            data,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_data
