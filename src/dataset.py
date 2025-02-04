from typing import Dict, List

import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class CustomDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        labels: List[List[int]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        train_flag: bool = True,
    ):
        self.labels = torch.tensor(labels)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.train_flag = train_flag
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
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return tokenized_data
