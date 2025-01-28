import argparse

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

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

    def _get_dataloader(self) -> DataLoader:
        dataset = CustomDataset(self.tokenizer, self.args.max_length, self.args.lang)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        return dataloader
