import argparse
import json
import logging
import os
import random
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from dataset.dataset import CustomDataset
from exp.exp_utils import intervention_indices
from model.bloom import CustomBloomForCausalLM
from model.xglm import CustomXGLMForCausalLM


class Exp_base:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

        self.xglm_list = [
            "facebook/xglm-564M",
            "facebook/xglm-1.7B",
            "facebook/xglm-2.9B",
        ]
        self.bloom_list = [
            "bigscience/bloom-560m",
            "bigscience/bloom-1b7",
            "bigscience/bloom-3b",
        ]
        self.lm_list = self.xglm_list + self.bloom_list

        self.model = self._build_model()
        self.tokenizer = self._bulid_tokenizer()

    def _build_model(self) -> PreTrainedModel:
        assert self.args.lm_name in self.lm_list, f"Invalid model name: {self.args.lm_name}"

        if self.args.lm_name in self.bloom_list:
            model = CustomBloomForCausalLM.from_pretrained(self.args.lm_name, torch_dtype=self.dtype)
        else:
            model = CustomXGLMForCausalLM.from_pretrained(self.args.lm_name, torch_dtype=self.dtype)
        model = model.to(self.device)
        return model

    def _bulid_tokenizer(self) -> PreTrainedTokenizer:
        tokenizer = AutoTokenizer.from_pretrained(self.args.lm_name)
        return tokenizer

    def get_dataloader(self) -> DataLoader:
        lang = ["en", "de", "fr", "es", "zh", "ja"]
        texts = []
        labels = []
        for i, l in enumerate(lang):
            # load paws-x dataset
            pawsx_ = load_dataset("google-research-datasets/paws-x", l)
            pawsx_ = pawsx_["train"]["sentence1"] + pawsx_["test"]["sentence1"] + pawsx_["validation"]["sentence1"]
            pawsx_ = [p for p in pawsx_ if p != ""]

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
            flores_ = [f for f in flores_ if f != ""]

            # sample 250 data from each dataset
            texts_ = random.sample(pawsx_, 250) + random.sample(flores_, 250)
            labels_ = [[int(j == i) for j in range(len(lang))] for _ in range(500)]

            texts.extend(texts_)
            labels.extend(labels_)

        dataset = CustomDataset(texts, labels, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True)
        return dataloader

    def save_outputs(
        self, indices: Dict[str, Dict[str, list]], neurons: np.ndarray, labels: np.ndarray, output_path: str
    ) -> None:
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        if self.args.lm_name in self.bloom_list:
            num_layers = self.model.config.n_layer
            d_model = self.model.config.hidden_size
        else:
            num_layers = self.model.config.num_layers
            d_model = self.model.config.d_model

        lang = ["en", "de", "fr", "es", "zh", "ja"]
        fixed_neurons = []
        neuron_indices = []
        hidden_indices = []
        for l in lang:
            top_bottom_indices = sorted(indices[l]["top"] + indices[l]["bottom"])
            top_bottom_neurons = neurons[labels[:, lang.index(l)] == 1][:, top_bottom_indices]
            fixed_neurons_ = np.median(top_bottom_neurons, axis=0)

            neuron_indices_, hidden_indices_ = intervention_indices(num_layers, d_model, top_bottom_indices)

            fixed_neurons.append(fixed_neurons_)
            neuron_indices.append(neuron_indices_)
            hidden_indices.append(hidden_indices_)

        np.save(os.path.join(output_path, "fixed_neurons.npy"), np.array(fixed_neurons))
        json.dump(neuron_indices, open(os.path.join(output_path, "neuron_indices.json"), "w"))
        json.dump(hidden_indices, open(os.path.join(output_path, "hidden_indices.json"), "w"))

    def plot_indices(self, indices: Dict[str, Dict[str, list]], plot_path: str) -> None:
        """
        Plot histogram of number of top, middle, and bottom neurons per layer for each language.
        Also, plot heatmap of the number of common top and bottom neurons between languages.
        """
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)

        if self.args.lm_name in self.bloom_list:
            num_layers = self.model.config.n_layer
            num_neurons = num_layers * self.model.config.hidden_size * 9
        else:
            num_layers = self.model.config.num_layers
            num_neurons = num_layers * self.model.config.d_model * 9

        lang = ["en", "de", "fr", "es", "zh", "ja"]
        for i, l in enumerate(lang):
            top = indices[l]["top"]
            middle = indices[l]["middle"]
            bottom = indices[l]["bottom"]

            plt.figure(figsize=(15, 5))
            for j, idx in enumerate([top, middle, bottom]):
                plt.subplot(1, 3, j + 1)
                plt.hist(idx, bins=num_layers, range=(0, num_neurons))
                plt.xticks(np.arange(num_layers), np.arange(1, num_layers + 1))
                plt.title(["top", "middle", "bottom"][j])
            plt.suptitle(l)
            plt.savefig(os.path.join(plot_path, f"{l}.png"))
            plt.close()

        corr = np.zeros((len(lang), len(lang)))
        for i, l1 in enumerate(lang):
            for j, l2 in enumerate(lang):
                l1_top_bottom = set(indices[l1]["top"]).union(set(indices[l1]["bottom"]))
                l2_top_bottom = set(indices[l2]["top"]).union(set(indices[l2]["bottom"]))
                corr[i, j] = len(l1_top_bottom.intersection(l2_top_bottom))

        sns.heatmap(corr, annot=True)
        plt.xticks(np.arange(len(lang)), lang)
        plt.yticks(np.arange(len(lang)), lang)
        plt.savefig(os.path.join(plot_path, "heatmap.png"))
        plt.close()
