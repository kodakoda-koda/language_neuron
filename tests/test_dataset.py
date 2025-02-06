import random
from typing import Tuple

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from src.dataset.dataset import CustomDataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/xglm-564M")


class TestCustomDataset:
    def test_dataset(self, tokenizer: AutoTokenizer):
        texts, labels = self._load_data()
        dataset = CustomDataset(texts=texts, labels=labels, tokenizer=tokenizer)

        assert len(dataset) == int(3000)

        for i in range(len(dataset)):
            batch = dataset[i]
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert "labels" in batch
            # assert batch["input_ids"].shape == (512,)
            # assert batch["attention_mask"].shape == (512,)
            assert batch["labels"].shape == (6,)

    def _load_data(self) -> Tuple[list, list]:
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

        return texts, labels
