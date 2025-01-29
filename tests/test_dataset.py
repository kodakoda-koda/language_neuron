import random

import pytest
from datasets import load_dataset
from transformers import AutoTokenizer

from src.dataset import CustomDataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/xglm-564M")


class TestCustomDataset:
    def test_dataset(self, tokenizer):
        data = self._load_data("en")
        dataset = CustomDataset(data=data, tokenizer=tokenizer, max_length=512, train_flag=True)

        assert len(dataset) == int(500 * 0.8)

        for i in range(len(dataset)):
            batch = dataset[i]
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert batch["input_ids"].shape == (512,)
            assert batch["attention_mask"].shape == (512,)

    def _load_data(self, lang):
        pawsx_ = load_dataset("google-research-datasets/paws-x", lang)
        pawsx = pawsx_["train"]["sentence1"] + pawsx_["test"]["sentence1"] + pawsx_["validation"]["sentence1"]

        # load flores200 dataset
        flores_lang_set = {
            "en": "eng_Latn-jpn_Jpan",
            "de": "deu_Latn-jpn_Jpan",
            "fr": "fra_Latn-jpn_Jpan",
            "es": "spa_Latn-jpn_Jpan",
            "zh": "zho_Hans-jpn_Jpan",
            "ja": "jpn_Jpan-eng_Latn",
        }
        flores_lang = flores_lang_set[lang]
        flores_ = load_dataset("facebook/flores", flores_lang)
        flores = (
            flores_["dev"][f"sentence_{flores_lang.split('-')[0]}"]
            + flores_["devtest"][f"sentence_{flores_lang.split('-')[0]}"]
        )

        # sample 250 data from each dataset
        data = random.sample(pawsx, 250) + random.sample(flores, 250)
        return data
