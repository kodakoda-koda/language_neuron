import pytest
from transformers import AutoTokenizer

from src.dataset import CustomDataset


@pytest.fixture
def tokenizer():
    return AutoTokenizer.from_pretrained("facebook/xglm-564M")


class TestCustomDataset:
    def test_dataset(self, tokenizer):
        dataset = CustomDataset(tokenizer=tokenizer, max_length=512, lang="en")

        assert len(dataset) == 500

        for i in range(len(dataset)):
            batch = dataset[i]
            assert "input_ids" in batch
            assert "attention_mask" in batch
            assert batch["input_ids"].shape == (512,)
            assert batch["attention_mask"].shape == (512,)

    def test_invalid_lang(self, tokenizer):
        with pytest.raises(AssertionError) as e:
            CustomDataset(tokenizer=tokenizer, max_length=512, lang="hi")

        assert str(e.value) == "lang must be one of ['en', 'de', 'fr', 'es', 'zh', 'ja']"
