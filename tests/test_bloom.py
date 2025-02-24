import pytest
import torch
from transformers import BloomConfig

from src.model.bloom import (
    CustomBloomAttention,
    CustomBloomBlock,
    CustomBloomForCausalLM,
    CustomBloomMLP,
    CustomBloomModel,
)


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


@pytest.fixture
def config() -> BloomConfig:
    return BloomConfig.from_pretrained("bigscience/bloom-560m")


class TestCustomBloom:
    def test_CustomBloomAttention(self, config: BloomConfig, device: torch.device, dtype: torch.dtype):
        attention = CustomBloomAttention(config, 1).to(device=device, dtype=dtype)

        x = torch.randn(10, 32, config.hidden_size).to(device=device, dtype=dtype)
        alibi = torch.randn(10 * config.n_head, 1, 32).to(device=device, dtype=dtype)
        mask = (torch.randn(10, 1, 32, 32) > 0.5).to(device=device, dtype=dtype)

        attn_output, _, attn_neurons = attention(
            hidden_states=x,
            alibi=alibi,
            residual=x,
            attention_mask=mask,
            output_neurons=True,
        )
        assert attn_output.size() == (10, 32, config.hidden_size)
        assert attn_neurons.size() == (10, 32, config.hidden_size * 4)

        attn_output, _ = attention(
            hidden_states=x,
            alibi=alibi,
            residual=x,
            attention_mask=mask,
            output_neurons=False,
        )
        assert attn_output.size() == (10, 32, config.hidden_size)

    def test_CustomBloomMLP(self, config: BloomConfig, device: torch.device, dtype: torch.dtype):
        mlp = CustomBloomMLP(config).to(device=device, dtype=dtype)

        x = torch.randn(10, 32, config.hidden_size).to(device=device, dtype=dtype)

        mlp_output, mlp_neurons = mlp(
            hidden_states=x,
            residual=x,
            output_neurons=True,
        )
        assert mlp_output.size() == (10, 32, config.hidden_size)
        assert mlp_neurons.size() == (10, 32, config.hidden_size * 5)

        mlp_output, _ = mlp(
            hidden_states=x,
            residual=x,
            output_neurons=False,
        )
        assert mlp_output.size() == (10, 32, config.hidden_size)

    def test_CustomBloomBlock(self, config: BloomConfig, device: torch.device, dtype: torch.dtype):
        block = CustomBloomBlock(config, 1).to(device=device, dtype=dtype)

        x = torch.randn(10, 32, config.hidden_size).to(device=device, dtype=dtype)
        alibi = torch.randn(10 * config.n_head, 1, 32).to(device=device, dtype=dtype)
        mask = (torch.randn(10, 1, 32, 32) > 0.5).to(device=device, dtype=dtype)

        block_output, block_neurons = block(
            hidden_states=x,
            alibi=alibi,
            attention_mask=mask,
            output_neurons=True,
        )
        assert block_output.size() == (10, 32, config.hidden_size)
        assert block_neurons.size() == (10, 32, config.hidden_size * 9)

        block_output, _ = block(
            hidden_states=x,
            alibi=alibi,
            attention_mask=mask,
            output_neurons=False,
        )
        assert block_output.size() == (10, 32, config.hidden_size)

    def test_CustomBloomModel(self, config, device, dtype):
        model = CustomBloomModel(config).to(device=device, dtype=dtype)

        input_ids = torch.randint(0, config.vocab_size, (10, 32)).to(device=device)
        attention_mask = (torch.randn(10, 32) > 0.5).to(device=device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=True)
        assert outputs.last_hidden_state.size() == (10, 32, config.hidden_size)
        assert outputs.neurons.size() == (10, config.hidden_size * 9 * config.n_layer)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=False)
        assert outputs.last_hidden_state.size() == (10, 32, config.hidden_size)

    def test_CustomXGLMForCausalLM(self, config, device, dtype):
        model = CustomBloomForCausalLM(config).to(device=device, dtype=dtype)

        input_ids = torch.randint(0, config.vocab_size, (10, 32)).to(device=device)
        attention_mask = (torch.randn(10, 32) > 0.5).to(device=device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=True)
        assert outputs.logits.size() == (10, 32, config.vocab_size)
        assert outputs.neurons.size() == (10, config.hidden_size * 9 * config.n_layer)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=False)
        assert outputs.logits.size() == (10, 32, config.vocab_size)
