import pytest
import torch
from transformers import XGLMConfig

from src.model.xglm import CustomXGLMAttention, CustomXGLMDecoderLayer, CustomXGLMForCausalLM, CustomXGLMModel


@pytest.fixture
def device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    return torch.bfloat16 if torch.cuda.is_available() else torch.float32


@pytest.fixture
def config() -> XGLMConfig:
    return XGLMConfig.from_pretrained("facebook/xglm-564M")


class TestCustomXGLM:
    def test_CustomXGLMAttention(self, config: XGLMConfig, device: torch.device, dtype: torch.dtype):
        attention = CustomXGLMAttention(
            embed_dim=config.d_model,
            num_heads=config.attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
            bias=True,
        ).to(device=device, dtype=dtype)

        x = torch.randn(10, 32, config.d_model).to(device=device, dtype=dtype)
        mask = (torch.randn(10, 1, 32, 32) > 0.5).to(device=device, dtype=dtype)
        attn_output, _, _, attn_neurons = attention(hidden_states=x, attention_mask=mask, output_neurons=True)

        assert attn_output.size() == (10, 32, config.d_model)
        assert attn_neurons.size() == (10, 32, config.d_model * 4)

    def test_CustomXGLMDecoderLayer(self, config: XGLMConfig, device: torch.device, dtype: torch.dtype):
        decoder_layer = CustomXGLMDecoderLayer(config).to(device=device, dtype=dtype)

        x = torch.randn(10, 32, config.d_model).to(device=device, dtype=dtype)
        mask = (torch.randn(10, 1, 32, 32) > 0.5).to(device=device, dtype=dtype)
        outputs = decoder_layer(hidden_states=x, attention_mask=mask, output_neurons=True)

        assert outputs[0].size() == (10, 32, config.d_model)
        assert outputs[2].size() == (10, config.d_model * 9)

    def test_CustomXGLMModel(self, config, device, dtype):
        model = CustomXGLMModel(config).to(device=device, dtype=dtype)

        input_ids = torch.randint(0, config.vocab_size, (10, 32)).to(device=device)
        attention_mask = (torch.randn(10, 32) > 0.5).to(device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=True)

        assert outputs.last_hidden_state.size() == (10, 32, config.d_model)
        assert len(outputs.neurons) == config.num_layers
        assert outputs.neurons[0].size() == (10, config.d_model * 9)

    def test_CustomXGLMForCausalLM(self, config, device, dtype):
        model = CustomXGLMForCausalLM(config).to(device=device, dtype=dtype)

        input_ids = torch.randint(0, config.vocab_size, (10, 32)).to(device=device)
        attention_mask = (torch.randn(10, 32) > 0.5).to(device=device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_neurons=True)

        assert outputs.logits.size() == (10, 32, config.vocab_size)
        assert len(outputs.neurons) == config.num_layers
        assert outputs.neurons[0].size() == (10, config.d_model * 9)
