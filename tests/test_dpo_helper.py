import pytest
import torch
from transformers import AutoTokenizer
from training.dpo_posttraining import _completion_lengths, _anneal_alpha_by_steps
from training.pretraining_dpo import collate_lm

@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.fixture(scope="module")
def tiny_model_and_tokenizer():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    config = AutoConfig.from_pretrained("gpt2")
    config.n_layer = 1
    config.n_head = 1
    config.n_embd = 64
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    return model, tokenizer

#test helper methods for dpo
class TestCompletionLengths:
    def test_completion_lengths_counts_non_masked(self):
        labels = torch.tensor([[1, 2, -100, -100], [1, -100, -100, -100]])
        result = _completion_lengths(labels)
        assert result.tolist() == [2, 1]

class TestAnnealAlphaBySteps:
    def test_anneal_alpha_by_steps_zero_steps_returns_alpha0(self):
        result = _anneal_alpha_by_steps(0, 0, alpha0=0.1, k=5)
        assert result == 0.1

    def test_anneal_alpha_by_steps_decreases(self):
        a1 = _anneal_alpha_by_steps(100, 1000, alpha0=0.1, k=5)
        a2 = _anneal_alpha_by_steps(900, 1000, alpha0=0.1, k=5)
        assert a1 > a2

class TestCollateLM:
    def test_collate_lm_adds_eos(self, tokenizer):
        batch = collate_lm(tokenizer, ["hello world"], max_length=32)
        eos_id = tokenizer.eos_token_id
        assert eos_id in batch["input_ids"][0].tolist()

    def test_collate_lm_pads_are_masked(self, tokenizer):
        batch = collate_lm(tokenizer, ["hi", "hello world this is longer"], max_length=32)
        for i in range(batch["labels"].shape[0]):
            attn = batch["attention_mask"][i]
            pad_positions = (attn == 0) 
            if pad_positions.any():
                assert (batch["labels"][i][pad_positions] == -100).all()

    def test_collate_lm_output_keys(self, tokenizer):
        batch = collate_lm(tokenizer, ["hello"], max_length=32)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch