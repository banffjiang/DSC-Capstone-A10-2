import pytest
import torch
from transformers import AutoTokenizer
from training.pretraining_dpo import collate_sft, tfidf_target
from training.dpo_posttraining import _mask_prompt_labels

@pytest.fixture
def tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("gpt2") #won't have access to ckpts
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

@pytest.fixture
def dummy_idf():
    return {"cat": 2.0, "sat": 1.5, "mat": 1.8, "the": 0.1, "on": 0.2}

@pytest.fixture
def dummy_examples():
    return [
        {"passage": "the cat sat on the mat"},
        {"passage": "the dog ran in the park"}
    ]

class TestMaskPromptLabels:
    def test_prompt_tokens_are_masked(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_lens = torch.tensor([3])
        labels = _mask_prompt_labels(input_ids, prompt_lens, pad_token_id=0)
        assert (labels[0, :3] == -100).all(), "Prompt tokens should be masked"

    def test_target_tokens_are_not_masked(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        prompt_lens = torch.tensor([3])
        labels = _mask_prompt_labels(input_ids, prompt_lens, pad_token_id=0)
        assert (labels[0, 3:] != -100).all(), "Target tokens should not be masked"

    def test_padding_tokens_are_masked(self):
        input_ids = torch.tensor([[1, 2, 0, 0]])
        prompt_lens = torch.tensor([1])
        labels = _mask_prompt_labels(input_ids, prompt_lens, pad_token_id=0)
        assert (labels[0, 2:] == -100).all(), "Padding should be masked"


class TestTfidfTarget:
    def test_returns_string(self, dummy_idf):
        result = tfidf_target("the cat sat on the mat", dummy_idf, top_k=3)
        assert isinstance(result, str)

    def test_respects_top_k(self, dummy_idf):
        result = tfidf_target("the cat sat on the mat", dummy_idf, top_k=2)
        assert len(result.split()) <= 2

    #prevents an error for empty string
    def test_empty_input_returns_fallback(self, dummy_idf):
        result = tfidf_target("", dummy_idf)
        assert result == "something"

    #filter out single character tokens
    def test_filters_short_tokens(self, dummy_idf):
        result = tfidf_target("a b c cat", dummy_idf, top_k=8)
        assert "a" not in result.split()
        assert "b" not in result.split()


class TestCollateSft:
    def test_no_fully_masked_rows(self, tokenizer, dummy_idf, dummy_examples):
        batch = collate_sft(dummy_examples, tokenizer, dummy_idf)
        for i in range(batch["labels"].shape[0]):
            has_target = (batch["labels"][i] != -100).any()
            assert has_target, f"Row {i} has all labels masked"

    def test_output_keys(self, tokenizer, dummy_idf, dummy_examples):
        batch = collate_sft(dummy_examples, tokenizer, dummy_idf)
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert "labels" in batch

    def test_batch_size_preserved(self, tokenizer, dummy_idf, dummy_examples):
        batch = collate_sft(dummy_examples, tokenizer, dummy_idf)
        assert batch["input_ids"].shape[0] == len(dummy_examples)