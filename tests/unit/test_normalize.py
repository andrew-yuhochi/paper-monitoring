"""Unit tests for normalize_concept_name."""
import pytest
from src.utils.normalize import normalize_concept_name


@pytest.mark.parametrize("input_name,expected", [
    ("attention mechanism", "attention_mechanism"),
    ("Self-Attention (QKV)", "self_attention_qkv"),
    ("backpropagation", "backpropagation"),
    ("  Variational  AutoEncoder  ", "variational_autoencoder"),
    ("RNN", "rnn"),
    ("Transformer Architecture", "transformer_architecture"),
    ("batch_normalization", "batch_normalization"),
    ("BERT: Pre-training", "bert_pre_training"),
    ("", ""),                             # edge case: empty string
    ("---", ""),                          # edge case: all special chars
])
def test_normalize_concept_name(input_name, expected):
    assert normalize_concept_name(input_name) == expected


def test_node_id_format():
    """Confirm the full concept node ID format used in seeding."""
    name = "attention mechanism"
    node_id = f"concept:{normalize_concept_name(name)}"
    assert node_id == "concept:attention_mechanism"
