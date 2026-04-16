"""Unit tests for OllamaClient.

All ollama.Client interactions are mocked — no live Ollama instance required.
"""

from unittest.mock import MagicMock, patch

import httpx
import ollama
import pytest
from pydantic import BaseModel

from src.integrations.ollama_client import OllamaClient, _RETRY_SUFFIX


# ---------------------------------------------------------------------------
# Helper model and factory
# ---------------------------------------------------------------------------


class _SimpleModel(BaseModel):
    value: int
    label: str


def _make_response(content: str) -> MagicMock:
    """Create a mock ollama ChatResponse with the given content string."""
    resp = MagicMock()
    resp.message.content = content
    return resp


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_success_first_attempt(mock_client_class: MagicMock) -> None:
    """Valid JSON returned on first attempt produces the correct model instance."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = _make_response('{"value": 42, "label": "hello"}')

    client = OllamaClient()
    result = client.chat("sys", "usr", _SimpleModel)

    assert result == _SimpleModel(value=42, label="hello")
    assert mock_instance.chat.call_count == 1


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_retries_on_invalid_json(mock_client_class: MagicMock) -> None:
    """Invalid JSON on attempt 1 triggers a retry; valid JSON on attempt 2 succeeds."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.side_effect = [
        _make_response("not json {{{"),
        _make_response('{"value": 7, "label": "retry"}'),
    ]

    client = OllamaClient()
    result = client.chat("sys", "usr", _SimpleModel)

    assert result == _SimpleModel(value=7, label="retry")
    assert mock_instance.chat.call_count == 2

    # Second call must include the retry suffix in the user message
    second_call_messages = mock_instance.chat.call_args_list[1].kwargs["messages"]
    user_message = next(m["content"] for m in second_call_messages if m["role"] == "user")
    assert _RETRY_SUFFIX in user_message


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_retries_on_validation_error(mock_client_class: MagicMock) -> None:
    """JSON with wrong shape on attempt 1 triggers a retry; correct shape on attempt 2 succeeds."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.side_effect = [
        _make_response('{"wrong_field": 1}'),
        _make_response('{"value": 99, "label": "valid"}'),
    ]

    client = OllamaClient()
    result = client.chat("sys", "usr", _SimpleModel)

    assert result == _SimpleModel(value=99, label="valid")
    assert mock_instance.chat.call_count == 2


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_returns_none_after_exhausted_retries(mock_client_class: MagicMock) -> None:
    """All 3 attempts returning invalid JSON causes chat() to return None."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.side_effect = [
        _make_response("bad"),
        _make_response("also bad"),
        _make_response("still bad"),
    ]

    client = OllamaClient()
    result = client.chat("sys", "usr", _SimpleModel)

    assert result is None
    assert mock_instance.chat.call_count == 3


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_raises_on_ollama_not_running(mock_client_class: MagicMock) -> None:
    """httpx.ConnectError is converted to RuntimeError with a clear message."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.side_effect = httpx.ConnectError("Connection refused")

    client = OllamaClient()
    with pytest.raises(RuntimeError, match="Ollama is not running"):
        client.chat("sys", "usr", _SimpleModel)


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_raises_on_model_not_found(mock_client_class: MagicMock) -> None:
    """ollama.ResponseError with 'not found' is converted to RuntimeError with pull hint."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.side_effect = ollama.ResponseError(
        "model 'xyz' not found", status_code=404
    )

    client = OllamaClient()
    with pytest.raises(RuntimeError) as exc_info:
        client.chat("sys", "usr", _SimpleModel)

    message = str(exc_info.value)
    assert "not found" in message
    assert "ollama pull" in message


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_propagates_unexpected_response_error(mock_client_class: MagicMock) -> None:
    """An ollama.ResponseError that is not 'not found' propagates without wrapping."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.side_effect = ollama.ResponseError(
        "some other error", status_code=500
    )

    client = OllamaClient()
    with pytest.raises(ollama.ResponseError):
        client.chat("sys", "usr", _SimpleModel)


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_passes_format_json_schema(mock_client_class: MagicMock) -> None:
    """The ollama.Client.chat call includes the response model's JSON schema as format."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = _make_response('{"value": 1, "label": "x"}')

    client = OllamaClient()
    client.chat("sys", "usr", _SimpleModel)

    call_kwargs = mock_instance.chat.call_args.kwargs
    assert call_kwargs.get("format") == _SimpleModel.model_json_schema()


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_passes_system_and_user_messages(mock_client_class: MagicMock) -> None:
    """The messages list contains exactly one system and one user message with correct content."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = _make_response('{"value": 1, "label": "x"}')

    client = OllamaClient()
    client.chat("my system prompt", "my user prompt", _SimpleModel)

    messages = mock_instance.chat.call_args.kwargs["messages"]
    system_msgs = [m for m in messages if m["role"] == "system"]
    user_msgs = [m for m in messages if m["role"] == "user"]

    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "my system prompt"
    assert len(user_msgs) == 1
    assert user_msgs[0]["content"] == "my user prompt"


@patch("src.integrations.ollama_client.ollama.Client")
def test_chat_uses_model_from_config(mock_client_class: MagicMock) -> None:
    """The model passed to ollama.Client.chat matches the value set in config."""
    mock_instance = MagicMock()
    mock_client_class.return_value = mock_instance
    mock_instance.chat.return_value = _make_response('{"value": 1, "label": "x"}')

    from src.config import Settings

    cfg = Settings()
    cfg.ollama_model = "custom-model:7b"

    client = OllamaClient(cfg=cfg)
    client.chat("sys", "usr", _SimpleModel)

    call_kwargs = mock_instance.chat.call_args.kwargs
    assert call_kwargs.get("model") == "custom-model:7b"
