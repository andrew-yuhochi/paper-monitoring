"""Low-level Ollama API wrapper with JSON retry logic.

This is the single point where ollama.Client.chat() is called.
All retry logic lives here; callers (OllamaClassifier) never call ollama directly.
"""

import json
import logging
from typing import Type, TypeVar

import httpx
import ollama
from pydantic import BaseModel, ValidationError

from src.config import Settings, settings as default_settings

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_RETRY_SUFFIX = "\nYour previous response was not valid JSON. Return ONLY a JSON object with the specified fields."


class OllamaClient:
    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or default_settings
        self._client = ollama.Client(
            host=self._cfg.ollama_host,
            timeout=self._cfg.ollama_timeout,
        )

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
    ) -> T | None:
        """Call Ollama with JSON mode and validate the response against response_model.

        Retries up to config.ollama_max_retries times (default 3) on JSON parse
        failure or Pydantic validation failure. On each retry the correction suffix
        is appended to the user prompt.

        Returns a validated instance of response_model, or None if all retries fail.

        Raises:
            RuntimeError: If Ollama is not running (connection refused).
            RuntimeError: If the configured model is not found (not pulled).
        """
        current_user_prompt = user_prompt
        raw_response = ""

        for attempt in range(1, self._cfg.ollama_max_retries + 1):
            try:
                response = self._client.chat(
                    model=self._cfg.ollama_model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": current_user_prompt},
                    ],
                    format=response_model.model_json_schema(),
                )
                raw_response = response.message.content
                data = json.loads(raw_response)
                return response_model.model_validate(data)

            except httpx.ConnectError as exc:
                raise RuntimeError(
                    "Ollama is not running. Start with: ollama serve"
                ) from exc

            except ollama.ResponseError as exc:
                msg = str(exc).lower()
                if "not found" in msg:
                    raise RuntimeError(
                        f"Model {self._cfg.ollama_model!r} not found. "
                        f"Pull with: ollama pull {self._cfg.ollama_model}"
                    ) from exc
                raise  # other ResponseError — don't retry, propagate

            except (json.JSONDecodeError, ValidationError) as exc:
                logger.warning(
                    "Ollama attempt %d/%d failed (%s: %s). Retrying with correction prompt.",
                    attempt,
                    self._cfg.ollama_max_retries,
                    type(exc).__name__,
                    exc,
                )
                current_user_prompt += _RETRY_SUFFIX

        logger.warning(
            "Ollama chat failed after %d attempts. Raw response: %r",
            self._cfg.ollama_max_retries,
            raw_response,
        )
        return None
