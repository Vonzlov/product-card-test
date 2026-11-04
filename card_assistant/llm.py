import logging
from typing import Protocol
from urllib.parse import urlparse

import openai

from .config import Settings
from .errors import LLMError

log = logging.getLogger(__name__)

Message = dict[str, str]


class LLMClient(Protocol):
    model: str

    def complete(self, messages: list[Message]) -> str: ...


class OpenAICompatibleClient:
    """Клиент для любого OpenAI-совместимого API: OpenAI, DeepSeek, OpenRouter, Ollama, vLLM, LM Studio."""

    def __init__(self, settings: Settings):
        if not settings.model:
            raise LLMError("Не задана модель: укажите LLM_MODEL в .env (см. .env.example).")
        api_key = settings.api_key or ("local" if _is_local(settings.base_url) else "")
        if not api_key:
            raise LLMError("Не задан API-ключ: укажите LLM_API_KEY в .env (для локальной Ollama ключ не нужен).")

        self.model = settings.model
        self._base_url = settings.base_url
        # SDK сам повторяет запрос при обрывах связи, 429 и 5xx с экспоненциальной паузой
        self._client = openai.OpenAI(base_url=settings.base_url, api_key=api_key, timeout=settings.timeout, max_retries=2)
        self._params: dict = {}
        if settings.temperature is not None:
            self._params["temperature"] = settings.temperature
        if settings.json_mode:
            self._params["response_format"] = {"type": "json_object"}

    def complete(self, messages: list[Message]) -> str:
        try:
            response = self._create(messages)
        except openai.AuthenticationError:
            raise LLMError("LLM API отклонил ключ (401): проверьте LLM_API_KEY.") from None
        except openai.PermissionDeniedError as e:
            raise LLMError(f"Нет доступа к модели или API (403): {_detail(e)}") from None
        except openai.NotFoundError:
            raise LLMError(f"Модель «{self.model}» или адрес {self._base_url} не найдены (404): проверьте LLM_MODEL и LLM_BASE_URL.") from None
        except openai.RateLimitError as e:
            raise LLMError(f"Превышен лимит запросов или закончилась квота (429): {_detail(e)}") from None
        except openai.APITimeoutError:
            raise LLMError("LLM не ответила вовремя: попробуйте ещё раз или увеличьте LLM_TIMEOUT.") from None
        except openai.APIConnectionError:
            raise LLMError(f"Нет соединения с LLM API ({self._base_url}): проверьте адрес, сеть или VPN.") from None
        except openai.APIStatusError as e:
            raise LLMError(f"LLM API вернул ошибку {e.status_code}: {_detail(e)}") from None

        content = response.choices[0].message.content if response.choices else None
        if not content or not content.strip():
            raise LLMError("LLM вернула пустой ответ.")
        return content

    def _create(self, messages: list[Message]):
        try:
            return self._client.chat.completions.create(model=self.model, messages=messages, **self._params)
        except openai.BadRequestError as e:
            # Часть моделей не принимает temperature (reasoning-модели) или response_format —
            # убираем отвергнутый параметр и повторяем запрос один раз.
            rejected = [name for name in self._params if name in str(e)]
            if not rejected:
                raise
            log.warning("Модель не поддерживает %s — повторяю запрос без них", ", ".join(rejected))
            for name in rejected:
                self._params.pop(name)
            return self._client.chat.completions.create(model=self.model, messages=messages, **self._params)


def _is_local(url: str) -> bool:
    return urlparse(url).hostname in {"localhost", "127.0.0.1", "0.0.0.0", "host.docker.internal"}


def _detail(error: openai.APIStatusError) -> str:
    body = error.body if isinstance(error.body, dict) else {}
    nested = body.get("error")
    message = nested.get("message") if isinstance(nested, dict) else body.get("message")
    return str(message or error.message)[:300]
