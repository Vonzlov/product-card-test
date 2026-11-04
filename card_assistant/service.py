"""Общие точки входа для CLI и веб-сервиса."""

import json
from typing import Any

from .config import Settings
from .errors import InputDataError
from .generator import generate_card
from .llm import LLMClient, OpenAICompatibleClient
from .render import render_markdown
from .schemas import load_product


def parse_json(text: str, origin: str = "Входные данные") -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise InputDataError(f"{origin}: некорректный JSON — {e.msg} (строка {e.lineno}, столбец {e.colno}).") from None


def make_llm(settings: Settings) -> LLMClient:
    return OpenAICompatibleClient(settings)


def run(data: Any, llm: LLMClient, settings: Settings) -> dict[str, Any]:
    """Сырые данные → проверенная карточка в двух видах: JSON и Markdown."""
    card = generate_card(load_product(data), llm, settings.max_repairs)
    return {"card": card.model_dump(mode="json"), "markdown": render_markdown(card)}
