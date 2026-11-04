import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .errors import CardAssistantError


@dataclass(frozen=True)
class Settings:
    base_url: str
    api_key: str
    model: str
    temperature: float | None
    json_mode: bool
    timeout: float
    max_repairs: int

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            base_url=_env("LLM_BASE_URL", "https://api.openai.com/v1"),
            api_key=_env("LLM_API_KEY"),
            model=_env("LLM_MODEL"),
            temperature=_number("LLM_TEMPERATURE", 0.2, float, allow_empty=True),
            json_mode=_env("LLM_JSON_MODE", "true").lower() not in {"0", "false", "no", "off"},
            timeout=_number("LLM_TIMEOUT", 90, float),
            max_repairs=_number("MAX_REPAIR_ATTEMPTS", 2, int),
        )


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


def _number(name, default, cast, allow_empty=False):
    raw = os.getenv(name)
    if raw is None:
        return default
    if not raw.strip():
        # Пустой LLM_TEMPERATURE= означает «не передавать параметр» (нужно для части reasoning-моделей)
        return None if allow_empty else default
    try:
        return cast(raw)
    except ValueError:
        raise CardAssistantError(f"Переменная {name} должна быть числом, сейчас: {raw!r}") from None
