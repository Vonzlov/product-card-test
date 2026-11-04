class CardAssistantError(Exception):
    """Базовая ошибка сервиса. Текст ошибки показывается пользователю как есть."""


class InputDataError(CardAssistantError):
    """Входные данные нельзя обработать: битый JSON, не тот тип, нет названия товара."""


class LLMError(CardAssistantError):
    """LLM API недоступен, не настроен или ответил ошибкой."""


class GenerationError(CardAssistantError):
    """Модель так и не вернула ответ в нужном формате."""
