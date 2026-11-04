from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import InputDataError

MISSING = "данных нет"

# Дополнительные поля входного JSON, которые полезны модели. Остальные (source, url, ...) только сохраняются.
LLM_EXTRA_FIELDS = ("category",)


# ---------- Входные данные ----------

class ProductInput(BaseModel):
    """Данные о товаре. Лишние поля допускаются и попадают в meta результата."""

    model_config = ConfigDict(extra="allow")

    name: str = ""
    manufacturer: str = ""
    specs: dict[str, str] = Field(default_factory=dict)
    description: str = ""

    @field_validator("name", "manufacturer", "description", mode="before")
    @classmethod
    def _text(cls, value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError("ожидается строка")
        return str(value).strip()

    @field_validator("specs", mode="before")
    @classmethod
    def _specs(cls, value: Any) -> dict[str, str]:
        return parse_specs(value)

    @property
    def extras(self) -> dict[str, Any]:
        return self.model_extra or {}

    def llm_payload(self) -> dict[str, Any]:
        payload = self.model_dump(include={"name", "manufacturer", "specs", "description"})
        payload.update({k: self.extras[k] for k in LLM_EXTRA_FIELDS if isinstance(self.extras.get(k), str)})
        return payload

    def corpus(self) -> str:
        """Весь текст, который видит модель: с ним сверяются числа и термины в ответе."""
        extra = [self.extras[k] for k in LLM_EXTRA_FIELDS if isinstance(self.extras.get(k), str)]
        return "\n".join([self.name, self.manufacturer, *self.specs, *self.specs.values(), self.description, *extra])


def load_product(data: Any) -> ProductInput:
    """Проверяет распарсенный JSON и превращает его в ProductInput с понятными ошибками."""
    if not isinstance(data, dict):
        raise InputDataError("Ожидается JSON-объект с полями name, manufacturer, specs, description.")
    try:
        product = ProductInput.model_validate(data)
    except ValidationError as e:
        details = "; ".join(
            f"{'.'.join(map(str, err['loc'])) or 'данные'}: {err['msg'].removeprefix('Value error, ')}"
            for err in e.errors()
        )
        raise InputDataError(f"Некорректные входные данные — {details}") from None
    if not product.name:
        raise InputDataError("Не заполнено название товара (name): без него карточку не собрать.")
    return product


def parse_specs(value: Any) -> dict[str, str]:
    """Приводит характеристики к плоскому словарю {название: значение}.

    Принимает объект, список пар/объектов {"name", "value"} или текст «характеристика: значение»
    построчно (так их удобно вводить в веб-форме). Вложенные объекты — это подстроки таблицы
    характеристик, они разворачиваются в ключи вида «Погрешность — ТС».
    """
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        pairs = _parse_lines(value)
    elif isinstance(value, dict):
        pairs = value.items()
    elif isinstance(value, list):
        pairs = [_pair(item) for item in value]
    else:
        raise ValueError("ожидается объект {характеристика: значение}, список пар или текст «характеристика: значение»")

    specs: dict[str, str] = {}
    for key, val in pairs:
        key = str(key).strip()
        if not key:
            continue
        if isinstance(val, dict):
            specs.update({f"{key} — {sub_key}": sub_val for sub_key, sub_val in parse_specs(val).items()})
        else:
            specs[key] = _spec_value(val)
    return specs


def _spec_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "да" if value else "нет"
    if isinstance(value, (list, tuple)):
        return "; ".join(filter(None, map(_spec_value, value)))
    return str(value).strip()


def _pair(item: Any) -> tuple[Any, Any]:
    if isinstance(item, dict):
        lowered = {str(k).lower(): v for k, v in item.items()}
        for key_field, value_field in (("name", "value"), ("key", "value"), ("название", "значение")):
            if key_field in lowered:
                return lowered[key_field], lowered.get(value_field)
        if len(item) == 1:
            return next(iter(item.items()))
    elif isinstance(item, (list, tuple)) and len(item) == 2:
        return item[0], item[1]
    raise ValueError(f"не удалось разобрать характеристику {item!r}")


def _parse_lines(text: str) -> list[tuple[str, str]]:
    pairs = []
    for line in text.splitlines():
        line = line.strip().lstrip("-•*").strip()
        if not line:
            continue
        for sep in (":", "\t", " — ", " – ", " - "):
            if sep in line:
                key, val = line.split(sep, 1)
                pairs.append((key, val))
                break
        else:
            raise ValueError(f"в строке «{line}» нет разделителя «:» между названием и значением")
    return pairs


# ---------- Ответ модели (как его просим в промпте) ----------

class _Lenient(BaseModel):
    model_config = ConfigDict(extra="ignore", coerce_numbers_to_str=True)


class LLMSpec(_Lenient):
    name: str = ""
    value: str = ""
    source: str = "specs"
    evidence: str = ""

    @field_validator("name", "value", "source", "evidence", mode="before")
    @classmethod
    def _none_to_empty(cls, value: Any) -> Any:
        return "" if value is None else value


class Commercial(_Lenient):
    audience: list[str] = Field(default_factory=list)
    tasks: list[str] = Field(default_factory=list)


class LLMAnswer(_Lenient):
    seo_title: str
    seo_description: str
    key_specs: list[LLMSpec] = Field(default_factory=list)
    missing_specs: list[str] = Field(default_factory=list)
    search_queries: list[str] = Field(default_factory=list)
    commercial: Commercial = Field(default_factory=Commercial)
    clarifications: list[str] = Field(default_factory=list)


# ---------- Результат сервиса ----------

class KeySpec(BaseModel):
    name: str
    value: str
    source: Literal["specs", "name", "description", "missing"]


class CardDraft(BaseModel):
    product: str
    manufacturer: str
    status: Literal["ok", "needs_review", "insufficient_data"]
    seo_title: str
    seo_description: str
    key_specs: list[KeySpec]
    search_queries: list[str]
    commercial: Commercial
    clarifications: list[str]
    warnings: list[str]
    meta: dict[str, Any]
