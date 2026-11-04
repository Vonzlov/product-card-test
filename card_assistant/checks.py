"""Детерминированные проверки: полнота входных данных и «заземлённость» ответа модели.

Промпт просит модель ничего не выдумывать, а этот модуль проверяет, что она послушалась:
числа и рискованные термины в текстах сверяются с входными данными, а значения
характеристик — с конкретным ключом specs или дословной цитатой.
"""

import re
from dataclasses import dataclass, field
from typing import Literal

from .schemas import KeySpec, LLMAnswer, LLMSpec, ProductInput

TITLE_MAX = 70
DESCRIPTION_MIN, DESCRIPTION_MAX = 800, 1200
QUERIES_MIN, QUERIES_MAX = 3, 5
SHORT_DESCRIPTION = 150

_DASHES = dict.fromkeys(map(ord, "‐‑‒–—−"), "-")
_SPACES = dict.fromkeys(map(ord, "    "), " ")
# Кириллические буквы, которые в каталогах путают с латинскими («Сu», «М20х1,5»)
_LOOKALIKES = str.maketrans("аеокрсхумтнв", "aeokpcxymthb")
_NUM = r"\d+(?:[.,]\d+)?"
_NUMBER = re.compile(_NUM)
_ALTERNATIVES = re.compile(r"[;|]|\s/\s|\sили\s")  # «1,5; 1,0», «0…1 / 1,6 / 2,5», «5 А или 30 А»
# Число или перечень/диапазон чисел и единица после него: «96…264 В», «от −60 до +60 °C», «0…1 / 1,6 кгс/см²»
_MEASURE = re.compile(rf"({_NUM}(?:\s*(?:\.\.\.|-|/|;|,|до|x|х)\s*[+-]?{_NUM})*)\s*(°\s*[CcСс]?|%|[A-Za-zА-Яа-я]+)?")
_UNITS = {
    "мпа": "МПа", "mpa": "МПа", "кпа": "кПа", "kpa": "кПа", "па": "Па", "pa": "Па", "бар": "бар", "bar": "бар",
    "мбар": "мбар", "кгс": "кгс/см²", "атм": "атм", "psi": "psi",
    "ма": "мА", "ma": "мА", "мка": "мкА", "мв": "мВ", "mv": "мВ", "кв": "кВ", "вт": "Вт", "w": "Вт", "квт": "кВт",
    "ом": "Ом", "ком": "кОм", "мом": "МОм", "гц": "Гц", "hz": "Гц", "кгц": "кГц",
    "мм": "мм", "mm": "мм", "см": "см", "м": "м", "мкм": "мкм", "км": "км",
    "сек": "с", "мс": "мс", "мин": "мин", "кг": "кг", "kg": "кг", "дб": "дБ", "ppm": "ppm", "%": "%",
}
# Однобуквенные единицы считаем только заглавными: строчные «в», «а», «с» в тексте — предлоги и союзы
_UPPER_UNITS = {"В": "В", "V": "В", "А": "А", "A": "А", "C": "°C", "С": "°C"}

# Сведения, которые модели любят «додумывать». Упоминать можно, только если они есть во входных данных.
CLAIM_PATTERNS = {
    "гарантия": r"гаранти",
    "сертификат": r"сертифик",
    "реестр СИ": r"реестр",
    "поверка": r"поверк|поверочн|\bповерен",
    "ГОСТ": r"\bгост\b",
    "взрывозащита": r"взрывозащ",
    "RS-485": r"rs-?485",
    "Modbus": r"modbus",
    "Wi-Fi": r"wi-?fi",
    "Bluetooth": r"bluetooth",
    "USB": r"\busb\b",
    "HART": r"\bhart\b",
    "Ethernet": r"ethernet",
    "нержавеющая сталь": r"нержаве",
    "латунь": r"латун",
}
# Цена, наличие и магазин — не свойства товара, в карточке им не место, даже если они есть в описании
STORE_PATTERN = re.compile(
    r"₽|\bруб(?:л[а-я]*)?\b|\bцен(?:а|ы|е|у|ой|ам|ами|ах)?\b|\bстоимост|\bскидк|\bдоставк|\bв наличии\b"
    r"|союз-прибор|интернет-магазин"
)


# ---------- Нормализация ----------

def _unify(text: str) -> str:
    """Тире, неразрывные пробелы, многоточие, знак умножения, степени — без смены регистра."""
    text = text.replace("ё", "е").replace("Ё", "Е").translate(_DASHES).translate(_SPACES)
    text = text.replace("…", "...").replace("×", "x").replace("²", "2").replace("³", "3")
    return re.sub(r"\s+", " ", text).strip()


def simplify(text: str) -> str:
    return _unify(text).casefold()


def fold(text: str) -> str:
    """simplify + сведение похожих кириллических букв к латинице — для сравнения строк."""
    return simplify(text).translate(_LOOKALIKES)


def numbers(text: str) -> set[str]:
    """Все числа в тексте в каноническом виде: «1,50» и «1.5» — одно и то же число."""
    return {_canonical(n) for n in _NUMBER.findall(simplify(text))}


def measures(text: str) -> set[tuple[str, str]]:
    """Пары (число, единица). Единица перечня или диапазона относится ко всем его числам."""
    result = set()
    for line in text.splitlines():  # число в конце одной строки не должно «склеиться» с единицей из следующей
        for sequence, token in _MEASURE.findall(_unify(line)):
            unit = "°C" if token.startswith("°") else _UPPER_UNITS.get(token) or _UNITS.get(token.lower(), "")
            if unit:
                result |= {(number, unit) for number in numbers(sequence)}
    return result


def _canonical(number: str) -> str:
    whole, _, frac = number.replace(",", ".").partition(".")
    whole, frac = whole.lstrip("0") or "0", frac.rstrip("0")
    return f"{whole}.{frac}" if frac else whole


def clean_list(items: list[str]) -> list[str]:
    """Убирает пустые строки и дубли без учёта регистра, сохраняя порядок."""
    seen, result = set(), []
    for item in (i.strip() for i in items):
        if item and fold(item) not in seen:
            seen.add(fold(item))
            result.append(item)
    return result


# ---------- Входные данные ----------

@dataclass
class InputReport:
    level: Literal["ok", "partial", "insufficient"]
    gaps: list[str] = field(default_factory=list)


def assess_input(product: ProductInput) -> InputReport:
    gaps = []
    if not product.manufacturer:
        gaps.append("Не указан производитель (manufacturer).")
    if not product.specs:
        gaps.append("Не переданы характеристики (specs).")
    empty = [key for key, value in product.specs.items() if not value]
    if empty:
        gaps.append("Пустые значения характеристик: " + ", ".join(f"«{key}»" for key in empty) + ".")
    if not product.description:
        gaps.append("Нет описания (description).")
    elif len(product.description) < SHORT_DESCRIPTION:
        gaps.append(f"Описание короткое ({len(product.description)} символов) — фактов для SEO-текста мало.")

    has_specs = any(product.specs.values())
    if not has_specs and len(product.description) < SHORT_DESCRIPTION:
        return InputReport("insufficient", gaps)
    return InputReport("partial" if gaps else "ok", gaps)


# ---------- Ответ модели ----------

def review_answer(answer: LLMAnswer, product: ProductInput, report: InputReport) -> list[str]:
    """Список проблем ответа; пустой список — ответ годен."""
    problems = []
    title, description = answer.seo_title.strip(), answer.seo_description.strip()
    if not title:
        problems.append("seo_title пустой.")
    elif len(title) > TITLE_MAX:
        problems.append(f"seo_title: {len(title)} символов при лимите {TITLE_MAX}.")

    # Если данных мало, короткое описание лучше раздутого: длину снизу тогда не требуем
    too_short = len(description) < DESCRIPTION_MIN and report.level != "insufficient"
    if too_short or len(description) > DESCRIPTION_MAX:
        problems.append(f"seo_description: {len(description)} символов при норме {DESCRIPTION_MIN}–{DESCRIPTION_MAX}.")

    queries = clean_list(answer.search_queries)
    if not QUERIES_MIN <= len(queries) <= QUERIES_MAX:
        problems.append(f"search_queries: {len(queries)} шт. при норме {QUERIES_MIN}–{QUERIES_MAX}.")

    problems += check_grounding(published_fields(answer), product.corpus())
    return problems + verify_specs(answer.key_specs, product)[1]


def published_fields(answer: LLMAnswer) -> dict[str, str]:
    """Тексты, которые уйдут на сайт или менеджеру, — их и проверяем на выдумки."""
    return {
        "seo_title": answer.seo_title,
        "seo_description": answer.seo_description,
        "search_queries": "\n".join(answer.search_queries),
        "commercial.audience": "\n".join(answer.commercial.audience),
        "commercial.tasks": "\n".join(answer.commercial.tasks),
    }


def check_grounding(fields: dict[str, str], corpus: str) -> list[str]:
    allowed, allowed_measures = numbers(corpus), measures(corpus)
    known_claims = {label for label, pattern in CLAIM_PATTERNS.items() if re.search(pattern, simplify(corpus))}
    problems = []
    for name, raw in fields.items():
        text = simplify(raw)
        unknown = sorted(numbers(text) - allowed, key=float)
        # Число может встречаться во входных данных, но с другой единицей: «20» из «M20x1,5» — не «20 мА»
        mismatched = sorted(f"{n} {u}" for n, u in measures(raw) - allowed_measures if n in allowed)
        if unknown or mismatched:
            problems.append(f"{name}: значения {', '.join(unknown + mismatched)} не найдены во входных данных.")
        terms = [label for label, p in CLAIM_PATTERNS.items() if label not in known_claims and re.search(p, text)]
        if terms:
            problems.append(f"{name}: упоминается {', '.join(terms)} — во входных данных этого нет.")
        if STORE_PATTERN.search(text):
            problems.append(f"{name}: упоминаются цена, наличие, доставка или магазин — это не свойства товара.")
    return problems


def verify_specs(items: list[LLMSpec], product: ProductInput) -> tuple[list[KeySpec], list[str]]:
    """Каждая характеристика из ответа должна ссылаться на ключ specs или дословную цитату."""
    by_key = {fold(key): key for key in product.specs}
    texts = {"name": product.name, "description": product.description}
    kept: list[KeySpec] = []
    problems = []

    for item in items:
        name, value, evidence = item.name.strip(), item.value.strip(), item.evidence.strip()
        if item.source == "specs":
            key = by_key.get(fold(evidence)) or by_key.get(fold(name))
            if key is None:
                problems.append(f"key_specs: «{evidence or name}» — такой характеристики нет в specs.")
                continue
            original = product.specs[key]
            if not original:
                continue  # пустые значения из входа добавятся отдельно с пометкой «данных нет»
            kept.append(KeySpec(name=name or key, value=_spec_value(value, original, product.name), source="specs"))
        elif item.source in texts:
            evidence = evidence or value
            if not name or not value:
                problems.append(f"key_specs: у характеристики из {item.source} нет названия или значения.")
            elif fold(evidence) not in fold(texts[item.source]):
                problems.append(f"key_specs: цитата «{evidence}» не найдена дословно в поле {item.source}.")
            elif not supported(value, evidence):
                problems.append(f"key_specs: значение «{value}» для «{name}» не подтверждается цитатой «{evidence}».")
            else:
                kept.append(KeySpec(name=name, value=value, source=item.source))
        else:
            problems.append(f"key_specs: неизвестный источник «{item.source}» у «{name}» (допустимо specs, name, description).")

    unique = {(fold(s.name), fold(s.value)): s for s in kept}
    return list(unique.values()), problems


def _spec_value(value: str, original: str, product_name: str) -> str:
    """Модель может взять часть значения (например, разделить две температуры), но не переписать его.

    Если значение — перечень вариантов серии («1,5; 1,0»), выбрать один можно только когда он явно
    записан в названии модели (как «G1/2» в «ТМ-510Р.00(0-2,5 МПа)G1/2»). Иначе — исходное значение целиком.
    """
    if not value or fold(value) not in fold(original):
        return original
    is_part = fold(value) != fold(original)
    if is_part and _ALTERNATIVES.search(original) and fold(value) not in fold(product_name):
        return original
    return value


def supported(value: str, evidence: str) -> bool:
    """Значение подтверждается цитатой: совпадают все числа и (с точностью до окончаний) все слова."""
    if fold(value) in fold(evidence):
        return True
    if not (numbers(value) <= numbers(evidence) and measures(value) <= measures(evidence)):
        return False
    evidence_words = re.findall(r"[a-zа-я]+", fold(evidence))
    return all(
        any(word.startswith(token[:5]) for word in evidence_words)
        for token in re.findall(r"[a-zа-я]+", fold(value))
        if len(token) >= 3
    )


# ---------- Финальная подгонка ----------

def shorten_title(title: str, limit: int = TITLE_MAX) -> str:
    """Обрезает заголовок по смысловой границе («—», «|», запятая), а если её нет — по слову."""
    if len(title) <= limit:
        return title
    head = title[: limit + 1]
    boundary = max(head.rfind(sep) for sep in (" — ", " – ", " - ", " | ", ", ", " для "))
    if boundary >= limit // 2:
        return head[:boundary].rstrip(" ,;:-–—(")
    cut = head.rsplit(" ", 1)[0] if " " in head[:limit] else title[:limit]
    return cut.rstrip(" ,;:-–—(")


def trim_to_sentence(text: str, limit: int = DESCRIPTION_MAX) -> str:
    """Обрезает текст по последнему концу предложения, не превышая лимит. Ничего не дописывает."""
    if len(text) <= limit:
        return text
    head = text[:limit]
    end = max(head.rfind(mark) for mark in (". ", "! ", "? ", ".\n"))
    if end > limit * 0.6:
        return head[: end + 1].rstrip()
    return head.rsplit(" ", 1)[0].rstrip(" ,;:-–—") + "…"
