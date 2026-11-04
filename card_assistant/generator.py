import logging
from datetime import datetime, timezone

from pydantic import ValidationError

from . import checks
from .checks import DESCRIPTION_MAX, DESCRIPTION_MIN, QUERIES_MAX, QUERIES_MIN, TITLE_MAX, InputReport, clean_list, fold
from .errors import GenerationError
from .llm import LLMClient
from .prompts import build_messages, build_repair_message
from .schemas import MISSING, CardDraft, Commercial, KeySpec, LLMAnswer, ProductInput

log = logging.getLogger(__name__)


def generate_card(product: ProductInput, llm: LLMClient, max_repairs: int = 2) -> CardDraft:
    """Запрос к модели → автопроверка → до max_repairs исправлений с обратной связью → сборка карточки."""
    report = checks.assess_input(product)
    messages = build_messages(product, report)
    answer, parse_error = None, ""

    for attempt in range(1, max_repairs + 2):
        raw = llm.complete(messages)
        try:
            answer = parse_answer(raw)
        except ValueError as e:
            parse_error = str(e)
            problems = [f"Ответ не разобран как JSON нужной структуры: {e}"]
        else:
            problems = checks.review_answer(answer, product, report)
        log.info("Попытка %d: проблем %d", attempt, len(problems))
        if not problems or attempt > max_repairs:
            break
        messages += [{"role": "assistant", "content": raw}, build_repair_message(problems)]

    if answer is None:
        raise GenerationError(f"Модель не вернула корректный JSON за {attempt} попыток: {parse_error}")
    return assemble(answer, product, report, model=llm.model, attempts=attempt)


def parse_answer(raw: str) -> LLMAnswer:
    # Модели иногда оборачивают JSON в ```json … ``` или добавляют фразу до или после
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("в ответе нет JSON-объекта")
    try:
        return LLMAnswer.model_validate_json(raw[start : end + 1])
    except ValidationError as e:
        raise ValueError("; ".join(f"{'.'.join(map(str, err['loc'])) or 'json'}: {err['msg']}" for err in e.errors()[:5])) from None


def assemble(answer: LLMAnswer, product: ProductInput, report: InputReport, model: str, attempts: int) -> CardDraft:
    """Финальная сборка: подгоняет длины без выдумок, а всё, что не прошло проверку, выносит в warnings."""
    title = checks.shorten_title(answer.seo_title.strip())
    description = checks.trim_to_sentence(answer.seo_description.strip())
    queries = clean_list(answer.search_queries)[:QUERIES_MAX]
    commercial = Commercial(audience=clean_list(answer.commercial.audience), tasks=clean_list(answer.commercial.tasks))
    final = answer.model_copy(update={"seo_title": title, "seo_description": description, "search_queries": queries, "commercial": commercial})

    warnings = []
    if title != answer.seo_title.strip():
        warnings.append(f"SEO-заголовок был длиннее {TITLE_MAX} символов и обрезан автоматически — проверьте формулировку.")
    if description != answer.seo_description.strip():
        warnings.append(f"SEO-описание было длиннее {DESCRIPTION_MAX} символов и обрезано по границе предложения.")
    if len(description) < DESCRIPTION_MIN:
        reason = "во входных данных мало фактов — дополните их" if report.level == "insufficient" else "модель не уложилась в объём — дополните текст"
        warnings.append(f"SEO-описание — {len(description)} символов при норме {DESCRIPTION_MIN}–{DESCRIPTION_MAX}: {reason}.")
    if len(queries) < QUERIES_MIN:
        warnings.append(f"Поисковых запросов {len(queries)} при норме {QUERIES_MIN}–{QUERIES_MAX}.")
    warnings += checks.check_grounding(checks.published_fields(final), product.corpus())

    specs, spec_problems = checks.verify_specs(answer.key_specs, product)
    warnings += [f"Отброшена неподтверждённая характеристика: {p.removeprefix('key_specs: ')}" for p in spec_problems]

    status = "insufficient_data" if report.level == "insufficient" else "needs_review" if warnings else "ok"
    return CardDraft(
        product=product.name,
        manufacturer=product.manufacturer or MISSING,
        status=status,
        seo_title=title,
        seo_description=description,
        key_specs=specs + _missing_specs(product, answer.missing_specs, specs),
        search_queries=queries,
        commercial=commercial,
        clarifications=clean_list(report.gaps + answer.clarifications),
        warnings=warnings,
        meta={
            "model": model,
            "attempts": attempts,
            "input_level": report.level,
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "input_extra": product.extras,
        },
    )


def _missing_specs(product: ProductInput, suggested: list[str], present: list[KeySpec]) -> list[KeySpec]:
    """Пустые значения из входа и важные характеристики, которых нет, — с явной пометкой «данных нет»."""
    known = {fold(spec.name) for spec in present} | {fold(key) for key, value in product.specs.items() if value}
    names = [key for key, value in product.specs.items() if not value] + suggested
    return [KeySpec(name=name, value=MISSING, source="missing") for name in clean_list(names) if fold(name) not in known]
