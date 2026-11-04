from .schemas import CardDraft

STATUS_LABELS = {
    "ok": "✅ черновик готов к вычитке",
    "needs_review": "⚠️ требует внимания: см. «Автоматическая проверка»",
    "insufficient_data": "⛔ мало исходных данных: см. «Что уточнить»",
}
SOURCE_LABELS = {"specs": "характеристики", "name": "название модели", "description": "описание", "missing": "—"}


def render_markdown(card: CardDraft) -> str:
    lines = [
        f"# {card.product}",
        "",
        f"**Производитель:** {card.manufacturer}  ",
        f"**Статус:** {STATUS_LABELS[card.status]}",
        "",
        f"## SEO-заголовок ({len(card.seo_title)}/70)",
        "",
        card.seo_title,
        "",
        f"## SEO-описание ({len(card.seo_description)} симв.)",
        "",
        card.seo_description,
        "",
        "## Ключевые характеристики",
        "",
        "| Характеристика | Значение | Источник |",
        "|---|---|---|",
        *(f"| {_cell(s.name)} | {_cell(s.value)} | {SOURCE_LABELS[s.source]} |" for s in card.key_specs),
        "",
        "## Поисковые запросы",
        "",
        *_bullets(card.search_queries),
        "",
        "## Для менеджера",
        "",
        "**Кому подходит:**",
        "",
        *_bullets(card.commercial.audience),
        "",
        "**Какие задачи решает:**",
        "",
        *_bullets(card.commercial.tasks),
        "",
        "## Что уточнить",
        "",
        *_bullets(card.clarifications),
        "",
        "## Автоматическая проверка",
        "",
        *_bullets(card.warnings, empty="замечаний нет"),
        "",
        f"<sub>Модель: {card.meta.get('model')} · попыток: {card.meta.get('attempts')} · {card.meta.get('generated_at')}</sub>",
        "",
    ]
    return "\n".join(lines)


def _bullets(items: list[str], empty: str = "данных нет") -> list[str]:
    return [f"- {item}" for item in items] or [f"- {empty}"]


def _cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")
