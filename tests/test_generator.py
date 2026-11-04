import json

import pytest

from card_assistant.errors import GenerationError
from card_assistant.generator import generate_card
from card_assistant.render import render_markdown
from card_assistant.schemas import MISSING

from .conftest import FakeLLM, manometer_answer


def test_good_answer_is_accepted_in_one_attempt(manometer):
    llm = FakeLLM(manometer_answer())
    card = generate_card(manometer, llm)

    assert card.status == "ok" and card.warnings == [] and card.meta["attempts"] == 1
    assert 800 <= len(card.seo_description) <= 1200 and len(card.seo_title) <= 70
    values = {s.name: s.value for s in card.key_specs}
    assert values["Класс точности"] == "1,5; 1,0"  # модель выбрала «1,5», но вариант серии не подтверждён названием
    assert values["Присоединительная резьба"] == "G1/2"
    assert values["Тип присоединения (радиальное/осевое) для этой модификации"] == MISSING
    assert "Класс точности" in values and list(values).count("Класс точности") == 1  # уже есть — не дублируется как «данных нет»


def test_problems_are_sent_back_for_repair(manometer):
    bad = manometer_answer(
        seo_title="Манометр показывающий ТМ-510Р.00 0–2,5 МПа G1/2 РОСМА — купить по выгодной цене с доставкой",
        commercial={"audience": ["АСУ ТП"], "tasks": ["передача сигнала 4–20 мА по RS-485"]},
    )
    llm = FakeLLM(bad, manometer_answer())
    card = generate_card(manometer, llm)

    assert card.status == "ok" and card.meta["attempts"] == 2
    repair_request = llm.calls[1][-1]["content"]
    assert "seo_title" in repair_request and "при лимите 70" in repair_request
    assert "20 мА" in repair_request and "RS-485" in repair_request
    assert llm.calls[1][-2]["role"] == "assistant"  # модели показываем её же прошлый ответ


def test_unfixed_problems_become_warnings(manometer):
    bad = manometer_answer(
        seo_title="Манометр показывающий ТМ-510Р.00 0–2,5 МПа G1/2 РОСМА — для систем отопления и водоснабжения",
        seo_description=manometer_answer()["seo_description"] + " Гарантия 3 года.",
        key_specs=[{"name": "Выходной сигнал", "value": "4–20 мА", "source": "description", "evidence": "4–20 мА"}],
    )
    llm = FakeLLM(bad)
    card = generate_card(manometer, llm, max_repairs=2)

    assert card.status == "needs_review" and card.meta["attempts"] == 3
    assert len(card.seo_title) <= 70
    warnings = " ".join(card.warnings)
    assert "обрезан автоматически" in warnings and "гарантия" in warnings and "Отброшена" in warnings
    assert all(s.name != "Выходной сигнал" for s in card.key_specs)


def test_invalid_json_is_retried(manometer):
    llm = FakeLLM("Вот карточка: {не json", "```json\n" + json.dumps(manometer_answer(), ensure_ascii=False) + "\n```")
    card = generate_card(manometer, llm)
    assert card.meta["attempts"] == 2 and card.status == "ok"
    assert "не разобран как JSON" in llm.calls[1][-1]["content"]


def test_never_valid_json_raises(manometer):
    with pytest.raises(GenerationError, match="за 3 попыток"):
        generate_card(manometer, FakeLLM('{"seo_title": "без описания"}'), max_repairs=2)


def test_insufficient_data_keeps_short_honest_text(feed_only):
    answer = {
        "seo_title": "Датчик давления ОВЕН ПД100-ДИ1,6-111-0,5",
        "seo_description": "Датчик давления общепромышленный ОВЕН ПД100-ДИ1,6-111-0,5 для объектов ЖКХ.",
        "key_specs": [],
        "missing_specs": ["Диапазон измерений", "Выходной сигнал", "Напряжение питания"],
        "search_queries": ["пд100-ди1,6-111-0,5", "датчик давления овен пд100", "датчик давления для жкх"],
        "commercial": {"audience": ["объекты ЖКХ"], "tasks": ["измерение давления"]},
        "clarifications": ["Нужны характеристики и описание из паспорта."],
    }
    llm = FakeLLM(answer)
    card = generate_card(feed_only, llm)

    assert card.status == "insufficient_data" and card.meta["attempts"] == 1  # за длину не наказываем
    assert "Данных очень мало" in llm.calls[0][1]["content"]
    assert all(s.value == MISSING for s in card.key_specs) and len(card.key_specs) == 3
    assert any("мало фактов" in w for w in card.warnings)
    assert "Не переданы характеристики (specs)." in card.clarifications


def test_markdown_render(manometer):
    card = generate_card(manometer, FakeLLM(manometer_answer()))
    md = render_markdown(card)
    assert md.startswith("# Манометр показывающий ТМ-510Р.00(0-2,5 МПа)G1/2")
    assert "| Класс точности | 1,5; 1,0 | характеристики |" in md
    assert "## Что уточнить" in md and "замечаний нет" in md
