import pytest

from card_assistant.checks import assess_input
from card_assistant.errors import InputDataError
from card_assistant.schemas import load_product, parse_specs


def test_specs_formats_are_normalized():
    assert parse_specs({"Класс точности": 1.5, "Питание": None, "Корпус": ["Щ2", "96×48 мм"]}) == {
        "Класс точности": "1.5", "Питание": "", "Корпус": "Щ2; 96×48 мм",
    }
    assert parse_specs([{"name": "Диапазон", "value": "0–1,6 МПа"}, ["IP", "IP65"]]) == {"Диапазон": "0–1,6 МПа", "IP": "IP65"}
    assert parse_specs("Диапазон: 0–1,6 МПа\n\n- Резьба: G1/2\nМатериал — сталь") == {
        "Диапазон": "0–1,6 МПа", "Резьба": "G1/2", "Материал": "сталь",
    }


def test_nested_specs_are_flattened(thermoregulator):
    assert thermoregulator.specs["Основная приведенная погрешность — ТС"] == "0,25 %"
    assert thermoregulator.specs["Выход 1 — ТРМ500-Щ2.30А"].startswith("Электромагнитное реле, в on/off режиме ~30 А/250 В")


def test_spec_line_without_separator_is_reported():
    with pytest.raises(InputDataError, match="нет разделителя"):
        load_product({"name": "Манометр", "specs": "просто текст без двоеточия"})


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "Ожидается JSON-объект"),
        ({"manufacturer": "ОВЕН"}, "Не заполнено название"),
        ({"name": "   "}, "Не заполнено название"),
        ({"name": "Манометр", "specs": 42}, "specs"),
        ({"name": {"ru": "Манометр"}}, "name: ожидается строка"),
    ],
)
def test_invalid_input_gives_readable_error(data, message):
    with pytest.raises(InputDataError, match=message):
        load_product(data)


def test_extra_fields_are_kept_and_category_goes_to_llm(manometer):
    assert manometer.extras["source"]["feed_offer_id"] == "191953"
    payload = manometer.llm_payload()
    assert payload["category"] == "Манометры показывающие"
    assert "source" not in payload


def test_assess_levels(manometer, feed_only):
    assert assess_input(manometer).level == "ok"

    report = assess_input(feed_only)
    assert report.level == "insufficient"
    assert any("характеристики" in gap for gap in report.gaps)
    assert any("Нет описания" in gap for gap in report.gaps)

    partial = assess_input(load_product({"name": "Датчик", "specs": {"Диапазон": "0–1 МПа", "Выход": ""}}))
    assert partial.level == "partial"
    assert any("«Выход»" in gap for gap in partial.gaps)
    assert any("производитель" in gap for gap in partial.gaps)
