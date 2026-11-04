from card_assistant import checks
from card_assistant.schemas import LLMSpec


def test_numbers_are_canonical():
    assert checks.numbers("0…1,6 МПа, класс 1,50, 0,25 %") == {"0", "1.6", "1.5", "0.25"}
    assert checks.numbers("кгс/см²") == checks.numbers("кгс/см2") == {"2"}


def test_measures_spread_unit_over_ranges():
    assert checks.measures("96…264 В переменного тока") == {("96", "В"), ("264", "В")}
    assert checks.measures("от −60 до +60 °C, до 150 C") == {("60", "°C"), ("150", "°C")}
    assert checks.measures("0…1 / 1,6 / 2,5 кгс/см²") == {("0", "кгс/см²"), ("1", "кгс/см²"), ("1.6", "кгс/см²"), ("2.5", "кгс/см²")}
    assert checks.measures("96×48×100 мм") == checks.measures("96х48х100 мм") == {("96", "мм"), ("48", "мм"), ("100", "мм")}
    # строчные «в», «с» — предлоги, а не вольты и секунды
    assert checks.measures("выход 3 в режиме ТТР, модель 2 с реле") == set()


def test_unit_mismatch_is_flagged_even_if_number_exists(manometer):
    problems = checks.check_grounding({"t": "давление до 2,5 МПа, резьба M20x1,5, ток 20 мА"}, manometer.corpus())
    assert len(problems) == 1 and "20 мА" in problems[0] and "2.5" not in problems[0]


def test_grounded_text_passes(manometer):
    text = "Манометр РОСМА ТМ-510 с диапазоном 0–2,5 МПа, резьбой G1/2 и корпусом 100 мм, IP40."
    assert checks.check_grounding({"seo_description": text}, manometer.corpus()) == []


def test_invented_numbers_claims_and_shop_are_flagged(manometer):
    text = "Выходной сигнал 4–20 мА, интерфейс RS-485, гарантия от производителя. Цена ниже рынка, доставка по России."
    problems = checks.check_grounding({"seo_description": text}, manometer.corpus())
    joined = " ".join(problems)
    # 4 и 20 во входных данных есть (диапазоны серии, резьба M20x1,5), но не в миллиамперах
    assert "20 мА" in joined and "4 мА" in joined
    assert "RS-485" in joined and "гарантия" in joined
    assert "цена, наличие, доставка или магазин" in joined


def test_claims_present_in_input_are_allowed(thermoregulator, manometer):
    # Wi-Fi и реестр СИ упомянуты во входных данных ТРМ500, а «гостиницы» — не ГОСТ
    assert checks.check_grounding({"t": "Wi-Fi, внесён в реестр СИ"}, thermoregulator.corpus()) == []
    assert checks.check_grounding({"t": "для гостиницы"}, thermoregulator.corpus()) == []
    assert "ГОСТ" in " ".join(checks.check_grounding({"t": "по ГОСТ"}, thermoregulator.corpus()))
    assert checks.check_grounding({"t": "по ГОСТ"}, manometer.corpus()) == []


def test_spec_values_come_from_input(manometer):
    items = [
        LLMSpec(name="Диаметр", value="100 мм", source="specs", evidence="Диаметр корпуса"),
        LLMSpec(name="Материал корпуса", value="нержавеющая сталь", source="specs", evidence="Материал корпуса"),
        LLMSpec(name="Окружающая среда", value="−60…+60 °C", source="specs", evidence="Температура окружающей среды"),
    ]
    specs, problems = checks.verify_specs(items, manometer)
    assert problems == []
    assert [s.value for s in specs] == ["100 мм", "Сталь", "−60…+60 °C"]  # переписанное значение заменено исходным


def test_series_variant_is_not_chosen_without_name(manometer):
    items = [
        LLMSpec(name="Класс точности", value="1,5", source="specs", evidence="Класс точности"),
        LLMSpec(name="Резьба", value="G1/2", source="specs", evidence="Присоединительная резьба"),
    ]
    specs, _ = checks.verify_specs(items, manometer)
    assert [s.value for s in specs] == ["1,5; 1,0", "G1/2"]  # G1/2 подтверждается названием модели


def test_quotes_from_name_and_description(manometer):
    ok = [
        LLMSpec(name="Диапазон", value="0-2,5 МПа", source="name", evidence="(0-2,5 МПа)"),
        LLMSpec(name="Чувствительный элемент", value="Трубка Бурдона", source="description",
                evidence="в качестве чувствительного элемента используется трубка Бурдона"),
    ]
    specs, problems = checks.verify_specs(ok, manometer)
    assert problems == [] and len(specs) == 2

    bad = [
        LLMSpec(name="Диапазон", value="0-4 МПа", source="name", evidence="0-2,5 МПа"),
        LLMSpec(name="Интерфейс", value="RS-485", source="description", evidence="поддерживает RS-485"),
        LLMSpec(name="Питание", value="24 В", source="specs", evidence="Напряжение питания"),
        LLMSpec(name="X", value="1", source="passport", evidence=""),
    ]
    specs, problems = checks.verify_specs(bad, manometer)
    assert specs == [] and len(problems) == 4


def test_cyrillic_lookalikes_match(thermoregulator):
    # во входных данных «Сu» написано с кириллической «С»
    items = [LLMSpec(name="ТС", value="50/100/500/1000 (М, Cu, П, Pt), 53М", source="specs", evidence="Поддержка типов датчиков — ТС")]
    specs, problems = checks.verify_specs(items, thermoregulator)
    assert problems == [] and specs[0].value.startswith("50/100")


def test_length_helpers():
    title = "Манометр показывающий ТМ-510Р.00 0–2,5 МПа G1/2 РОСМА — для систем отопления и водоснабжения"
    assert checks.shorten_title(title) == "Манометр показывающий ТМ-510Р.00 0–2,5 МПа G1/2 РОСМА"
    no_separators = "Терморегулятор ОВЕН ТРМ500-Щ2.30А щитовой с мощным реле и крупным индикатором температуры"
    short = checks.shorten_title(no_separators)
    assert len(short) <= 70 and no_separators.startswith(short) and not short.endswith(" ")

    text = ("Первое предложение. " * 70).strip()
    trimmed = checks.trim_to_sentence(text)
    assert len(trimmed) <= 1200 and trimmed.endswith(".")
