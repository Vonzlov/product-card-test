import json
from pathlib import Path

import pytest

from card_assistant.schemas import ProductInput, load_product

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples" / "input"

MANOMETER_DESCRIPTION = (
    "Манометр показывающий ТМ-510Р.00 производства РОСМА предназначен для измерения избыточного давления "
    "неагрессивных к медным сплавам жидких и газообразных, не вязких и не кристаллизующихся сред с температурой "
    "до +150 °C. Исполнение с диапазоном 0–2,5 МПа и присоединительной резьбой G1/2 указано в названии модели.\n\n"
    "Принцип действия основан на деформации чувствительного элемента — трубки Бурдона: под давлением её свободный "
    "конец перемещается и через трибко-секторный механизм поворачивает стрелку. Корпус диаметром 100 мм выполнен "
    "из стали, степень пылевлагозащиты — IP40, виброзащита — L3. Штуцер и трубчатая пружина изготовлены из медного "
    "сплава, циферблат алюминиевый, шкала черная на белом фоне. Прибор рассчитан на температуру окружающей среды "
    "от −60 до +60 °C, межповерочный интервал — 2 года.\n\n"
    "Манометр применяется во всех отраслях промышленности, включая теплоснабжение, водоснабжение, вентиляцию "
    "и машиностроение. Класс точности серии — 1,5 или 1,0 в зависимости от исполнения, поэтому при заказе его стоит уточнить."
)


def manometer_answer(**overrides) -> dict:
    answer = {
        "seo_title": "Манометр показывающий ТМ-510Р.00 0–2,5 МПа G1/2 РОСМА",
        "seo_description": MANOMETER_DESCRIPTION,
        "key_specs": [
            {"name": "Диапазон измерений", "value": "0-2,5 МПа", "source": "name", "evidence": "0-2,5 МПа"},
            {"name": "Присоединительная резьба", "value": "G1/2", "source": "specs", "evidence": "Присоединительная резьба"},
            {"name": "Класс точности", "value": "1,5", "source": "specs", "evidence": "Класс точности"},
            {"name": "Диаметр корпуса", "value": "100 мм", "source": "specs", "evidence": "Диаметр корпуса"},
            {"name": "Степень защиты", "value": "IP40", "source": "specs", "evidence": "Степень пылевлагозащиты"},
            {"name": "Чувствительный элемент", "value": "трубка Бурдона", "source": "description",
             "evidence": "В качестве чувствительного элемента используется трубка Бурдона"},
        ],
        "missing_specs": ["Тип присоединения (радиальное/осевое) для этой модификации", "Класс точности"],
        "search_queries": ["манометр тм-510", "манометр росма тм-510р", "манометр 0-2,5 мпа g1/2", "манометр показывающий 100 мм"],
        "commercial": {
            "audience": ["предприятия теплоснабжения и водоснабжения", "машиностроительные производства"],
            "tasks": ["контроль избыточного давления в трубопроводах", "визуальный контроль давления на месте установки"],
        },
        "clarifications": ["Уточнить класс точности для модификации: в данных указано «1,5; 1,0»."],
    }
    answer.update(overrides)
    return answer


class FakeLLM:
    """Возвращает заранее заданные ответы по очереди и запоминает присланные сообщения."""

    model = "fake-llm"

    def __init__(self, *answers):
        self.answers = [a if isinstance(a, str) else json.dumps(a, ensure_ascii=False) for a in answers]
        self.calls: list[list[dict]] = []

    def complete(self, messages):
        self.calls.append([dict(m) for m in messages])
        return self.answers[min(len(self.calls), len(self.answers)) - 1]


def load_example(name: str) -> ProductInput:
    return load_product(json.loads((EXAMPLES / f"{name}.json").read_text(encoding="utf-8")))


@pytest.fixture
def manometer() -> ProductInput:
    return load_example("manometer_tm510")


@pytest.fixture
def thermoregulator() -> ProductInput:
    return load_example("thermoregulator_trm500")


@pytest.fixture
def feed_only() -> ProductInput:
    return load_example("feed_only_pd100")
