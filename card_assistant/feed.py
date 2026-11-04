"""Превращает оффер из YML-фида во входной JSON сервиса.

В фиде souz-pribor.ru у офферов нет ни описания, ни технических характеристик — только название,
тип, производитель, цена и артикулы. Поэтому JSON «только из фида» почти всегда получает статус
insufficient_data: сервис честно перечислит, чего не хватает, вместо того чтобы додумывать.
"""

import http.client
import urllib.request
import xml.etree.ElementTree as ET
from typing import IO, Any

from .errors import InputDataError

DEFAULT_FEED = "https://www.souz-pribor.ru/yml/any/feed.xml"
SERVICE_PARAMS = {"Артикул", "Код товара"}  # учётные коды, а не характеристики прибора


def load_offer(offer_id: str, feed: str = DEFAULT_FEED) -> dict[str, Any]:
    with _open(feed) as stream:
        categories: dict[str, tuple[str | None, str]] = {}
        try:
            # Фид весит десятки мегабайт, поэтому читаем потоком и чистим разобранные элементы
            for _, element in ET.iterparse(stream):
                if element.tag == "category":
                    categories[element.get("id", "")] = (element.get("parentId"), (element.text or "").strip())
                elif element.tag == "offer":
                    if element.get("id") == offer_id:
                        return offer_to_input(element, categories)
                    element.clear()
        except ET.ParseError as e:
            raise InputDataError(f"Не удалось разобрать фид {feed}: {e}") from None
        except (OSError, http.client.HTTPException) as e:
            raise InputDataError(f"Чтение фида {feed} прервалось: {e}") from None
    raise InputDataError(f"Оффер {offer_id} не найден в фиде {feed}")


def offer_to_input(offer: ET.Element, categories: dict[str, tuple[str | None, str]]) -> dict[str, Any]:
    def text(tag: str) -> str:
        return (offer.findtext(tag) or "").strip()

    prefix, name = text("typePrefix"), text("name")
    params = {p.get("name", "").strip(): (p.text or "").strip() for p in offer.findall("param")}
    category = categories.get(text("categoryId"), (None, ""))[1]
    return {
        "name": f"{prefix[:1].upper()}{prefix[1:]} {name}".strip(),
        "manufacturer": text("vendor"),
        "category": category,
        "specs": {k: v for k, v in params.items() if k not in SERVICE_PARAMS},
        "description": text("description"),
        "source": {
            "feed_offer_id": offer.get("id"),
            "url": text("url"),
            "sku": text("vendorCode") or params.get("Артикул", ""),
            "note": "Собрано только из YML-фида.",
        },
    }


def _open(feed: str) -> IO[bytes]:
    if not feed.startswith(("http://", "https://")):
        try:
            return open(feed, "rb")
        except OSError as e:
            raise InputDataError(f"Не удалось открыть файл фида {feed}: {e.strerror}") from None
    request = urllib.request.Request(feed, headers={"User-Agent": "Mozilla/5.0 (product-card-assistant)"})
    try:
        return urllib.request.urlopen(request, timeout=60)
    except OSError as e:
        raise InputDataError(f"Не удалось скачать фид {feed}: {e}") from None
