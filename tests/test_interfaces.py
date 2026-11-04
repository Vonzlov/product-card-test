"""Веб-API, CLI, конвертер фида и обработка ошибок LLM API."""

import json
from pathlib import Path

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from card_assistant import __main__ as cli
from card_assistant import web
from card_assistant.config import Settings
from card_assistant.errors import InputDataError, LLMError
from card_assistant.feed import load_offer
from card_assistant.llm import OpenAICompatibleClient

from .conftest import EXAMPLES, FakeLLM, manometer_answer

FEED = Path(__file__).parent / "fixtures" / "feed_sample.xml"
SETTINGS = Settings(base_url="https://api.example.com/v1", api_key="test", model="m", temperature=0.2, json_mode=True, timeout=5, max_repairs=2)


# ---------- Веб ----------

@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(web, "get_settings", lambda: SETTINGS)
    monkeypatch.setattr(web, "get_llm", lambda: FakeLLM(manometer_answer()))
    return TestClient(web.app)


def test_web_generate(client):
    response = client.post("/api/generate", content=(EXAMPLES / "manometer_tm510.json").read_bytes())
    assert response.status_code == 200
    body = response.json()
    assert body["card"]["status"] == "ok" and body["markdown"].startswith("# Манометр")


def test_web_form_style_specs_text(client):
    data = {"name": "Манометр показывающий ТМ-510Р.00(0-2,5 МПа)G1/2", "specs": "Диаметр корпуса: 100 мм", "description": ""}
    assert client.post("/api/generate", json=data).status_code == 200


@pytest.mark.parametrize(
    ("payload", "message"),
    [((EXAMPLES / "broken_input.json").read_bytes(), "некорректный JSON"), (b'{"specs": {}}', "Не заполнено название")],
)
def test_web_input_errors(client, payload, message):
    response = client.post("/api/generate", content=payload)
    assert response.status_code == 422 and message in response.json()["error"]


def test_web_llm_error(client, monkeypatch):
    def broken():
        raise LLMError("Не задан API-ключ")
    monkeypatch.setattr(web, "get_llm", broken)
    response = client.post("/api/generate", json={"name": "Манометр"})
    assert response.status_code == 502 and "API-ключ" in response.json()["error"]


def test_web_pages(client):
    assert "Черновик карточки" in client.get("/").text
    assert {e["id"] for e in client.get("/api/examples").json()} >= {"manometer_tm510", "thermoregulator_trm500", "broken_input"}


# ---------- CLI ----------

def test_cli_generate_continues_after_bad_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: SETTINGS))
    monkeypatch.setattr(cli, "make_llm", lambda settings: FakeLLM(manometer_answer()))
    code = cli.main(["generate", str(EXAMPLES / "broken_input.json"), str(EXAMPLES / "manometer_tm510.json"), "-o", str(tmp_path)])

    assert code == 1  # один файл с ошибкой, второй обработан
    assert json.loads((tmp_path / "manometer_tm510.json").read_text(encoding="utf-8"))["status"] == "ok"
    assert (tmp_path / "manometer_tm510.md").exists()
    assert "broken_input.json: некорректный JSON" in capsys.readouterr().err


def test_cli_expands_masks_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(Settings, "from_env", classmethod(lambda cls: SETTINGS))
    monkeypatch.setattr(cli, "make_llm", lambda settings: FakeLLM(manometer_answer()))
    cli.main(["generate", str(EXAMPLES / "*.json"), "-o", str(tmp_path)])
    assert {p.stem for p in tmp_path.glob("*.md")} == {"manometer_tm510", "thermoregulator_trm500", "feed_only_pd100"}


def test_cli_from_feed(tmp_path):
    out = tmp_path / "offer.json"
    assert cli.main(["from-feed", "191953", "--feed", str(FEED), "-o", str(out)]) == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["name"] == "Манометр показывающий ТМ-510Р.00(0-2,5 МПа)G1/2"
    assert data["manufacturer"] == "РОСМА" and data["category"] == "Манометры показывающие" and data["specs"] == {}


def test_feed_errors():
    with pytest.raises(InputDataError, match="не найден"):
        load_offer("000", str(FEED))
    with pytest.raises(InputDataError, match="Не удалось открыть"):
        load_offer("1", "нет-такого-файла.xml")


# ---------- LLM-клиент ----------

def _error(cls, status, message):
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    return cls(message, response=httpx.Response(status, request=request), body={"message": message})


def _response(content):
    return openai.types.chat.ChatCompletion.model_validate({
        "id": "1", "object": "chat.completion", "created": 0, "model": "m",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
    })


def test_llm_config_errors():
    with pytest.raises(LLMError, match="LLM_MODEL"):
        OpenAICompatibleClient(Settings(**{**SETTINGS.__dict__, "model": ""}))
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        OpenAICompatibleClient(Settings(**{**SETTINGS.__dict__, "api_key": ""}))
    OpenAICompatibleClient(Settings(**{**SETTINGS.__dict__, "api_key": "", "base_url": "http://localhost:11434/v1"}))


def test_llm_drops_unsupported_params(monkeypatch):
    client = OpenAICompatibleClient(SETTINGS)
    calls = []

    def create(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise _error(openai.BadRequestError, 400, "Unsupported value: 'temperature' does not support 0.2")
        return _response('{"ok": true}')

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    assert client.complete([{"role": "user", "content": "hi"}]) == '{"ok": true}'
    assert "temperature" not in calls[-1] and "response_format" in calls[-1]


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (_error(openai.AuthenticationError, 401, "bad key"), "LLM_API_KEY"),
        (_error(openai.NotFoundError, 404, "no model"), "не найдены"),
        (_error(openai.RateLimitError, 429, "quota exceeded"), "quota exceeded"),
        (_error(openai.InternalServerError, 500, "boom"), "ошибку 500"),
        (openai.APIConnectionError(request=httpx.Request("POST", "https://x")), "Нет соединения"),
    ],
)
def test_llm_errors_are_readable(monkeypatch, error, message):
    client = OpenAICompatibleClient(SETTINGS)

    def create(**kwargs):
        raise error

    monkeypatch.setattr(client._client.chat.completions, "create", create)
    with pytest.raises(LLMError, match=message):
        client.complete([{"role": "user", "content": "hi"}])


def test_llm_empty_answer(monkeypatch):
    client = OpenAICompatibleClient(SETTINGS)
    monkeypatch.setattr(client._client.chat.completions, "create", lambda **kw: _response(""))
    with pytest.raises(LLMError, match="пустой"):
        client.complete([{"role": "user", "content": "hi"}])
