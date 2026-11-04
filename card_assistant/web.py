from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, JSONResponse

from .config import Settings
from .errors import CardAssistantError, GenerationError, InputDataError, LLMError
from .llm import LLMClient
from .service import make_llm, parse_json, run

STATIC = Path(__file__).parent / "static"
EXAMPLES = Path(__file__).resolve().parent.parent / "examples" / "input"

app = FastAPI(title="Помощник по карточкам товаров КИП")


@lru_cache
def get_settings() -> Settings:
    return Settings.from_env()


@lru_cache
def get_llm() -> LLMClient:
    # Создаётся при первом запросе: форма открывается даже без ключа, а ошибка настройки видна в ответе
    return make_llm(get_settings())


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (STATIC / "index.html").read_text(encoding="utf-8")


@app.get("/api/examples")
def examples() -> list[dict[str, str]]:
    # Отдаём как текст: среди примеров есть намеренно битый JSON
    return [{"id": p.stem, "text": p.read_text(encoding="utf-8")} for p in sorted(EXAMPLES.glob("*.json"))]


@app.post("/api/generate")
async def generate(request: Request) -> JSONResponse:
    try:
        data = parse_json((await request.body()).decode("utf-8-sig", errors="replace"))
        result = await run_in_threadpool(run, data, get_llm(), get_settings())
    except InputDataError as e:
        return JSONResponse({"error": str(e)}, status_code=422)
    except (LLMError, GenerationError) as e:
        return JSONResponse({"error": str(e)}, status_code=502)
    except CardAssistantError as e:  # ошибка настройки .env
        return JSONResponse({"error": str(e)}, status_code=500)
    return JSONResponse(result)
