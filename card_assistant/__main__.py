import argparse
import glob
import json
import logging
import sys
from pathlib import Path

from .config import Settings
from .errors import CardAssistantError, InputDataError, LLMError
from .feed import DEFAULT_FEED, load_offer
from .service import make_llm, parse_json, run


def main(argv: list[str] | None = None) -> int:
    # Чтобы русский текст и значки не ломались при выводе в файл и чтении из канала на Windows
    for stream, encoding in ((sys.stdout, "utf-8"), (sys.stderr, "utf-8"), (sys.stdin, "utf-8-sig")):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding=encoding, errors="replace")

    parser = argparse.ArgumentParser(prog="python -m card_assistant", description="Черновики карточек товаров КИП с помощью LLM.")
    commands = parser.add_subparsers(dest="command", required=True)

    generate = commands.add_parser("generate", help="сгенерировать карточки по JSON-файлам")
    generate.add_argument("files", nargs="+", help="JSON-файлы с товаром или списком товаров; «-» — читать из stdin")
    generate.add_argument("-o", "--out", type=Path, help="папка для результатов (.json и .md); без неё Markdown печатается в консоль")
    generate.add_argument("-v", "--verbose", action="store_true", help="показывать ход попыток")

    serve = commands.add_parser("serve", help="запустить веб-форму")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    feed = commands.add_parser("from-feed", help="собрать входной JSON из оффера YML-фида")
    feed.add_argument("offer_id", help="id оффера в фиде")
    feed.add_argument("--feed", default=DEFAULT_FEED, help="URL или путь к файлу фида")
    feed.add_argument("-o", "--out", type=Path, help="куда сохранить JSON (по умолчанию — в консоль)")

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if getattr(args, "verbose", False) else logging.WARNING, format="%(levelname)s: %(message)s")

    try:
        if args.command == "generate":
            return generate_files(args.files, args.out)
        if args.command == "serve":
            import uvicorn

            uvicorn.run("card_assistant.web:app", host=args.host, port=args.port)
            return 0
        payload = json.dumps(load_offer(args.offer_id, args.feed), ensure_ascii=False, indent=2) + "\n"
        _emit(payload, args.out)
        return 0
    except LLMError as e:
        print(f"Ошибка LLM: {e}", file=sys.stderr)
        return 3
    except CardAssistantError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        return 2


def generate_files(files: list[str], out: Path | None) -> int:
    """Обрабатывает файлы по очереди: ошибка в одном товаре не останавливает остальные."""
    settings = Settings.from_env()
    llm = make_llm(settings)
    failed = 0
    # PowerShell и cmd не раскрывают маски вроде examples/input/*.json — делаем это сами
    files = [match for pattern in files for match in (sorted(glob.glob(pattern)) or [pattern])]
    for file in files:
        stem = "stdin" if file == "-" else Path(file).stem
        try:
            text = sys.stdin.read() if file == "-" else Path(file).read_text(encoding="utf-8-sig")
            data = parse_json(text, origin=file)
        except OSError as e:
            print(f"✗ {file}: не удалось прочитать файл — {e.strerror}", file=sys.stderr)
            failed += 1
            continue
        except InputDataError as e:  # в тексте ошибки уже есть имя файла
            print(f"✗ {e}", file=sys.stderr)
            failed += 1
            continue

        items = data if isinstance(data, list) else [data]
        for index, item in enumerate(items, 1):
            name = f"{stem}_{index}" if isinstance(data, list) else stem
            try:
                result = run(item, llm, settings)
            except LLMError:
                raise  # ключ, сеть, квота — дальше пробовать бессмысленно
            except CardAssistantError as e:
                print(f"✗ {name}: {e}", file=sys.stderr)
                failed += 1
                continue
            if out:
                out.mkdir(parents=True, exist_ok=True)
                (out / f"{name}.json").write_text(json.dumps(result["card"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                (out / f"{name}.md").write_text(result["markdown"], encoding="utf-8")
                print(f"✓ {name}: {result['card']['status']} → {out / name}.md", file=sys.stderr)
            else:
                print(result["markdown"])
    return 1 if failed else 0


def _emit(text: str, out: Path | None) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"✓ сохранено в {out}", file=sys.stderr)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    sys.exit(main())
