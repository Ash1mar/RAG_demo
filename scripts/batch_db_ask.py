"""
Batch helper to sequentially query /tasks/ask.

用法示例：

  # 在命令行直接给出多个问题
  python scripts/batch_db_ask.py "张三的E3D接口联调现在什么状态？" "张三还有多少任务未完成？"

  # 从文件读取问题（每行一个，自然语言中文）
  python scripts/batch_db_ask.py --file scripts/questions.txt
  python scripts/batch_db_ask.py --file scripts/qs_old.txt
  python scripts/batch_db_ask.py --file scripts/FACT_TASK_ASSIGN_questions_real.txt

  # 自定义服务地址
  python scripts/batch_db_ask.py --endpoint http://localhost:8000/tasks/ask --file questions.txt
"""
import argparse
import json
from pathlib import Path
from urllib.parse import urlparse, urlunparse
from typing import Any, Dict, Iterable, List

import requests


def load_questions(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"questions file not found: {path}")
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def iter_questions(args: argparse.Namespace) -> Iterable[str]:
    questions: List[str] = []
    if args.file:
        questions.extend(load_questions(Path(args.file)))
    if args.questions:
        questions.extend([q.strip() for q in args.questions if q.strip()])
    if not questions:
        raise ValueError("no questions provided; pass positional strings or --file")
    return questions


def ask(endpoint: str, question: str, timeout: int, debug_trace: bool = False) -> dict:
    params = {"q": question}
    if debug_trace:
        params["debug"] = "true"
    resp = requests.get(endpoint, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def _print_json_block(title: str, value: Any) -> None:
    print(title)
    print(json.dumps(value, ensure_ascii=False, indent=2))


def print_debug_trace(payload: Dict[str, Any]) -> None:
    trace = payload.get("debug_trace") or []
    if not trace:
        print("--- Debug trace: <empty>")
        return

    print("\n--- Debug trace start ---")
    for idx, step in enumerate(trace, start=1):
        stage = step.get("stage") or f"step_{idx}"
        function = step.get("function") or "unknown"
        print(f"[{idx}] {stage} :: {function}")
        if "note" in step:
            print("  note:")
            print(f"    {step.get('note')}")
        if "inputs" in step:
            _print_json_block("  inputs:", step.get("inputs"))
        if "output" in step:
            _print_json_block("  output:", step.get("output"))
    print("--- Debug trace end ---")

def fetch_health(endpoint: str, timeout: int) -> dict:
    parsed = urlparse(endpoint)
    path = parsed.path or ""
    for suffix in ("/tasks/ask", "/db/ask", "/tasks/ask/", "/db/ask/"):
        if path.endswith(suffix):
            path = path[: -len(suffix)] or "/"
            break
    if not path.endswith("/"):
        path = path + "/"
    health_path = f"{path}health"
    health_url = urlunparse(parsed._replace(path=health_path, query="", params="", fragment=""))
    resp = requests.get(health_url, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sequentially query /tasks/ask with prepared questions"
    )
    parser.add_argument(
        "questions",
        nargs="*",
        help="可选：直接在命令行给出的问题（UTF-8）",
    )
    parser.add_argument(
        "--file",
        help="可选：UTF-8 文本文件路径，每行一个问题",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000/tasks/ask",
        help="完整接口地址 (默认: http://localhost:8000/tasks/ask)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="HTTP 超时时间（秒），默认 300",
    )
    parser.add_argument(
        "--debug-trace",
        action="store_true",
        help="请求服务端返回 step-by-step 调试信息，并在每个问题最后打印完整链路",
    )
    args = parser.parse_args()

    try:
        questions = list(iter_questions(args))
    except Exception as exc:
        raise SystemExit(str(exc))

    try:
        health = fetch_health(args.endpoint, args.timeout)
        tasks_store = health.get("tasks_store")
        tasks_ready = health.get("tasks_ready")
        print(f"--- Tasks store: {tasks_store}")
        print(f"--- Tasks ready: {tasks_ready}")
    except Exception as exc:
        print(f"Health check failed: {exc}")

    for idx, question in enumerate(questions, start=1):
        print(f"\n=== Q{idx}: {question}")
        try:
            payload = ask(args.endpoint, question, args.timeout, debug_trace=args.debug_trace)
        except Exception as exc:
            print(f"Request failed: {exc}")
            continue

        # /tasks/ask 返回的是完整回答，这里打印几个关键字段
        print("--- Answer:", payload.get("answer"))
        print("--- Resolver mode:", payload.get("resolver_mode"))
        print("--- Intent:", payload.get("intent"))
        print("--- KG enabled:", payload.get("kg_enabled"))

        nl_ir = payload.get("nl_ir") or {}
        extra = nl_ir.get("extra") or {}
        if extra:
            print("--- NL IR source:", extra.get("nl2sql_source"))
            if "kg_person_source" in extra:
                print("--- KG person source:", extra.get("kg_person_source"))
            if "kg_category_source" in extra:
                print("--- KG category source:", extra.get("kg_category_source"))
            if "nl2sql_llm_error" in extra:
                print("--- NL2SQL LLM error:", extra.get("nl2sql_llm_error"))

        if payload.get("resolver_mode") == "text2sql":
            model = payload.get("text2sql_model")
            if model:
                provider = payload.get("text2sql_provider")
                if provider:
                    print(f"--- Text2SQL model: {model} (provider={provider})")
                else:
                    print(f"--- Text2SQL model: {model}")
            err = payload.get("error")
            if err:
                print("--- Text2SQL error:", err)
                if payload.get("reason"):
                    print("--- Text2SQL reason:", payload.get("reason"))
                if payload.get("sql"):
                    print("--- Text2SQL SQL:", payload.get("sql"))
                if "params" in payload:
                    print("--- Text2SQL params:", payload.get("params"))

            if payload.get("error") == "text2sql_invalid_sql":
                print("--- Invalid SQL:", payload.get("invalid_sql"))
            if payload.get("error") == "text2sql_llm_failed":
                reason = payload.get("reason")
                if reason:
                    print("--- Text2SQL failure reason:", reason)
                raw = payload.get("text2sql_raw_response")
                if raw:
                    print("--- LLM raw response:", raw)
            text2sql = payload.get("text2sql") or []
            for ix, item in enumerate(text2sql, start=1):
                print(f"--- Text2SQL query #{ix}:")
                print("    SQL:", item.get("sql"))
                if item.get("description"):
                    print("    Description:", item.get("description"))
                if item.get("generated_sql"):
                    print("    Generated SQL:", item.get("generated_sql"))
                if item.get("rewritten_sql"):
                    print("    Rewritten SQL:", item.get("rewritten_sql"))

        if args.debug_trace:
            print_debug_trace(payload)


if __name__ == "__main__":
    main()
