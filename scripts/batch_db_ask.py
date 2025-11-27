"""
Batch helper to sequentially query /tasks/ask.

用法示例：

  # 在命令行直接给出多个问题
  python scripts/batch_db_ask.py "张三的E3D接口联调现在什么状态？" "张三还有多少任务未完成？"

  # 从文件读取问题（每行一个，自然语言中文）
  python scripts/batch_db_ask.py --file scripts/questions.txt

  # 自定义服务地址
  python scripts/batch_db_ask.py --endpoint http://localhost:8000/tasks/ask --file questions.txt
"""
import argparse
from pathlib import Path
from typing import Iterable, List

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


def ask(endpoint: str, question: str, timeout: int) -> dict:
    resp = requests.get(endpoint, params={"q": question}, timeout=timeout)
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
    args = parser.parse_args()

    try:
        questions = list(iter_questions(args))
    except Exception as exc:
        raise SystemExit(str(exc))

    for idx, question in enumerate(questions, start=1):
        print(f"\n=== Q{idx}: {question}")
        try:
            payload = ask(args.endpoint, question, args.timeout)
        except Exception as exc:
            print(f"Request failed: {exc}")
            continue

        # /tasks/ask 返回的是完整回答，这里打印几个关键字段
        print("--- Answer:", payload.get("answer"))
        print("--- Resolver mode:", payload.get("resolver_mode"))
        print("--- Intent:", payload.get("intent"))

        nl_ir = payload.get("nl_ir") or {}
        extra = nl_ir.get("extra") or {}
        if extra:
            print("--- NL IR source:", extra.get("nl2sql_source"))
            if "nl2sql_llm_error" in extra:
                print("--- NL2SQL LLM error:", extra.get("nl2sql_llm_error"))

        if payload.get("resolver_mode") == "text2sql":
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


if __name__ == "__main__":
    main()
