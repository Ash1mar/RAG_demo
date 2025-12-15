from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import requests

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency guard
    yaml = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).with_name("config.yaml")


def _strip_inline_comment(value: str) -> str:
    result: List[str] = []
    in_quote: Optional[str] = None
    prev = ""
    for ch in value:
        if in_quote:
            result.append(ch)
            if ch == in_quote and prev != "\\":
                in_quote = None
            prev = ch
            continue
        if ch in ("'", '"'):
            in_quote = ch
            result.append(ch)
            prev = ch
            continue
        if ch == "#":
            break
        result.append(ch)
        prev = ch
    return "".join(result).rstrip()


def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    root: Dict[str, Any] = {}
    stack: List[tuple[int, Dict[str, Any]]] = [(0, root)]

    for raw_line in text.splitlines():
        if not raw_line.strip():
            continue
        stripped = raw_line.lstrip(" ")
        if stripped.startswith("#"):
            continue

        indent = len(raw_line) - len(stripped)
        if indent % 2 != 0:
            raise SystemExit(f"Unsupported indentation in config: {raw_line!r}")

        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            raise SystemExit(f"Invalid indentation sequence near: {raw_line}")

        parent = stack[-1][1]
        if ":" not in stripped:
            raise SystemExit(f"Invalid config line (missing colon): {raw_line}")

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = _strip_inline_comment(value).strip()

        if not value:
            new_dict: Dict[str, Any] = {}
            parent[key] = new_dict
            stack.append((indent + 2, new_dict))
            continue

        parent[key] = _coerce_value(value)

    return root


def _coerce_value(value: str) -> Any:
    if value.startswith(("'", '"')) and value.endswith(value[0]):
        return value[1:-1]

    lowered = value.lower()
    if lowered in {"true", "yes"}:
        return True
    if lowered in {"false", "no"}:
        return False
    if lowered in {"null", "none"}:
        return None

    try:
        if value.startswith("0") and value != "0":
            raise ValueError
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def load_config() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        raise SystemExit(f"Config file missing: {CONFIG_PATH}")
    data = CONFIG_PATH.read_text(encoding="utf-8")
    if yaml is not None:
        return yaml.safe_load(data)
    print("PyYAML not found; falling back to minimal parser (limited YAML features).")
    return _parse_simple_yaml(data)


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_subprocess(command: List[str] | str, *, shell: bool = False, env: Optional[Dict[str, str]] = None) -> None:
    subprocess.run(command, check=True, shell=shell, env=env)


def read_questions(path: Path) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Question file not found: {path}")
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def query_endpoint(endpoint: str, question: str, timeout: int) -> dict:
    resp = requests.get(endpoint, params={"q": question}, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False))
            fh.write("\n")


def read_jsonl(path: Path) -> List[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Artifact not found: {path}")
    data: List[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def seed_database(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    seed_cfg = config.get("seed", {})
    script_location = resolve_path(args.script or seed_cfg.get("script", "scripts/init_tasks_sqlite.py"))
    if not script_location.exists():
        raise SystemExit(f"Seed script not found: {script_location}")

    env = os.environ.copy()
    configured_env = seed_cfg.get("env", {})
    for key, value in configured_env.items():
        env[key] = str(value)
    run_subprocess([sys.executable, str(script_location)], env=env)


def run_baseline(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    baseline_cfg = config.get("models", {}).get("baseline", {})
    command_template = args.command or baseline_cfg.get("command") or ""
    output_path = resolve_path(args.output or baseline_cfg.get("output_path", "experiments/artifacts/baseline_latest.jsonl"))

    if not command_template.strip():
        print("Baseline command not configured. Update `models.baseline.command` in config.yaml.")
        return

    context = {
        "question_file": str(resolve_path(args.question_file or config.get("data", {}).get("eval_questions", "scripts/questions.txt"))),
        "output_path": str(output_path),
    }
    expanded = command_template.format(**context)
    command = expanded if args.shell or baseline_cfg.get("shell", False) else shlex.split(expanded)

    ensure_dir(output_path.parent)
    env = os.environ.copy()
    env["EXPERIMENT_BASELINE_OUTPUT"] = context["output_path"]
    run_subprocess(command, shell=args.shell or baseline_cfg.get("shell", False), env=env)


def run_system(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    system_cfg = config.get("models", {}).get("system", {})
    endpoint = args.endpoint or system_cfg.get("endpoint", "http://localhost:8000/tasks/ask")
    timeout = args.timeout or system_cfg.get("timeout", 120)
    question_file = resolve_path(args.question_file or system_cfg.get("question_file", "scripts/questions.txt"))
    output_path = resolve_path(args.output or system_cfg.get("output_path", "experiments/artifacts/system_latest.jsonl"))

    questions = read_questions(question_file)
    print(f"Loaded {len(questions)} questions from {question_file}")

    results: List[dict] = []
    for idx, question in enumerate(questions, start=1):
        record = {
            "question": question,
            "index": idx,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        try:
            payload = query_endpoint(endpoint, question, timeout)
            record["response"] = payload
            record["status"] = "ok"
        except Exception as exc:  # pragma: no cover - network failure path
            record["response"] = {"error": str(exc)}
            record["status"] = "error"
            print(f"[WARN] Q{idx} failed: {exc}")
        results.append(record)

    write_jsonl(output_path, results)
    print(f"Wrote system responses to {output_path}")


def summarize_records(records: List[dict]) -> Dict[str, Any]:
    total = len(records)
    answered = 0
    resolver_counts: Counter[str] = Counter()
    error_counts: Counter[str] = Counter()

    for item in records:
        response = item.get("response") or {}
        if response.get("answer"):
            answered += 1
        resolver = response.get("resolver_mode") or "unknown"
        resolver_counts[resolver] += 1
        if response.get("error"):
            error_counts[response.get("error")] += 1

    return {
        "total": total,
        "answered": answered,
        "answer_rate": (answered / total) if total else 0.0,
        "resolver_mode_counts": dict(resolver_counts),
        "error_counts": dict(error_counts),
    }


def load_reference_answers(path: Path) -> Dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Reference answers file not found: {path}")
    references: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"Invalid JSONL at {path}:{idx}: {exc}") from exc
            question = payload.get("question")
            answer = payload.get("answer")
            if not question or answer is None:
                raise SystemExit(f"Missing `question`/`answer` fields at {path}:{idx}")
            references[str(question)] = str(answer)
    return references


def normalize_answer(value: Optional[str]) -> str:
    if value is None:
        return ""
    return " ".join(value.strip().lower().split())


def evaluate_against_refs(records: List[dict], references: Dict[str, str]) -> Dict[str, Any]:
    if not references:
        return {}

    covered = 0
    matches = 0
    missing_predictions = 0

    for record in records:
        question = record.get("question")
        if not question:
            continue
        target = references.get(str(question))
        if target is None:
            continue
        covered += 1
        predicted = ((record.get("response") or {}).get("answer"))
        if predicted is None:
            missing_predictions += 1
            continue
        if normalize_answer(predicted) == normalize_answer(target):
            matches += 1

    return {
        "reference_questions": len(references),
        "evaluated": covered,
        "missing_predictions": missing_predictions,
        "exact_match": matches,
        "exact_match_rate": (matches / covered) if covered else 0.0,
    }


def evaluate_runs(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    artifacts_cfg = config.get("artifacts", {})
    metrics_path = resolve_path(args.metrics or artifacts_cfg.get("metrics_path", "experiments/artifacts/metrics.json"))
    system_path = resolve_path(args.system_output or config.get("models", {}).get("system", {}).get("output_path", "experiments/artifacts/system_latest.jsonl"))
    baseline_path = resolve_path(args.baseline_output or config.get("models", {}).get("baseline", {}).get("output_path", "experiments/artifacts/baseline_latest.jsonl"))
    ref_path_value = args.reference or config.get("data", {}).get("reference_answers")

    references: Dict[str, str] = {}
    if ref_path_value:
        ref_path = resolve_path(ref_path_value)
        if ref_path.exists():
            references = load_reference_answers(ref_path)
        else:
            print(f"[WARN] Reference file missing ({ref_path}); skipping exact-match evaluation.")

    metrics: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "pipeline": {
            "retriever_top_k": config.get("pipeline", {}).get("retriever_top_k"),
            "sql_retry_limit": config.get("pipeline", {}).get("sql_retry_limit"),
            "answer_temperature": config.get("pipeline", {}).get("answer_temperature"),
        },
    }

    system_records = read_jsonl(system_path)
    system_summary = summarize_records(system_records)
    if references:
        system_summary["reference_eval"] = evaluate_against_refs(system_records, references)
    metrics["system"] = system_summary

    if baseline_path.exists():
        try:
            baseline_records = read_jsonl(baseline_path)
            baseline_summary = summarize_records(baseline_records)
            if references:
                baseline_summary["reference_eval"] = evaluate_against_refs(baseline_records, references)
            metrics["baseline"] = baseline_summary
        except FileNotFoundError:
            pass

    ensure_dir(metrics_path.parent)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Metrics written to {metrics_path}")


def render_report(metrics: Dict[str, Any], config: Dict[str, Any]) -> str:
    report_cfg = config.get("report", {})
    title = report_cfg.get("title", "Experiment Report")
    author = report_cfg.get("author", "Unknown")

    runs: List[tuple[str, Dict[str, Any]]] = [("System", metrics.get("system", {}))]
    if "baseline" in metrics:
        runs.append(("Baseline", metrics.get("baseline", {})))

    def format_reference_line(run_metrics: Dict[str, Any]) -> Optional[str]:
        ref = run_metrics.get("reference_eval")
        if not ref:
            return None
        exact_rate = ref.get("exact_match_rate", 0.0)
        exact_cnt = ref.get("exact_match", 0)
        evaluated = ref.get("evaluated", 0)
        return (
            f"Exact match: {exact_rate:.2%} "
            f"({exact_cnt}/{evaluated} vs {ref.get('reference_questions', 0)} reference questions)"
        )

    lines = [
        f"# {title}",
        "",
        f"- Author: {author}",
        f"- Generated at: {metrics.get('generated_at', 'N/A')}",
        "",
        "## Run summary",
        "| Run | Total | Answer rate | Exact match |",
        "| --- | --- | --- | --- |",
    ]

    for name, data in runs:
        answer_rate = data.get("answer_rate", 0.0)
        ref = data.get("reference_eval")
        exact = ref.get("exact_match_rate") if ref else None
        exact_display = f"{exact:.2%}" if exact is not None else "N/A"
        lines.append(f"| {name} | {data.get('total', 0)} | {answer_rate:.2%} | {exact_display} |")

    for name, data in runs:
        lines.extend(
            [
                "",
                f"### {name}",
                f"- Resolver modes: {data.get('resolver_mode_counts', {})}",
                f"- Error counts: {data.get('error_counts', {})}",
            ]
        )
        ref_line = format_reference_line(data)
        if ref_line:
            lines.append(f"- {ref_line}")

    lines.extend(
        [
            "",
            "## Pipeline configuration",
            f"- Retriever top-k: {config.get('pipeline', {}).get('retriever_top_k')}",
            f"- SQL retry limit: {config.get('pipeline', {}).get('sql_retry_limit')}",
            f"- Answer temperature: {config.get('pipeline', {}).get('answer_temperature')}",
        ]
    )
    return "\n".join(lines)


def generate_report(args: argparse.Namespace, config: Dict[str, Any]) -> None:
    metrics_path = resolve_path(args.metrics or config.get("artifacts", {}).get("metrics_path", "experiments/artifacts/metrics.json"))
    if not metrics_path.exists():
        raise SystemExit(f"Metrics file missing: {metrics_path}. Run `evaluate` first.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    report_path = resolve_path(args.output or config.get("artifacts", {}).get("report_path", "experiments/artifacts/report.md"))
    report_content = render_report(metrics, config)
    ensure_dir(report_path.parent)
    report_path.write_text(report_content, encoding="utf-8")
    print(f"Report saved to {report_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Experiments runner for NL->IR->SQL pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    seed_cmd = subparsers.add_parser("seed", help="Reinitialize the SQLite demo database.")
    seed_cmd.add_argument("--script", help="Override seed script path")
    seed_cmd.set_defaults(func=seed_database)

    base_cmd = subparsers.add_parser("run_baseline", help="Execute the configured baseline pipeline.")
    base_cmd.add_argument("--command", help="Command string overriding config value.")
    base_cmd.add_argument("--question-file", help="Path to questions file.")
    base_cmd.add_argument("--output", help="Output JSONL path.")
    base_cmd.add_argument("--shell", action="store_true", help="Run the command via shell=True.")
    base_cmd.set_defaults(func=run_baseline)

    system_cmd = subparsers.add_parser("run_system", help="Run the production system via HTTP and capture responses.")
    system_cmd.add_argument("--question-file", help="Override question file path.")
    system_cmd.add_argument("--output", help="Output JSONL path.")
    system_cmd.add_argument("--endpoint", help="Override endpoint URL.")
    system_cmd.add_argument("--timeout", type=int, help="HTTP timeout in seconds.")
    system_cmd.set_defaults(func=run_system)

    eval_cmd = subparsers.add_parser("evaluate", help="Compute metrics using stored artifacts.")
    eval_cmd.add_argument("--system-output", help="Path to system JSONL output.")
    eval_cmd.add_argument("--baseline-output", help="Path to baseline JSONL output.")
    eval_cmd.add_argument("--metrics", help="Destination metrics JSON path.")
    eval_cmd.add_argument("--reference", help="Path to reference answers JSONL (question/answer).")
    eval_cmd.set_defaults(func=evaluate_runs)

    report_cmd = subparsers.add_parser("report", help="Generate a Markdown report from metrics.")
    report_cmd.add_argument("--metrics", help="Metrics file to load.")
    report_cmd.add_argument("--output", help="Report file to write.")
    report_cmd.set_defaults(func=generate_report)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = load_config()
    args.func(args, config)


if __name__ == "__main__":
    main()
