from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import httpx
from pydantic import ValidationError

from app.config import llm_settings


class LLMClient(Protocol):
    """Abstract LLM client for NL鈫扟SON task query parsing.

    Implementations should take a natural鈥憀anguage task query and return a
    JSON鈥憇erializable dict that can be fed into TaskQuerySpec.parse_obj(...)
    (see nl2sql_engine.TaskQuerySpec for the expected schema).
    """

    def generate_task_query_spec(self, q: str) -> Dict[str, Any]:
        """Generate a TaskQuerySpec鈥慶ompatible JSON dict from a NL query."""
        ...


class LLMParseError(Exception):
    """Raised when the LLM output cannot be parsed into TaskQuerySpec."""


@dataclass
class DummyLLMClient:
    """Dummy implementation used before wiring a real LLM provider.

    Current behavior:
    - Raises NotImplementedError to force callers to fall back to the
      rule鈥慴ased parser in nl2sql_engine.

    TODO:
    - Replace or extend this class with real HTTP SDK calls to OpenAI /
      DeepSeek / 鍥藉唴澶фā鍨嬬瓑锛屾牴鎹幆澧冨彉閲忛€夋嫨 provider / model銆?    """

    provider: str = "dummy"
    model: str = "dummy"

    def generate_task_query_spec(self, q: str) -> Dict[str, Any]:
        raise NotImplementedError(
            "DummyLLMClient cannot generate TaskQuerySpec; "
            "configure a real LLM client implementation and update get_llm_client()."
        )


@dataclass
class OllamaLLMClient:
    """LLM client implementation backed by a local Ollama server."""

    base_url: str
    model: str

    def generate_task_query_spec(self, q: str) -> Dict[str, Any]:
        """Call Ollama /api/chat with structured outputs targeting TaskQuerySpec."""
        # Import here to avoid circular import: nl2sql_engine -> llm_client -> nl2sql_engine
        from app.services.nl2sql_engine import TaskQuerySpec

        schema = TaskQuerySpec.schema()

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": TASK_QUERY_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": build_task_query_user_prompt(q),
                },
            ],
            "stream": False,
            "format": schema,
            # Use deterministic decoding for NL鈫扞R so that the same
            # question produces a stable TaskQuerySpec.
            "options": {
                "temperature": 0.0,
            },
        }

        url = f"{self.base_url.rstrip('/')}/api/chat"

        try:
            resp = httpx.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # includes network errors and non鈥?xx
            raise LLMParseError(f"Failed to call Ollama /api/chat: {exc}") from exc

        try:
            data = resp.json()
        except ValueError as exc:
            raise LLMParseError("Ollama response is not valid JSON") from exc

        try:
            message = data["message"]
            content = message["content"]
        except (TypeError, KeyError) as exc:
            raise LLMParseError("Ollama response missing expected message.content") from exc

        try:
            if isinstance(content, str):
                spec = TaskQuerySpec.parse_raw(content)
            else:
                # Be tolerant if Ollama already returns structured JSON
                spec = TaskQuerySpec.parse_obj(content)
        except (ValueError, TypeError, ValidationError) as exc:
            raise LLMParseError(f"Failed to parse TaskQuerySpec from LLM output: {exc}") from exc

        return spec.dict()


def build_task_query_user_prompt(q: str) -> str:
    """Build the user prompt for TaskQuerySpec generation."""

    return (
        "Here is a natural-language question about task status or task lists. "
        "Convert it into a JSON object that satisfies the TaskQuerySpec JSON schema.\n\n"
        "Requirements:\n"
        "1) Output only JSON (no explanations).\n"
        "2) Field names must come from the schema: intent, raw_query, is_supported, intent_confidence, raw_intent_nl, person, task, "
        "task_keywords, project, tags, status, time_range, due_range, created_range, "
        "order_by, limit, answer_mode, filters, extra.\n"
        "3) raw_query must contain the original user query verbatim.\n"
        "4) Do not invent people/tasks/projects; leave fields null/empty when unspecified.\n"
        "5) When multiple people/tasks appear, move them into filters such as {\"field\":\"person\",\"op\":\"in\",\"values\":[...]} and clear the top-level person/task.\n"
        "6) Fill priority/time_range/due_range when the query mentions priority keywords "
        "(P0/P1/high/low) or time phrases (this week/month, last N days, deadline/due). "
        "Use normalized strings like now-7d or start_of_week.\n"
        "7) answer_mode must be one of \"default\", \"completion_time_latest\", "
        "\"task_count_by_status\", \"person_summary_by_project\", \"overdue_count_by_person\": "
        "current status -> default; when-finished questions -> completion_time_latest "
        "(also set status=[\"DONE\"], limit=1, order_by ts desc); status-count questions "
        "-> task_count_by_status (supply status/time_range/due_range); project/person summaries "
        "-> person_summary_by_project; overdue stats -> overdue_count_by_person (provide "
        "due_range/time_range and non-DONE status).\n"
        "8) status must use enum strings (DONE/TODO/IN_PROGRESS/BLOCKED/ANY).\n"
        "9) LIMIT defaults to 10 (task_list_by_person can use 50) unless the question demands "
        "otherwise; order_by defaults to ts desc then priority asc.\n"
        "10) filters should express remaining constraints with eq/in/like/gte/lte "
        "(project, tags, status, priority, etc.).\n"
        "11) Provide is_supported=true only when the query clearly maps to an existing simple intent; otherwise set is_supported=false and prefer intent=\"unknown\".\n"
        "12) intent_confidence must be a float between 0 and 1 (e.g. 0.9 for strong matches); raw_intent_nl should briefly summarize the user intent in natural language.\n\n"
        "Return a single JSON object and nothing else.\n\n"
        f"User query:\n{q}"
    )


TASK_QUERY_SYSTEM_PROMPT = (
    "You are a strict JSON schema parser for task-status queries.\n"
    "Convert each Chinese task question into a JSON object that matches the TaskQuerySpec schema (intent, raw_query, is_supported, intent_confidence, raw_intent_nl, person, task, task_keywords, project, tags, status, time_range, due_range, created_range, order_by, limit, answer_mode, filters, extra).\n\n"
    "Requirements:\n"
    "- Output ONLY a JSON object, no extra text.\n"
    "- Field names must exactly match the schema (case-sensitive).\n"
    "- If the query omits a field, leave it null/empty or use a safe default; never invent people/tasks/projects.\n"
    "- raw_query must be the original user query.\n"
    "- intent must be one of task_status_single | task_status_list | task_list_by_person | task_history | person_summary | unknown.\n"
    "- answer_mode must be one of default | completion_time_latest | task_count_by_status | person_summary_by_project | overdue_count_by_person.\n"
    "- status values must use the enum strings (DONE, TODO, IN_PROGRESS, BLOCKED, ANY).\n"
    "- When the query asks \"when was it finished\" (phrases meaning finished/done/completed), set answer_mode=completion_time_latest, status=[\"DONE\"], limit=1, order_by ts desc.\n"
    "- When the query asks \"how many tasks remain\" or requests counts per status, set answer_mode=task_count_by_status and include the relevant status/time_range/due_range filters.\n"
    "- When the query asks for project/person summaries, set answer_mode=person_summary_by_project and provide the necessary project/person filters.\n"
    "- When the query asks for overdue counts per person, set answer_mode=overdue_count_by_person and provide due_range/time_range plus non-DONE status buckets.\n"
    "- When multiple people appear, use {\"field\":\"person\",\"op\":\"in\",\"values\":[...]} and clear the top-level person field.\n"
    "- Populate project/tags/priority/time_range/due_range whenever the query mentions them (P0/P1, this week/month, deadline/due, etc.).\n"
    "- Keep LIMIT reasonable (default 10; list-by-person up to 50) unless the user explicitly asks otherwise.\n"
    "- Use filters with eq/in/like/gte/lte for additional constraints.\n"
    "- Provide is_supported=true only when the question clearly belongs to the supported simple intents; otherwise set is_supported=false and prefer intent=\"unknown\".\n"
    "- intent_confidence must be 0~1 (float) and raw_intent_nl should briefly describe the perceived intent in Chinese.\n"
    "You MUST respect the JSON Schema provided via the `format` parameter."
)


def get_llm_client() -> LLMClient:
    """Factory for LLMClient.

    Decides implementation based on environment鈥慸riven llm_settings.
    """

    if not llm_settings.enabled or llm_settings.provider == "dummy":
        return DummyLLMClient(provider=llm_settings.provider, model=llm_settings.model)

    if llm_settings.provider == "ollama":
        return OllamaLLMClient(
            base_url=llm_settings.ollama_base_url,
            model=llm_settings.model,
        )

    # TODO: add support for other providers (OpenAI / DeepSeek / etc.)
    raise NotImplementedError(f"Unsupported LLM provider: {llm_settings.provider}")
