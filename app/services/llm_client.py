from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol

import httpx
from pydantic import ValidationError

from app.config import llm_settings


class LLMClient(Protocol):
    """Abstract LLM client for NL→JSON task query parsing.

    Implementations should take a natural‑language task query and return a
    JSON‑serializable dict that can be fed into TaskQuerySpec.parse_obj(...)
    (see nl2sql_engine.TaskQuerySpec for the expected schema).
    """

    def generate_task_query_spec(self, q: str) -> Dict[str, Any]:
        """Generate a TaskQuerySpec‑compatible JSON dict from a NL query."""
        ...


class LLMParseError(Exception):
    """Raised when the LLM output cannot be parsed into TaskQuerySpec."""


@dataclass
class DummyLLMClient:
    """Dummy implementation used before wiring a real LLM provider.

    Current behavior:
    - Raises NotImplementedError to force callers to fall back to the
      rule‑based parser in nl2sql_engine.

    TODO:
    - Replace or extend this class with real HTTP SDK calls to OpenAI /
      DeepSeek / 国内大模型等，根据环境变量选择 provider / model。
    """

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
            # Use deterministic decoding for NL→IR so that the same
            # question produces a stable TaskQuerySpec.
            "options": {
                "temperature": 0.0,
            },
        }

        url = f"{self.base_url.rstrip('/')}/api/chat"

        try:
            resp = httpx.post(url, json=payload, timeout=30.0)
            resp.raise_for_status()
        except httpx.HTTPError as exc:  # includes network errors and non‑2xx
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
    """Build the user prompt for TaskQuerySpec generation.

    The content is bilingual (ZH/EN) to keep behavior explicit even when
    the underlying model is primarily Chinese‑tuned.

    TODO: Refine wording and/or move to a dedicated prompts module.
    """

    return (
        "下面是一句关于任务状态/任务列表的自然语言查询，请你根据查询内容，"
        "严格抽取出一个符合 TaskQuerySpec JSON Schema 的 JSON 对象。\n\n"
        "必须遵守：\n"
        "1) 只输出 JSON，不要输出任何解释性文字；\n"
        "2) 所有字段名必须来自 Schema，例如：intent, raw_query, person, task, "
        "task_keywords, status, time_range, order_by, limit, answer_mode, filters, extra；\n"
        "3) 如果某些信息在问句中没有明确给出，可以留空或使用合理默认值，"
        "但不要胡乱臆造任务或人员；\n"
        "4) raw_query 字段必须原样填入用户的原始问句；\n"
        "5) 当用户问“什么时候完成/搞定/结束”时，answer_mode 必须是 "
        '"completion_time_latest"，并将 status 设为 ["DONE"]、limit=1、'
        "order_by 按 ts 降序；\n"
        "6) 当问题涉及多个姓名/实体时，使用 filters 中 "
        '{"field":"person","op":"in","values":[...]} 表达范围，并清空顶层 person；\n'
        "7) 当问题包含优先级（P0/P1/高优/低优）或截止/到期/最近 N 天/本周/本月等时间短语时，"
        "必须填充 priority、time_range 或 due_range。\n\n"
        "The following is a natural language query about task status or task list. "
        "You MUST output ONLY one JSON object that conforms to the TaskQuerySpec "
        "JSON Schema. Do not add any explanations.\n\n"
        f"用户问句 / user query:\n{q}"
    )


TASK_QUERY_SYSTEM_PROMPT = (
    "You are a strict JSON schema parser for task-status queries.\n"
    "Convert each Chinese task question into a JSON object that matches the "
    "TaskQuerySpec schema (intent, raw_query, person, task, task_keywords, "
    "project, tags, status, time_range, due_range, created_range, order_by, "
    "limit, answer_mode, filters, extra).\n\n"
    "Requirements:\n"
    "- Output ONLY a JSON object, no extra text.\n"
    "- Field names must exactly match the schema (case-sensitive).\n"
    "- If the query omits a field, leave it null/empty or use a safe default; "
    "  never invent people/tasks/projects.\n"
    "- raw_query must be the original user query.\n"
    "- intent must be one of: task_status_single | task_status_list | "
    "  task_list_by_person | task_history | person_summary.\n"
    "- answer_mode must be one of: default | completion_time_latest | "
    "  task_count_by_status.\n"
    "- status values must use the enum strings: DONE, TODO, IN_PROGRESS, "
    "  BLOCKED, ANY.\n"
    "- When the question is about *when a task was finished* "
    "  (Chinese phrasing meaning “finished / done / completed”), set "
    "  answer_mode=completion_time_latest, status=[\"DONE\"], limit=1, and "
    "  order_by ts DESC.\n"
    "- When the question is about *how many tasks remain* or asks for counts "
    "  per status, set answer_mode=task_count_by_status and include the "
    "  appropriate status/time_range/due_range filters.\n"
    "- When multiple people appear, use a filter "
    "  {\"field\":\"person\",\"op\":\"in\",\"values\":[...]} and clear the "
    "  top-level person field.\n"
    "- Populate project/tags/priority/time_range/due_range whenever the query "
    "  mentions them (e.g., P0/P1, “this week/this month”, “deadline/due by …”).\n"
    "- Keep LIMIT reasonable (default 10; list-by-person up to 50) unless the "
    "  user explicitly asks otherwise.\n"
    "- Use filters with eq / in / like / gte / lte for additional constraints.\n"
    "You MUST respect the JSON Schema provided via the `format` parameter."
)


def get_llm_client() -> LLMClient:
    """Factory for LLMClient.

    Decides implementation based on environment‑driven llm_settings.
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
