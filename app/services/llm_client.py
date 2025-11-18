from __future__ import annotations

from dataclasses import dataclass
from os import getenv
from typing import Any, Dict, Protocol


class LLMClient(Protocol):
    """Abstract LLM client for NL→JSON task query parsing.

    Implementations should take a natural‑language task query and return a
    JSON‑serializable dict that can be fed into TaskQuerySpec.parse_obj(...)
    (see nl2sql_engine.TaskQuerySpec for the expected schema).
    """

    def generate_task_query_spec(self, q: str) -> Dict[str, Any]:
        """Generate a TaskQuerySpec‑compatible JSON dict from a NL query."""
        ...


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


def get_llm_client() -> LLMClient:
    """Factory for LLMClient.

    Currently returns DummyLLMClient; this is the single place to change when
    integrating a real LLM provider.

    Env hints (reserved for future use):
    - TASKS_NL2SQL_LLM_PROVIDER: e.g., "openai", "deepseek", "dashscope"
    - TASKS_NL2SQL_LLM_MODEL:    model name / deployment id
    """
    provider = getenv("TASKS_NL2SQL_LLM_PROVIDER", "dummy")
    model = getenv("TASKS_NL2SQL_LLM_MODEL", "dummy")
    return DummyLLMClient(provider=provider, model=model)

