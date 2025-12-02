from __future__ import annotations

from os import getenv
from typing import Any, Dict

from pydantic import BaseModel, Field


class LLMSettings(BaseModel):
    """LLM-related settings loaded from environment variables."""

    enabled: bool = Field(True, alias="LLM_ENABLED")
    provider: str = Field("dummy", alias="LLM_PROVIDER")
    model: str = Field("qwen2.5-coder:7b", alias="LLM_MODEL")
    ollama_base_url: str = Field("http://localhost:11434", alias="LLM_OLLAMA_BASE_URL")
    openai_base_url: str = Field(
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
        alias="LLM_OPENAI_BASE_URL",
    )
    api_key: str = Field("", alias="LLM_API_KEY")


def _load_llm_settings() -> LLMSettings:
    raw: Dict[str, Any] = {}

    v = getenv("LLM_ENABLED")
    if v is not None:
        raw["LLM_ENABLED"] = v

    v = getenv("LLM_PROVIDER")
    if v is not None:
        raw["LLM_PROVIDER"] = v

    v = getenv("LLM_MODEL")
    if v is not None:
        raw["LLM_MODEL"] = v

    v = getenv("LLM_OLLAMA_BASE_URL")
    if v is not None:
        raw["LLM_OLLAMA_BASE_URL"] = v

    v = getenv("LLM_OPENAI_BASE_URL")
    if v is not None:
        raw["LLM_OPENAI_BASE_URL"] = v

    v = getenv("LLM_API_KEY")
    if v is not None:
        raw["LLM_API_KEY"] = v

    # Backward compatibility with earlier TASKS_NL2SQL_* env vars
    if "LLM_PROVIDER" not in raw:
        legacy_provider = getenv("TASKS_NL2SQL_LLM_PROVIDER")
        if legacy_provider is not None:
            raw["LLM_PROVIDER"] = legacy_provider

    if "LLM_MODEL" not in raw:
        legacy_model = getenv("TASKS_NL2SQL_LLM_MODEL")
        if legacy_model is not None:
            raw["LLM_MODEL"] = legacy_model

    return LLMSettings(**raw)


llm_settings = _load_llm_settings()
