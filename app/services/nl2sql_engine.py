from __future__ import annotations

from enum import Enum
from os import getenv
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from app.services.llm_client import get_llm_client


class TaskQueryIntent(str, Enum):
    """任务查询语义意图类型（只覆盖当前任务场景）。

    后续如果需要，可以在不破坏已有接口的前提下扩展：
    - task_status_single: 询问单个任务当前状态（当前默认）
    - task_status_list: 查询符合某些条件的任务列表
    - task_list_by_person: 按人员列出任务
    - unknown: 暂无法识别
    """

    task_status_single = "task_status_single"
    task_status_list = "task_status_list"
    task_list_by_person = "task_list_by_person"
    unknown = "unknown"


class TaskStatus(str, Enum):
    """任务状态枚举，对应 tasks.status 字段的语义层封装。"""

    DONE = "DONE"
    TODO = "TODO"
    ANY = "ANY"  # 不限定状态（仅在语义 IR 中使用）


class OrderByDirection(str, Enum):
    asc = "asc"
    desc = "desc"


class TimeRange(BaseModel):
    """时间范围（后续可扩展为真正的时间解析/标准化）。"""

    start: Optional[str] = Field(
        None,
        description="起始时间（ISO-8601 或保留的自然语言片段），含边界",
    )
    end: Optional[str] = Field(
        None,
        description="结束时间（ISO-8601 或保留的自然语言片段），含边界",
    )


class OrderBySpec(BaseModel):
    """排序字段 + 方向，供后续 NL→SQL 映射使用。"""

    field: str = Field(..., description="排序字段，如 ts、id、person 等")
    direction: OrderByDirection = Field(
        OrderByDirection.desc, description="排序方向（默认按时间倒序）"
    )


class TaskQuerySpec(BaseModel):
    """任务查询语义 IR（中间表示）。

    说明：
    - 该结构只负责承载“从自然语言中抽取出的任务查询语义”，不涉及 SQL 生成和数据库访问。
    - 未来可以由单独的 NL→SQL 模块，将 TaskQuerySpec 安全地映射为具体 SQL + 参数。
    - 设计尽量贴近当前 tasks 表结构，便于后续迁移：
        - person / task / status / time_range 等字段与 WHERE 子句强相关；
        - order_by / limit 与 ORDER BY / LIMIT 强相关。
    """

    intent: TaskQueryIntent = Field(
        TaskQueryIntent.task_status_single, description="识别出的查询意图"
    )
    raw_query: str = Field(..., description="原始自然语言查询句子")

    person: Optional[str] = Field(
        None, description="任务相关的人员名，未识别则为 None"
    )
    task: Optional[str] = Field(
        None, description="任务名称（解析出的最可能候选），未识别则为 None"
    )
    task_keywords: List[str] = Field(
        default_factory=list,
        description="从任务描述中抽取得到的关键字列表，用于模糊匹配/扩展",
    )
    status: List[TaskStatus] = Field(
        default_factory=list,
        description="要求的任务状态过滤，如 [DONE]、[TODO] 等；为空表示不限定",
    )

    time_range: Optional[TimeRange] = Field(
        None, description="查询时间范围，如最近一周、9 月等"
    )
    order_by: List[OrderBySpec] = Field(
        default_factory=list, description="排序字段列表"
    )
    limit: Optional[int] = Field(
        10, ge=1, le=200, description="返回条数上限，默认 10"
    )

    extra: Dict[str, Any] = Field(
        default_factory=dict,
        description="预留扩展字段（调试信息、模型打分、解析细节等）",
    )


_USE_LLM_FOR_NL2SQL = getenv("TASKS_NL2SQL_LLM", "0") == "1"


def parse_task_query_nl(q: str) -> TaskQuerySpec:
    """从自然语言任务查询句子中，抽取结构化语义 IR（TaskQuerySpec）。

    当前实现说明：
    - 默认使用轻量规则解析 `_rule_based_parse_task_query_nl`，不访问数据库、不生成 SQL；
    - 当环境变量 `TASKS_NL2SQL_LLM=1` 时，会优先尝试通过 LLMClient 抽象
      （app.services.llm_client.get_llm_client）生成 IR，失败时自动回退到规则解析。

    未来演进方向：
    - 在 LLMClient 实现中接入真实 LLM（如 OpenAI/DeepSeek/国产大模型），
      通过 prompt 让模型直接输出符合 TaskQuerySpec JSON Schema 的结构化对象，
      再用 TaskQuerySpec.parse_obj(...) 做校验；
    - 对上层调用方（TaskQueryEngine、SQL 编译器）保持接口稳定，只关心 TaskQuerySpec。
    """

    text = (q or "").strip()
    if not text:
        return TaskQuerySpec(intent=TaskQueryIntent.unknown, raw_query=q)

    # 可选：优先尝试通过 LLMClient 解析 IR（由环境变量控制）
    if _USE_LLM_FOR_NL2SQL:
        try:
            client = get_llm_client()
            raw = client.generate_task_query_spec(text)
            spec = TaskQuerySpec.parse_obj(raw)
            # 确保保留原始 query
            if not spec.raw_query:
                spec.raw_query = q
            # 标注来源，便于调试
            spec.extra.setdefault("nl2sql_source", "llm")
            return spec
        except Exception:
            # 安全兜底：LLM 解析失败时退回规则解析
            pass

    # 规则版本解析（当前主实现）
    spec = _rule_based_parse_task_query_nl(text)
    spec.raw_query = q
    spec.extra.setdefault("nl2sql_source", "rules")
    return spec


def _rule_based_parse_task_query_nl(q: str) -> TaskQuerySpec:
    """规则版 NL→JSON 解析实现。

    只做非常轻量的规则解析，便于先打通 NL→JSON→SQL 的整体流程；
    不访问数据库、不生成 SQL；可被 LLM 版本替换或作为兜底。
    """
    text = q.strip()

    # 1) 粗略识别意图
    intent = TaskQueryIntent.task_status_single
    if any(kw in text for kw in ("列表", "有哪些", "所有", "全部")):
        intent = TaskQueryIntent.task_status_list
    if any(kw in text for kw in ("张三", "李四", "老王", "老张")) and "有哪些" in text:
        intent = TaskQueryIntent.task_list_by_person

    # 2) 粗略解析人名 & 任务文本
    person: Optional[str] = None
    task: Optional[str] = None

    if "的" in text:
        # 示例：张三的E3D接口联调现在什么状态？
        left, _, right = text.partition("的")
        if left:
            person = left.strip()
        # 暂时把“的”右边剩余文本整体视为任务片段
        task = right.strip() or None
    else:
        # 没有“的”，暂不做复杂切分，全部作为任务描述
        task = text

    # 3) 粗略识别状态过滤
    status: List[TaskStatus] = []
    if any(kw in text for kw in ("完成了吗", "完成了没", "搞定了没", "搞定没有", "done")):
        status = [TaskStatus.DONE]
    elif any(kw in text for kw in ("未完成", "没完成", "还没做", "待办", "todo")):
        status = [TaskStatus.TODO]

    # 4) 默认排序 & limit
    order_by = [OrderBySpec(field="ts", direction=OrderByDirection.desc)]
    limit = 10

    return TaskQuerySpec(
        intent=intent,
        raw_query=q,
        person=person or None,
        task=(task or "").strip() or None,
        task_keywords=[],
        status=status,
        time_range=None,
        order_by=order_by,
        limit=limit,
        extra={},
    )

