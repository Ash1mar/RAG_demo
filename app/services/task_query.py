from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import faiss
import numpy as np

from app.services.embeddings import Embedder
from app.services.nl2sql_engine import (
    parse_task_query_nl,
    build_task_query_plan,
    TaskQueryIntent,
)
from app.services.sql_compiler import compile_tasks_sql, TaskSqlCompileError
from app.tasks_store.base import TasksStore


def _norm_text(s: str) -> str:
    s = s.strip()
    s = re.sub(r"[\s\t\n\r]+", "", s)
    s = re.sub(r"[，。！？,.!?()（）:：；\-_/]", "", s)
    return s


INTENT_STATUS_KWS = [
    "完成",
    "未完成",
    "是否完成",
    "状态",
    "进度",
    "搞定",
    "结束",
    "done",
    "todo",
]


@dataclass
class ResolverConfig:
    topk: int = 3
    alpha_vec: float = 1.0   # default; may be tuned per mode
    thresh: float = 0.58     # default; may be overridden per mode
    mode: str = "hybrid"     # one of: "rules" | "embeddings" | "hybrid" | "hybrid_plus_rules"
    # Fine-grained controls (hybrid/hybrid_plus_rules)
    thresh_person: Optional[float] = None
    thresh_task: Optional[float] = None
    delta_min: Optional[float] = None        # Top1-Top2 margin for weak accept
    weak_task_min: Optional[float] = None    # low bar for weak accept
    rules_assist_min: Optional[float] = None # relaxed low bar when rules strongly agree


@dataclass
class EntityResolver:
    embedder: Embedder
    persons: List[str] = field(default_factory=list)
    tasks: List[str] = field(default_factory=list)
    cfg: ResolverConfig = field(default_factory=ResolverConfig)

    # internals
    _idx_person: Optional[faiss.Index] = field(init=False, default=None)
    _idx_task: Optional[faiss.Index] = field(init=False, default=None)
    _pers_vecs: Optional[np.ndarray] = field(init=False, default=None)
    _task_vecs: Optional[np.ndarray] = field(init=False, default=None)

    # simple alias dictionary for people (extendable)
    alias_map: Dict[str, str] = field(default_factory=lambda: {
        "老张": "张三",
    })

    def build(self) -> None:
        # Use original strings (keep case/punct for embeddings), normalization is only for rules
        p_vecs = self.embedder.encode(self.persons).astype(np.float32) if self.persons else np.zeros(
            (0, self.embedder.dim), dtype=np.float32
        )
        t_vecs = self.embedder.encode(self.tasks).astype(np.float32) if self.tasks else np.zeros(
            (0, self.embedder.dim), dtype=np.float32
        )
        if p_vecs.size:
            faiss.normalize_L2(p_vecs)
        if t_vecs.size:
            faiss.normalize_L2(t_vecs)
        self._pers_vecs = p_vecs
        self._task_vecs = t_vecs
        self._idx_person = faiss.IndexFlatIP(p_vecs.shape[1]) if p_vecs.size else None
        self._idx_task = faiss.IndexFlatIP(t_vecs.shape[1]) if t_vecs.size else None
        if self._idx_person is not None and p_vecs.shape[0] > 0:
            self._idx_person.add(p_vecs)
        if self._idx_task is not None and t_vecs.shape[0] > 0:
            self._idx_task.add(t_vecs)

    def _kw_score(self, query: str, cand: str) -> float:
        qn = _norm_text(query)
        cn = _norm_text(cand)
        if not qn or not cn:
            return 0.0
        if qn == cn:
            return 1.0
        if qn in cn or cn in qn:
            return 0.8
        qset = set(qn)
        cset = set(cn)
        inter = len(qset & cset)
        union = len(qset | cset) or 1
        return inter / union

    def _search(
        self,
        idx: Optional[faiss.Index],
        vecs: Optional[np.ndarray],
        cands: List[str],
        query: str,
        *,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        if idx is None or vecs is None or not cands:
            return []
        qv = self.embedder.encode([query]).astype(np.float32)
        faiss.normalize_L2(qv)
        topk = min(self.cfg.topk if k is None else k, len(cands))
        D, I = idx.search(qv, topk)
        scores = D[0]
        ids = I[0]
        out: List[Tuple[str, float]] = []
        for s, i in zip(scores, ids):
            if i < 0 or i >= len(cands):
                continue
            out.append((cands[i], float(s)))
        return out

    def _rule_rank(self, cands: List[str], query: str, *, k: Optional[int] = None) -> List[Tuple[str, float]]:
        items: List[Tuple[str, float]] = []
        for cand in cands:
            items.append((cand, float(self._kw_score(query, cand))))
        items.sort(key=lambda x: x[1], reverse=True)
        kk = self.cfg.topk if k is None else k
        return items[:kk]

    def _vector_rank_with_focus(
        self,
        vecs: Optional[np.ndarray],
        cands: List[str],
        query: str,
        focus: List[str],
        *,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        embeddings-only 的“聚焦查询”实现：
        1) 对完整 query 编码；
        2) 若存在规则法高置信候选（>=0.8），则将这些候选文本也编码；
        3) 用 [query] + focus 的多路向量与候选向量矩阵计算相似度，逐候选取最大值；
        4) 返回 Top‑k。
        说明：build() 已对库向量做 L2 归一化，此处对查询向量再次归一化。
        """
        if vecs is None or vecs.size == 0 or not cands:
            return []
        queries: List[str] = [query]
        if focus:
            queries.extend(focus)
        embs = self.embedder.encode(queries).astype(np.float32)
        faiss.normalize_L2(embs)
        sims = embs @ vecs.T  # (m, N) 余弦相似（已归一化）= 点积
        best = sims.max(axis=0)  # (N,)
        topk = min(self.cfg.topk if k is None else k, len(cands))
        order = np.argsort(-best)[:topk]
        return [(cands[int(i)], float(best[int(i)])) for i in order]

    def _faiss_rank_with_focus(
        self,
        idx: Optional[faiss.Index],
        vecs: Optional[np.ndarray],
        cands: List[str],
        query: str,
        focus: List[str],
        *,
        k: Optional[int] = None,
    ) -> List[Tuple[str, float]]:
        """
        FAISS 版本的“聚焦查询”：
        - 用 [query] + focus 编码并归一化；
        - 对每一路查询向量用 FAISS 在候选索引上检索 K_all=len(cands)；
        - 对同一候选取多路分数的最大值；
        - 返回 Top‑k。
        说明：IndexFlatIP + 归一化 = 余弦相似，和 _vector_rank_with_focus 的度量保持一致。
        """
        if idx is None or vecs is None or not cands:
            return []
        queries: List[str] = [query]
        if focus:
            queries.extend(focus)
        embs = self.embedder.encode(queries).astype(np.float32)
        faiss.normalize_L2(embs)
        n = len(cands)
        best = np.full(n, -1.0, dtype=np.float32)
        K_all = n  # 小规模集合取全量，避免截断导致的漏召回
        for i in range(embs.shape[0]):
            D, I = idx.search(embs[i : i + 1], K_all)
            scores = D[0]
            ids = I[0]
            for s, j in zip(scores, ids):
                if 0 <= j < n and s > best[j]:
                    best[j] = float(s)
        topk = min(self.cfg.topk if k is None else k, n)
        order = np.argsort(-best)[:topk]
        return [(cands[int(i)], float(best[int(i)])) for i in order]

    def resolve_person(self, query: str) -> List[Tuple[str, float]]:
        # alias substitution first
        q = query
        for alias, real in self.alias_map.items():
            if alias in q:
                q = q.replace(alias, real)
        mode = (self.cfg.mode or "hybrid").lower()
        if mode == "rules":
            return self._rule_rank(self.persons, q)
        if mode == "embeddings":
            # 规则粗提（>=0.8）作为聚焦词，提升短实体名对齐的分数稳定性
            focus = [cand for cand in self.persons if self._kw_score(q, cand) >= 0.8]
            return self._vector_rank_with_focus(self._pers_vecs, self.persons, q, focus)
        # hybrid（向量 only）：采用 FAISS 的聚焦查询，与 embeddings 行为保持一致
        focus = [cand for cand in self.persons if self._kw_score(q, cand) >= 0.8]
        return self._faiss_rank_with_focus(self._idx_person, self._pers_vecs, self.persons, q, focus, k=self.cfg.topk)

    def resolve_task(self, query: str) -> List[Tuple[str, float]]:
        mode = (self.cfg.mode or "hybrid").lower()
        if mode == "rules":
            return self._rule_rank(self.tasks, query)
        if mode == "embeddings":
            focus = [cand for cand in self.tasks if self._kw_score(query, cand) >= 0.8]
            return self._vector_rank_with_focus(self._task_vecs, self.tasks, query, focus)
        # hybrid（向量 only）：采用 FAISS 的聚焦查询，与 embeddings 行为保持一致
        focus = [cand for cand in self.tasks if self._kw_score(query, cand) >= 0.8]
        return self._faiss_rank_with_focus(self._idx_task, self._task_vecs, self.tasks, query, focus, k=self.cfg.topk)


def is_status_intent(q: str) -> bool:
    s = q.lower()
    return any(kw in s for kw in INTENT_STATUS_KWS)


def ts_to_str(ms: int) -> str:
    try:
        dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc).astimezone()
        return dt.strftime("%Y-%m-%d %H:%M:%S %Z")
    except Exception:
        return str(ms)


@dataclass
class TaskQueryEngine:
    tasks_store: TasksStore
    embedder: Embedder
    resolver: Optional[EntityResolver] = None
    resolver_mode: str = "hybrid"

    def ensure_built(self) -> None:
        if self.resolver is not None:
            return
        persons = self.tasks_store.list_persons()
        tasks = self.tasks_store.list_tasks()

        # 模式自适应阈值（可被运行时 thresh 覆盖）
        mode = self.resolver_mode

        def _default_thresh(m: str) -> float:
            m = (m or "hybrid").lower()
            if m == "rules":
                return 0.8
            if m == "embeddings":
                return 0.45
            # hybrid redefined as vector-only; align threshold with embeddings
            return 0.45  # hybrid

        cfg = ResolverConfig(mode=mode, thresh=_default_thresh(mode))
        res = EntityResolver(
            embedder=self.embedder,
            persons=persons,
            tasks=tasks,
            cfg=cfg,
        )
        res.build()
        # Per-mode defaults
        m = (mode or "hybrid").lower()
        if m == "hybrid" or m == "hybrid_llm":
            # vector-only with FAISS focus; split thresholds and delta logic
            res.cfg.thresh_person = 0.45
            res.cfg.thresh_task = 0.40
            res.cfg.delta_min = 0.09
            res.cfg.weak_task_min = 0.40
            res.cfg.alpha_vec = 1.0
        elif m == "hybrid_plus_rules":
            # vector-only ranking + rules-assisted gating (no linear fusion)
            res.cfg.thresh_person = 0.45
            res.cfg.thresh_task = 0.40
            res.cfg.delta_min = 0.09
            res.cfg.weak_task_min = 0.40
            res.cfg.rules_assist_min = 0.37
            res.cfg.alpha_vec = 0.9
        self.resolver = res

    def reload(self) -> Dict[str, Any]:
        self.resolver = None
        self.ensure_built()
        return {
            "persons": len(self.resolver.persons if self.resolver else []),
            "tasks": len(self.resolver.tasks if self.resolver else []),
        }

    def answer(self, q: str, topk: int = 3, thresh: Optional[float] = None) -> Dict[str, Any]:
        """Main entry for non‑LLM task status queries.

        说明：
        - 默认行为（rules/embeddings/hybrid/hybrid_plus_rules）保持不变，作为 legacy 解析路径；
        - 当 resolver_mode 显式设置为 "nl2sql" 时，会优先尝试 NL→JSON→SQL 分支，
          失败时回退到 legacy 行为。该模式仅用于本地调试和灰度实验。
        """
        mode_raw = (self.resolver_mode or "hybrid").lower()
        nl2sql_attempted = False
        nl2sql_error: Optional[Dict[str, Any]] = None
        if mode_raw == "nl2sql":
            nl2sql_attempted = True
            try:
                return self._answer_via_nl2sql(q)
            except Exception as exc:  # pragma: no cover - defensive fallback
                nl2sql_error = {
                    "resolver_mode": "nl2sql_failed_fallback_legacy",
                    "nl2sql_error": "unexpected_failure",
                    "nl2sql_reason": str(exc),
                }
                # fall through to legacy behavior

        # LLM 抽取 + hybrid 对齐的实验模式：统一走 TaskQuerySpec → SQL compiler。
        if mode_raw == "hybrid_llm":
            return self._answer_via_hybrid_llm(q, topk=topk, thresh=thresh)

        self.ensure_built()
        assert self.resolver is not None

        # runtime overrides
        if topk != self.resolver.cfg.topk:
            self.resolver.cfg.topk = int(topk)
        if thresh is not None and abs(float(thresh) - self.resolver.cfg.thresh) > 1e-9:
            self.resolver.cfg.thresh = float(thresh)

        intent = "status_query" if is_status_intent(q) else "unknown"
        person_hits = self.resolver.resolve_person(q)[:topk]
        task_hits = self.resolver.resolve_task(q)[:topk]

        # In hybrid_plus_rules, let rules lightly influence task ranking (vector still dominates)
        mode_decide_rank = (self.resolver.cfg.mode or "hybrid").lower()
        if mode_decide_rank == "hybrid_plus_rules" and task_hits:
            lambda_rule = 0.15  # small weight for rule score
            kw_tokens = ["接口", "联调"]
            enriched: List[Tuple[str, float, float]] = []
            for idx, (val, score_vec) in enumerate(task_hits):
                rule_s = float(self.resolver._kw_score(q, val))
                kw_bonus = 0.0
                if any(tok in q and tok in val for tok in kw_tokens):
                    kw_bonus = 0.05
                score_final = float(score_vec) + lambda_rule * rule_s + kw_bonus
                # strong rule priority: if very high rule score and currently in Top2/Top3, give extra boost
                if rule_s >= 0.8 and idx in (1, 2):
                    score_final += 0.2
                enriched.append((val, float(score_vec), score_final))
            enriched.sort(key=lambda x: x[2], reverse=True)
            task_hits = [(v, sv) for (v, sv, _) in enriched]

        payload: Dict[str, Any] = {
            "intent": intent,
            "resolver_mode": self.resolver.cfg.mode,
            "alpha_vec": round(float(self.resolver.cfg.alpha_vec), 4),
            "thresh": round(float(self.resolver.cfg.thresh), 4),
            "candidates": {
                "persons": [{"value": v, "score": round(float(s), 4)} for v, s in person_hits],
                "tasks": [{"value": v, "score": round(float(s), 4)} for v, s in task_hits],
            },
            "sql": "SELECT id, person, task, status, ts FROM tasks WHERE person = ? AND task = ? ORDER BY ts DESC, id DESC LIMIT 1",
        }

        best_p = person_hits[0] if person_hits else None
        best_t = task_hits[0] if task_hits else None
        if not best_p or not best_t:
            payload["answer"] = "未识别出人员或任务，请从候选中选择"
            return payload

        # New acceptance logic for vector-only modes (hybrid, hybrid_plus_rules)
        low_conf = False
        mode_decide = (self.resolver.cfg.mode or "hybrid").lower()
        if mode_decide in ("hybrid", "hybrid_plus_rules", "hybrid_llm"):
            tp = 0.45 if getattr(self.resolver.cfg, "thresh_person", None) is None else float(self.resolver.cfg.thresh_person)
            tt = 0.40 if getattr(self.resolver.cfg, "thresh_task", None) is None else float(self.resolver.cfg.thresh_task)
            delta_min = 0.09 if getattr(self.resolver.cfg, "delta_min", None) is None else float(self.resolver.cfg.delta_min)
            weak_min = tt if getattr(self.resolver.cfg, "weak_task_min", None) is None else float(self.resolver.cfg.weak_task_min)

            p_ok = best_p[1] >= tp
            t_ok = best_t[1] >= tt
            if not t_ok:
                second_t = task_hits[1][1] if len(task_hits) > 1 else -1.0
                if best_t[1] >= weak_min and (second_t >= 0) and (best_t[1] - second_t) >= delta_min:
                    t_ok = True
                    low_conf = True
                elif mode_decide == "hybrid_plus_rules":
                    # rules-assisted gating: strong rule agreement slightly relaxes the low bar
                    rule_s = float(self.resolver._kw_score(q, best_t[0]))
                    assist_min = weak_min if getattr(self.resolver.cfg, "rules_assist_min", None) is None else float(
                        self.resolver.cfg.rules_assist_min
                    )
                    if rule_s >= 0.8 and best_t[1] >= assist_min:
                        t_ok = True
                        low_conf = True

            if not p_ok or not t_ok:
                payload["answer"] = "识别置信度不足，请确认候选是否正确"
                return payload
            # prevent legacy single-threshold check below from blocking
            self.resolver.cfg.thresh = 0.0

        if best_p[1] < self.resolver.cfg.thresh or best_t[1] < self.resolver.cfg.thresh:
            payload["answer"] = "识别置信度不足，请确认候选是否正确"
            return payload

        person = best_p[0]
        task = best_t[0]
        rec = self.tasks_store.get_latest_status(person, task)
        if not rec:
            payload["answer"] = "未在任务库中找到对应记录"
            payload["person"] = person
            payload["task"] = task
            return payload

        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ts_to_str(ts) if ts >= 0 else "未知时间"

        zh_status = "已完成" if status == "DONE" else "未完成/待办"
        answer = f"{person} 的「{task}」{zh_status}（最近更新时间：{ts_str}）"

        payload.update(
            {
                "answer": answer,
                "person": person,
                "task": task,
                "status": status,
                "ts": ts,
            }
        )
        # annotate low confidence if accepted via weak rules
        try:
            if "low_conf" in locals() and low_conf:
                payload["answer"] = str(payload.get("answer", "")) + "，低置信度"
        except Exception:
            pass
        if nl2sql_attempted and nl2sql_error is not None:
            payload.update(nl2sql_error)
        return payload

    def _answer_via_hybrid_llm(self, q: str, topk: int = 3, thresh: Optional[float] = None) -> Dict[str, Any]:
        """NL->JSON (LLM/rules) + hybrid entity alignment + SQL compiler."""
        payload: Dict[str, Any] = {
            "intent": "unknown",
            "resolver_mode": "hybrid_llm",
        }

        # 1) NL -> IR (LLM/rules)
        try:
            spec = parse_task_query_nl(q)
            payload["nl_ir"] = spec.dict()
        except Exception as exc:
            payload["error"] = "hybrid_llm_parse_failed"
            payload["reason"] = str(exc)
            return payload

        # Map fine-grained TaskQueryIntent to coarse-grained label.
        spec_intent = getattr(spec, "intent", None)
        if isinstance(spec_intent, TaskQueryIntent):
            if spec_intent == TaskQueryIntent.task_list_by_person:
                payload["intent"] = "task_list"
            elif spec_intent == TaskQueryIntent.task_history:
                payload["intent"] = "task_history"
            elif spec_intent in (TaskQueryIntent.task_status_single, TaskQueryIntent.task_status_list):
                payload["intent"] = "status_query"
            elif spec_intent == TaskQueryIntent.person_summary:
                payload["intent"] = "person_summary"
            else:
                payload["intent"] = "unknown"

        # 2) 使用现有 EntityResolver（hybrid 向量逻辑）在候选列表上对齐 LLM 抽取的 person / task
        self.ensure_built()
        assert self.resolver is not None

        # runtime overrides（保持与 answer 一致的 topk / thresh 行为）
        if topk != self.resolver.cfg.topk:
            self.resolver.cfg.topk = int(topk)
        if thresh is not None and abs(float(thresh) - self.resolver.cfg.thresh) > 1e-9:
            self.resolver.cfg.thresh = float(thresh)

        # 优先使用 IR 中的 person/task 作为查询文本，否则退回整句
        q_person = spec.person or q
        q_task = spec.task or q

        person_hits = self.resolver.resolve_person(q_person)[:topk]
        task_hits = self.resolver.resolve_task(q_task)[:topk]

        payload.update(
            {
                "alpha_vec": round(float(self.resolver.cfg.alpha_vec), 4),
                "thresh": round(float(self.resolver.cfg.thresh), 4),
                "candidates": {
                    "persons": [{"value": v, "score": round(float(s), 4)} for v, s in person_hits],
                    "tasks": [{"value": v, "score": round(float(s), 4)} for v, s in task_hits],
                },
            }
        )

        best_p = person_hits[0] if person_hits else None
        best_t = task_hits[0] if task_hits else None
        if not best_p or not best_t:
            payload["error"] = "hybrid_llm_no_candidates"
            payload["answer"] = "未识别出人员或任务，请从候选中选择"
            return payload

        # 复用 hybrid 的接受逻辑（阈值 / margin 等），但 resolver_mode 为 hybrid_llm
        low_conf = False
        mode_decide = (self.resolver.cfg.mode or "hybrid").lower()
        if mode_decide in ("hybrid", "hybrid_plus_rules", "hybrid_llm"):
            tp = 0.45 if getattr(self.resolver.cfg, "thresh_person", None) is None else float(self.resolver.cfg.thresh_person)
            tt = 0.40 if getattr(self.resolver.cfg, "thresh_task", None) is None else float(self.resolver.cfg.thresh_task)
            delta_min = 0.09 if getattr(self.resolver.cfg, "delta_min", None) is None else float(self.resolver.cfg.delta_min)
            weak_min = tt if getattr(self.resolver.cfg, "weak_task_min", None) is None else float(self.resolver.cfg.weak_task_min)

            p_ok = best_p[1] >= tp
            t_ok = best_t[1] >= tt
            if not t_ok:
                second_t = task_hits[1][1] if len(task_hits) > 1 else -1.0
                if best_t[1] >= weak_min and (second_t >= 0) and (best_t[1] - second_t) >= delta_min:
                    t_ok = True
                    low_conf = True

            if not p_ok or not t_ok:
                payload["answer"] = "识别置信度不足，请确认候选是否正确"
                return payload

        person = best_p[0]
        task = best_t[0]

        if getattr(spec, "intent", None) == TaskQueryIntent.task_list_by_person:
            # For task_list_by_person, keep task unset so SQL lists all tasks of this person.
            spec.person = person
            spec.task = None
        else:
            # Default: single-task style queries still set both person and task.
            spec.person = person
            spec.task = task

        try:
            compiled = compile_tasks_sql(spec)
        except TaskSqlCompileError as exc:
            payload["error"] = "hybrid_llm_compile_failed"
            payload["reason"] = str(exc)
            payload["person"] = person
            payload["task"] = task
            return payload

        payload["sql"] = compiled.sql
        payload["params"] = compiled.params

        query_fn = getattr(self.tasks_store, "query", None)
        if query_fn is None:
            payload["error"] = "hybrid_llm_query_not_supported_by_tasks_store"
            return payload

        try:
            rows = query_fn(compiled.sql, compiled.params)
        except Exception as exc:  # pragma: no cover - defensive
            payload["error"] = "hybrid_llm_db_query_failed"
            payload["reason"] = str(exc)
            return payload

        payload["rows"] = rows

        if not rows:
            payload["answer"] = "未在任务库中找到匹配记录（hybrid_llm）"
            payload["person"] = person
            payload["task"] = task
            return payload

        spec_intent = getattr(spec, "intent", None)

        # 1) task_list_by_person: summarize multiple tasks for this person
        if spec_intent == TaskQueryIntent.task_list_by_person:
            count = len(rows)
            preview_tasks = []
            for rec in rows[:5]:
                t_name = str(rec.get("task", ""))
                t_status = str(rec.get("status", "")).upper()
                preview_tasks.append(f"{t_name}({t_status})")
            preview = "；".join(preview_tasks) if preview_tasks else "无任务"
            payload.update(
                {
                    "answer": f"{person} 当前共有 {count} 个任务：{preview}",
                    "person": person,
                    "task": None,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + "（低置信度）"
            return payload

        # 2) task_history: show count + latest status
        if spec_intent == TaskQueryIntent.task_history:
            count = len(rows)
            rec = rows[0]
            status = str(rec.get("status", "")).upper()
            ts = int(rec.get("ts", -1))
            ts_str = ts_to_str(ts) if ts >= 0 else "未知时间"
            payload.update(
                {
                    "answer": f"{person} 的「{task}」共有 {count} 条状态记录，最近一次为 {status}（时间：{ts_str}）",
                    "person": person,
                    "task": task,
                    "status": status,
                    "ts": ts,
                }
            )
            if low_conf:
                payload["answer"] = str(payload.get("answer", "")) + "（低置信度）"
            return payload

        # 3) default: keep the old single-latest behavior (status_query)
        rec = rows[0]
        status = str(rec.get("status", "")).upper()
        ts = int(rec.get("ts", -1))
        ts_str = ts_to_str(ts) if ts >= 0 else "未知时间"
        zh_status = "已完成" if status == "DONE" else "未完成/待办"
        payload.update(
            {
                "answer": f"{person} 的「{task}」{zh_status}（最近更新时间：{ts_str}）",
                "person": person,
                "task": task,
                "status": status,
                "ts": ts,
            }
        )
        if low_conf:
            payload["answer"] = str(payload.get("answer", "")) + "（低置信度）"
        return payload

    def _answer_via_nl2sql(self, q: str) -> Dict[str, Any]:
        """Experimental NL→JSON→SQL resolver path.

        流程：
        1. 调用 parse_task_query_nl(q) 得到 TaskQuerySpec 语义 IR；
        2. 调用 compile_tasks_sql(spec) 生成只读 SQL 和参数；
        3. 使用 tasks_store.query(sql, params) 执行查询；

        该分支是未来切换到 NL2SQL 的扩展点，目前仅在 resolver_mode="nl2sql"
        时用于本地调试/灰度，不改变默认 /tasks/ask 行为。
        """
        intent = "status_query" if is_status_intent(q) else "unknown"
        payload: Dict[str, Any] = {
            "intent": intent,
            "resolver_mode": "nl2sql",
        }

        # 1) NL -> IR
        try:
            spec = parse_task_query_nl(q)
            payload["nl_ir"] = spec.dict()
        except Exception as exc:
            payload["error"] = "nl2sql_parse_failed"
            payload["reason"] = str(exc)
            return payload

        # 2) IR -> SQL
        try:
            compiled = compile_tasks_sql(spec)
        except TaskSqlCompileError as exc:
            payload["error"] = "nl2sql_compile_failed"
            payload["reason"] = str(exc)
            return payload

        payload["sql"] = compiled.sql
        payload["params"] = compiled.params

        # 3) 执行 SQL（要求 tasks_store 提供 query(...) 辅助方法）
        query_fn = getattr(self.tasks_store, "query", None)
        if query_fn is None:
            payload["error"] = "nl2sql_query_not_supported_by_tasks_store"
            return payload

        try:
            rows = query_fn(compiled.sql, compiled.params)
        except Exception as exc:  # pragma: no cover - defensive
            payload["error"] = "nl2sql_db_query_failed"
            payload["reason"] = str(exc)
            return payload

        payload["rows"] = rows

        # 有结果时，构造一个简要的人类可读回答；否则仅返回 rows
        if rows:
            rec = rows[0]
            person = rec.get("person")
            task = rec.get("task")
            status = str(rec.get("status", "")).upper()
            ts = int(rec.get("ts", -1))
            ts_str = ts_to_str(ts) if ts >= 0 else "未知时间"
            zh_status = "已完成" if status == "DONE" else "未完成/待办"
            payload.update(
                {
                    "answer": f"{person} 的「{task}」{zh_status}（最近更新时间：{ts_str}）",
                    "person": person,
                    "task": task,
                    "status": status,
                    "ts": ts,
                }
            )
        else:
            payload["answer"] = "未在任务库中找到匹配记录（NL2SQL）"
        return payload
