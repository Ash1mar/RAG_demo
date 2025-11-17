# 无模型问数（Step 2）说明书

本说明聚焦“自然语言询问某人某任务是否完成”的非大模型实现：
- 意图识别
- 实体解析（人名 / 任务名）
- 固定 SQL 查询
- 中文应答

---

## 功能概览

- 意图识别：关键词匹配（完成 / 未完成 / 状态 / 进度 / 是否完成 / 搞定 / 结束）。
- 实体解析（多种模式）：
  - `rules`：规则 + 关键词模糊匹配（字符归一化 + 等价 / 包含 / 字符重叠率）。
  - `embeddings`：中文句向量检索 + 矩阵法 “Focus Query”（[query] + focus 编码，逐候选取最大相似度）。
  - `hybrid`：向量‑only + FAISS Focus Query，使用拆分阈值（person/task）与 Top1‑Top2 margin 逻辑，不再做线性融合。
  - `hybrid_plus_rules`：在 `hybrid` 基础上引入“规则助推”，在任务排序和 gating 上利用高规则得分做小幅加权（向量仍为主导）。
- 数据源：SQLite 只读任务库（`tasks` 表），取该人该任务的最新一条记录。
- 应答：中文结论（`DONE` → 已完成；`TODO` → 未完成 / 待办）+ 最近更新时间；置信度不足时返回候选而非报错。
- 调试输出：返回 SQL 模板与候选 Top‑k 打分、解析模式与阈值。

---

## 运行配置与 API

- 端点：`GET /tasks/ask?q=...&topk=3&thresh=0.45`
- 主要环境变量：
  - `RESOLVER=rules|embeddings|hybrid|hybrid_plus_rules`（默认 `hybrid`）。
  - `MOCK_EMB=1|0`（mock 向量开关）。
  - `MODEL_NAME`（如 `BAAI/bge-small-zh-v1.5` / `sentence-transformers/all-MiniLM-L6-v2`）。
  - `EMB_DIM`（bge-small-zh=512；MiniLM=384）。
  - `EMB_URL`（容器化嵌入服务地址，如 `http://localhost:8080/embeddings`）。
  - `TASKS_DB`（默认 `data/tasks.db`）。
- 运行时参数：
  - `topk`（默认 3）：候选返回个数。
  - `thresh`：整体置信度阈值；若省略，则按模式采用“自适应默认阈值”（见后文）。

---

## 数据库要求

`tasks` 表字段：
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `person TEXT NOT NULL`
- `task TEXT NOT NULL`
- `status TEXT NOT NULL`（期望值：`DONE` / `TODO`）
- `ts INTEGER NOT NULL`（epoch 毫秒）

读取语句（固定模板，查单条最新状态）：
```sql
SELECT id, person, task, status, ts
FROM tasks
WHERE person = ? AND task = ?
ORDER BY ts DESC, id DESC LIMIT 1
```

---

## 返回字段（/tasks/ask 示例）

```json
{
  "answer": "张三 的「提交9月周报」已完成（最近更新时间：2024-09-30 18:00:00 CST）",
  "status": "DONE",
  "person": "张三",
  "task": "提交9月周报",
  "ts": 1727690400000,
  "sql": "SELECT id, person, task, status, ts FROM tasks WHERE person = ? AND task = ? ORDER BY ts DESC, id DESC LIMIT 1",
  "intent": "status_query",
  "resolver_mode": "hybrid",
  "alpha_vec": 1.0,
  "thresh": 0.45,
  "candidates": {
    "persons": [{"value": "张三", "score": 0.91}],
    "tasks":   [{"value": "提交9月周报", "score": 0.88}]
  },
  "nl_ir": {
    "...": "TaskQuerySpec (见下)"
  }
}
```

当置信度不足或未命中时，`answer` 会给出提示，并返回 `candidates` 供人工确认。

---

## 自测用例（/tasks/ask）

请先初始化样例库：

```bash
python scripts/init_tasks_sqlite.py
```

典型请求（预期结果）：

- `GET /tasks/ask?q=张三的提交9月周报完成了吗？` → 期望 `DONE`（已完成）。
- `GET /tasks/ask?q=张三的E3D接口联调现在什么状态？` → 期望 `TODO`（未完成 / 待办）。
- `GET /tasks/ask?q=李四的整理工艺包V2是否已完成？` → 期望 `DONE`（已完成）。
- `GET /tasks/ask?q=老张九月报搞定了没？` → 别名映射“老张 → 张三”，口语可命中或返回合理候选。

遇到召回偏低可尝试：
- 将 `RESOLVER` 设为 `embeddings` / `hybrid` / `hybrid_plus_rules`（三者均为 Focus Query，只是融合逻辑不同）。
- 适度下调 `thresh`（如 `0.5`）。
- 扩充别名词表（`EntityResolver.alias_map`）或标准化任务名称。

---

## 模式与打分融合细节

- 规则分（`rules`）：
  - 文本归一化（去空白 / 标点）；完全相等 = 1.0；包含关系 = 0.8；其余按字符集合交并比。
- 向量分（`embeddings`）：
  - 使用 `sentence-transformers` 生成句向量，L2 归一化；FAISS `IndexFlatIP` 实现余弦相似度 Top‑k。
- 模式行为：
  - `rules`：仅用规则排序，单一阈值（默认 0.8）。
  - `embeddings`：矩阵法 Focus Query（`[query] + focus` 编码，逐候选取最大相似度），默认阈值 0.45。
  - `hybrid`：向量‑only + FAISS Focus Query，内部使用 person/task 拆分阈值和 Top1‑Top2 margin 逻辑，默认整体行为等价于 `embeddings`（0.45）。
  - `hybrid_plus_rules`：在 `hybrid` 的候选上附加规则助推逻辑：
    - 若规则分较高（例如任务名高度匹配，且包含“接口”“联调”等关键词），对该候选的向量得分做小幅加分或放宽任务阈值；
    - 仍然保持向量分为主导。
- 自适应默认阈值（当未显式传 `thresh` 时）：
  - `rules`: 0.8
  - `embeddings`: 0.45
  - `hybrid`: 0.45
  - `hybrid_plus_rules`: 0.45（内部再拆分 person/task 阈值和 margin）

你仍可以通过 `&thresh=...` 显式覆盖阈值。

---

## 性能与只读安全

- SQLite 只读连接（URI 模式），超时（默认 2s），查询均 `LIMIT` 控制。
- 嵌入来源三选一：`MOCK_EMB=1`（mock）、本地 SBERT、容器化嵌入服务（`EMB_URL`）。
- Windows 推荐只用 FAISS；Milvus 通过 compose profile `milvus` 显式开启。

---

## NL→JSON 语义 IR（TaskQuerySpec）

为后续 NL→JSON→SQL 改造，项目在服务层新增了一个轻量语义 IR 模块：

- 模块：`app/services/nl2sql_engine.py`
- 核心模型：`TaskQuerySpec`，字段包括但不限于：
  - `intent`：如 `task_status_single` / `task_status_list` / `task_list_by_person` / `unknown`。
  - `person` / `task`：解析出的人员名和任务名（可空）。
  - `task_keywords`：任务相关关键词列表（预留给模糊匹配）。
  - `status`：状态过滤枚举列表，如 `[DONE]` / `[TODO]`。
  - `time_range`：时间范围（起止字符串，后续可升级为标准时间）。
  - `order_by`：排序字段 + 方向列表。
  - `limit`：返回上限。
- 接口函数：
  - `parse_task_query_nl(q: str) -> TaskQuerySpec`
  - 职责：只做 NL→JSON/IR 解析，不生成 SQL、不访问数据库；当前实现为简易规则版，后续可替换为小模型 / LLM。

当前 `/tasks/ask` 在保持原有 `TaskQueryEngine` 行为的基础上，会调用 `parse_task_query_nl` 并在响应 JSON 中附带一个 `nl_ir` 字段，便于调试和后续的 NL→SQL 重构。

---

## NL→JSON→SQL 闭环试验端点：`/db/ask`

为验证 NL→JSON→SQL 整体链路，本项目提供一个实验性接口：`GET /db/ask?q=...`。

### 处理流程

1. 接收自然语言参数 `q`。
2. 调用 `parse_task_query_nl(q)` 得到 `TaskQuerySpec`（语义 IR）。
3. 调用 `compile_tasks_sql(spec)`（模块 `app/services/sql_compiler.py`）生成只读 SQL + 参数。
4. 使用 `SQLiteTasksStore.query(sql, params)` 执行查询。
5. 返回 JSON payload，包括：
   - `query`：原始 NL。
   - `ir`：序列化后的 `TaskQuerySpec`。
   - `sql`：生成的 SQL 字符串。
   - `params`：参数元组。
   - `rows`：从 `tasks` 表查出的原始记录列表。

### 注意事项

- `/db/ask` 不生成自然语言回答，只返回结构化 JSON，用于调试 NL→JSON→SQL 链路。
- 不会影响 `/tasks/ask` 的现有逻辑。
- 若 IR 不完整或无法安全编译为 SQL，会返回 4xx，并在 `detail.reason` 中给出原因。

