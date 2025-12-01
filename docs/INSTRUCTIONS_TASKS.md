# 任务问数（Step 2）说明书

本说明聚焦“自然语言询问某人某任务是否完成”的任务问答实现，涵盖：
- 传统非大模型模式（rules / embeddings / hybrid / hybrid_plus_rules）
- LLM 驱动的 NL→JSON→SQL 模式（`hybrid_llm` / `/db/ask`）
- 意图识别、实体解析、SQL 查询与中文应答

---

## 功能概览

- 意图识别：关键词匹配（完成 / 未完成 / 状态 / 进度 / 是否完成 / 搞定 / 结束）
- 实体解析（多种模式）
  - `rules`：规则 + 关键词模糊匹配（字符归一化 + 等价 / 包含 / 字符重叠率）
  - `embeddings`：中文句向量检索 + 矩阵式 “Focus Query”（[query] + focus 编码，逐候选取最大相似度）
  - `hybrid`：向量‑only + FAISS Focus Query，使用拆分阈值（person/task）与 Top1‑Top2 margin 逻辑，不再做线性融合
  - `hybrid_plus_rules`：在 `hybrid` 基础上引入“规则助推”，在任务排序和 gating 上利用高规则得分做小幅加权（向量仍为主导）
  - `hybrid_llm`：LLM 先生成 `TaskQuerySpec`（NL→JSON），再用 `hybrid` 的小模型 + FAISS 在候选列表上对齐 person/task，并统一走 SQL compiler
- 数据源：SQLite 只读任务库（`tasks` 表），取该人该任务的最新一条记录（或通过 NL→SQL 返回多条）
- 应答：中文结论（`DONE` → 已完成；`TODO` → 未完成/待办）+ 最近更新时间；置信度不足时返回候选而非报错
- 调试输出：返回 SQL 模板与候选 Top‑k 打分、解析模式与阈值，以及 `nl_ir`（语义 IR）

---

## 运行配置与 API

- 端点：`GET /tasks/ask?q=...&topk=3&thresh=0.45`
- 主要环境变量：
  - `RESOLVER=rules|embeddings|hybrid|hybrid_plus_rules|hybrid_llm`（默认 `hybrid`）
  - `MOCK_EMB=1|0`（mock 向量开关）
  - `MODEL_NAME`（如 `BAAI/bge-small-zh-v1.5` / `sentence-transformers/all-MiniLM-L6-v2`）
  - `EMB_DIM`（bge-small-zh=512；MiniLM=384）
  - `EMB_URL`（容器化嵌入服务地址，如 `http://localhost:8080/embeddings`）
  - `TASKS_DB`（默认 `data/tasks.db`）
  - `TASKS_NL2SQL_LLM=1`：在 NL→JSON→SQL 流程中启用 LLM 优先解析（失败时回退规则）

- 运行时参数：
  - `topk`（默认 3）：候选返回个数
  - `thresh`：整体置信度阈值；若省略，则按模式采用“自适应默认阈值”（见后文）

---

## 数据库要求

`tasks` 表字段：
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `person TEXT NOT NULL`
- `task TEXT NOT NULL`
- `status TEXT NOT NULL`（期望值：`DONE` / `TODO`）
- `ts INTEGER NOT NULL`（epoch 毫秒）

传统固定查询模板（单条最新状态）：

```sql
SELECT id, person, task, status, ts
FROM tasks
WHERE person = ? AND task = ?
ORDER BY ts DESC, id DESC LIMIT 1
```

在 NL→SQL 模式下（`/db/ask` / `hybrid_llm`），SQL 将由 `compile_tasks_sql(TaskQuerySpec)` 统一生成，但仍遵守只读、带 `LIMIT` 的约束。

---

## 返回字段示例：`/tasks/ask`

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
    "intent": "task_status_single",
    "raw_query": "张三的提交9月周报完成了吗？",
    "...": "TaskQuerySpec 其它字段略"
  }
}
```

当置信度不足或未命中时，`answer` 会给出提示，并返回 `candidates` 供人工确认。

---

## 自测用例：`/tasks/ask`

请先初始化样例库：

```bash
python scripts/init_tasks_sqlite.py
```

典型请求（预期结果）：
- `GET /tasks/ask?q=张三的提交9月周报完成了吗？` → 期望 `DONE`（已完成）
- `GET /tasks/ask?q=张三的E3D接口联调现在什么状态？` → 期望 `TODO`（未完成 / 待办）
- `GET /tasks/ask?q=李四的整理工艺包V2是否已完成？` → 期望 `DONE`（已完成）
- `GET /tasks/ask?q=老张九月报搞定了没？` → 别名映射“老张 → 张三”，口语可命中或返回合理候选

遇到召回偏低可尝试：
- 将 `RESOLVER` 设为 `embeddings` / `hybrid` / `hybrid_plus_rules` / `hybrid_llm`
- 适度下调 `thresh`（如 `0.5`）
- 扩充别名词表（`EntityResolver.alias_map`）或标准化任务名称

---

## 模式与打分融合细节

- 规则分（`rules`）：
  - 文本归一化（去空白 / 标点）；完全相等 = 1.0；包含关系 ≈ 0.8；其余按字符集合交并比

- 向量分（`embeddings`）：
  - 使用 `sentence_transformers` 生成句向量，L2 归一化；FAISS `IndexFlatIP` 实现余弦相似 Top‑k；
  - Focus Query：将完整 query 与高置信规则候选一起编码，对每个候选取这些向量的最大相似度。

- 模式行为：
  - `rules`：仅用规则排序，单一阈值（默认 0.8）。
  - `embeddings`：矩阵法 Focus Query，默认阈值 0.45。
  - `hybrid`：向量‑only + FAISS Focus Query，内部使用 person/task 拆分阈值和 Top1‑Top2 margin 逻辑，整体行为等价于 `embeddings`（0.45）。
  - `hybrid_plus_rules`：在 `hybrid` 的候选上附加规则助推逻辑：
    - 若规则分较高（例如任务名高度匹配，且包含“接口”“联调”等关键词），对该候选的向量得分做小幅加分或放宽任务阈值；
    - 仍然保持向量分为主导。
  - `hybrid_llm`：
    - 先通过 `parse_task_query_nl(q)` 得到 `TaskQuerySpec`（可由 LLM 生成）；
    - 再用 `EntityResolver` 的向量逻辑在候选 person/task 列表上对齐 IR 中的 `person` / `task`；
    - 最终通过 `compile_tasks_sql(spec)` 生成只读 SQL 并查 SQLite。

- 自适应默认阈值（当未显式传入 `thresh` 时）：
  - `rules`: 0.8
  - `embeddings`: 0.45
  - `hybrid`: 0.45
  - `hybrid_plus_rules`: 0.45（内部再拆分 person/task 阈值和 margin）

---

## 性能与只读安全

- SQLite 只读连接（URI 模式），超时（默认 2s），查询均有 `LIMIT` 控制。
- 嵌入来源三选一：`MOCK_EMB=1`（mock）、本地 SBERT、容器化嵌入服务（`EMB_URL`）。
- Windows 推荐只用 FAISS；Milvus 通过 compose profile `milvus` 显式开启。

---

## NL→JSON 语义 IR（TaskQuerySpec）

为后续 NL→JSON→SQL 改造，项目在服务层提供了一个轻量语义 IR 模块：

- 模块：`app/services/nl2sql_engine.py`
- 核心模型：`TaskQuerySpec`，字段包括但不限于：
  - `intent`：如 `task_status_single` / `task_status_list` / `task_list_by_person` / `unknown`
  - `answer_mode`?可选提示回答模式，如 `completion_time_latest`
  - `person` / `task`：解析出的人名和任务名（可空）
  - `task_keywords`：任务相关关键词列表
  - `status`：状态过滤枚举列表，如 `[DONE]` / `[TODO]`
  - `time_range`：时间范围
  - `order_by`：排序字段 + 方向列表
  - `limit`：返回上限
  - `filters`: flexible list of `{field, op, value/values}` entries for multi-person/task scopes and advanced conditions.

- 接口函数：
  - `parse_task_query_nl(q: str) -> TaskQuerySpec`
  - 职责：只做 NL→JSON/IR 解析，不生成 SQL、不访问数据库；当前实现为规则+LLM 混合版（启用 LLM 时优先用 LLM，否则回退规则）。

当前 `/tasks/ask` 在保持原有 `TaskQueryEngine` 行为的基础上，会调用 `parse_task_query_nl` 并在响应 JSON 中附带一个 `nl_ir` 字段，便于调试和后续 NL→SQL 重构。

### NL→IR 规则约束原则

为防止规则层继续膨胀，我们沉淀出三条 guardrail 并已在代码层落地：

1. **只保留通用 heuristics**：规则层仅负责正则化（空白、别名）、基础 intent 归类，以及通用的时间范围 / due_range / multi-person filter 等“所有实现都需要兜底”的信息，避免为具体业务问法写死字符串。
2. **复杂语义交给 LLM / 调用方**：例如 `answer_mode=task_count_by_status` 这些聚合语义，只会在 LLM 结构化输出或调用方显式设置时触发；纯规则 parser 不再凭“还有多少任务未完成”之类的中文提示自动猜测，默认 `answer_mode=default`。
3. **扩展优先走 IR / SQL 通路**：新增能力优先扩展 `TaskQuerySpec` 字段、`build_task_query_plan`、answer formatter 这条可复用链路，而非在 `_post_process_intent` 中不断追加新的关键词 heuristics。

#### Phase 3：冻结 heuristics 边界（仅保留下列表格中的兜底逻辑）

| 分类 | 触发条件 | 作用 / 位置 |
| --- | --- | --- |
| Completion-time detection | 问句包含 “什么时候完成/搞定/结束” | `_post_process_intent`：`intent=task_history`、`answer_mode=completion_time_latest`、`status=[DONE]`、`limit=1` |
| Multi-person filter | 多个姓名（“张三和李四” 等） | `_ensure_person_filter_from_text`：插入 `filters=[{"field":"person","op":"in","values":[...]}]` 并清空顶层 person |
| Time / due range hints | “最近 N 天 / 本周 / 本月 / 截止” 等 | `_detect_time_range` / `_detect_due_range`：返回 `now-7d`、`start_of_week` 等标准表达，写入 `time_range` / `due_range` |
| Priority hints | “P1 / 高优 / 低优” 等 | `_detect_priority`：填充 `priority` |
| Limit / order safety | 缺省或异常的 limit、order_by | `_post_process_intent`：重置 `limit`（10/50/1）和 `order_by`（`ts desc`, `priority asc`） |

除上述条目之外不再新增 heuristics；所有新语义能力都应通过 LLM IR（intent/answer_mode/filters 等）或下游 SQL / formatter 来实现。

---

## NL→JSON→SQL 闭环试验端点：`/db/ask`

为验证 NL→JSON→SQL 整体链路，本项目提供一个实验性接口：`GET /db/ask?q=...`。

### 处理流程

1. 接收自然语言参数 `q`。
2. 调用 `parse_task_query_nl(q)` 得到 `TaskQuerySpec`（语义 IR）。
3. 调用 `compile_tasks_sql(spec)`（模块 `app/services/sql_compiler.py`）生成只读 SQL + 参数。
4. 使用 `SQLiteTasksStore.query(sql, params)` 执行查询。
5. 返回 JSON payload，包括：
   - `query`：原始 NL；
   - `ir`：序列化后的 `TaskQuerySpec`；
   - `sql`：生成的 SQL 字符串；
   - `params`：参数元组；
   - `rows`：从 `tasks` 表查出的原始记录列表。

### 注意事项

- `/db/ask` 不生成自然语言回答，只返回结构化 JSON，用于调试 NL→JSON→SQL 链路。
- 不会影响 `/tasks/ask` 的现有逻辑。
- 当 IR 不完整或无法安全编译为 SQL 时，会返回 4xx，并在 `detail.reason` 中给出原因。

### 对应测试：`tests/test_nl2sql_db_ask.py`

测试覆盖三个层面：
- IR 层：验证 `parse_task_query_nl` 对典型句子的解析是否符合预期。
- SQL 编译层：验证 `compile_tasks_sql` 生成的 SQL 是否只读且参数正确。
- API 层：验证 `/db/ask` 对合法/非法输入的行为是否符合约定。

---

## 模式选择建议

- 只想演示“无模型问数”：推荐 `RESOLVER=rules` 或 `hybrid`，`TASKS_NL2SQL_LLM` 关闭。
- 想对比“小模型 vs 规则”：使用 `rules` / `embeddings` / `hybrid` / `hybrid_plus_rules` 做横向对比。
- 想演示“LLM NL→JSON + 小模型对齐 + SQL compiler”的完整链路：设置
  - `RESOLVER=hybrid_llm`
  - `TASKS_NL2SQL_LLM=1`
  并结合 `/db/ask` 观察 IR / SQL / rows。
---

## Text2SQL 校验与 sqlglot AST（概览）

在 `resolver_mode="text2sql"` 或 `hybrid_llm` 下，任务问答还会走一条“LLM 直接生成 SQL，再由后端校验”的 Text2SQL 管线。为避免 LLM 生成的 SQL 破坏安全性或因方言差异频繁报错，本项目使用 `sqlglot` 做 AST 级解析与校验：

- **解析与白名单约束**
  - 使用 `sqlglot.parse_one(sql, read="sqlite")` 将 LLM 返回的 SQL 解析为 AST。
  - 只接受顶层为 `SELECT` 的语句（拒绝 UPDATE / INSERT / DELETE / DDL）。
  - 通过遍历 AST 中的 `Table` 节点，强制要求只访问 `task_latest` / `tasks` 两张表。

- **结构规范化**
  - 在 AST 上自动补齐 / 裁剪 `LIMIT`：
    - 未设置 `LIMIT` 时补 `LIMIT 100`；
    - 超过 100 行时强制改成 `LIMIT 100`。
  - 通过 AST 保证 `ORDER BY` 与 `LIMIT` 的位置合法，而不是用正则猜测。

- **轻量重写**
  - 对一些方言/语义小坑，结合 IR hint 做轻量重写（仍然保持只读语义不变）：
    - 时间范围：将 IR 中的 `time_range` / `due_range` / `created_range`（如 `now-7d` / `start_of_week`）解析成当前时间对应的毫秒区间；
    - 标签字段：在 IR `tags` 非空而 SQL 未使用 tags 时，自动注入 `tags LIKE '%...%'` 条件；
    - 优先级：在 IR `priority` 或问句中出现“高优P1”等提示时，自动注入 `priority = 1`，并移除把“高优P1任务”当成 task 文本的等值匹配。

- **安全守护与错误上报**
  - 拒绝占位符 `?` 和命名参数（`:_name` 等），防止 LLM 生成“留给应用层填充”的未绑定 SQL。
  - 拒绝跨方言函数（如 `DATE_SUB` / `CURDATE`）以及不合理的列比较（如 `ts > created_ts`）。
  - 当 AST 解析失败或触发上述规则时，会抛出 `Text2SQLValidationError`，在 `/tasks/ask` 的响应中体现为：
    - `error: "text2sql_invalid_sql"`；
    - `reason`: 包含具体原因（例如 “Failed to parse SQL: ...” / “Query references table '...' which is not allowed” 等）；
    - `invalid_sql` 或 `text2sql_raw_response`: 方便你直接复制 SQL / LLM 输出做进一步调试。

建议在调优 Text2SQL 时配合以下工具使用：

- `GET /db/ask?q=...`：对比 IR → SQL（`compile_tasks_sql`）路径与纯 Text2SQL 的差异。
- `scripts/batch_db_ask.py`：批量跑一组 NL 问题，打印 `answer` / `resolver_mode` / `intent` / `error` / `reason` / `sql` / `text2sql`，快速定位是 JSON、SQL 还是数据本身的问题。
