# 无模型问数（Step 2）说明书

本说明聚焦“自然语言询问某人某任务是否完成”的非大模型实现：意图识别 + 实体解析（人名/任务名）+ 固定 SQL 查询 + 中文应答。

---

## 功能概览

- 意图识别：关键词匹配（完成/未完成/状态/进度/是否完成/搞定/结束）
- 实体解析（可切换三种模式）：
  - rules：规则/关键词/模糊匹配（字符归一化 + 等值/包含/字符重叠率）
  - embeddings：中文句向量检索 + 矩阵法“聚焦查询”（[query]+focus 编码，逐候选取最大相似度）
  - hybrid：向量分与规则分加权融合（`alpha_vec`，默认 0.65）
- 数据源：SQLite 只读任务库（`tasks` 表），取该人该任务的最新一条记录
- 应答：中文结论（DONE→已完成；TODO→未完成/待办）+ 最近更新时间；置信度不足时返回候选而非报错
- 调试输出：返回 SQL 模板与候选 Top‑k 打分、解析模式与阈值

---

## 运行配置（与 API）

- 端点：`GET /tasks/ask?q=...&topk=3&thresh=0.45`
- 主要环境变量：
  - `RESOLVER=rules|embeddings|hybrid`（默认 `hybrid`）
  - `MOCK_EMB=1|0`（mock 向量开关）
  - `MODEL_NAME`（如 `BAAI/bge-small-zh-v1.5` / `sentence-transformers/all-MiniLM-L6-v2`）
  - `EMB_DIM`（bge-small-zh=512；MiniLM=384）
  - `EMB_URL`（容器化嵌入服务地址，如 `http://localhost:8080/embeddings`）
  - `TASKS_DB`（默认 `data/tasks.db`）
- 运行时参数：
  - `topk`（默认 3）：候选返回个数
  - `thresh`（默认 0.58）：人名/任务分数阈值，低于则给候选提示

---

## 数据库要求

`tasks` 表字段：
- `id INTEGER PRIMARY KEY AUTOINCREMENT`
- `person TEXT NOT NULL`
- `task TEXT NOT NULL`
- `status TEXT NOT NULL`（期望 `DONE` 或 `TODO`）
- `ts INTEGER NOT NULL`（epoch 毫秒）

读取语句（固定模板）：
```
SELECT id, person, task, status, ts
FROM tasks
WHERE person = ? AND task = ?
ORDER BY ts DESC, id DESC LIMIT 1
```

---

## 返回字段（示例）

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
    "persons": [{"value": "张三", "score": 0.91}, ...],
    "tasks":   [{"value": "提交9月周报", "score": 0.88}, ...]
  }
}
```

当置信度不足或未命中时，`answer` 给出提示并返回 `candidates` 供确认。

---

## 自测用例

请先初始化样例库：
```bash
python scripts/init_tasks_sqlite.py
```

- `GET /tasks/ask?q=张三的提交9月周报完成了吗？` → 期望 DONE（已完成）
- `GET /tasks/ask?q=张三的E3D接口联调现在什么状态？` → 期望 TODO（未完成/待办）
- `GET /tasks/ask?q=李四的整理工艺包V2是否已完成？` → 期望 DONE（已完成）
- `GET /tasks/ask?q=老张九月报搞定了没？` → 别名映射“老张→张三”，口语可命中或返回合理候选

遇到召回偏低可尝试：
- 设为 `embeddings` 或 `hybrid`（两者均为“聚焦查询”：embeddings 用矩阵法，hybrid 用 FAISS，实现等价）
- 适度下调 `thresh`（如 `0.5`）
- 扩充别名词表（`EntityResolver.alias_map`）或任务名标准化
- 将 `RESOLVER` 设为 `hybrid`，并适度下调 `thresh`（如 `0.5`）
- 扩充别名词表（`EntityResolver.alias_map`）或任务名标准化

---

## 模式与打分融合细节

- 规则分（rules）：
  - 文本归一化（去空白/标点），完全相等=1.0；包含关系=0.8；否则按字符集合交并比
- 向量分（embeddings）：
  - `sentence-transformers` 生成句向量，L2 归一化，FAISS `IndexFlatIP` 实现余弦相似 Top‑k
- 融合（hybrid）：
  - `score = alpha_vec * vec + (1 - alpha_vec) * rule`（默认 `alpha_vec=0.65`）
  - Top‑k×2 召回再融合，最后取 Top‑k

---

## 性能与只读安全

- SQLite 只读连接（URI 模式）+ 超时（默认 2s），查询 `LIMIT 1`
- 嵌入来源三选一：`MOCK_EMB=1`、本地 SBERT、容器化嵌入服务（`EMB_URL`）
- Windows 可仅用 FAISS；Milvus 通过 compose profile `milvus` 显式开启

---

## 故障排查

- 结果为空：先初始化任务库；确认 `TASKS_DB` 路径
- 维度不匹配：切换模型后需 `/reset`（文档索引）与 `/tasks/reload`（实体索引）
- 容器嵌入不可达：检查 `docker compose up -d embedder`、端口与 `EMB_URL`
- 下载过慢：先用 `MOCK_EMB=1` 跑通流程

# 更新与调参建议（新增）

本页补充 Step 2（无模型问数）的两项改进：

- 聚焦查询（embeddings-only）：
  - 先用规则法粗提候选（≥0.8），将这些候选与整句并行编码，逐候选取最大相似度；
  - 解决“整句问句对齐短实体名”导致余弦分数偏低的问题，改善 embeddings-only 的稳定性；
  - 与原有 FAISS Top‑k 检索兼容，不影响 rules/hybrid 逻辑。

- 模式自适应阈值：
  - 当未显式传 `thresh` 时，系统按模式取默认值：rules=0.8，embeddings=0.45，hybrid=0.58；
  - 仍可在请求中通过 `&thresh=...` 覆盖；
  - 若切换模型/语料特性发生变化，建议先用默认阈值观测候选分布，再按需微调。

提示：使用 `BAAI/bge-small-zh-v1.5` 时，维度为 512；若本地直载，请确保 `EMB_DIM=512`；切换模型/维度后请执行 `/reset`（文档索引）与 `/tasks/reload`（实体索引）。
