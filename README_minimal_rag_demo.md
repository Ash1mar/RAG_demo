# Minimal RAG Demo (CNPE VSCode Workshop Edition)

一个针对 CNPE VSCode 工作坊定制的、尽量贴近真实场景的 Minimal RAG Demo：
- FastAPI 后端 + 可切换的向量库（FAISS 默认，Milvus 可选）
- 中文 embedding 支持（mock / 本地模型 / 远程 bge-small-zh 容器）
- 基于 SQLite 的任务状态问答接口（`/tasks/ask`），支持无 LLM 与 LLM‑驱动两种模式
- 一个面向 `tasks` 表的 NL→JSON→SQL 实验端点（`/db/ask`），可接 LLM 生成 `TaskQuerySpec`

详细的启动与测试流程请看 `docs/START_AND_TEST.md`。

---

## Project Structure

顶层目录结构（简化）：

```text
RAG_demo/
├─ app/
│  ├─ demo_app.py               # FastAPI 应用（HTTP 路由）
│  ├─ __init__.py
│  ├─ services/
│  │  ├─ embeddings.py          # Embedder：mock / 本地 SentenceTransformer / 远程 HTTP
│  │  ├─ chunking.py            # 简单文本分块
│  │  ├─ answer.py              # 简单中文回答拼装
│  │  ├─ keyword.py             # 关键词 / BM25 风格索引
│  │  ├─ hybrid.py              # 向量 + 关键词融合逻辑
│  │  ├─ task_query.py          # 任务状态解析与候选排序（Step 2 主逻辑）
│  │  ├─ nl2sql_engine.py       # 任务查询 NL→JSON 语义 IR（TaskQuerySpec）
│  │  ├─ sql_compiler.py        # TaskQuerySpec → 只读 SQL 编译器（tasks 单表）
│  │  └─ llm_client.py          # LLMClient 抽象、Dummy/Ollama 实现与工厂
│  ├─ vector_store/
│  │  ├─ faiss_store.py         # 基于 FAISS 的向量库（Windows 默认）
│  │  └─ milvus_store.py        # 基于 Milvus 的向量库（通过 compose profile 启用）
│  └─ tasks_store/
│     ├─ base.py                # TasksStore 抽象接口
│     └─ sqlite_store.py        # SQLiteTasksStore 实现（只读任务库 + query/sql 辅助）
├─ data/
│  ├─ faiss/                    # FAISS 索引与元数据
│  └─ tasks.db                  # Demo 任务 SQLite 库（scripts/init_tasks_sqlite.py 生成）
├─ embedder_server/
│  ├─ server.py                 # Docker 中运行的 bge-small-zh embedding HTTP 服务
│  └─ requirements.txt          # embedder 容器依赖
├─ scripts/
│  ├─ init_tasks_sqlite.py      # 初始化 demo 任务数据到 SQLite
│  └─ demo_queries.http         # VSCode REST Client 示例请求
├─ tests/
│  ├─ conftest.py               # 把项目根目录加入 sys.path，方便导入 app.*
│  └─ test_nl2sql_db_ask.py     # NL→JSON→SQL 闭环测试（FastAPI TestClient）
├─ docs/
│  ├─ START_AND_TEST.md         # 从 0 开始的启动 / 测试指南
│  └─ INSTRUCTIONS_TASKS.md     # Step 2：任务问答与 NL→SQL 设计说明
├─ docker-compose.yml           # embedder（必选）+ Milvus（可选 profile）
├─ Dockerfile                   # API 服务镜像
├─ Dockerfile.embedder          # Embedding 服务镜像（CPU/GPU）
├─ requirements.txt             # Python 依赖（API 侧）
├─ .env.example                 # 环境变量示例
└─ README_minimal_rag_demo.md   # 本文件
```

---

## Key Features

- **RAG 栈最小实现**
  - `/ingest`：按 `doc_id` ingest 文本 → 分块 → embedding → 写入向量库 + 关键词索引。
  - `/search`：只用向量检索（FAISS / Milvus），支持按 `doc_id` / `source` / 时间过滤。
  - `/reset`：清空向量库与关键词索引（不影响 `tasks.db`）。

- **任务状态问答 `/tasks/ask`**
  - 使用 `TaskQueryEngine`（`app/services/task_query.py`）对中文问句做：
    - 意图识别（完成了吗 / 状态 / 进度 / 是否完成 / 搞定 / 结束 等）。
    - 实体解析（人名 / 任务名），支持多种 resolver 模式：
      - `rules` / `embeddings` / `hybrid` / `hybrid_plus_rules`（非 LLM baseline）
      - `hybrid_llm`：LLM NL→JSON + 小模型 + FAISS 对齐 + 统一 SQL compiler
  - 底层通过 `SQLiteTasksStore` 查询最新任务记录，并返回：
    - `answer`（简短中文回答）、`status`、`person`、`task`、`ts`、`id`
    - `sql`、`resolver_mode`、`thresh`
    - `candidates`（候选列表及分数）
    - `nl_ir`（轻量级 NL→JSON 语义 IR，用于调试）

- **NL→JSON→SQL 实验 `/db/ask`**
  - 专门用于 `tasks` 表的只读 NL→SQL 闭环调试端点，不生成自然语言回答：
    - `query`：原始自然语言。
    - `ir`：`TaskQuerySpec` 的 JSON 结构（由 `parse_task_query_nl` 生成，可由 LLM 提供）。
    - `sql`：通过 `compile_tasks_sql` 从 IR 编译出来的只读 SQL。
    - `params`：SQL 参数元组。
    - `rows`：`TASKS.query(sql, params)` 直接返回的记录列表。

- **灵活的 Embedding 模式**
  - **mock 模式**：`MOCK_EMB=1` 时使用纯 Python 的「哈希投影词袋」生成确定性向量，无需下载模型。
  - **本地模型模式**：`MOCK_EMB=0` 且未设置 `EMB_URL` 时，`embeddings.py` 会加载 `SentenceTransformer(MODEL_NAME)`（如 MiniLM 或本地 bge-small-zh 权重）。
  - **远程 HTTP 模式**：设置 `EMB_URL=http://...` 后，优先调用 HTTP embedding 服务；推荐使用 `docker-compose` 启动的 bge-small-zh 容器。

---

## Configuration (Env Vars)

常用环境变量（详细说明见 `docs/START_AND_TEST.md` 与 `.env.example`）：

- **向量库 / 存储**
  - `STORE=faiss|milvus`（默认 `faiss`）
  - `DATA_DIR`：向量数据目录（FAISS 使用，默认 `data`）
  - `MILVUS_HOST` / `MILVUS_PORT` / `MILVUS_COLLECTION`：在 `STORE=milvus` 时使用

- **任务解析模式**
  - `RESOLVER=rules|embeddings|hybrid|hybrid_plus_rules|hybrid_llm`（默认 `hybrid`）

- **Embedding 相关**
  - `MOCK_EMB=1|0`：是否启用 mock embedding。
  - `MODEL_NAME`：本地 SentenceTransformer 模型 ID（如 `sentence-transformers/all-MiniLM-L6-v2` / `BAAI/bge-small-zh-v1.5`）。
  - `EMB_DIM`：embedding 维度（MiniLM=384，bge-small-zh=512）。
  - `EMB_URL`：远程 embedding HTTP 服务地址，如 `http://localhost:8080/embeddings`。
  - `EMB_TIMEOUT`：HTTP 调用超时时间（秒，默认 12）。

- **任务存储 / DB**
  - `TASKS_BACKEND=sqlite`（当前仅支持 sqlite）
  - `TASKS_DB`：SQLite 文件路径，默认 `data/tasks.db`。

- **LLM / NL2SQL 相关**
- `LLM_ENABLED=true|false`
- `LLM_PROVIDER=dummy|ollama|openai`（`openai` 代表所有 OpenAI-Compatible 服务，如通义千问 DashScope）
- `LLM_MODEL`：默认 `qwen2.5-coder:7b`，也可指定其他模型
- `LLM_TEXT2SQL_MODEL`：单独指定 Text2SQL 使用的模型（默认 `qwen3-coder:480b-cloud`）
- `LLM_TEXT2SQL_PROVIDER` / `LLM_TEXT2SQL_OLLAMA_BASE_URL` / `LLM_TEXT2SQL_OPENAI_BASE_URL` / `LLM_TEXT2SQL_API_KEY`：若 Text2SQL 需要独立的 provider 或端点，可通过这些变量覆盖，未设置时沿用 `LLM_PROVIDER` 对应配置
- `LLM_OLLAMA_BASE_URL`：Ollama HTTP 端点（默认 `http://localhost:11434`）
- `LLM_OPENAI_BASE_URL`：OpenAI-Compatible API 根路径（默认 `https://dashscope.aliyuncs.com/compatible-mode/v1`）
- `LLM_API_KEY`：当 `LLM_PROVIDER=openai`/`dashscope` 时必填，用于鉴权
  - `TASKS_NL2SQL_LLM=1`：在 NL→JSON→SQL 流程中优先使用 LLM 抽取 `TaskQuerySpec`，失败时自动回退规则解析。

---

## Development / Run Modes

典型开发方式：

- **本机 API + mock embedding（最快起跑）**
  1. 创建并激活虚拟环境，安装依赖：
     - `python -m venv venv`
     - `./venv/Scripts/Activate`
     - `pip install -r requirements.txt`
  2. 设置环境：
     - `STORE=faiss`
     - `MOCK_EMB=1`
  3. 启动 API：
     - `uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload`

- **本机 API + Docker 中的 bge-small-zh**
  1. 构建并启动 embedder 容器：
     - `docker compose build embedder`
     - `docker compose up -d embedder`
  2. 设置环境：
     - `STORE=faiss`
     - `RESOLVER=hybrid` 或 `embeddings` 或 `hybrid_llm`
     - `MOCK_EMB=0`
     - `EMB_URL=http://localhost:8080/embeddings`
     - `EMB_DIM=512`
  3. 启动 API 同上。

- **Milvus 模式（可选，推荐在 Linux/WSL）**
  - `docker compose --profile milvus up -d`
  - 设置 `STORE=milvus` 并配置 Milvus 相关环境变量。

---

## Where to Read Next

- 从 0 开始启动 / 测试整个 demo：`docs/START_AND_TEST.md`
- 深入理解任务问答与 NL→SQL 的实现细节：
  - `docs/INSTRUCTIONS_TASKS.md`
  - `app/services/task_query.py`
  - `app/services/nl2sql_engine.py`
  - `app/services/sql_compiler.py`
  - `app/services/llm_client.py`
---

## Text2SQL & SQL AST 概览

本 demo 在任务子系统中还内置了一个可选的 Text2SQL 管线，用于“让 LLM 生成 SQL，但仍由后端严格控制 / 重写”：

- **入口与模式**  
  - `/tasks/ask` 在 `resolver_mode="text2sql"` 或 `hybrid_llm` 下，会走 Text2SQL 分支：  
    1. `parse_task_query_nl` 生成 `TaskQuerySpec`（语义 IR）。  
    2. 构造 Text2SQL prompt，将 schema + IR hint 一并交给 LLM，要求返回 `{"queries":[{"sql":...,"description":...}]}` 结构的 JSON。  
    3. 对每条 SQL 先做 **重写 + AST 校验**，再交给 `SQLiteTasksStore.query(sql, params)` 执行。  

- **安全校验与重写（`app/services/task_query.py`）**  
  - 使用 `sqlglot` 将 LLM 返回的 SQL 解析成 AST：  
    - 只允许只读 `SELECT`，且只允许访问 `task_latest` / `tasks` 两张表。  
    - 自动补齐 / 裁剪 `LIMIT`（最多 100 行），并规范 `ORDER BY` 位置。  
    - 拒绝危险关键字和跨方言函数（如 `DATE_SUB` / `CURDATE`）、占位符 `?`、命名参数等。  
  - 在 AST 之外，对一些常见坑做轻量修正：  
    - 时间窗口：将 `now-7d`、`start_of_week` 等符号解析成实际时间戳，修剪 `NOW() - 7d` 这类方言函数。  
    - 标签（`tags`）：自动注入 `tags LIKE '%...%'` 条件，避免把标签词误当人名。  
    - 优先级（`priority`）：将“高优P1任务”类问句映射到 `priority = 1`，并移除对“高优P1任务”作为任务名的硬匹配。  

- **调试字段**  
  - 当 Text2SQL 失败时，`/tasks/ask` 的 JSON 中会包含：  
    - `error`：如 `text2sql_invalid_sql` / `text2sql_db_query_failed` / `text2sql_llm_failed`；  
    - `reason`：AST 或 DB 抛出的详细错误信息；  
    - `text2sql_raw_response`：LLM 原始输出（便于改 prompt）；  
    - `text2sql`：每条 SQL 的执行结果（包括 `sql`、`description`、`rows`）。  
  - 配合 `scripts/batch_db_ask.py` 可以批量跑一组自然语言问题，观察 Text2SQL 行为，并据此调优 prompt 与解析规则。
