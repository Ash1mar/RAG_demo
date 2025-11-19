# Minimal RAG Demo (CNPE VSCode Workshop Edition)

一个针对 CNPE VSCode 工作坊定制的、尽量贴近真实场景的 Minimal RAG Demo：

- FastAPI 后端 + 可切换的向量库（FAISS 默认，Milvus 可选）
- 中文 embedding 支持（mock / 本地模型 / 远程 bge-small-zh 容器）
- 基于 SQLite 的「非 LLM」任务状态问答接口 (`/tasks/ask`)
- 一个面向 `tasks` 表的 NL→JSON→SQL 实验端点 (`/db/ask`)

详细的启动与测试流程请看 `docs/START_AND_TEST.md`。

---

## Project Structure

顶层目录结构（简化）

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
│  │  ├─ task_query.py          # 非 LLM 任务状态解析与候选排序（Step 2 主逻辑）
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
└─ README_minimal_rag_demo.md   # 本文档
```

---

## Key Features

- **RAG 栈最小实现**
  - `/ingest`：按 `doc_id` ingest 文本 → 分块 → embedding → 写入向量库 + 关键词索引。
  - `/search`：只用向量检索（FAISS / Milvus），支持按 `doc_id` / `source` / 时间过滤。
  - `/reset`：清空向量库与关键词索引（不影响 `tasks.db`）。
- **非 LLM 任务状态问答 `/tasks/ask`**
  - 使用 `TaskQueryEngine`（`app/services/task_query.py`）对中文问句做：
    - 意图识别（完成了没 / 状态 / 进度 / 是否完成 / 搞定 / 结束 等）
    - 实体解析（人名 / 任务名），支持多种 resolver 模式：`rules` / `embeddings` / `hybrid` / `hybrid_plus_rules`
  - 底层通过 `SQLiteTasksStore` 查询最新任务记录，并返回：
    - `answer`（简短中文回答）、`status`、`person`、`task`、`ts`、`id`
    - `sql`、`resolver_mode`、`thresh`
    - `candidates`（候选列表及分数）
    - `nl_ir`（轻量级 NL→JSON 语义 IR，用于调试）
- **NL→JSON→SQL 实验 `/db/ask`**
  - 专门用于 `tasks` 表的只读 NL→SQL 闭环调试端点，不生成自然语言回答。
  - 返回字段包括：
    - `query`：原始自然语言
    - `ir`：`TaskQuerySpec` 的 JSON 结构（由 `parse_task_query_nl` 生成）
    - `sql`：通过 `compile_tasks_sql` 从 IR 编译出来的只读 SQL
    - `params`：SQL 参数元组
    - `rows`：`TASKS.query(sql, params)` 返回的记录列表
- **灵活的 Embedding 模式**
  - **mock 模式**：`MOCK_EMB=1` 时使用纯 Python 的「哈希投影词袋」生成确定性向量，无需下载模型，适合初次运行和单测。
  - **本地模型模式**：`MOCK_EMB=0` 且未设置 `EMB_URL` 时，`embeddings.py` 会加载 `SentenceTransformer(MODEL_NAME)`（如 MiniLM 或本地 bge-small-zh 权重）。
  - **远程 HTTP 模式**：设置 `EMB_URL=http://...` 后，优先调用 HTTP embedding 服务；本仓库推荐使用 `docker compose` 启动 bge-small-zh 容器。
- **Milvus 可选**
  - 默认 `STORE=faiss`，在 Windows 上避免直接配置 Milvus。
  - 如需 Milvus：通过 compose profile 启动 Milvus，并将 `STORE` 设置为 `milvus`，由 `MilvusVectorStore` 负责读写。
- **可选：基于 Ollama 的 NL→JSON 解析（TaskQuerySpec）**
  - 可使用本地 Ollama + `deepseek-r1:7b` 等模型，对任务查询问句做 NL→JSON 结构化解析，生成 `TaskQuerySpec`，再由现有 SQL 编译器生成只读 SQL。
  - LLM 解析是可选的：不开启时自动使用规则解析 `_rule_based_parse_task_query_nl` 作为主路径。

---

## Configuration (Env Vars)

常用环境变量（详细可参考 `.env.example` 和 `docs/START_AND_TEST.md`）：

- **向量库 / 存储**
  - `STORE=faiss|milvus`（默认 `faiss`）。
  - `DATA_DIR`：向量数据目录（FAISS 使用，默认 `data`）。
  - `MILVUS_HOST`、`MILVUS_PORT`、`MILVUS_COLLECTION`：在 `STORE=milvus` 时必需。
- **任务解析模式**
  - `RESOLVER=rules|embeddings|hybrid|hybrid_plus_rules`（默认 `hybrid`）：
    - `rules`：只用规则和关键词。
    - `embeddings`：只用 embedding（带 Focus Query 机制）。
    - `hybrid`：以向量为主的 hybrid，仍使用 Focus Query，不再简单「规则 + 向量」加权。
    - `hybrid_plus_rules`：在向量排名基础上，对高置信规则结果给轻量加成。
- **Embedding 相关**
  - `MOCK_EMB=1|0`：是否启用 mock embedding。
  - `MODEL_NAME`：本地 SentenceTransformer 模型 ID，例如：
    - `sentence-transformers/all-MiniLM-L6-v2`
    - `BAAI/bge-small-zh-v1.5`
  - `EMB_DIM`：embedding 维度（MiniLM=384，bge-small-zh=512）。
  - `EMB_URL`：远程 embedding HTTP 服务地址，如 `http://localhost:8080/embeddings`。
  - `EMB_TIMEOUT`：HTTP 调用超时时间（秒，默认 12）。
- **任务存储 / DB**
  - `TASKS_BACKEND`：目前仅 `sqlite`。
  - `TASKS_DB`：SQLite 文件路径，默认 `data/tasks.db`。
- **可选：NL→JSON LLM 解析（基于 Ollama）**
  - `LLM_ENABLED`：`true`/`false`，是否启用 LLM 客户端工厂（默认 `true`，但仍需要 `TASKS_NL2SQL_LLM=1` 才会在 NL→SQL 中生效）。
  - `LLM_PROVIDER`：LLM 提供方标识，目前支持 `dummy`（默认）和 `ollama`：
    - `dummy`：使用 `DummyLLMClient`，所有 NL→JSON 请求都会抛出 `NotImplementedError`，`parse_task_query_nl` 会自动回退到规则解析。
    - `ollama`：使用本地 Ollama 服务（`/api/chat` + `format` 结构化输出）。
  - `LLM_MODEL`：Ollama 中对应的模型 tag，例如 `deepseek-r1:7b`（或你在 Ollama 里拉取并起的名称）。
  - `LLM_OLLAMA_BASE_URL`：Ollama HTTP 地址，默认 `http://localhost:11434`。
  - `TASKS_NL2SQL_LLM`：是否在 `parse_task_query_nl` 中优先使用 LLM 解析 TaskQuerySpec：
    - `TASKS_NL2SQL_LLM=1`：先尝试通过 `LLMClient.generate_task_query_spec` → `TaskQuerySpec.parse_obj`，失败时自动回退到规则版本。
    - 其他值或未设置：始终使用规则解析 `_rule_based_parse_task_query_nl`。

---

## Development / Run Modes

结合 `docs/START_AND_TEST.md`，典型开发方式如下：

- **本机 API + mock embedding（最快起跑）**
  1. 创建并激活虚拟环境，安装依赖：
     - `python -m venv venv`
     - `./venv/Scripts/Activate`
     - `pip install -r requirements.txt`
  2. 设置环境：`$env:STORE='faiss'; $env:MOCK_EMB='1'`
  3. 启动 API：`uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload`
- **本机 API + Docker 中的 bge-small-zh**
  1. 构建并启动 embedder 容器：
     - `docker compose build embedder`
     - `docker compose up -d embedder`
  2. 设置环境：
     - `STORE='faiss'`
     - `RESOLVER='hybrid'` 或 `embeddings`
     - `MOCK_EMB='0'`
     - `EMB_URL='http://localhost:8080/embeddings'`
     - `EMB_DIM='512'`
  3. 启动 API 同上。
- **Milvus 模式（可选，推荐在 Linux/WSL 下启用）**
  - `docker compose --profile milvus up -d`
  - 设置 `STORE='milvus'` 并配置 Milvus 相关环境变量。
- **可选：本地 Ollama + LLM NL→JSON 解析**
  - 安装并启动 Ollama，确保可以通过 `http://localhost:11434` 访问。
  - 在 Ollama 中拉取所需模型，例如：`ollama pull deepseek-r1:7b`。
  - 设置环境变量，例如：
    - `LLM_ENABLED=true`
    - `LLM_PROVIDER=ollama`
    - `LLM_MODEL=deepseek-r1:7b`
    - `LLM_OLLAMA_BASE_URL=http://localhost:11434`
    - `TASKS_NL2SQL_LLM=1`  （启用 NL→JSON LLM 路径）
  - 之后通过 `/db/ask` 调试 NL→JSON→SQL 闭环；LLM 解析失败时会自动回退到规则解析。

---

## Task Q&A & NL→SQL Notes

- 任务问答链路说明：`docs/INSTRUCTIONS_TASKS.md`。
- 初始化 demo 任务数据：
  - 在项目根目录下执行：`python scripts/init_tasks_sqlite.py`，生成/更新 `data/tasks.db`。
- 对比不同 resolver 模式：
  - 修改 `RESOLVER` 环境变量后，通过 `/tasks/ask` 观察：
    - `answer` / `status` / `candidates` 的变化；
    - `resolver_mode`、`thresh` 的默认策略；
    - `nl_ir` 中语义解析结构的差异。
- NL→SQL 闭环实验：
  - 使用 `/db/ask` 查看完整的 IR / SQL / rows：
    - 更适合用于调试/教学；
    - 当前 LLM NL→JSON（基于 Ollama）也只接入在这条链路上。

---

## Testing

- 使用 `pytest` 运行测试：
  - 在项目根目录、激活 venv 后执行：`pytest -q tests/test_nl2sql_db_ask.py`。
  - `tests/conftest.py` 会自动把项目根目录加入 `sys.path`，确保 `from app.demo_app import app, TASKS` 能被正常导入。
- 在离线环境或网络受限环境运行时，推荐：
  - 设置 `MOCK_EMB='1'`，或
  - 提前启动本地 embedding 服务并配置 `EMB_URL`，避免运行时尝试下载外部模型。

---

## Where to Read Next

- 从 0 开始启动 / 测试整个 demo：`docs/START_AND_TEST.md`
- 深入理解任务问答与 NL→SQL 的实现细节：
  - `docs/INSTRUCTIONS_TASKS.md`
  - `app/services/task_query.py`
  - `app/services/nl2sql_engine.py`
  - `app/services/sql_compiler.py`
  - `app/services/llm_client.py`（Ollama LLMClient 与工厂逻辑）
*** End Patch ***!
