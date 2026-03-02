# Windows 内网离线运行说明（FastAPI）

适用场景：内网无法联网；依赖通过 U 盘拷贝；FastAPI 跑在一台内网 Windows 电脑上；另一台 Windows 服务器主机只作为 HTTP 调用方。

---

## 1) 需要提前用 U 盘带到内网电脑的东西

- 项目代码目录（整个项目根目录）
- `wheelhouse/`（离线依赖目录，放在项目根目录：`wheelhouse\`）
- Python Windows 安装包（建议固定版本，例如 `python-3.11.x-amd64.exe`）
- **ODBC Driver 17 for SQL Server** 离线安装包（内网电脑没有就必须安装）
-（建议）Microsoft Visual C++ Redistributable 2015-2022 x64（遇到 DLL/运行库报错时使用）

---

## 2) 内网电脑一次性环境准备

### 2.1 安装 Python

安装完成后用 PowerShell 验证：

```powershell
python --version
python -m pip --version
```

### 2.2 安装 ODBC Driver 17

确认内网电脑已安装 “ODBC Driver 17 for SQL Server”。（没有就用离线包安装）

---

## 3) 放置项目目录（示例）

假设拷贝到：`D:\RAG_demo\`

你应看到类似结构：

```text
D:\RAG_demo\
  app\
  config\
    app.env
  docs\
  scripts\
    install_offline.bat
    run_dev.bat
    check_net.bat
  wheelhouse\
  requirements.txt
```

---

## 4) 修改配置（到内网后手动改）

编辑：`config\app.env`

至少把 SQL Server 相关项改成内网真实环境，并把驱动改为 17：

```env
TASKS_BACKEND=mssql
TASKS_DIALECT=mssql

TASKS_MSSQL_SERVER=10.0.0.12,1433
TASKS_MSSQL_DATABASE=fact_tasks
TASKS_MSSQL_USER=sa
TASKS_MSSQL_PASSWORD=你的密码
TASKS_MSSQL_DRIVER=ODBC Driver 17 for SQL Server

# 常见内网证书问题：建议先用这组组合跑通
TASKS_MSSQL_ENCRYPT=yes
TASKS_MSSQL_TRUST_CERT=yes
```

如果你们安全策略要求不加密：把 `TASKS_MSSQL_ENCRYPT=no`（按你们要求来）。

### 4.1 内网模型/Embedding 配置建议（Nginx Model Proxy Gateway）

你们内网可用的 AI 网关（示例）：

- Health：`http://10.27.118.221/health`
- Chat Completions（OpenAI-compatible 风格）：
  - DeepSeek-R1-32B：`http://10.27.118.221/deepseek32b/v1/chat/completions`
  - Qwen3-32B：`http://10.27.118.221/qwen32b/v1/chat/completions`
  - Qwen3-235B：`http://10.27.118.221/qwen235b/v1/chat/completions`
- Embeddings：
  - bge-large：`http://10.27.118.221/bge-large/embed`
  - bge-m3：`http://10.27.118.221/bge-m3/embed`

#### A) 启用内网 Embedding（替代 MOCK_EMB）

> 推荐：先用 `MOCK_EMB=1` 跑通 API + 数据库，再切到远端 embedding。

```env
# 关闭 mock，使用内网 embedding 服务
MOCK_EMB=0
EMB_URL=http://10.27.118.221/bge-large/embed

# EMB_DIM 必须与 embedding 返回向量维度一致，否则 FAISS 会报维度错误
# 不确定维度时，用下面命令测一次，再把长度填进来：
# powershell -c "$r=irm http://10.27.118.221/bge-large/embed -Method Post -ContentType 'application/json' -Body '{\"inputs\":[\"hello\"],\"normalize\":true}'; ($r.embeddings[0].Length)"
EMB_DIM=1024
EMB_TIMEOUT=12
```

#### B) 启用内网大模型（用于 NL->JSON / Text2SQL 等能力）

本项目把 “OpenAI-compatible API” 当作 `LLM_PROVIDER=openai`（只要接口是 `/v1/chat/completions` 风格即可）。

注意：当前实现要求 `LLM_API_KEY` **非空**；如果网关不校验 key，填任意非空字符串即可（例如 `local`）。

**推荐选择：**
- 日常/性价比：Qwen3-32B
- 更强推理（更慢）：DeepSeek-R1-32B 或 Qwen3-235B

示例（统一把 NL->JSON 与 Text2SQL 都走同一个网关模型）：

```env
LLM_ENABLED=true
LLM_PROVIDER=openai
LLM_API_KEY=local

# 这里填 “/v1” 的基地址；程序会自动拼成 /chat/completions
LLM_OPENAI_BASE_URL=http://10.27.118.221/qwen32b/v1
LLM_MODEL=qwen32b

# Text2SQL 单独指定（可选；不写则复用上面的 LLM_*）
LLM_TEXT2SQL_PROVIDER=openai
LLM_TEXT2SQL_OPENAI_BASE_URL=http://10.27.118.221/qwen32b/v1
LLM_TEXT2SQL_MODEL=qwen32b
```

---

## 5) 离线安装依赖（首次/依赖变化时执行）

在项目根目录运行：

```bat
scripts\install_offline.bat
```

说明：

- 会在项目根目录创建本机 venv：`.venv\`
- 会从 `wheelhouse\` 离线安装 `requirements.txt` 的依赖（不联网）

---

## 6) 启动 FastAPI（调试，局域网可访问）

运行：

```bat
scripts\run_dev.bat
```

默认行为：

- 监听 `0.0.0.0:8000`（局域网可访问）
- 启动时会通过 `APP_CONFIG=config\app.env` 加载集中配置

改端口（例如 8001）：

```powershell
set PORT=8001
scripts\run_dev.bat
```

---

## 7) Windows 防火墙放行端口（让“服务器主机”能访问）

在运行 FastAPI 的内网电脑上（管理员 PowerShell）：

```powershell
netsh advfirewall firewall add rule name="FastAPI 8000" dir=in action=allow protocol=TCP localport=8000
```

如果你改了端口，把 `8000` 换成实际端口。

---

## 8) 连通性验证

### 8.1 内网电脑本机验证

```bat
scripts\check_net.bat
```

或直接访问：

```powershell
curl http://127.0.0.1:8000/docs
```

### 8.2 从“服务器主机（调用方）”验证

```powershell
Test-NetConnection <内网电脑IP> -Port 8000
curl http://<内网电脑IP>:8000/docs
```

---

## 9) 常见问题

- 离线安装失败：通常是 `wheelhouse\` 不全 / Python 版本不匹配（生成 wheelhouse 的 Python 版本要与内网电脑一致）
- SQL Server 连接报证书/加密错误：优先确认 `TASKS_MSSQL_TRUST_CERT=yes`；或按策略改 `TASKS_MSSQL_ENCRYPT=no`
- 只能本机访问：确认用 `scripts\run_dev.bat` 启动（绑定 `0.0.0.0`）并且防火墙端口已放行
