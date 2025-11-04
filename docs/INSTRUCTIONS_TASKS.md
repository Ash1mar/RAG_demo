无模型问数（SQLite + 规则 + FAISS 实体解析）

一、准备与运行
- 初始化样例库：`python scripts/init_tasks_sqlite.py`
- 启动服务：`uvicorn app.demo_app:app --host 0.0.0.0 --port 8000 --reload`

二、连通性
- `GET /tasks/status?person=张三&task=提交9月周报`
- 期望：`{"found":true, ... "status":"DONE"}`

三、自然语言问数
- `GET /tasks/ask?q=张三的提交9月周报完成了吗？`
- `GET /tasks/ask?q=张三的E3D接口联调现在什么状态？`
- `GET /tasks/ask?q=李四的整理工艺包V2是否已完成？`
- `GET /tasks/ask?q=老张九月报搞定了没？`

返回包含：
- `answer`：中文应答（含最近更新时间）
- `status`：标准化状态（DONE/TODO）
- `sql`：查询模板（调试用）
- `candidates`：人名/任务 Top-k 候选及打分（调试用）

四、实现要点
- 候选来源：SQLite `tasks` 表的去重 `person`/`task`
- 实体解析：`Embedder` + FAISS 近邻，融合关键词/字符交并比分数
- 置信度：默认阈值 `0.58`，不足时返回候选而非报错
- 别名词表：内置示例 `老张→张三`，可扩展
- 后续扩展位：保留 `TasksStore` 抽象接口，便于 Milvus/KG 替换

五、常见问题
- PowerShell 中 `curl` 是 `Invoke-WebRequest` 别名，中文参数建议用 `curl.exe --get --data-urlencode ...` 或 `Invoke-WebRequest -Proxy $null` 并用 `EscapeDataString` 编码。

