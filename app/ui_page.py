from __future__ import annotations


UI_HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>RAG Task Ask UI</title>
  <style>
    :root {
      --bg: #f3efe5;
      --panel: #fffaf0;
      --panel-strong: #fff;
      --ink: #1c1c1c;
      --muted: #6a6a6a;
      --line: #d8cfbf;
      --accent: #0d6b5c;
      --accent-soft: #dcefe9;
      --danger: #a03333;
      --shadow: 0 18px 45px rgba(40, 35, 25, 0.12);
      --radius: 18px;
      --mono: "Cascadia Code", "Consolas", monospace;
      --sans: "Segoe UI", "Microsoft YaHei UI", sans-serif;
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(13, 107, 92, 0.14), transparent 28%),
        radial-gradient(circle at right center, rgba(156, 122, 78, 0.12), transparent 22%),
        linear-gradient(180deg, #f8f4ea 0%, var(--bg) 100%);
      min-height: 100vh;
    }

    .shell {
      width: min(1180px, calc(100vw - 32px));
      margin: 28px auto;
      padding: 28px;
      border: 1px solid rgba(216, 207, 191, 0.9);
      border-radius: 28px;
      background: rgba(255, 250, 240, 0.92);
      box-shadow: var(--shadow);
      backdrop-filter: blur(10px);
    }

    .hero {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 22px;
    }

    .hero h1 {
      margin: 0 0 8px;
      font-size: clamp(28px, 4vw, 42px);
      line-height: 1.05;
    }

    .hero p {
      margin: 0;
      color: var(--muted);
      max-width: 720px;
    }

    .badge {
      white-space: nowrap;
      border-radius: 999px;
      padding: 10px 14px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 13px;
      font-weight: 700;
    }

    .tabs {
      display: flex;
      gap: 10px;
      margin-bottom: 18px;
    }

    .tab {
      border: 1px solid var(--line);
      background: rgba(255, 255, 255, 0.7);
      color: var(--ink);
      padding: 12px 18px;
      border-radius: 999px;
      cursor: pointer;
      font-weight: 700;
      transition: 0.2s ease;
    }

    .tab.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      transform: translateY(-1px);
    }

    .tab-panel {
      display: none;
      gap: 18px;
      animation: fadeIn 0.22s ease;
    }

    .tab-panel.active {
      display: grid;
    }

    .grid {
      grid-template-columns: minmax(280px, 420px) minmax(320px, 1fr);
    }

    .card {
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--panel-strong);
      padding: 18px;
      box-shadow: 0 10px 24px rgba(37, 34, 26, 0.06);
    }

    .card h2 {
      margin: 0 0 8px;
      font-size: 18px;
    }

    .card p {
      margin: 0 0 14px;
      color: var(--muted);
      line-height: 1.5;
    }

    label {
      display: block;
      font-size: 13px;
      font-weight: 700;
      margin-bottom: 8px;
    }

    textarea,
    input[type="text"] {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      font: inherit;
      color: var(--ink);
      background: #fffdf8;
      resize: vertical;
      min-height: 52px;
    }

    textarea {
      min-height: 148px;
    }

    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      margin-top: 14px;
    }

    .checkbox {
      display: inline-flex;
      align-items: center;
      gap: 8px;
      color: var(--muted);
      font-size: 14px;
      font-weight: 600;
    }

    .checkbox input {
      accent-color: var(--accent);
    }

    button {
      border: 0;
      border-radius: 14px;
      padding: 12px 16px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      transition: 0.18s ease;
    }

    .primary {
      background: var(--accent);
      color: #fff;
    }

    .secondary {
      background: #efe7d8;
      color: var(--ink);
    }

    button:hover {
      transform: translateY(-1px);
    }

    button:disabled {
      opacity: 0.6;
      cursor: wait;
      transform: none;
    }

    .status {
      min-height: 22px;
      margin-top: 12px;
      font-size: 14px;
      color: var(--muted);
    }

    .status.error {
      color: var(--danger);
    }

    .result-card,
    .batch-item {
      border: 1px solid var(--line);
      border-radius: 16px;
      background: #fffdf9;
      padding: 16px;
    }

    .result-card + .result-card,
    .batch-item + .batch-item {
      margin-top: 12px;
    }

    .headline {
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      margin-bottom: 10px;
      flex-wrap: wrap;
    }

    .headline strong {
      font-size: 16px;
    }

    .meta {
      display: flex;
      gap: 8px;
      flex-wrap: wrap;
      margin: 10px 0;
    }

    .chip {
      background: var(--accent-soft);
      color: var(--accent);
      border-radius: 999px;
      padding: 6px 10px;
      font-size: 12px;
      font-weight: 700;
    }

    pre {
      margin: 12px 0 0;
      padding: 14px;
      background: #1f2428;
      color: #e9f0ee;
      border-radius: 14px;
      overflow: auto;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
    }

    details summary {
      cursor: pointer;
      color: var(--muted);
      font-weight: 700;
    }

    .empty {
      border: 1px dashed var(--line);
      border-radius: 16px;
      padding: 22px;
      color: var(--muted);
      background: rgba(255, 255, 255, 0.55);
      text-align: center;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(4px); }
      to { opacity: 1; transform: translateY(0); }
    }

    @media (max-width: 900px) {
      .shell {
        width: min(100vw - 18px, 100%);
        margin: 10px auto;
        padding: 16px;
      }

      .hero {
        flex-direction: column;
      }

      .grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div>
        <h1>Task Ask Unified UI</h1>
        <p>这个页面只复用现有的 <code>/tasks/ask</code> 链路。单问 Tab 发起一次请求，批量 Tab 按行顺序逐条请求，行为尽量对齐 <code>scripts/batch_db_ask.py</code>。</p>
      </div>
      <div class="badge" id="health-badge">Checking health...</div>
    </section>

    <div class="tabs" role="tablist" aria-label="Task Ask Tabs">
      <button class="tab active" type="button" data-tab-target="single-panel">单问</button>
      <button class="tab" type="button" data-tab-target="batch-panel">批量</button>
    </div>

    <section id="single-panel" class="tab-panel grid active">
      <div class="card">
        <h2>单条提问</h2>
        <p>输入一个问题，页面会直接调用 <code>/tasks/ask?q=...</code> 并展示原始返回内容。</p>
        <label for="single-question">问题</label>
        <textarea id="single-question" placeholder="例如：张三的 E3D 接口联调现在什么状态？"></textarea>
        <div class="controls">
          <button id="single-submit" class="primary" type="button">发送</button>
          <label class="checkbox"><input id="single-debug" type="checkbox" /> 显示调试信息</label>
        </div>
        <div id="single-status" class="status"></div>
      </div>

      <div class="card">
        <h2>单问结果</h2>
        <p>先展示关键字段，再保留完整 JSON。</p>
        <div id="single-result" class="empty">还没有结果。</div>
      </div>
    </section>

    <section id="batch-panel" class="tab-panel grid">
      <div class="card">
        <h2>批量提问</h2>
        <p>每行一个问题。页面会像 <code>batch_db_ask.py --file ...</code> 一样顺序逐条执行。</p>
        <label for="batch-questions">问题列表</label>
        <textarea id="batch-questions" placeholder="张三的 E3D 接口联调现在什么状态？&#10;张三还有多少任务未完成？"></textarea>
        <div class="controls">
          <button id="batch-submit" class="primary" type="button">批量执行</button>
          <button id="batch-clear" class="secondary" type="button">清空</button>
          <label class="checkbox"><input id="batch-debug" type="checkbox" /> 显示调试信息</label>
        </div>
        <div id="batch-status" class="status"></div>
      </div>

      <div class="card">
        <h2>批量结果</h2>
        <p>每条问题独立显示状态和完整返回，便于和脚本输出对照。</p>
        <div id="batch-results" class="empty">还没有结果。</div>
      </div>
    </section>
  </main>

  <script>
    const singleQuestion = document.getElementById("single-question");
    const singleDebug = document.getElementById("single-debug");
    const singleSubmit = document.getElementById("single-submit");
    const singleStatus = document.getElementById("single-status");
    const singleResult = document.getElementById("single-result");

    const batchQuestions = document.getElementById("batch-questions");
    const batchDebug = document.getElementById("batch-debug");
    const batchSubmit = document.getElementById("batch-submit");
    const batchClear = document.getElementById("batch-clear");
    const batchStatus = document.getElementById("batch-status");
    const batchResults = document.getElementById("batch-results");
    const healthBadge = document.getElementById("health-badge");

    function escapeHtml(value) {
      return String(value)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
    }

    function prettyJson(value) {
      return escapeHtml(JSON.stringify(value, null, 2));
    }

    function buildResultCard(question, payload, requestState) {
      const answer = payload && payload.answer ? payload.answer : "<empty>";
      const resolverMode = payload && payload.resolver_mode ? payload.resolver_mode : "-";
      const intent = payload && payload.intent ? payload.intent : "-";
      const statusText = requestState || "ok";
      return `
        <article class="result-card">
          <div class="headline">
            <strong>${escapeHtml(question)}</strong>
            <span class="chip">${escapeHtml(statusText)}</span>
          </div>
          <div><strong>Answer:</strong> ${escapeHtml(answer)}</div>
          <div class="meta">
            <span class="chip">resolver_mode: ${escapeHtml(resolverMode)}</span>
            <span class="chip">intent: ${escapeHtml(intent)}</span>
          </div>
          <details>
            <summary>完整 JSON</summary>
            <pre>${prettyJson(payload)}</pre>
          </details>
        </article>
      `;
    }

    function setStatus(target, message, isError) {
      target.textContent = message || "";
      target.classList.toggle("error", Boolean(isError));
    }

    async function askTask(question, debug) {
      const params = new URLSearchParams({ q: question });
      if (debug) {
        params.set("debug", "true");
      }
      const response = await fetch(`/tasks/ask?${params.toString()}`, {
        method: "GET",
        headers: { "Accept": "application/json" }
      });

      let data = null;
      try {
        data = await response.json();
      } catch (error) {
        data = { error: "invalid_json_response", detail: String(error) };
      }

      if (!response.ok) {
        throw { status: response.status, payload: data };
      }
      return data;
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/health", { headers: { "Accept": "application/json" } });
        const data = await response.json();
        healthBadge.textContent = `tasks_store=${data.tasks_store} | resolver=${data.resolver_mode} | ready=${data.tasks_ready}`;
      } catch (error) {
        healthBadge.textContent = "Health check failed";
      }
    }

    singleSubmit.addEventListener("click", async () => {
      const question = singleQuestion.value.trim();
      if (!question) {
        setStatus(singleStatus, "请输入问题。", true);
        return;
      }

      singleSubmit.disabled = true;
      setStatus(singleStatus, "请求中...", false);
      singleResult.innerHTML = '<div class="empty">正在请求...</div>';

      try {
        const payload = await askTask(question, singleDebug.checked);
        singleResult.innerHTML = buildResultCard(question, payload, "ok");
        setStatus(singleStatus, "请求完成。", false);
      } catch (error) {
        const payload = error && error.payload ? error.payload : { error: "request_failed", detail: String(error) };
        const code = error && error.status ? `HTTP ${error.status}` : "request_failed";
        singleResult.innerHTML = buildResultCard(question, payload, code);
        setStatus(singleStatus, `请求失败: ${code}`, true);
      } finally {
        singleSubmit.disabled = false;
      }
    });

    batchSubmit.addEventListener("click", async () => {
      const questions = batchQuestions.value
        .split(/\\r?\\n/)
        .map((item) => item.trim())
        .filter(Boolean);

      if (!questions.length) {
        setStatus(batchStatus, "请至少输入一条问题。", true);
        return;
      }

      batchSubmit.disabled = true;
      batchClear.disabled = true;
      batchResults.innerHTML = "";
      setStatus(batchStatus, `准备执行 ${questions.length} 条问题...`, false);

      for (let index = 0; index < questions.length; index += 1) {
        const question = questions[index];
        setStatus(batchStatus, `执行中 ${index + 1}/${questions.length}: ${question}`, false);
        try {
          const payload = await askTask(question, batchDebug.checked);
          batchResults.insertAdjacentHTML("beforeend", buildResultCard(question, payload, "ok"));
        } catch (error) {
          const payload = error && error.payload ? error.payload : { error: "request_failed", detail: String(error) };
          const code = error && error.status ? `HTTP ${error.status}` : "request_failed";
          batchResults.insertAdjacentHTML("beforeend", buildResultCard(question, payload, code));
        }
      }

      if (!batchResults.innerHTML.trim()) {
        batchResults.innerHTML = '<div class="empty">没有生成结果。</div>';
      }

      setStatus(batchStatus, `执行完成，共 ${questions.length} 条。`, false);
      batchSubmit.disabled = false;
      batchClear.disabled = false;
    });

    batchClear.addEventListener("click", () => {
      batchQuestions.value = "";
      batchResults.innerHTML = '<div class="empty">还没有结果。</div>';
      setStatus(batchStatus, "", false);
    });

    document.querySelectorAll(".tab").forEach((button) => {
      button.addEventListener("click", () => {
        const target = button.dataset.tabTarget;
        document.querySelectorAll(".tab").forEach((item) => item.classList.remove("active"));
        document.querySelectorAll(".tab-panel").forEach((panel) => panel.classList.remove("active"));
        button.classList.add("active");
        document.getElementById(target).classList.add("active");
      });
    });

    refreshHealth();
  </script>
</body>
</html>
"""
