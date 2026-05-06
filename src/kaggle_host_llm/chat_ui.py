from __future__ import annotations


CHAT_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Kaggle Host LLM Chat</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --surface: #ffffff;
      --surface-2: #eef2f5;
      --border: #d9dee6;
      --text: #151a22;
      --muted: #627083;
      --accent: #0f766e;
      --accent-strong: #0b5f59;
      --danger: #b42318;
      --shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
    }

    * {
      box-sizing: border-box;
    }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--text);
      font-family:
        Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
      font-size: 15px;
      letter-spacing: 0;
    }

    .shell {
      display: grid;
      grid-template-columns: 320px minmax(0, 1fr);
      min-height: 100vh;
    }

    aside {
      border-right: 1px solid var(--border);
      background: var(--surface);
      padding: 20px;
    }

    main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-height: 100vh;
    }

    .brand {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      margin-bottom: 22px;
    }

    h1 {
      margin: 0;
      font-size: 18px;
      line-height: 1.2;
      font-weight: 700;
    }

    .status {
      display: inline-flex;
      align-items: center;
      gap: 7px;
      min-height: 28px;
      padding: 4px 9px;
      border: 1px solid var(--border);
      border-radius: 999px;
      color: var(--muted);
      font-size: 12px;
      white-space: nowrap;
    }

    .dot {
      width: 8px;
      height: 8px;
      border-radius: 999px;
      background: #98a2b3;
    }

    .status.ok .dot {
      background: #16a34a;
    }

    .status.bad .dot {
      background: var(--danger);
    }

    .field {
      margin: 14px 0;
    }

    label {
      display: block;
      margin-bottom: 6px;
      color: #344054;
      font-size: 13px;
      font-weight: 600;
    }

    input,
    textarea {
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: #fff;
      color: var(--text);
      font: inherit;
      padding: 10px 11px;
      outline: none;
    }

    input:focus,
    textarea:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.16);
    }

    .row {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
    }

    .sidebar-actions {
      display: flex;
      gap: 10px;
      margin-top: 18px;
    }

    button {
      min-height: 40px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
      font-weight: 650;
      padding: 9px 12px;
      cursor: pointer;
    }

    button:hover {
      border-color: #b8c0cc;
      background: #f9fafb;
    }

    button.primary {
      border-color: var(--accent);
      background: var(--accent);
      color: #fff;
    }

    button.primary:hover {
      background: var(--accent-strong);
    }

    button:disabled {
      cursor: not-allowed;
      opacity: 0.6;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
      min-height: 68px;
      padding: 16px 22px;
      border-bottom: 1px solid var(--border);
      background: rgba(255, 255, 255, 0.88);
      backdrop-filter: blur(10px);
    }

    .topbar-title {
      min-width: 0;
    }

    .topbar-title strong {
      display: block;
      font-size: 16px;
      line-height: 1.3;
    }

    .topbar-title span {
      display: block;
      margin-top: 2px;
      color: var(--muted);
      font-size: 13px;
    }

    .messages {
      overflow-y: auto;
      padding: 22px;
    }

    .message {
      max-width: 880px;
      margin: 0 auto 14px;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--surface);
      box-shadow: var(--shadow);
    }

    .message.user {
      border-color: #b9d5d2;
      background: #f0fdfa;
    }

    .message-head {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      padding: 10px 12px;
      border-bottom: 1px solid var(--border);
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
    }

    .message-body {
      padding: 13px 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      line-height: 1.55;
    }

    .composer {
      border-top: 1px solid var(--border);
      background: var(--surface);
      padding: 16px 22px 18px;
    }

    .composer-inner {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      max-width: 980px;
      margin: 0 auto;
      align-items: end;
    }

    #prompt {
      min-height: 54px;
      max-height: 180px;
      resize: vertical;
    }

    .error {
      margin-top: 12px;
      color: var(--danger);
      font-size: 13px;
      line-height: 1.45;
      white-space: pre-wrap;
    }

    .muted {
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }

    @media (max-width: 820px) {
      .shell {
        grid-template-columns: 1fr;
      }

      aside {
        border-right: 0;
        border-bottom: 1px solid var(--border);
      }

      main {
        min-height: 70vh;
      }

      .composer-inner {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="shell">
    <aside>
      <div class="brand">
        <h1>Kaggle Host LLM</h1>
        <div id="status" class="status"><span class="dot"></span><span>Checking</span></div>
      </div>

      <div class="field">
        <label for="apiKey">Gateway API key</label>
        <input id="apiKey" type="password" autocomplete="off" placeholder="Bearer token">
      </div>

      <div class="field">
        <label for="model">Model</label>
        <input id="model" value="qwen2.5-9b-quantized" autocomplete="off">
      </div>

      <div class="row">
        <div class="field">
          <label for="maxTokens">Max tokens</label>
          <input id="maxTokens" type="number" min="1" max="8192" value="512">
        </div>
        <div class="field">
          <label for="temperature">Temperature</label>
          <input id="temperature" type="number" min="0" max="2" step="0.1" value="0.7">
        </div>
      </div>

      <div class="row">
        <div class="field">
          <label for="topP">Top p</label>
          <input id="topP" type="number" min="0.01" max="1" step="0.01" value="0.9">
        </div>
        <div class="field">
          <label for="timeoutNote">Workers</label>
          <input id="timeoutNote" value="best effort" disabled>
        </div>
      </div>

      <div class="sidebar-actions">
        <button id="saveSettings" type="button">Save</button>
        <button id="clearChat" type="button">Clear</button>
      </div>

      <p class="muted">
        Settings are stored in this browser only. The chat endpoint uses the same
        gateway URL that served this page.
      </p>
    </aside>

    <main>
      <div class="topbar">
        <div class="topbar-title">
          <strong>Chat</strong>
          <span id="workerSummary">No worker status loaded</span>
        </div>
        <button id="refreshHealth" type="button">Refresh</button>
      </div>

      <section id="messages" class="messages" aria-live="polite"></section>

      <form id="chatForm" class="composer">
        <div class="composer-inner">
          <textarea id="prompt" placeholder="Type a message" required></textarea>
          <button id="send" class="primary" type="submit">Send</button>
        </div>
        <div id="error" class="error" role="alert"></div>
      </form>
    </main>
  </div>

  <script>
    const state = {
      messages: [],
      busy: false,
    };

    const settingsStorageKey = "kaggleHostChatSettings";
    const historyStorageKey = "kaggleHostChatHistory";
    const maxStoredMessages = 100;

    const els = {
      apiKey: document.getElementById("apiKey"),
      model: document.getElementById("model"),
      maxTokens: document.getElementById("maxTokens"),
      temperature: document.getElementById("temperature"),
      topP: document.getElementById("topP"),
      prompt: document.getElementById("prompt"),
      messages: document.getElementById("messages"),
      form: document.getElementById("chatForm"),
      send: document.getElementById("send"),
      error: document.getElementById("error"),
      status: document.getElementById("status"),
      workerSummary: document.getElementById("workerSummary"),
      saveSettings: document.getElementById("saveSettings"),
      clearChat: document.getElementById("clearChat"),
      refreshHealth: document.getElementById("refreshHealth"),
    };

    function loadSettings() {
      const saved = JSON.parse(localStorage.getItem(settingsStorageKey) || "{}");
      for (const key of ["apiKey", "model", "maxTokens", "temperature", "topP"]) {
        if (saved[key]) {
          els[key].value = saved[key];
        }
      }
    }

    function saveSettings() {
      localStorage.setItem(
        settingsStorageKey,
        JSON.stringify({
          apiKey: els.apiKey.value.trim(),
          model: els.model.value.trim(),
          maxTokens: els.maxTokens.value,
          temperature: els.temperature.value,
          topP: els.topP.value,
        })
      );
    }

    function loadHistory() {
      try {
        const saved = JSON.parse(localStorage.getItem(historyStorageKey) || "[]");
        if (!Array.isArray(saved)) {
          return;
        }
        state.messages = saved
          .filter((message) => {
            return (
              message &&
              ["user", "assistant"].includes(message.role) &&
              typeof message.content === "string"
            );
          })
          .slice(-maxStoredMessages);
      } catch (error) {
        state.messages = [];
      }
    }

    function saveHistory() {
      localStorage.setItem(
        historyStorageKey,
        JSON.stringify(state.messages.slice(-maxStoredMessages))
      );
    }

    function setError(message) {
      els.error.textContent = message || "";
    }

    function setBusy(value) {
      state.busy = value;
      els.send.disabled = value;
      els.send.textContent = value ? "Sending" : "Send";
    }

    function roleLabel(role) {
      return role === "assistant" ? "Assistant" : "User";
    }

    function renderMessages() {
      els.messages.innerHTML = "";
      if (state.messages.length === 0) {
        const empty = document.createElement("div");
        empty.className = "message";
        empty.innerHTML = "<div class=\\"message-head\\">Ready</div><div class=\\"message-body\\">Send a message when at least one worker is active.</div>";
        els.messages.appendChild(empty);
        return;
      }
      for (const message of state.messages) {
        const item = document.createElement("article");
        item.className = `message ${message.role}`;

        const head = document.createElement("div");
        head.className = "message-head";
        head.textContent = roleLabel(message.role);

        const body = document.createElement("div");
        body.className = "message-body";
        body.textContent = message.content;

        item.appendChild(head);
        item.appendChild(body);
        els.messages.appendChild(item);
      }
      els.messages.scrollTop = els.messages.scrollHeight;
    }

    async function refreshHealth() {
      try {
        const response = await fetch("/health", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Health returned HTTP ${response.status}`);
        }
        const health = await response.json();
        const active = Number(health.active_workers || 0);
        els.status.className = active > 0 ? "status ok" : "status";
        els.status.lastElementChild.textContent = `${active} active`;
        els.workerSummary.textContent = `${active} active worker${active === 1 ? "" : "s"}`;
      } catch (error) {
        els.status.className = "status bad";
        els.status.lastElementChild.textContent = "Offline";
        els.workerSummary.textContent = "Health check failed";
      }
    }

    async function sendMessage(event) {
      event.preventDefault();
      if (state.busy) {
        return;
      }
      const content = els.prompt.value.trim();
      if (!content) {
        return;
      }

      setError("");
      saveSettings();
      state.messages.push({ role: "user", content });
      saveHistory();
      els.prompt.value = "";
      renderMessages();
      setBusy(true);

      try {
        const headers = { "Content-Type": "application/json" };
        const apiKey = els.apiKey.value.trim();
        if (apiKey) {
          headers.Authorization = `Bearer ${apiKey}`;
        }

        const response = await fetch("/v1/chat/completions", {
          method: "POST",
          headers,
          body: JSON.stringify({
            model: els.model.value.trim() || "qwen2.5-9b-quantized",
            messages: state.messages,
            max_tokens: Number(els.maxTokens.value || 512),
            temperature: Number(els.temperature.value || 0.7),
            top_p: Number(els.topP.value || 0.9),
          }),
        });

        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
          throw new Error(payload.detail || payload.error?.message || `HTTP ${response.status}`);
        }

        const answer = payload.choices?.[0]?.message?.content || "";
        state.messages.push({ role: "assistant", content: answer || "(empty response)" });
        saveHistory();
        renderMessages();
        await refreshHealth();
      } catch (error) {
        saveHistory();
        setError(error.message || String(error));
      } finally {
        setBusy(false);
        els.prompt.focus();
      }
    }

    els.form.addEventListener("submit", sendMessage);
    els.saveSettings.addEventListener("click", saveSettings);
    els.clearChat.addEventListener("click", () => {
      state.messages = [];
      saveHistory();
      setError("");
      renderMessages();
    });
    els.refreshHealth.addEventListener("click", refreshHealth);
    els.prompt.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        els.form.requestSubmit();
      }
    });

    loadSettings();
    loadHistory();
    renderMessages();
    refreshHealth();
  </script>
</body>
</html>
"""
