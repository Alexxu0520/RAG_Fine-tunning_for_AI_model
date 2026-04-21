const questionEl = document.getElementById("question");
const modeEl = document.getElementById("mode");
const adapterPathEl = document.getElementById("adapterPath");
const askBtn = document.getElementById("askBtn");
const exampleBtn = document.getElementById("exampleBtn");
const statusEl = document.getElementById("status");
const resultsEl = document.getElementById("results");

console.log("app.js loaded");
console.log({ questionEl, modeEl, adapterPathEl, askBtn, exampleBtn, statusEl, resultsEl });

function setStatus(text) {
  if (statusEl) statusEl.textContent = text;
}

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");
}

function renderResults(data) {
  if (!resultsEl) return;
  resultsEl.innerHTML = "";

  if (data.error) {
    const div = document.createElement("div");
    div.className = "card error";
    div.innerHTML = `<h3>Error</h3><div class="answer">${escapeHtml(data.error)}</div>`;
    resultsEl.appendChild(div);
    return;
  }

  const results = data.results || [];
  for (const item of results) {
    const div = document.createElement("div");
    div.className = "card";
    div.innerHTML = `
      <h3>${escapeHtml(item.name || "")}</h3>
      <div class="answer">${escapeHtml(item.answer || "")}</div>
    `;
    resultsEl.appendChild(div);
  }
}

async function runQuery() {
  console.log("Run button clicked");

  const question = questionEl ? questionEl.value.trim() : "";
  const mode = modeEl ? modeEl.value : "all";
  const adapter_path = adapterPathEl ? adapterPathEl.value.trim() : "";

  if (!question) {
    setStatus("Please enter a question.");
    return;
  }

  if (askBtn) askBtn.disabled = true;
  setStatus("Running...");
  if (resultsEl) resultsEl.innerHTML = "";

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        question,
        mode,
        adapter_path,
      }),
    });

    const data = await res.json();
    console.log("API response:", data);
    renderResults(data);

    if (data.error) {
      setStatus("Request failed.");
    } else {
      setStatus("Done.");
    }
  } catch (err) {
    console.error("Fetch error:", err);
    renderResults({ error: err.message || "Unknown error" });
    setStatus("Request failed.");
  } finally {
    if (askBtn) askBtn.disabled = false;
  }
}

if (askBtn) {
  askBtn.addEventListener("click", runQuery);
}

if (exampleBtn) {
  exampleBtn.addEventListener("click", () => {
    if (questionEl) questionEl.value = "Where is Saint Denis?";
    if (modeEl) modeEl.value = "all";
    if (adapterPathEl) adapterPathEl.value = "outputs/qwen25_rdr2_lora";
  });
}