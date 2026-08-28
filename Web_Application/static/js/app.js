/**
 * Analytica AI — Client Application Script
 * Handles settings persistence, dataset uploads, live table rendering,
 * and multi-agent copilot chat interaction.
 */

(() => {
  // -------------------------------------------------------------------
  // 1. SETTINGS & LOCAL STORAGE
  // -------------------------------------------------------------------
  function initSettings() {
    const apiKeyInput = document.getElementById("settings-api-key");
    const modelSelect = document.getElementById("settings-model");

    if (apiKeyInput) {
      const savedKey = localStorage.getItem("analytica_api_key") || "";
      apiKeyInput.value = savedKey;
      apiKeyInput.addEventListener("input", (e) => {
        localStorage.setItem("analytica_api_key", e.target.value.trim());
      });
    }

    if (modelSelect) {
      const savedModel = localStorage.getItem("analytica_model") || "gemini-2.5-flash";
      modelSelect.value = savedModel;
      modelSelect.addEventListener("change", (e) => {
        localStorage.setItem("analytica_model", e.target.value);
      });
    }
  }

  function getApiKey() {
    const apiKeyInput = document.getElementById("settings-api-key");
    return apiKeyInput ? apiKeyInput.value.trim() : (localStorage.getItem("analytica_api_key") || "");
  }

  function getSelectedModel() {
    const modelSelect = document.getElementById("settings-model");
    return modelSelect ? modelSelect.value : (localStorage.getItem("analytica_model") || "gemini-2.5-flash");
  }

  // -------------------------------------------------------------------
  // 2. DATASET UPLOAD LOGIC
  // -------------------------------------------------------------------
  async function uploadFile(file) {
    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch("/api/v1/dataset/upload", {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        alert("Upload Error: " + (data.detail || "Failed to upload dataset"));
        return;
      }

      // Dataset upload success handled by local page component
      if (window.location.pathname === "/workspace-empty") {
        if (window.AnalyticaRouter && typeof window.AnalyticaRouter.navigate === "function") {
          window.AnalyticaRouter.navigate("/workspace-empty");
        }
      }
    } catch (err) {
      console.error("[Upload] Error uploading dataset:", err);
      alert("Error uploading dataset: " + err.message);
    }
  }

  function initUploadEvents() {
    const dropZone = document.querySelector(".upload-dashed") || document.getElementById("upload-zone");
    const fileInput = document.getElementById("dataset-file-input") || document.getElementById("dataset-upload-input");

    if (dropZone) {
      dropZone.addEventListener("click", () => {
        if (fileInput) fileInput.click();
      });

      dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("border-primary");
      });

      dropZone.addEventListener("dragleave", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-primary");
      });

      dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("border-primary");
        if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
          uploadFile(e.dataTransfer.files[0]);
        }
      });
    }

    if (fileInput) {
      fileInput.addEventListener("change", (e) => {
        if (e.target.files && e.target.files.length > 0) {
          uploadFile(e.target.files[0]);
        }
      });
    }
  }

  // -------------------------------------------------------------------
  // 3. POPULATED WORKSPACE LIVE DATA RENDERING
  // -------------------------------------------------------------------
  async function loadPopulatedWorkspaceData() {
    const tableBody = document.querySelector("#data-table-body") || document.querySelector("#raw-data-table tbody");
    const tableHeader = document.querySelector("#data-table-head tr") || document.querySelector("#raw-data-table thead tr");
    const filenameLabel = document.querySelector("#data-filename");

    if (!tableBody) return;

    try {
      const res = await fetch("/api/v1/dataset/preview");
      const data = await res.json();

      if (!data.columns || data.columns.length === 0) return;

      // Render Header
      if (tableHeader) {
        tableHeader.innerHTML = data.columns
          .slice(0, 4)
          .map((col) => `<th class="py-2 font-normal uppercase">${col}</th>`)
          .join("");
      }

      // Render Rows
      tableBody.innerHTML = data.data
        .slice(0, 25)
        .map((row) => {
          const cells = data.columns
            .slice(0, 4)
            .map((col, idx) => {
              const val = row[col] !== null ? row[col] : "null";
              return `<td class="py-2 ${idx === 0 ? "text-primary font-semibold" : ""}">${val}</td>`;
            })
            .join("");
          return `<tr class="hover:bg-surface-bright transition-colors cursor-pointer group">${cells}</tr>`;
        })
        .join("");
    } catch (err) {
      console.error("[Workspace] Failed to load dataset preview:", err);
    }
  }

  // -------------------------------------------------------------------
  // 4. COPILOT CHAT INTERACTION
  // -------------------------------------------------------------------
  function formatMarkdown(text) {
    if (!text) return "";
    let html = text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/\*(.*?)\*/g, "<em>$1</em>")
      .replace(/`([^`]+)`/g, "<code class='bg-surface-variant px-1 rounded'>$1</code>")
      .replace(/^### (.*$)/gim, "<h3 class='text-sm font-bold text-primary mt-3 mb-1'>$1</h3>")
      .replace(/^## (.*$)/gim, "<h4 class='text-xs font-bold text-primary mt-2 mb-1'>$1</h4>")
      .replace(/^- (.*$)/gim, "<li class='ml-4 list-disc'>$1</li>\n")
      .replace(/\n\n/g, "<br/><br/>");
    return html;
  }

  async function sendChatMessage(inputElement, chatContainer) {
    if (!inputElement || !chatContainer) return;

    const message = inputElement.value.trim();
    if (!message) return;

    // Clear input
    inputElement.value = "";

    // Append User Message
    const userMsgEl = document.createElement("div");
    userMsgEl.className = "self-end max-w-[85%] border border-[#333] p-3 rounded-bl-lg rounded-tl-lg rounded-tr-lg bg-surface-container";
    userMsgEl.innerHTML = `<p class="text-on-surface-variant text-xs">${message}</p>`;
    chatContainer.appendChild(userMsgEl);

    // Scroll chat down
    chatContainer.scrollTop = chatContainer.scrollHeight;

    // Append Thinking Indicator
    const thinkingEl = document.createElement("div");
    thinkingEl.className = "self-start max-w-[90%] border border-outline-variant bg-[#1a1a1a] p-3 rounded-br-lg rounded-tr-lg rounded-tl-lg font-label-mono text-[11px] leading-relaxed";
    thinkingEl.innerHTML = `<div class="flex items-center space-x-2 text-primary animate-pulse"><span class="material-symbols-outlined text-sm">smart_toy</span><span>>> AGENTS PROCESSING QUERY...</span></div>`;
    chatContainer.appendChild(thinkingEl);
    chatContainer.scrollTop = chatContainer.scrollHeight;

    const apiKey = getApiKey();
    const model = getSelectedModel();

    try {
      const res = await fetch("/api/v1/copilot/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message, api_key: apiKey, model }),
      });

      const data = await res.json();
      thinkingEl.remove();

      const aiMsgEl = document.createElement("div");
      aiMsgEl.className = "self-start max-w-[95%] border border-[#444] bg-[#1a1a1a] p-3 rounded-br-lg rounded-tr-lg rounded-tl-lg font-label-mono text-[11px] leading-relaxed";

      if (data.status === "error") {
        aiMsgEl.innerHTML = `<p class="text-error font-semibold mb-1">>> ERROR</p><div class="text-on-surface-variant">${formatMarkdown(data.response)}</div>`;
      } else if (data.status === "warning") {
        aiMsgEl.innerHTML = `<p class="text-amber-400 font-semibold mb-1">>> NOTICE</p><div class="text-on-surface-variant">${formatMarkdown(data.response)}</div>`;
      } else {
        aiMsgEl.innerHTML = `<p class="text-primary font-semibold mb-1">>> ANALYSIS COMPLETE</p><div class="text-on-surface-variant">${formatMarkdown(data.response)}</div>`;
      }

      chatContainer.appendChild(aiMsgEl);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    } catch (err) {
      thinkingEl.remove();
      const errEl = document.createElement("div");
      errEl.className = "self-start max-w-[95%] border border-error bg-[#1a1a1a] p-3 rounded-br-lg rounded-tr-lg rounded-tl-lg font-label-mono text-[11px]";
      errEl.innerHTML = `<p class="text-error font-semibold">>> CONNECTION ERROR</p><p class="text-on-surface-variant">${err.message}</p>`;
      chatContainer.appendChild(errEl);
      chatContainer.scrollTop = chatContainer.scrollHeight;
    }
  }

  function initChatEvents() {
    const chatInputs = document.querySelectorAll(".copilot-chat-input");

    chatInputs.forEach((input) => {
      const parent = input.closest(".p-4") || input.parentElement;
      const sendBtn = parent ? parent.querySelector("button") : null;
      const chatContainer = input.closest("aside, section")?.querySelector(".overflow-y-auto");

      if (sendBtn && chatContainer) {
        sendBtn.addEventListener("click", () => sendChatMessage(input, chatContainer));
      }

      input.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && chatContainer) {
          e.preventDefault();
          sendChatMessage(input, chatContainer);
        }
      });
    });
  }

  // -------------------------------------------------------------------
  // 5. GLOBAL INITIALIZER FOR DYNAMIC SWAPS
  // -------------------------------------------------------------------
  function initPage() {
    initSettings();
    initUploadEvents();
    initChatEvents();
    loadPopulatedWorkspaceData();
  }

  // Run on initial load
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initPage);
  } else {
    initPage();
  }

  // Expose hook for client-side router
  window.AnalyticaApp = {
    initPage,
    uploadFile,
    getApiKey,
    getSelectedModel,
  };
})();
