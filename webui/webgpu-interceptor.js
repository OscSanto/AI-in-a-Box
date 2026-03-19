// ── WebGPU Interceptor — injected into existing llama.cpp webui ───────────────
// Silently loads WebLLM (from local IIAB server), injects an opt-in toggle,
// and monkey-patches fetch to intercept /v1/chat/completions.
// When GPU mode is ON: calls /context for RAG context, then generates
// the answer locally via WebGPU. Falls back to server when OFF or unavailable.

(async function () {
  "use strict";

  const WEBLLM_MODEL = "SmolLM2-360M-Instruct-q4f16_1-MLC";
  const WEBLLM_SRC   = "/webllm/webllm.js";

  // ── State ──────────────────────────────────────────────────────────────────
  let engine      = null;   // WebLLM MLC engine
  let gpuReady    = false;  // true once model is loaded
  let gpuEnabled  = false;  // user toggle state
  let loading     = false;  // model currently loading

  // ── 1. Inject toggle UI ───────────────────────────────────────────────────
  const style = document.createElement("style");
  style.textContent = `
    #webgpu-toggle-bar {
      position: fixed; bottom: 12px; right: 12px; z-index: 9999;
      background: #1e1e2e; color: #cdd6f4; border-radius: 8px;
      padding: 6px 12px; font-size: 12px; font-family: system-ui, sans-serif;
      display: flex; align-items: center; gap: 8px;
      box-shadow: 0 2px 8px rgba(0,0,0,0.4); user-select: none;
    }
    #webgpu-toggle-bar label { cursor: pointer; display: flex; align-items: center; gap: 6px; }
    #webgpu-toggle-bar input[type=checkbox] { cursor: pointer; width: 14px; height: 14px; }
    #webgpu-status { font-size: 10px; color: #a6adc8; }
    #webgpu-toggle-bar.gpu-active { background: #1a3a2a; color: #a6e3a1; }
    #webgpu-progress {
      position: fixed; bottom: 44px; right: 12px; z-index: 9998;
      background: #1e1e2e; color: #cdd6f4; border-radius: 6px;
      padding: 6px 10px; font-size: 11px; font-family: system-ui, sans-serif;
      display: none; min-width: 200px;
    }
    #webgpu-progress progress { width: 100%; height: 4px; margin-top: 4px; }
  `;
  document.head.appendChild(style);

  const bar = document.createElement("div");
  bar.id = "webgpu-toggle-bar";
  bar.innerHTML = `
    <label>
      <input type="checkbox" id="webgpu-checkbox" />
      ⚡ Device GPU
    </label>
    <span id="webgpu-status">checking…</span>
  `;
  document.body.appendChild(bar);

  const progressDiv = document.createElement("div");
  progressDiv.id = "webgpu-progress";
  progressDiv.innerHTML = `<div id="webgpu-progress-label">Loading model…</div><progress id="webgpu-progress-bar" value="0" max="100"></progress>`;
  document.body.appendChild(progressDiv);

  const checkbox   = document.getElementById("webgpu-checkbox");
  const statusSpan = document.getElementById("webgpu-status");
  const progLabel  = document.getElementById("webgpu-progress-label");
  const progBar    = document.getElementById("webgpu-progress-bar");

  function setStatus(msg) { statusSpan.textContent = msg; }

  // ── 2. WebGPU capability check ────────────────────────────────────────────
  const hasWebGPU = !!navigator.gpu && !!(await navigator.gpu.requestAdapter().catch(() => null));

  if (!hasWebGPU) {
    setStatus("not supported");
    checkbox.disabled = true;
    return; // nothing more to do — fetch unchanged
  }

  // ── 3. Fetch admin config — sets default mode and toggle lock ────────────
  let inferenceConfig = { inference_default_mode: "server", inference_allow_user_toggle: true };
  try {
    const cfgRes = await fetch("/config");
    if (cfgRes.ok) inferenceConfig = await cfgRes.json();
  } catch (_) { /* use defaults if server unreachable */ }

  const userCanToggle = inferenceConfig.inference_allow_user_toggle;
  const defaultClient = inferenceConfig.inference_default_mode === "client";

  if (!userCanToggle) {
    // Admin locked the mode — hide toggle, honour default
    bar.style.display = "none";
  }

  if (defaultClient) {
    // Admin default is client inference — start loading immediately
    gpuEnabled = true;
    checkbox.checked = true;
    loadModel(); // async, don't await — progress bar will show
  } else {
    setStatus("available");
    checkbox.disabled = false;
  }

  // ── 4. Load WebLLM model when user enables toggle ─────────────────────────
  async function loadModel() {
    if (engine || loading) return;
    loading = true;
    setStatus("loading…");
    progressDiv.style.display = "block";

    try {
      // Dynamic import from local IIAB server
      const webllm = await import(WEBLLM_SRC);

      const APP_CONFIG = {
        model_list: [{
          model_id: WEBLLM_MODEL,
          model:     `/models/${WEBLLM_MODEL}/`,
          model_lib: `/models/${WEBLLM_MODEL}/`,
        }],
      };

      engine = await webllm.CreateMLCEngine(WEBLLM_MODEL, {
        appConfig: APP_CONFIG,
        initProgressCallback: (p) => {
          const pct = Math.round((p.progress || 0) * 100);
          progBar.value = pct;
          progLabel.textContent = p.text || `Loading… ${pct}%`;
        },
      });

      gpuReady = true;
      progressDiv.style.display = "none";
      setStatus("ready");
      bar.classList.add("gpu-active");
    } catch (err) {
      progressDiv.style.display = "none";
      setStatus("load failed");
      checkbox.checked = false;
      gpuEnabled = false;
      console.warn("[WebGPU interceptor] model load failed:", err);
    }
    loading = false;
  }

  checkbox.addEventListener("change", async () => {
    gpuEnabled = checkbox.checked;
    if (gpuEnabled) {
      await loadModel();
      if (!gpuReady) { gpuEnabled = false; checkbox.checked = false; }
    } else {
      bar.classList.remove("gpu-active");
      setStatus(gpuReady ? "ready" : "available");
    }
  });

  // ── 5. Vague query detection (rule-based, no LLM call) ───────────────────
  const VAGUE_PRONOUNS = /\b(it|its|they|their|them|that|this|those|these|he|she|his|her)\b/i;

  function isVague(q) {
    const words = q.trim().split(/\s+/);
    return words.length <= 5 || VAGUE_PRONOUNS.test(q);
  }

  // ── 6. Query rewrite using WebGPU + prior conversation turns ─────────────
  // Only called when isVague(query) is true and there are prior messages.
  // Uses the webui's own messages[] (already has full history) — no extra storage.
  async function rewriteQuery(rawQuery, priorMessages) {
    if (!engine || priorMessages.length < 2) return rawQuery;

    // Build a compact summary of the last prior exchange (last assistant + user pair)
    let lastQ = "", lastA = "";
    for (let i = priorMessages.length - 1; i >= 0; i--) {
      if (!lastA && priorMessages[i].role === "assistant") {
        const c = priorMessages[i].content;
        lastA = (typeof c === "string" ? c : c?.[0]?.text || "").slice(0, 300);
      }
      if (lastA && !lastQ && priorMessages[i].role === "user") {
        const c = priorMessages[i].content;
        lastQ = typeof c === "string" ? c : (c?.[0]?.text || "");
        break;
      }
    }
    if (!lastQ) return rawQuery;

    const prompt = `Previous question: ${lastQ}
Previous answer: ${lastA}
Follow-up question: ${rawQuery}

Rewrite the follow-up as a complete standalone question. Reply with only the rewritten question, nothing else.`;

    try {
      const resp = await engine.chat.completions.create({
        messages: [
          { role: "system", content: "You rewrite vague follow-up questions into clear standalone questions using the provided context." },
          { role: "user",   content: prompt },
        ],
        stream: false,
        max_tokens: 48,
        temperature: 0.0,
      });
      const rewritten = resp.choices[0]?.message?.content?.trim();
      if (rewritten && rewritten.length > 4) return rewritten;
    } catch (_) {}

    return rawQuery;
  }

  // ── 7. Monkey-patch fetch ─────────────────────────────────────────────────
  const _origFetch = window.fetch.bind(window);

  window.fetch = async function (input, init, ...rest) {
    const url = typeof input === "string" ? input : (input instanceof URL ? input.href : input.url);

    // Only intercept chat completions when GPU is active
    if (!gpuEnabled || !gpuReady || !engine) {
      return _origFetch(input, init, ...rest);
    }
    if (!url || !url.includes("/v1/chat/completions")) {
      return _origFetch(input, init, ...rest);
    }

    // Parse the request body to extract messages
    let body = {};
    try {
      const rawBody = init?.body;
      body = rawBody ? JSON.parse(rawBody) : {};
    } catch (_) {
      return _origFetch(input, init, ...rest);
    }

    const messages = body.messages || [];

    // Extract the last user message as the raw query
    let rawQuery = "";
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        const c = messages[i].content;
        rawQuery = typeof c === "string" ? c : (c?.[0]?.text || "");
        break;
      }
    }
    if (!rawQuery) return _origFetch(input, init, ...rest);

    // Step A: rewrite vague query using prior turns (WebGPU, ~48 tokens)
    // priorMessages = everything except the last user message
    const priorMessages = messages.slice(0, -1);
    const query = isVague(rawQuery) ? await rewriteQuery(rawQuery, priorMessages) : rawQuery;

    // Step B: get RAG context from server (stages 1–10)
    let ctxData;
    try {
      const ctxRes = await _origFetch("/context", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query }),
      });
      ctxData = await ctxRes.json();
    } catch (err) {
      console.warn("[WebGPU interceptor] /exp01/context failed, falling back:", err);
      return _origFetch(input, init, ...rest);
    }

    // Step C: cache hit — stream cached answer directly, no generation needed
    if (ctxData.cache_hit) {
      return _makeFakeSSEResponse(ctxData.answer, query, null, /*remember=*/false);
    }

    // Step D: generate answer locally via WebGPU
    const SYS = "Answer using only the provided context. Be detailed and thorough.";
    const genMessages = [
      { role: "system", content: SYS },
      { role: "user",   content: ctxData.prompt },
    ];

    return _makeFakeSSEResponse(null, query, { genMessages, ctxData }, /*remember=*/true);
  };

  // ── 8. Fake SSE Response helpers ──────────────────────────────────────────
  // Returns a Response whose body is a ReadableStream of SSE chunks,
  // matching exactly what the webui expects from the real /v1/chat/completions.

  function _sseChunk(text, finish = false) {
    const payload = JSON.stringify({
      id: "chatcmpl-webgpu",
      object: "chat.completion.chunk",
      created: Math.floor(Date.now() / 1000),
      model: WEBLLM_MODEL,
      choices: [{ index: 0, delta: { content: text }, finish_reason: finish ? "stop" : null }],
    });
    return `data: ${payload}\n\n`;
  }

  function _makeFakeSSEResponse(cachedAnswer, query, genCtx, remember) {
    const encoder = new TextEncoder();

    const stream = new ReadableStream({
      async start(controller) {
        try {
          if (cachedAnswer !== null) {
            // Stream cached answer word by word
            for (const word of cachedAnswer.split(" ")) {
              controller.enqueue(encoder.encode(_sseChunk(word + " ")));
              await _sleep(8);
            }
          } else {
            // Generate via WebGPU
            const { genMessages, ctxData } = genCtx;
            let fullAnswer = "";
            const genStream = await engine.chat.completions.create({
              messages: genMessages,
              stream: true,
              max_tokens: 512,
            });
            for await (const chunk of genStream) {
              const text = chunk.choices[0]?.delta?.content || "";
              if (text) {
                fullAnswer += text;
                controller.enqueue(encoder.encode(_sseChunk(text)));
              }
            }

            // POST answer back to server for query memory
            if (remember && fullAnswer) {
              _origFetch("/remember", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, answer: fullAnswer, query_vec: ctxData.query_vec }),
              }).catch(() => {});
            }
          }

          controller.enqueue(encoder.encode(_sseChunk("", true)));
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        } catch (err) {
          console.warn("[WebGPU interceptor] generation error:", err);
          controller.enqueue(encoder.encode(_sseChunk(`[GPU error: ${err.message}]`, true)));
          controller.enqueue(encoder.encode("data: [DONE]\n\n"));
        }
        controller.close();
      },
    });

    return new Response(stream, {
      status: 200,
      headers: { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" },
    });
  }

  function _sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

})();

// ── Citation + Footnote linkifier ─────────────────────────────────────────────
// Handles two patterns in streamed LLM output:
//   [Article | Section]  → clickable link to Kiwix article
//   [17]                 → superscript link to footnote anchor in the same article
// currentArticle tracks the most recently seen [Article | Section] so footnote
// [N] links know which article they belong to.
(async function () {
  let zimId = "wikipedia_en_all_maxi_2025-08"; // fallback default
  try {
    const cfg = await fetch("/config").then(r => r.json());
    if (cfg.zim_content_id) zimId = cfg.zim_content_id;
  } catch (_) {}

  let currentArticle = ""; // updated whenever [Article | Section] is seen

  // Group 1+2 = [Article | Section],  Group 3 = [N] footnote
  const COMBINED_RE = /\[([^\|\]\n]+)\|([^\]\n]+)\]|\[(\d+)\]/g;

  function linkify(textNode) {
    const text = textNode.textContent;
    COMBINED_RE.lastIndex = 0;
    if (!COMBINED_RE.test(text)) return;
    COMBINED_RE.lastIndex = 0;

    const span = document.createElement("span");
    let last = 0, m;
    while ((m = COMBINED_RE.exec(text)) !== null) {
      if (m.index > last) span.appendChild(document.createTextNode(text.slice(last, m.index)));

      if (m[1] && m[2]) {
        // ── [Article | Section] citation ──────────────────────────────────
        const article = m[1].trim();
        const section = m[2].trim();
        currentArticle = article; // remember for following [N] footnotes
        const slug = encodeURIComponent(article.replace(/ /g, "_"));
        const a = document.createElement("a");
        a.href        = `/kiwix/${zimId}/A/${slug}`;
        a.target      = "_blank";
        a.rel         = "noopener";
        a.title       = `${article} — ${section}`;
        a.textContent = `[${article} | ${section}]`;
        a.style.cssText = "color:#89b4fa;text-decoration:none;font-size:0.85em;";
        a.onmouseover = () => { a.style.textDecoration = "underline"; };
        a.onmouseout  = () => { a.style.textDecoration = "none"; };
        span.appendChild(a);

      } else if (m[3] && currentArticle) {
        // ── [N] footnote → anchor in the current article ──────────────────
        const num  = m[3];
        const slug = encodeURIComponent(currentArticle.replace(/ /g, "_"));
        const a = document.createElement("a");
        a.href        = `/kiwix/${zimId}/A/${slug}#cite_note-${num}`;
        a.target      = "_blank";
        a.rel         = "noopener";
        a.title       = `Footnote ${num} in ${currentArticle}`;
        a.textContent = `[${num}]`;
        a.style.cssText = "color:#a6e3a1;text-decoration:none;font-size:0.75em;vertical-align:super;";
        a.onmouseover = () => { a.style.textDecoration = "underline"; };
        a.onmouseout  = () => { a.style.textDecoration = "none"; };
        span.appendChild(a);

      } else {
        // [N] with no article context yet — render as plain text
        span.appendChild(document.createTextNode(m[0]));
      }

      last = m.index + m[0].length;
    }
    if (last < text.length) span.appendChild(document.createTextNode(text.slice(last)));
    textNode.parentNode.replaceChild(span, textNode);
  }

  function processNode(root) {
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(linkify);
  }

  const observer = new MutationObserver(mutations => {
    for (const m of mutations) {
      for (const node of m.addedNodes) {
        if (node.nodeType === Node.ELEMENT_NODE) processNode(node);
        else if (node.nodeType === Node.TEXT_NODE)  linkify(node);
      }
    }
  });

  observer.observe(document.body, { childList: true, subtree: true });
})();
