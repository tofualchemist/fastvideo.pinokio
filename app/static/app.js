/* FastVideo MLX front end. Vanilla JS, no dependencies, works offline. */
(() => {
  "use strict";
  const $ = (sel, root = document) => root.querySelector(sel);
  const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
  const el = (tag, attrs = {}, ...children) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "html") node.innerHTML = v;
      else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
      else if (v === false || v == null) continue;
      else node.setAttribute(k, v === true ? "" : v);
    }
    for (const c of children.flat()) if (c != null) node.append(c.nodeType ? c : document.createTextNode(String(c)));
    return node;
  };
  const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const fmtDur = (s) => {
    if (s == null || isNaN(s)) return "";
    s = Math.round(s);
    if (s < 60) return `${s} s`;
    const m = Math.floor(s / 60), r = s % 60;
    if (m < 60) return r ? `${m} min ${r} s` : `${m} min`;
    return `${Math.floor(m / 60)} h ${m % 60} min`;
  };
  const fmtEta = (s) => (s == null ? "" : s < 60 ? "under a minute" : `~${Math.max(1, Math.round(s / 60))} min`);
  const fmtApprox = (s) => { if (s == null) return ""; if (s < 120) return fmtDur(s); const m = Math.round(s / 60); return m < 60 ? `${m} min` : `${Math.floor(m / 60)} h ${m % 60 ? (m % 60) + " min" : ""}`.trim(); };

  const TEMPLATE = `(S1) A [who] in [where], [what they do], [camera move], [lighting and mood].
<d>[English] Spoken line goes here.</d>
Soundscape: [ambient sound], [effects], [music if any].`;

  const TEMPLATE_SILENT = `A [who or what] in [where], [what happens], [camera move], [lighting and mood], [style words].`;

  const state = {
    projects: [], pid: null, project: null, models: null, queue: null, settings: null, fake: false,
    composer: { prompt: "", model: null, quality: "draft", duration: 5, seed: "", locked: false, refs: [], parent: null, canvas: "480p", draftMode: "fast", vsaImpl: "" },
    estimate: null, installs: {}, showDiff: {}, logLines: [],
  };

  // -- api -----------------------------------------------------------------------------
  let lastEventAt = 0;
  async function api(method, url, body) {
    const opts = { method, headers: {} };
    if (body !== undefined) { opts.headers["Content-Type"] = "application/json"; opts.body = JSON.stringify(body); }
    const sentAt = Date.now();
    const res = await fetch(url, opts);
    if (!res.ok) {
      let msg = res.statusText;
      try { const j = await res.json(); msg = j.detail || j.error || msg; } catch (_) { /* ignore */ }
      throw new Error(msg);
    }
    if (method !== "GET") {
      // If the event stream is dead we would never hear about this change: resync unless an event arrives.
      setTimeout(() => { if (lastEventAt < sentAt) resync(); }, 1500);
    }
    return res.json();
  }
  async function resync() {
    try {
      const s = await api("GET", "/api/state");
      Object.assign(state, { projects: s.projects, models: s.models, queue: s.queue, settings: s.settings, active: s.queue.active || null });
      if (state.pid) { try { state.project = await api("GET", `/api/projects/${state.pid}`); } catch (_) { state.pid = null; state.project = null; } }
      renderAll();
    } catch (e) { console.error("resync failed", e); }
  }
  async function upload(pid, file) {
    const fd = new FormData(); fd.append("file", file);
    const res = await fetch(`/api/projects/${pid}/uploads`, { method: "POST", body: fd });
    if (!res.ok) throw new Error((await res.json()).detail || "upload failed");
    return res.json();
  }
  let toastTimer;
  function toast(msg, error = false) {
    const t = $("#toast"); t.textContent = msg; t.classList.toggle("error", error); t.hidden = false;
    clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), error ? 6000 : 3000);
  }

  // -- bootstrap -----------------------------------------------------------------------
  async function init() {
    const s = await api("GET", "/api/state");
    Object.assign(state, { projects: s.projects, models: s.models, queue: s.queue, settings: s.settings, fake: s.fake, active: s.queue.active || null });
    state.composer.model = s.settings.default_model;
    state.composer.canvas = s.settings.canvas || "480p";
    const last = localStorage.getItem("fasth3.pid");
    if (last && s.projects.some((p) => p.id === last)) await openProject(last);
    else if (s.projects.length) await openProject(s.projects[0].id);
    renderAll();
    connect();
    bindUI();
    if (localStorage.getItem("fasth3.side") === "collapsed") toggleSide(true);
  }

  let es = null, everConnected = false;
  function connect() {
    if (es) { try { es.close(); } catch (_) { /* ignore */ } }
    es = new EventSource("/api/events");
    let dropped = false;
    es.onmessage = (m) => { lastEventAt = Date.now(); try { handleEvent(JSON.parse(m.data)); } catch (e) { console.error(e); } };
    es.onerror = () => { dropped = true; };
    es.onopen = () => {
      lastEventAt = Date.now();
      if (dropped || everConnected) resync();   // the server restarted or the stream dropped
      dropped = false; everConnected = true;
    };
  }
  // Watchdog: the server pings every 15 s; a silent stream is a dead stream.
  setInterval(() => { if (es && Date.now() - lastEventAt > 45000) { connect(); } }, 10000);
  document.addEventListener("visibilitychange", () => { if (document.visibilityState === "visible") resync(); });
  window.addEventListener("focus", () => { if (Date.now() - lastEventAt > 20000) resync(); });

  function handleEvent(ev) {
    switch (ev.type) {
      case "turn":
        if (state.project && ev.project_id === state.pid) {
          const i = state.project.turns.findIndex((t) => t.id === ev.turn.id);
          if (i >= 0) state.project.turns[i] = ev.turn; else state.project.turns.push(ev.turn);
          renderCard(ev.turn);
          renderEmpty();
          if (ev.turn.status === "done") { api("POST", `/api/projects/${state.pid}/seen`).catch(() => {}); renderGenerateButton(); }
        }
        refreshProjects();
        break;
      case "turn_deleted":
        if (ev.project_id === state.pid && state.project) {
          state.project.turns = state.project.turns.filter((t) => t.id !== ev.turn_id);
          $(`#turn-${ev.turn_id}`)?.remove(); renderEmpty();
        }
        refreshProjects();
        break;
      case "projects": refreshProjects(); break;
      case "queue": state.queue = ev; state.active = ev.active || (ev.current ? state.active : null); renderQueueMeta(); renderQueuedCards(); break;
      case "active": state.active = ev; renderQueueMeta(); renderQueuedCards(); break;
      case "models": refreshModels(); break;
      case "install": state.installs[ev.model] = ev; renderQueueMeta(); if (ev.phase === "done" || ev.phase === "failed") toast(ev.phase === "done" ? `${state.models?.models?.[ev.model]?.name || ev.model} downloaded` : `Download failed: ${ev.message}`, ev.phase === "failed"); break;
      case "settings": state.settings = ev.settings; break;
      case "log": state.logLines.push(ev.line); if (state.logLines.length > 400) state.logLines.shift(); renderLog(); break;
      case "notify": notify(ev); break;
      case "ping": break;
      default: break;
    }
  }
  async function refreshProjects() { state.projects = await api("GET", "/api/projects"); renderProjects(); }
  async function refreshModels() { state.models = await api("GET", "/api/models"); renderComposer(); }

  function notify(ev) {
    if (!state.settings?.notifications) return;
    if (typeof Notification === "undefined" || Notification.permission !== "granted") return;
    if (document.hasFocus() && ev.project_id === state.pid) return;
    try { new Notification(ev.title || "Clip ready", { body: ev.body || "" }); } catch (_) { /* ignore */ }
  }

  // -- projects ------------------------------------------------------------------------
  async function openProject(pid) {
    state.pid = pid; localStorage.setItem("fasth3.pid", pid);
    state.project = await api("GET", `/api/projects/${pid}`);
    api("POST", `/api/projects/${pid}/seen`).catch(() => {});
    state.composer.parent = null;
    const lastTurn = state.project.turns[state.project.turns.length - 1];
    if (lastTurn) loadIntoComposer(lastTurn, { keepParent: true });
    renderAll();
    const tl = $("#timeline"); tl.scrollTop = tl.scrollHeight;
  }
  function renderProjects() {
    const list = $("#project-list"); list.innerHTML = "";
    for (const p of state.projects) {
      const dot = el("i", { class: "dot " + (p.active ? "run" : p.unseen ? "new" : "") });
      const row = el("button", { class: "proj" + (p.id === state.pid ? " active" : ""), onclick: () => openProject(p.id), title: p.name },
        dot, el("span", { class: "n" }, p.name), el("span", { class: "c" }, p.turn_count || ""));
      list.append(row);
    }
    const cur = state.projects.find((p) => p.id === state.pid);
    if (cur) $("#project-title").textContent = cur.name;
  }

  // -- timeline ------------------------------------------------------------------------
  function renderAll() { renderProjects(); renderTimeline(); renderComposer(); renderQueueMeta(); renderSettingsForm(); }
  function renderEmpty() { $("#empty-state").hidden = !!(state.project && state.project.turns.length); }
  function renderTimeline() {
    const tl = $("#timeline");
    $$(".turn", tl).forEach((n) => n.remove());
    if (!state.project) { $("#empty-state").hidden = false; $("#composer").hidden = true; return; }
    $("#composer").hidden = false;
    renderEmpty();
    for (const t of state.project.turns) renderCard(t);
  }
  function turnById(id) { return state.project?.turns.find((t) => t.id === id); }

  function renderCard(turn) {
    const tl = $("#timeline");
    let card = $(`#turn-${turn.id}`);
    const pending = ["queued", "preparing", "running"].includes(turn.status);
    if (card && card.dataset.status === "done" && turn.status === "done") return; // keep the video element alive
    const fresh = buildCard(turn);
    if (card) { const atBottom = tl.scrollHeight - tl.scrollTop - tl.clientHeight < 80; card.replaceWith(fresh); if (atBottom) tl.scrollTop = tl.scrollHeight; }
    else {
      // insert in order of index
      const after = $$(".turn", tl).filter((n) => Number(n.dataset.index) < turn.index).pop();
      if (after) after.after(fresh); else tl.append(fresh);
      tl.scrollTop = tl.scrollHeight;
    }
    void pending;
  }

  function buildCard(turn) {
    const pending = ["queued", "preparing", "running"].includes(turn.status);
    const parent = turn.parent_id ? turnById(turn.parent_id) : null;
    const spec = state.models?.models?.[turn.model];
    const card = el("div", { class: "turn" + (pending ? " pending" : ""), id: `turn-${turn.id}`, "data-index": turn.index, "data-status": turn.status });

    // left: video / progress
    let media;
    if (turn.status === "done") {
      const video = el("video", { muted: true, loop: true, playsinline: true, preload: "metadata", src: `/files/projects/${turn.project_id}/${turn.result.clip}#t=0.5` });
      video.muted = true;
      media = el("button", { class: "vid thumb", title: "Open", "aria-label": `Open turn ${turn.index}`, onclick: () => openViewer(turn.id) },
        el("span", { class: "badge" + (turn.quality === "final" ? " final" : "") }, turn.quality === "final" ? "Final" : "Draft"),
        video,
        el("span", { class: "play", "aria-hidden": "true", html: '<svg width="44" height="44" viewBox="0 0 44 44"><circle cx="22" cy="22" r="21" fill="rgba(0,0,0,.55)"/><path d="M17 13 L31 22 L17 31 Z" fill="#fff"/></svg>' }),
        el("span", { class: "dur" }, `${turn.duration_s || 5} s`));
      media.addEventListener("mouseenter", () => { video.currentTime = 0; video.play().catch(() => {}); });
      media.addEventListener("mouseleave", () => { video.pause(); video.currentTime = 0.5; });
    } else if (pending) {
      const p = turn.progress || {};
      if (p.preview) {
        media = el("div", { class: "vid preview" },
          el("span", { class: "badge" + (turn.quality === "final" ? " final" : "") }, turn.quality === "final" ? "Final" : "Draft"),
          el("img", { src: `/files/projects/${turn.project_id}/${p.preview}?v=${p.preview_step || 0}`, alt: "preview" }),
          el("span", { class: "badge right" }, `preview · step ${p.preview_step || "?"}`));
      } else {
        media = el("div", { class: "vid pending" },
          el("span", { class: "badge" + (turn.quality === "final" ? " final" : "") }, turn.quality === "final" ? "Final" : "Draft"),
          el("div", { class: "spinner" }),
          el("div", {}, turn.status === "queued" ? "Queued" : phaseLabel(p.phase)));
      }
    } else {
      media = el("div", { class: "vid failed" }, turn.status === "cancelled" ? "Cancelled" : "Failed");
    }

    // right: header
    const headLeft = el("span", {}, `Turn ${turn.index} · ${spec ? spec.short : turn.model}`,
      parent ? ` · ${turn.kind === "reroll" ? "reroll of" : turn.kind === "finalize" ? "finalize of" : "from"} turn ${parent.index}` : "",
      ` · seed ${turn.seed}`, parent && parent.seed === turn.seed ? el("span", { class: "lock", title: "same seed as parent" }, " 🔒") : "",
      turn.duration_s && turn.duration_s !== 5 ? ` · ${turn.duration_s} s` : "", turn.canvas && turn.canvas !== "480p" ? ` · ${turn.canvas}` : "");
    let headRight = "";
    if (turn.status === "done") headRight = fmtDur(turn.result?.wall_s);
    else if (turn.status === "running") {
      const p = turn.progress || {};
      headRight = (p.steps ? `step ${p.step || 0} of ${p.steps}` : phaseLabel(p.phase)) + (p.eta_s != null ? ` · ${fmtEta(p.eta_s)} left` : "");
    } else if (turn.status === "preparing") headRight = (turn.progress?.message || "Preparing").slice(0, 90);
    else if (turn.status === "queued") headRight = queuePosition(turn);
    if (turn.status === "queued") {
      const a = state.active;
      if (a && a.turn_id !== turn.id) {
        const where = a.project_id === state.pid ? `turn ${a.turn_index}` : `${a.project_name} · turn ${a.turn_index}`;
        const prog = a.steps ? `step ${a.step || 0} of ${a.steps}` : phaseLabel(a.phase);
        headRight = `${queuePosition(turn)} · waiting for ${where} (${prog}${a.eta_s != null ? `, ${fmtEta(a.eta_s)} left` : ""})`;
      }
    }
    const head = el("div", { class: "turn-head" }, headLeft, el("span", {}, headRight));

    // prompt with diff
    const showDiff = state.showDiff[turn.id] !== false && parent && parent.prompt !== turn.prompt;
    const prompt = el("p", { class: "prompt" });
    if (showDiff) prompt.innerHTML = diffHtml(parent.prompt, turn.prompt); else prompt.textContent = turn.prompt;
    const body = [head, prompt];
    if (parent && parent.prompt !== turn.prompt) {
      const n = countChanges(parent.prompt, turn.prompt);
      body.push(el("button", { class: "link diff-toggle", onclick: () => { state.showDiff[turn.id] = !showDiff; card.replaceWith(buildCard(turn)); } },
        showDiff ? `Hide changes from turn ${parent.index}` : `${n} change${n === 1 ? "" : "s"} from turn ${parent.index}`));
    }
    if (turn.refs?.length) {
      body.push(el("div", { class: "refs-line" }, turn.refs.map((r) => el("span", { class: "chip" }, `${r.media_type}: ${r.name}`))));
    }
    if (pending && turn.status !== "queued") {
      const p = turn.progress || {};
      const frac = p.fraction;
      const bar = el("div", { class: "progress" + (frac == null ? " indeterminate" : "") }, el("i", { style: `width:${Math.round((frac || 0) * 100)}%` }));
      body.push(bar);
      const decoderNote = turn.engine === "wan" ? (turn.quality === "final" ? (turn.wan_vae ? "all frames · full Wan VAE" : "all frames · fast decoder") : "draft · half the frames, interpolated") : (turn.quality === "final" ? "full quality · 36-layer decoder" : "draft · 8-layer decoder");
      body.push(el("div", { class: "meta" }, p.message ? p.message.slice(0, 140) : "", turn.status === "running" && turn.quality ? ` · ${decoderNote} · ${turn.width}×${turn.height} · ${turn.num_frames} frames${turn.fps ? ` at ${turn.fps} fps` : ""}` : ""));
    }
    if (turn.status === "failed") body.push(el("div", { class: "err" }, turn.error || "Unknown error"));

    // actions
    const acts = el("div", { class: "acts" });
    if (turn.status === "done") {
      acts.append(el("button", { onclick: () => reroll(turn), title: "Same prompt and settings, new seed" }, "🎲 Reroll"));
      acts.append(el("button", { onclick: () => edit(turn), title: "Load this turn into the composer" }, "✎ Edit"));
      if (turn.quality === "draft") acts.append(el("button", { onclick: () => finalize(turn), title: "Render this prompt and seed at full quality (all frames, 36-layer decoder). Drafts that used the temporal shortcut will produce a different take; use draft mode 'none' for a matching pair." }, "✦ Finalize"));
      if (spec?.supports_references || Object.values(state.models?.models || {}).some((m) => m.supports_references))
        acts.append(el("button", { onclick: () => useAsRef(turn, "video"), title: "Add this clip as a video reference (Omni)" }, "⇥ Use as reference"));
      acts.append(el("a", { class: "chip", href: `/files/projects/${turn.project_id}/${turn.result.clip}`, download: `turn-${turn.index}-seed-${turn.seed}.mp4`, title: "Download" }, "⤓"));
      acts.append(el("button", { class: "danger", onclick: () => removeTurn(turn), title: "Delete this turn and its clip" }, "✕"));
    } else if (pending) {
      acts.append(el("button", { onclick: () => cancel(turn) }, turn.status === "queued" ? "Remove from queue" : "✕ Cancel"));
    } else {
      acts.append(el("button", { onclick: () => edit(turn) }, "✎ Edit"));
      acts.append(el("button", { onclick: () => retry(turn) }, "↻ Retry"));
      acts.append(el("button", { class: "danger", onclick: () => removeTurn(turn) }, "✕"));
    }
    body.push(acts);
    card.append(media, el("div", {}, body));
    return card;
  }

  function renderQueuedCards() {
    for (const t of state.project?.turns || []) if (t.status === "queued") renderCard(t);
  }
  function queuePosition(turn) {
    const q = state.queue?.queued || [];
    const i = q.findIndex((x) => x[1] === turn.id);
    return i >= 0 ? `queued · position ${i + 1}` : "queued";
  }
  function phaseLabel(phase) {
    return { prepare: "Preparing", download: "Downloading model", convert: "Converting weights", load: "Loading", condition: "Encoding prompt", denoise: "Rendering", decode: "Decoding", mux: "Finishing", done: "Done" }[phase] || "Working";
  }

  // -- word diff -----------------------------------------------------------------------
  function tokens(s) { return s.split(/(\s+)/).filter((x) => x !== ""); }
  function diffOps(a, b) {
    const A = tokens(a), B = tokens(b), n = A.length, m = B.length;
    const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
    for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
    const ops = []; let i = 0, j = 0;
    while (i < n && j < m) {
      if (A[i] === B[j]) { ops.push(["=", A[i]]); i++; j++; }
      else if (dp[i + 1][j] >= dp[i][j + 1]) ops.push(["-", A[i++]]);
      else ops.push(["+", B[j++]]);
    }
    while (i < n) ops.push(["-", A[i++]]);
    while (j < m) ops.push(["+", B[j++]]);
    return ops;
  }
  function diffHtml(a, b) {
    let out = "", run = null, buf = "";
    const flush = () => { if (buf) out += run === "=" ? esc(buf) : `<span class="${run === "+" ? "add" : "del"}">${esc(buf)}</span>`; buf = ""; };
    for (const [op, tok] of diffOps(a, b)) {
      const kind = op === "=" || /^\s+$/.test(tok) && run !== null && op !== run ? (op === "=" ? "=" : run) : op;
      if (kind !== run) { flush(); run = kind; }
      buf += tok;
    }
    flush();
    return out;
  }
  function countChanges(a, b) {
    let n = 0, prev = "=";
    for (const [op, tok] of diffOps(a, b)) { if (op !== "=" && !/^\s+$/.test(tok) && prev === "=") n++; if (!/^\s+$/.test(tok)) prev = op; }
    return n;
  }

  // -- composer ------------------------------------------------------------------------
  function loadIntoComposer(turn, { keepParent = false } = {}) {
    const c = state.composer;
    c.prompt = turn.prompt; c.model = turn.model; c.quality = turn.quality || "draft"; c.duration = turn.duration_s || 5;
    c.seed = String(turn.seed); c.locked = true; c.canvas = turn.canvas || "480p";
    c.refs = (turn.refs || []).map((r) => ({ ...r }));
    c.parent = turn.id;
    void keepParent;
  }
  function renderComposer() {
    const c = state.composer, models = state.models?.models || {};
    if (!state.project) return;
    const parent = c.parent ? turnById(c.parent) : null;
    $("#composer-from").textContent = parent ? `Continuing from turn ${parent.index}` : "New turn";
    $("#prompt").value = c.prompt;
    if (!models[c.model]) c.model = Object.keys(models)[0];
    const spec = models[c.model];
    $("#model-chip-label").textContent = spec ? `${spec.name}${spec.audio === false ? " · silent" : ""}` : "Model";
    // duration
    const ds = $("#duration-select"); ds.innerHTML = "";
    for (const d of spec?.durations_s || [5]) ds.append(el("option", { value: d }, `${d} s`));
    if (!(spec?.durations_s || [5]).includes(c.duration)) c.duration = (spec?.durations_s || [5])[0];
    ds.value = c.duration;
    $("#quality-select").value = c.quality;
    $("#seed").value = c.seed; $(".chip.seed").classList.toggle("locked", c.locked);
    $("#seed").placeholder = c.locked ? "" : "random";
    // refs
    const refsRow = $("#refs-row"); refsRow.classList.toggle("disabled", !spec?.supports_references);
    $("#refs-hint").textContent = spec?.supports_references ? "image · video · voice, in order" : spec?.audio === false ? "silent video · no references" : "references need FastH3 Omni";
    $("#prompt").placeholder = spec?.audio === false
      ? "Describe the shot. Subject, action, camera, lighting. This model makes silent video."
      : "Describe the shot. Subject, action, camera, lighting. Add dialogue as (S1) ... <d>[English] words</d> and finish with the soundscape.";
    const cs = $("#canvas-select"); const canvases = spec?.canvases || ["480p"];
    for (const opt of cs.options) opt.disabled = !canvases.includes(opt.value);
    if (!canvases.includes(c.canvas)) c.canvas = canvases[0];
    const chips = $("#ref-chips"); chips.innerHTML = "";
    c.refs.forEach((r, i) => chips.append(el("span", { class: "chip" }, `${r.media_type}: ${r.name}`, el("button", { onclick: () => { c.refs.splice(i, 1); renderComposer(); }, "aria-label": "remove" }, "✕"))));
    $("#canvas-select").value = c.canvas; $("#draft-mode").value = c.draftMode; $("#vsa-impl").value = c.vsaImpl;
    $("#draft-mode").disabled = !spec?.supports_fast_modes;
    $("#wan-vae-label").hidden = spec?.engine !== "wan"; $("#wan-vae").value = c.wanVae ? "1" : "";
    $("#model-hint").textContent = spec ? spec.description : "";
    renderGenerateButton();
  }
  let estTimer;
  function renderGenerateButton() {
    const c = state.composer, spec = state.models?.models?.[c.model];
    const btn = $("#generate");
    const installed = spec?.installed;
    btn.textContent = installed ? "Generate" : `Download & generate (${spec ? spec.download_gb : "?"} GB)`;
    if (!installed) return;
    clearTimeout(estTimer);
    estTimer = setTimeout(async () => {
      try {
        const clips = c.refs.filter((r) => r.media_type === "video").length;
        const e = await api("GET", `/api/estimate?model=${c.model}&quality=${c.quality}&refs=${c.refs.length}&ref_clips=${clips}&duration=${Number(c.duration)}&canvas=${c.canvas}&fast=${c.draftMode === "fast" || c.draftMode === "both" ? 1 : 0}&fast_spatial=${c.draftMode === "spatial" || c.draftMode === "both" ? 1 : 0}`);
        if (e.total_s == null) return;
        btn.textContent = `Generate · ${fmtEta(e.total_s)}${e.basis === "measured" ? "" : " est."}`;
        $("#model-chip-label").textContent = `${spec.name}${spec.audio === false ? " · silent" : ""} · ${fmtEta(e.total_s)}`;
      } catch (_) { /* ignore */ }
    }, 150);
  }

  async function generate(extra = {}) {
    const c = state.composer;
    if (!state.pid) return toast("Create a project first", true);
    const prompt = $("#prompt").value.trim();
    if (!prompt && !extra.prompt) return toast("Write a prompt first", true);
    if (typeof Notification !== "undefined" && Notification.permission === "default") Notification.requestPermission().catch(() => {});
    const body = {
      prompt, model: c.model, quality: c.quality, duration_s: Number(c.duration), canvas: c.canvas,
      seed: c.locked && c.seed !== "" ? Number(c.seed) : null, parent_id: c.parent, kind: "generate",
      refs: c.refs.map((r) => (r.from_turn ? { media_type: r.media_type, from_turn: r.from_turn } : { media_type: r.media_type, path: r.path, name: r.name })),
      fast: c.draftMode === "fast" || c.draftMode === "both", fast_spatial: c.draftMode === "spatial" || c.draftMode === "both",
      vsa_impl: c.vsaImpl || null, wan_vae: !!c.wanVae, ...extra,
    };
    try {
      const turn = await api("POST", `/api/projects/${state.pid}/turns`, body);
      if (state.project && !state.project.turns.some((t) => t.id === turn.id)) { state.project.turns.push(turn); renderCard(turn); renderEmpty(); }
      // carry state forward
      c.parent = turn.id; c.seed = String(turn.seed); c.locked = true; c.prompt = turn.prompt;
      renderComposer();
    } catch (e) { toast(e.message, true); }
  }
  function paramsOf(turn) {
    return { prompt: turn.prompt, model: turn.model, quality: turn.quality, duration_s: turn.duration_s, canvas: turn.canvas,
      refs: (turn.refs || []).map((r) => (r.from_turn ? { media_type: r.media_type, from_turn: r.from_turn } : { media_type: r.media_type, path: r.path, name: r.name })),
      fast: turn.fast, fast_spatial: turn.fast_spatial, vsa_impl: turn.vsa_impl || null, wan_vae: !!turn.wan_vae };
  }
  async function reroll(turn) {
    try { await api("POST", `/api/projects/${state.pid}/turns`, { ...paramsOf(turn), seed: null, parent_id: turn.id, kind: "reroll" }); }
    catch (e) { toast(e.message, true); }
  }
  async function finalize(turn) {
    try { await api("POST", `/api/projects/${state.pid}/turns`, { ...paramsOf(turn), quality: "final", seed: turn.seed, parent_id: turn.id, kind: "finalize" }); }
    catch (e) { toast(e.message, true); }
  }
  async function retry(turn) {
    try { await api("POST", `/api/projects/${state.pid}/turns`, { ...paramsOf(turn), seed: turn.seed, parent_id: turn.parent_id, kind: turn.kind }); }
    catch (e) { toast(e.message, true); }
  }
  function edit(turn) { loadIntoComposer(turn); renderComposer(); $("#prompt").focus(); }
  function useAsRef(turn, type) {
    const c = state.composer, models = state.models?.models || {};
    if (!models[c.model]?.supports_references) {
      const omni = Object.values(models).find((m) => m.supports_references);
      if (omni) { c.model = omni.id; toast(`Switched model to ${omni.name} for references`); }
    }
    const videos = c.refs.filter((r) => r.media_type === "video").length;
    if (type === "video" && videos >= 3) return toast("At most 3 video references", true);
    c.refs.push({ media_type: type, from_turn: turn.id, name: `turn ${turn.index}` });
    renderComposer();
  }
  async function cancel(turn) { try { await api("POST", `/api/projects/${state.pid}/turns/${turn.id}/cancel`); } catch (e) { toast(e.message, true); } }
  async function removeTurn(turn) {
    if (!confirm(`Delete turn ${turn.index} and its clip?`)) return;
    try { await api("DELETE", `/api/projects/${state.pid}/turns/${turn.id}`); } catch (e) { toast(e.message, true); }
  }

  // -- model picker --------------------------------------------------------------------
  function currentNeed() {
    const c = state.composer;
    if (c.refs.length) return "refs";
    const text = $("#prompt").value || c.prompt || "";
    if (/\(S\d\)|<d>/.test(text)) return "audio";
    return "any";
  }
  async function openPicker() {
    const c = state.composer;
    let data;
    try { data = await api("GET", `/api/picker?quality=${c.quality}&need=${currentNeed()}`); } catch (e) { return toast(e.message, true); }
    const m = data.machine;
    $("#picker-machine").textContent = `This Mac: ${m.chip.replace("Apple ", "")}, ${Math.round(m.memory_gib)} GB, ${Math.round(m.free_disk_gb)} GB free`;
    $("#picker-note").textContent = m.note + (m.runs ? ` · ${m.runs} run${m.runs === 1 ? "" : "s"} recorded` : "");
    const list = $("#picker-list"); list.innerHTML = "";
    for (const r of data.models) {
      const facts = [];
      const d = r.est_s.draft, f = r.est_s.final;
      facts.push(el("span", {}, "draft ", el("b", {}, d == null ? "?" : fmtApprox(d)), r.measured.draft ? " measured" : " est."));
      facts.push(el("span", {}, "final ", el("b", {}, f == null ? "?" : fmtApprox(f)), r.measured.final ? " measured" : " est."));
      facts.push(el("span", {}, "peak ", el("b", {}, `${Math.round(r.peak_gib)} GB`)));
      if (r.installed) facts.push(el("span", {}, "installed ", el("b", {}, `${r.size_gb} GB`)));
      else facts.push(el("span", {}, "download ", el("b", {}, `${r.download_gb} GB`), r.convert ? `, converts to ${r.disk_gb} GB (one-time, a few minutes)` : ""));
      const fitText = r.fit === "well" ? "Runs well" : r.fit === "slow" ? "Runs, slow" : "Won't fit";
      const fit = el("div", { class: "fit" }, el("span", { class: "dot " + (r.fit === "well" ? "" : r.fit === "slow" ? "half" : "no") }), fitText,
        el("small", {}, r.fit_reason || (r.installed ? "" : "not installed")));
      if (!r.installed && r.fit !== "no") fit.append(el("button", { title: "Download now, without generating. Selecting the model and pressing Generate downloads it too.", onclick: (e) => { e.stopPropagation(); api("POST", `/api/models/${r.id}/install`).then(() => toast(`Download of ${r.name} queued`)).catch((err) => toast(err.message, true)); } }, "Download now"));
      if (r.installed) fit.append(el("button", { class: "ghost", title: "Delete the weights from disk", onclick: (e) => { e.stopPropagation(); if (!confirm(`Remove ${r.name} (${r.size_gb} GB) from disk?`)) return; api("DELETE", `/api/models/${r.id}`).then(() => { toast(`${r.name} removed`); refreshModels(); openPicker(); }).catch((err) => toast(err.message, true)); } }, "Remove"));
      const tags = [];
      if (r.audio) tags.push(el("span", { class: "tag" }, "Audio + speech"));
      if (r.supports_references) tags.push(el("span", { class: "tag" }, "References"));
      if (r.id === data.recommended) tags.push(el("span", { class: "tag rec" }, data.recommended_tag));
      const row = el("button", { class: "prow" + (r.id === c.model ? " sel" : "") + (r.fit === "no" ? " nofit" : ""), onclick: () => { if (r.fit === "no") return; c.model = r.id; $("#picker").hidden = true; renderComposer(); } },
        el("div", {}, el("div", { class: "nm" }, r.name, ...tags), el("div", { class: "what" }, r.output), el("div", { class: "facts" }, ...facts)), fit);
      list.append(row);
    }
    $("#picker").hidden = false;
  }

  // -- viewer --------------------------------------------------------------------------
  let viewerTurn = null;
  function doneTurns() { return (state.project?.turns || []).filter((t) => t.status === "done"); }
  function openViewer(tid) {
    const turn = turnById(tid); if (!turn || turn.status !== "done") return;
    viewerTurn = tid;
    $$("video", $("#timeline")).forEach((v) => v.pause());
    const src = `/files/projects/${turn.project_id}/${turn.result.clip}`;
    const video = $("#viewer-video");
    if (video.getAttribute("src") !== src) { video.src = src; }
    const spec = state.models?.models?.[turn.model];
    $("#viewer-title").textContent = `Turn ${turn.index} · ${spec ? spec.short : turn.model} · seed ${turn.seed} · ${turn.quality === "final" ? "Final" : "Draft"} · ${turn.width}×${turn.height}`;
    $("#viewer-prompt").textContent = turn.prompt;
    const dl = $("#viewer-download"); dl.href = src; dl.download = `turn-${turn.index}-seed-${turn.seed}.mp4`;
    const list = doneTurns(); const i = list.findIndex((t) => t.id === tid);
    $("#viewer-prev").disabled = i <= 0; $("#viewer-next").disabled = i < 0 || i >= list.length - 1;
    $("#viewer").hidden = false;
    video.play().catch(() => {});
  }
  function closeViewer() { const v = $("#viewer-video"); v.pause(); $("#viewer").hidden = true; viewerTurn = null; }
  function viewerStep(delta) {
    const list = doneTurns(); const i = list.findIndex((t) => t.id === viewerTurn);
    const next = list[i + delta]; if (next) openViewer(next.id);
  }

  // -- header / queue -------------------------------------------------------------------
  function renderQueueMeta() {
    const q = state.queue || {};
    const parts = [];
    if (q.current) parts.push(q.current[0] === "__install__" ? `installing ${state.models?.models?.[q.current[1]]?.name || q.current[1]}` : "1 rendering");
    if (q.queued?.length) parts.push(`${q.queued.length} queued`);
    $("#queue-meta").textContent = parts.length ? "· " + parts.join(" · ") : "";
    const chip = $("#runner-chip");
    const a = state.active;
    if (q.current && q.current[0] !== "__install__") {
      const p = state.projects.find((x) => x.id === q.current[0]);
      const prog = a && a.turn_id === q.current[1] ? (a.steps ? ` · step ${a.step || 0}/${a.steps}` : ` · ${phaseLabel(a.phase)}`) + (a.eta_s != null ? ` · ${fmtEta(a.eta_s)}` : "") : "";
      chip.textContent = `rendering in ${p ? p.name : "another project"}${prog}`; chip.hidden = q.current[0] === state.pid;
      chip.onclick = () => openProject(q.current[0]);
      chip.style.cursor = "pointer";
    } else chip.hidden = true;
  }

  // -- settings ------------------------------------------------------------------------
  function renderSettingsForm() {
    const s = state.settings; if (!s) return;
    const f = $("#settings-form");
    const ms = $("#settings-model"); ms.innerHTML = "";
    for (const m of Object.values(state.models?.models || {})) ms.append(el("option", { value: m.id }, m.name));
    for (const [k, v] of Object.entries(s)) {
      const input = f.elements[k]; if (!input) continue;
      if (input.type === "checkbox") input.checked = !!v; else input.value = v == null ? "" : v;
    }
  }
  function renderLog() { const pre = $("#log"); if (!pre) return; pre.textContent = state.logLines.slice(-150).join("\n"); pre.scrollTop = pre.scrollHeight; }

  // -- ui wiring -----------------------------------------------------------------------
  function toggleSide(force) {
    const app = $("#app"); const collapsed = force ?? !app.classList.contains("collapsed");
    app.classList.toggle("collapsed", collapsed); $("#open-side").hidden = !collapsed;
    localStorage.setItem("fasth3.side", collapsed ? "collapsed" : "open");
  }
  const THEMES = ["system", "light", "dark"];
  function applyTheme(theme) {
    if (theme === "system") document.documentElement.removeAttribute("data-theme"); else document.documentElement.setAttribute("data-theme", theme);
    const btn = $("#theme-toggle"); if (btn) { btn.textContent = theme === "dark" ? "☾" : theme === "light" ? "☼" : "◐"; btn.title = `Theme: ${theme}`; }
    localStorage.setItem("fasth3.theme", theme);
  }
  function bindUI() {
    applyTheme(localStorage.getItem("fasth3.theme") || "system");
    $("#theme-toggle").onclick = () => { const cur = localStorage.getItem("fasth3.theme") || "system"; applyTheme(THEMES[(THEMES.indexOf(cur) + 1) % THEMES.length]); };
    $("#collapse-side").onclick = () => toggleSide(true);
    $("#open-side").onclick = () => toggleSide(false);
    $("#new-project").onclick = async () => {
      const n = state.projects.length + 1;
      const p = await api("POST", "/api/projects", { name: `Project ${n}` });
      state.composer = { ...state.composer, prompt: "", refs: [], parent: null, seed: "", locked: false };
      await refreshProjects(); await openProject(p.id);
      const title = $("#project-title"); title.focus();
      const range = document.createRange(); range.selectNodeContents(title);
      const sel = window.getSelection(); sel.removeAllRanges(); sel.addRange(range);
    };
    $("#project-title").addEventListener("blur", async (e) => {
      const name = e.target.textContent.trim(); if (!state.pid || !name) return renderProjects();
      await api("PATCH", `/api/projects/${state.pid}`, { name }); refreshProjects();
    });
    $("#project-title").addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); e.target.blur(); } });
    $("#prompt").addEventListener("input", (e) => (state.composer.prompt = e.target.value));
    $("#prompt").addEventListener("keydown", (e) => { if ((e.metaKey || e.ctrlKey) && e.key === "Enter") generate(); });
    $("#generate").onclick = () => generate();
    $("#insert-template").onclick = () => {
      const p = $("#prompt"); if (p.value.trim() && !confirm("Replace the prompt with the template?")) return;
      const silent = state.models?.models?.[state.composer.model]?.audio === false;
      const t = silent ? TEMPLATE_SILENT : TEMPLATE;
      p.value = t; state.composer.prompt = t; p.focus();
    };
    $("#clear-composer").onclick = () => { state.composer = { ...state.composer, prompt: "", refs: [], parent: null, seed: "", locked: false }; renderComposer(); };
    $("#model-chip").onclick = () => openPicker();
    $("#duration-select").onchange = (e) => { state.composer.duration = Number(e.target.value); renderGenerateButton(); };
    $("#quality-select").onchange = (e) => { state.composer.quality = e.target.value; renderGenerateButton(); };
    $("#seed").addEventListener("input", (e) => { state.composer.seed = e.target.value.replace(/\D/g, ""); e.target.value = state.composer.seed; if (state.composer.seed) { state.composer.locked = true; $(".chip.seed").classList.add("locked"); } });
    $("#seed-lock").onclick = () => { const c = state.composer; c.locked = !c.locked; if (!c.locked) c.seed = ""; renderComposer(); };
    $("#toggle-advanced").onclick = () => { const a = $("#advanced"); a.hidden = !a.hidden; };
    $("#canvas-select").onchange = (e) => { state.composer.canvas = e.target.value; renderGenerateButton(); };
    $("#draft-mode").onchange = (e) => (state.composer.draftMode = e.target.value);
    $("#vsa-impl").onchange = (e) => (state.composer.vsaImpl = e.target.value);
    $("#wan-vae").onchange = (e) => (state.composer.wanVae = e.target.value === "1");
    $("#ref-file").addEventListener("change", async (e) => {
      for (const file of e.target.files) {
        try { const r = await upload(state.pid, file); state.composer.refs.push({ media_type: r.media_type, path: r.path, name: r.name }); }
        catch (err) { toast(err.message, true); }
      }
      e.target.value = ""; renderComposer();
    });

    $("#open-settings").onclick = () => { renderSettingsForm(); renderLog(); $("#settings-modal").hidden = false; };
    $$("[data-close]").forEach((b) => (b.onclick = () => (b.closest(".modal").hidden = true)));
    $$(".modal").forEach((m) => m.addEventListener("click", (e) => { if (e.target === m) m.hidden = true; }));
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") { closeViewer(); $$(".modal").forEach((m) => (m.hidden = true)); }
      if (!$("#viewer").hidden) {
        if (e.key === "ArrowLeft") viewerStep(-1);
        if (e.key === "ArrowRight") viewerStep(1);
        if (e.key === " " && e.target === document.body) { e.preventDefault(); const v = $("#viewer-video"); v.paused ? v.play() : v.pause(); }
      }
    });
    $("#viewer-prev").onclick = () => viewerStep(-1);
    $("#viewer-next").onclick = () => viewerStep(1);
    $("#viewer").addEventListener("click", (e) => { if (e.target === $("#viewer")) closeViewer(); });
    $("#viewer [data-close]").onclick = closeViewer;
    $("#settings-form").addEventListener("submit", async (e) => {
      e.preventDefault();
      const f = e.target, body = {};
      for (const k of Object.keys(state.settings)) {
        const input = f.elements[k]; if (!input) continue;
        if (input.type === "checkbox") body[k] = input.checked;
        else if (input.type === "number") body[k] = input.value === "" ? 0 : Number(input.value);
        else body[k] = input.value === "" ? (k === "vsa_impl" ? null : "auto") : input.value;
      }
      if (body.metal_wired_limit_gib && body.metal_wired_limit_gib !== "auto" && body.metal_wired_limit_gib !== "off") body.metal_wired_limit_gib = Number(body.metal_wired_limit_gib) || "auto";
      try { state.settings = await api("PUT", "/api/settings", body); toast("Settings saved"); $("#settings-modal").hidden = true; }
      catch (err) { toast(err.message, true); }
    });
  }

  init().catch((e) => { console.error(e); toast("Could not load: " + e.message, true); });
})();
