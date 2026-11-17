/* MaDoc MLOps: the interface (Alpine.js component, no build step). One page, a view per part of the platform:
   the pipeline, the data, the page editor, training, evaluation, experiments, jobs and monitoring.
   The server keeps everything (every long task is a job of the queue): this file shows it and sends the changes. */
(function () {
  const POLL = 5000;
  const STAGE = { w: 168, h: 78, col: 196, left: 16 };

  async function api(method, path, body) {
    const options = { method, headers: {} };
    if (body instanceof FormData) options.body = body;
    else if (body !== undefined) { options.body = JSON.stringify(body); options.headers["Content-Type"] = "application/json"; }
    let answer;
    try { answer = await fetch(path, options); } catch (_) { throw new Error("The server does not answer (network)."); }
    if (!answer.ok) {
      let detail = answer.statusText;
      try { detail = (await answer.json()).detail || detail; } catch (_) { /* not JSON */ }
      const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
      error.status = answer.status;
      throw error;
    }
    return answer.json();
  }
  const esc = (text) => String(text ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const enc = encodeURIComponent;

  // where each stage of the pipeline is drawn: [column, top]
  const PLACES = {
    export: [0, 16], ingest: [1, 16], normalize: [2, 16], edit: [3, 16], split: [4, 16],
    version: [0, 200], augment: [1, 330], experiments: [1, 440],
    train_blla: [2, 150], train_htr: [2, 250], evaluate_blla: [3, 150], evaluate_htr: [3, 250],
    release: [4, 200], madoc: [5, 200],
  };

  window.mlops = function () {
    let ed = null;            // the LineEditor: out of Alpine's reactive state (it holds DOM nodes)
    let original = "";        // the lines of the page as saved, to know what changed
    let timer = null, lastHash = "", skipRoute = false, known = {};
    return {
      // ---- state ----
      view: "pipeline", sub: "overview", meta: null, theme: "system", systemDark: false, toasts: [], lightbox: null,
      nav: [
        { id: "pipeline", label: "Pipeline", icon: "flow" }, { id: "data", label: "Data", icon: "database" },
        { id: "editor", label: "Editor", icon: "edit" }, { id: "training", label: "Training", icon: "play" },
        { id: "evaluation", label: "Evaluation", icon: "target" }, { id: "experiments", label: "Experiments", icon: "flask" },
        { id: "jobs", label: "Jobs", icon: "list" }, { id: "monitor", label: "Monitoring", icon: "activity" },
      ],
      dataSections: [
        { id: "overview", label: "Overview", icon: "database" }, { id: "vocabulary", label: "Vocabulary", icon: "type" },
        { id: "writers", label: "Writers", icon: "users" }, { id: "split", label: "Split", icon: "split" },
        { id: "normalization", label: "Normalisation", icon: "wand" }, { id: "augmentation", label: "Augmentation", icon: "layers" },
        { id: "missing", label: "Missing pages", icon: "search" }, { id: "preview", label: "Preview", icon: "eye" },
      ],
      active: [],
      // pipeline, data
      graph: null, data: null, zip: null, dropping: false, heldOut: false, uploading: false,
      vocab: null, vocabScope: "all", writers: null, alias: { variant: "", canonical: "" },
      splitForm: null, splitResult: null, rules: [], rule: { source: "", replacement: "" }, normPreview: "",
      aug: null, augForm: { copies: 2, seed: 42 }, missingRows: [], previewInfo: null, busyAction: null,
      // editor
      pageList: [], pageNote: "", ef: { split: "train", type: "all", search: "", names: false, edited: false },
      page: null, editLines: [], selected: null, mode: "select", canUndo: false, dirty: false, saving: false, sideTab: "lines",
      // training, evaluation, experiments
      trainForm: null, models: [], modelFigures: null,
      evalPage: null, evaluation: null, evalForm: { model: "", seg: "production", rec: "production" }, override: false,
      experimentsList: [], experimentDetail: null,
      xf: { recognizer: "kraken", scheme: "writers", seeds: "", options: "", segmentation: false },
      // jobs, monitoring
      jobsData: null, jf: { status: "", kind: "" }, jobDetail: null, mon: null, minutes: 15,

      // ---- start, routes ----
      async init() {
        let theme = null;
        try { theme = localStorage.getItem("mlops-theme"); } catch (_) { /* storage blocked */ }
        this.theme = ["light", "dark"].includes(theme) ? theme : "system";
        const dark = window.matchMedia ? window.matchMedia("(prefers-color-scheme: dark)") : null;
        if (dark) {
          this.systemDark = dark.matches;
          dark.addEventListener("change", (e) => { this.systemDark = e.matches; this.$nextTick(() => ed && ed.render()); });
        }
        ed = new window.LineEditor(this.$refs.canvas, {
          onChange: (lines) => this.linesChanged(lines),
          onSelect: (id) => { this.selected = id; this.scrollToLine(id); },
          onHover: () => {},
          onState: () => { this.canUndo = ed.canUndo(); this.selected = ed.selected; },
          onCrop: (box) => this.crop(box),
        });
        window.addEventListener("hashchange", () => this.route());
        window.addEventListener("resize", () => ed.render());
        window.addEventListener("keydown", (e) => this.key(e));
        window.addEventListener("beforeunload", (e) => { if (this.dirty) { e.preventDefault(); e.returnValue = ""; } });
        document.addEventListener("visibilitychange", () => { if (!document.hidden) this.tick(); });
        try { this.meta = await api("GET", "/api/meta"); } catch (e) { this.fail(e); }
        this.route();
        timer = setInterval(() => { if (!document.hidden) this.tick(); }, POLL);
      },
      go(route) { location.hash = "#/" + route; },
      route() {
        if (skipRoute) { skipRoute = false; return; }
        const hash = location.hash || "#/pipeline";
        const parts = hash.replace(/^#\/?/, "").split("/").map(decodeURIComponent);
        const view = this.nav.some((n) => n.id === parts[0]) ? parts[0] : "pipeline";
        // leaving the editor (or another page of it) with changes not saved
        if (this.dirty && this.view === "editor" && (view !== "editor" || (parts[1] && this.page && parts[1] !== (this.page.key || this.page.id)))) {
          if (!confirm("The changes of this page are not saved. Leave them?")) { skipRoute = true; location.hash = lastHash; return; }
          this.dirty = false;
        }
        lastHash = hash;
        this.view = view;
        const id = parts[1];
        if (view === "pipeline") { this.loadGraph(); this.loadData(); }
        if (view === "data") { this.sub = this.dataSections.some((s) => s.id === id) ? id : "overview"; this.loadSection(); }
        if (view === "editor") { if (!this.pageList.length) this.loadPages(); if (id && (!this.page || id !== (this.page.key || this.page.id))) this.openPage(id); }
        if (view === "training") this.loadTraining();
        if (view === "evaluation") { this.loadEvaluations(); if (id) this.loadEvaluation(id); else this.evaluation = null; }
        if (view === "experiments") { this.loadExperiments(); if (id) this.loadExperiment(id); else this.experimentDetail = null; }
        if (view === "jobs") { this.loadJobs(); if (id) this.loadJob(id); else this.jobDetail = null; }
        if (view === "monitor") this.loadMonitor();
        if (view === "editor") this.$nextTick(() => ed.render());
        this.loadActive();
      },
      tick() {
        // what this view shows that changes by itself; the jobs badge everywhere
        if (this.view === "pipeline") this.loadGraph();
        if (this.view === "jobs") { this.loadJobs(); if (this.jobDetail && ["queued", "running"].includes(this.jobDetail.status)) this.loadJob(this.jobDetail.id, true); }
        if (this.view === "monitor") this.loadMonitor();
        this.loadActive();
      },
      async loadActive() {
        // the jobs queued or running; a job that ends tells it and refreshes the view it changes
        let jobs;
        try { jobs = (await api("GET", "/api/jobs?limit=40")).jobs; } catch (_) { return; }
        this.active = jobs.filter((j) => j.status === "queued" || j.status === "running");
        let ended = false;
        for (const j of jobs) {
          const before = known[j.id];
          if ((before === "queued" || before === "running") && !["queued", "running"].includes(j.status)) {
            ended = true;
            this.toast(`Job #${j.id} ${j.kind}${j.task ? " " + j.task : ""}: ${j.status}.`, j.status === "done" ? "ok" : (j.status === "failed" ? "bad" : ""));
          }
          known[j.id] = j.status;
        }
        if (ended) this.refreshView();
      },
      refreshView() {
        if (this.view === "data") this.loadSection();
        if (this.view === "training") this.loadTraining();
        if (this.view === "evaluation") { this.loadEvaluations(); if (this.evaluation) this.loadEvaluation(this.evaluation.id); }
        if (this.view === "experiments") this.loadExperiments();
        if (this.view === "editor" && !this.dirty) this.loadPages();
        if (this.view === "pipeline") this.loadData();
      },
      activeCount() { return this.active.length; },
      lock() {
        // a job reading the pages: an edit is refused until it ends (mlops/editing.py), so Save waits too
        const kinds = this.meta ? this.meta.reading_jobs : [];
        const job = this.active.find((j) => j.status === "running" && kinds.includes(j.kind));
        return job ? `Job #${job.id} (${job.kind}${job.task ? " " + job.task : ""}) is reading the pages: the changes can be saved once it ends.` : "";
      },
      key(e) {
        const field = e.target && ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName);
        if (this.view !== "editor" || !this.page || this.page.readonly) return;
        if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "s") { e.preventDefault(); if (this.dirty) this.saveLines(); return; }
        if (!field && !this.saving) ed.key(e);
      },

      // ---- theme, messages, formats ----
      setTheme(theme) {
        this.theme = theme;
        if (theme === "system") document.documentElement.removeAttribute("data-theme");
        else document.documentElement.setAttribute("data-theme", theme);
        try { if (theme === "system") localStorage.removeItem("mlops-theme"); else localStorage.setItem("mlops-theme", theme); }
        catch (_) { /* storage blocked: the choice lasts until the page is closed */ }
        this.$nextTick(() => ed && ed.render());
      },
      isDark() { return this.theme === "dark" || (this.theme === "system" && this.systemDark); },
      toast(text, kind) {
        const id = Date.now() + Math.random();
        this.toasts.push({ id, text, kind: kind || "" });
        setTimeout(() => { this.toasts = this.toasts.filter((x) => x.id !== id); }, kind === "bad" ? 9000 : 4000);
      },
      fail(error) { this.toast(error.message, "bad"); },
      async post(path, body, message) {
        try { const answer = await api("POST", path, body); this.toast(message || answer.message || "Done.", "ok"); this.loadActive(); return answer; }
        catch (e) { this.fail(e); return null; }
      },
      async copyText(text) {
        try { await navigator.clipboard.writeText(text); this.toast("Copied.", "ok"); } catch (_) { this.toast("The browser refused to copy.", "bad"); }
      },
      fmt(n) { return n === null || n === undefined ? "" : Number(n).toLocaleString("en"); },
      pct(v, digits) { return v === null || v === undefined ? "" : (100 * v).toFixed(digits ?? 1) + " %"; },
      num(v, digits) { return v === null || v === undefined ? "—" : (typeof v === "number" ? v.toFixed(digits ?? 2) : v); },
      signed(v) { return v === null || v === undefined ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(2); },
      ci(pair, sign) { return pair ? "[" + pair.map((v) => (sign ? this.signed(v) : this.num(v))).join(", ") + "]" : ""; },
      date(iso) { return iso ? new Date(iso).toLocaleString("en-GB", { dateStyle: "medium", timeStyle: "short" }) : ""; },
      since(iso) { return this.duration((Date.now() - new Date(iso).getTime()) / 1000); },
      duration(seconds) {
        const s = Math.max(0, Math.round(seconds)), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60);
        return h ? `${h} h ${m} min` : (m ? `${m} min ${s % 60} s` : `${s} s`);
      },
      lozenge(status) { return { running: "progress", queued: "", done: "success", failed: "removed", cancelled: "moved" }[status] || ""; },
      metricsText(m) { return Object.entries(m || {}).map(([k, v]) => k.replace("val_", "") + " " + (v < 1.0001 ? (100 * v).toFixed(1) + " %" : v.toFixed(2))).join(" · ") || "—"; },
      figuresHtml(owner, kind) {
        if (!owner || !owner.figures || !owner.figures.length) return "";
        const base = `/api/figures/${kind}/${owner.id}/`;
        return '<div class="figures">' + owner.figures.map((f) => `<figure class="figure"><img src="${base}${enc(f.name)}" alt="${esc(f.caption)}" loading="lazy" data-zoom="1">` +
          `<figcaption><span>${esc(f.caption)}</span>${f.pdf ? `<a href="${base}${enc(f.name.replace(/\.png$/, ".pdf"))}" target="_blank" rel="noopener">PDF</a>` : ""}</figcaption></figure>`).join("") + "</div>";
      },
      zoomFigure(e) { if (e.target && e.target.dataset && e.target.dataset.zoom) this.lightbox = e.target.src; },

      // ---- pipeline ----
      async loadGraph() { try { this.graph = await api("GET", "/api/pipeline"); } catch (e) { if (!this.graph) this.fail(e); } },
      graphSvg() {
        if (!this.graph) return "";
        const nodes = Object.fromEntries(this.graph.nodes.map((n) => [n.id, n]));
        const box = (id) => { const [c, top] = PLACES[id] || [0, 0]; return { x: STAGE.left + c * STAGE.col, y: top }; };
        const live = (id) => nodes[id] && ["running", "queued"].includes(nodes[id].state);
        let edges = "";
        for (const e of this.graph.edges) {
          if (!nodes[e.from] || !nodes[e.to]) continue;
          const a = box(e.from), b = box(e.to);
          let d;
          if (e.from === "split" && e.to === "version") {        // from the end of the first row to the start of the second
            const x1 = a.x + STAGE.w / 2, x2 = b.x + STAGE.w / 2, mid = a.y + STAGE.h + 28;
            d = `M${x1} ${a.y + STAGE.h} V${mid} H${x2} V${b.y - 2}`;
          } else {
            const x1 = a.x + STAGE.w, y1 = a.y + STAGE.h / 2, x2 = b.x - 2, y2 = b.y + STAGE.h / 2;
            const turn = e.from === "version" ? a.x + STAGE.w + 14 : b.x - 14;
            d = y1 === y2 ? `M${x1} ${y1} H${x2}` : `M${x1} ${y1} H${turn} V${y2} H${x2}`;
          }
          const cls = "edge" + (e.from === "augment" ? " optional" : "") + (live(e.to) ? " live" : "");
          edges += `<path class="${cls}" d="${d}" marker-end="url(#${live(e.to) ? "arr-live" : "arr"})" stroke-linejoin="round"/>`;
        }
        let shapes = "";
        for (const n of this.graph.nodes) {
          const { x, y } = box(n.id);
          const state = n.state === "idle" ? "" : n.state;
          const cut = (t) => (String(t).length > 26 ? String(t).slice(0, 25) + "…" : t);
          shapes += `<g class="node ${esc(n.state)}${n.route ? "" : " static"}${n.id === "madoc" ? " brand" : ""}" ${n.route ? `data-route="${esc(n.route)}" tabindex="0" role="link"` : ""} transform="translate(${x} ${y})">` +
            `<title>${esc(n.title)}${n.route ? ": open" : ""}</title>` +
            `<rect class="box" width="${STAGE.w}" height="${STAGE.h}" rx="6"/><rect class="bar" width="4" height="${STAGE.h - 12}" y="6" rx="2"/>` +
            `<text class="title" x="14" y="24">${esc(cut(n.title))}</text>` +
            `<text class="line" x="14" y="46">${esc(cut(n.lines[0] || ""))}</text><text class="line" x="14" y="64">${esc(cut(n.lines[1] || ""))}</text>` +
            (state ? `<text class="state" x="${STAGE.w - 10}" y="18" text-anchor="end">${esc(state)}</text>` : "") + "</g>";
        }
        const marker = (id, cls) => `<marker id="${id}" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path class="${cls}" d="M0 0L8 4L0 8z"/></marker>`;
        const width = STAGE.left * 2 + 5 * STAGE.col + STAGE.w, height = 440 + STAGE.h + 16;
        return `<svg viewBox="0 0 ${width} ${height}" role="img" aria-label="The stages of the platform"><defs>${marker("arr", "arrow")}${marker("arr-live", "arrow live")}</defs>${edges}${shapes}</svg>`;
      },
      graphClick(e) {
        const node = e.target.closest && e.target.closest("[data-route]");
        if (node) this.go(node.getAttribute("data-route"));
      },

      // ---- data ----
      async loadData() { try { this.data = await api("GET", "/api/data"); } catch (e) { this.fail(e); } },
      loadSection() {
        const s = this.sub;
        if (s === "overview") this.loadData();
        if (s === "vocabulary") this.loadVocab();
        if (s === "writers") this.loadWriters();
        if (s === "split") this.loadSplit();
        if (s === "normalization") api("GET", "/api/data/normalization").then((a) => { this.rules = a.rules; }, (e) => this.fail(e));
        if (s === "augmentation") api("GET", "/api/data/augmentation").then((a) => { this.aug = a; if (!this.augLoaded) { this.augForm = { ...a.defaults }; this.augLoaded = true; } }, (e) => this.fail(e));
        if (s === "missing") api("GET", "/api/data/missing").then((a) => { this.missingRows = a.rows; }, (e) => this.fail(e));
        if (s === "preview") api("GET", "/api/data/preview").then((a) => { this.previewInfo = a; }, (e) => this.fail(e));
        if (!this.data) this.loadData();
      },
      pickZip(file) {
        if (!file) return;
        if (!file.name.toLowerCase().endsWith(".zip")) return this.toast("A .zip file is expected (the eScriptorium export).", "bad");
        const limit = (this.data ? this.data.max_upload_mb : 2048) * 1048576;
        if (file.size > limit) return this.toast(`The zip is larger than ${this.data.max_upload_mb} MB.`, "bad");
        this.zip = file;
      },
      async uploadZip() {
        if (!this.zip || this.uploading) return;
        const form = new FormData();
        form.append("file", this.zip); form.append("held_out", this.heldOut ? "true" : "false");
        this.uploading = true;
        try { const answer = await api("POST", "/api/data/upload", form); this.toast(answer.message, "ok"); this.zip = null; this.heldOut = false; this.loadData(); this.loadActive(); }
        catch (e) { this.fail(e); }
        this.uploading = false;
      },
      async batchAction(batch, action) {
        let body = {};
        if (action === "delete" && !confirm(`Take the ${batch.pages} pages of batch ${batch.id} out of the dataset? (Restore brings them back.)`)) return;
        if (action === "purge") {
          const typed = prompt(`Remove batch ${batch.id} for good (files and revisions)? Type PURGE ${batch.id} to confirm.`);
          if (typed === null) return;
          body = { confirmation: typed };
        }
        if (await this.post(`/api/data/batches/${batch.id}/${action}`, body)) this.loadData();
      },
      async loadVocab() {
        this.vocab = null;
        try { this.vocab = await api("GET", "/api/data/vocabulary"); } catch (e) { this.fail(e); }
      },
      vocabShown() {
        if (!this.vocab) return [];
        return this.vocab.chars.filter((c) => !c.invisible).map((c) => ({ ...c, n: this.vocabScope === "all" ? c.total : c.counts[this.vocabScope] }))
          .filter((c) => c.n > 0).sort((a, b) => b.n - a.n);
      },
      vocabInvisible() { return this.vocab ? this.vocab.chars.filter((c) => c.invisible).map((c) => ({ ...c, n: c.total })) : []; },
      async loadWriters() { try { this.writers = await api("GET", "/api/data/writers"); } catch (e) { this.fail(e); } },
      async pinWriter(w, split) {
        try { this.writers = await api("POST", "/api/data/writers/pin", { writer: w.writer, split }); this.toast(`${w.label}: ${split === "auto" ? "split chosen at the next split" : "always in " + split}.`, "ok"); }
        catch (e) { this.fail(e); this.loadWriters(); }
      },
      async addAlias() {
        try { this.writers = await api("POST", "/api/data/writers/alias", this.alias); this.alias = { variant: "", canonical: "" }; this.toast("Merged: split again to apply it.", "ok"); }
        catch (e) { this.fail(e); }
      },
      async removeAlias(variant) {
        try { this.writers = await api("DELETE", "/api/data/writers/alias", { variant }); } catch (e) { this.fail(e); }
      },
      async loadSplit() { try { this.splitForm = await api("GET", "/api/data/split"); } catch (e) { this.fail(e); } },
      async splitAction(action) {
        if (action === "apply" && !confirm("Split the dataset again with these settings? Trainings already made keep their own dataset version.")) return;
        this.busyAction = action;
        try {
          const answer = await api("POST", "/api/data/split/" + action, this.splitForm);
          if (answer.settings) this.splitForm = answer.settings;
          if (answer.result) this.splitResult = { ...answer.result, applied: action === "apply" };
          if (answer.message) this.toast(answer.message, "ok");
          if (action === "apply") this.loadData();
        } catch (e) { this.fail(e); }
        this.busyAction = null;
      },
      async ruleAction(action, source, replacement) {
        try {
          this.rules = (await api("POST", "/api/data/normalization/" + action, { source, replacement: replacement || "" })).rules;
          if (action === "add") this.rule = { source: "", replacement: "" };
          this.normPreview = "";
        } catch (e) { this.fail(e); }
      },
      async normalizePreview() {
        this.busyAction = "normalize";
        try { this.normPreview = (await api("POST", "/api/data/normalization/preview", {})).text; } catch (e) { this.fail(e); }
        this.busyAction = null;
      },

      // ---- editor ----
      async loadPages() {
        const q = new URLSearchParams({ split: this.ef.split, type: this.ef.type, names: this.ef.names, search: this.ef.search, edited: this.ef.edited });
        try { const answer = await api("GET", "/api/pages?" + q); this.pageList = answer.pages; this.pageNote = answer.note; } catch (e) { this.fail(e); }
      },
      async openPage(key, keepView) {
        if (!keepView && this.dirty && this.page && key !== (this.page.key || this.page.id) && !confirm("The changes of this page are not saved. Leave them?")) return;
        try { this.applyPage(await api("GET", `/api/pages/${enc(key)}?names=${this.ef.names}`)); }
        catch (e) { this.fail(e); return; }
        const hash = "#/editor/" + enc(key);
        if (location.hash !== hash) { history.replaceState(null, "", hash); lastHash = hash; }
      },
      applyPage(page) {
        this.page = page;
        this.editLines = page.lines.map((l) => this.asLine(l));
        original = this.snapshot(this.editLines);
        this.dirty = false; this.selected = null;
        if (page.readonly) this.sideTab = "lines";
        ed.setImage(`/api/pages/${enc(page.key || page.id)}/image?r=${page.revision}`, page.width, page.height);
        this.setMode(page.readonly ? "view" : (["view", "crop"].includes(this.mode) ? "select" : this.mode));
        ed.setLines(this.editLines);
        const row = this.pageList.find((p) => p.key === (page.key || page.id));
        if (row && !page.readonly) { row.revision = page.revision; row.lines = page.lines.length; }
      },
      asLine(l) {
        // ground truth: a line without text is shown red, a line with text green, a changed line amber
        return { id: l.id, polygon: l.polygon, baseline: l.baseline, text: l.text || "", confidence: (l.text || "").trim() ? 1 : 0, dirty: !!l.dirty };
      },
      snapshot(lines) { return JSON.stringify(lines.map((l) => [l.id, l.polygon, l.baseline, l.text])); },
      linesChanged(lines) {
        const before = Object.fromEntries(JSON.parse(original).map((l) => [l[0], JSON.stringify(l)]));
        this.editLines = lines.map((l) => ({ ...l, confidence: (l.text || "").trim() ? 1 : 0,
          dirty: before[l.id] !== JSON.stringify([l.id, l.polygon, l.baseline, l.text]) }));
        this.dirty = this.snapshot(this.editLines) !== original;
        ed.setLines(this.editLines, true);
      },
      setText(id, text) {
        const line = this.editLines.find((l) => l.id === id);
        if (!line) return;
        line.text = text;
        const target = ed.line(id);
        if (target) { target.text = text; target.confidence = text.trim() ? 1 : 0; target.dirty = true; ed.paint(); }
        line.confidence = text.trim() ? 1 : 0; line.dirty = true;
        this.dirty = this.snapshot(this.editLines) !== original;
      },
      setMode(mode) { this.mode = mode; ed.setMode(mode); },
      select(id, fromTable) { this.selected = id; ed.setSelected(id, fromTable); },
      hover(id) { ed.setHover(id); },
      scrollToLine(id) {
        if (!id) return;
        this.sideTab = "lines";
        this.$nextTick(() => { const field = document.getElementById("line-" + id); if (field) field.scrollIntoView({ block: "nearest" }); });
      },
      boxLine() { ed.boxSelected(); },
      deleteLine() { ed.deleteSelected(); },
      undoEdit() { ed.undo(); },
      zoom(step) { ed.zoom(step); },
      discard() { if (confirm("Discard the changes of this page?")) this.applyPage(this.page); },
      async saveLines() {
        if (!this.page || this.page.readonly || this.saving || this.lock()) return;
        this.saving = true;
        const lines = this.editLines.map((l) => ({ id: l.id, polygon: l.polygon, baseline: l.baseline, text: l.text }));
        try {
          const answer = await api("PUT", `/api/pages/${enc(this.page.id)}/lines`, { lines, revision: this.page.revision, names: this.ef.names });
          this.applyPage(answer.page);
          this.toast(answer.result.unchanged ? "Nothing changed." : `Saved as revision ${answer.result.revision}.`, "ok");
        } catch (e) { this.conflict(e); }
        this.saving = false;
      },
      async transform(op) {
        if (!this.page || this.dirty || this.saving || this.lock()) return;
        this.saving = true;
        try {
          const answer = await api("POST", `/api/pages/${enc(this.page.id)}/transform`, { op, revision: this.page.revision, names: this.ef.names });
          this.applyPage(answer.page);
          this.toast(`Saved as revision ${answer.result.revision} (${answer.result.lines} lines).`, "ok");
        } catch (e) { this.conflict(e); }
        this.saving = false;
      },
      crop(box) {
        this.setMode("select");
        if (confirm(`Crop the page to ${box[2] - box[0]} × ${box[3] - box[1]} px? Lines outside are removed, the others cut; it is saved at once (History restores it).`)) this.transform({ op: "crop", box });
      },
      async restoreRevision(revision) {
        if (!confirm(`Bring back revision ${revision}? It is saved as a new revision: nothing is lost.`)) return;
        this.saving = true;
        try {
          const answer = await api("POST", `/api/pages/${enc(this.page.id)}/restore`, { restore: revision, revision: this.page.revision, names: this.ef.names });
          this.applyPage(answer.page);
          this.toast(`Revision ${revision} restored as revision ${answer.result.revision}.`, "ok");
        } catch (e) { this.conflict(e); }
        this.saving = false;
      },
      conflict(e) {
        this.fail(e);
        if (e.status === 409 && /revision/i.test(e.message) && confirm("The page changed meanwhile. Load its last revision (your changes are lost)?")) this.openPage(this.page.id, true);
      },

      // ---- training ----
      async loadTraining() {
        try { const answer = await api("GET", "/api/training"); if (!this.trainForm) this.trainForm = answer.form; this.models = answer.models; }
        catch (e) { this.fail(e); }
      },
      async train(task) {
        const body = Object.fromEntries(this.trainForm[task].fields.map((f) => [f.key, f.value]));
        await this.post("/api/training/" + task, body);
      },
      async resetTraining(task) {
        try { const answer = await api("DELETE", "/api/training/" + task); this.trainForm[task] = answer.form; this.toast(answer.message, "ok"); }
        catch (e) { this.fail(e); }
      },
      async showModel(m) {
        try { this.modelFigures = { id: m.id, ...(await api("GET", "/api/models/" + m.id)) }; if (!this.modelFigures.figures.length) this.toast("No figure for this model yet.", ""); }
        catch (e) { this.fail(e); }
      },

      // ---- evaluation ----
      async loadEvaluations() { try { this.evalPage = await api("GET", "/api/evaluation"); } catch (e) { this.fail(e); } },
      async loadEvaluation(id) { this.override = false; try { this.evaluation = await api("GET", "/api/evaluations/" + id); } catch (e) { this.fail(e); } },
      async release() {
        const e = this.evaluation;
        if (!confirm(`Open a pull request on MaDoc with model #${e.model_id}${e.passed ? "" : " although it is not better"}? A person still merges it.`)) return;
        await this.post(`/api/evaluations/${e.id}/release`, { override: this.override });
      },

      // ---- experiments ----
      async loadExperiments() { try { this.experimentsList = (await api("GET", "/api/experiments")).experiments; } catch (e) { this.fail(e); } },
      async loadExperiment(id) { try { this.experimentDetail = await api("GET", "/api/experiments/" + id); } catch (e) { this.fail(e); } },
      async runExperiment() { await this.post("/api/experiments", this.xf); },

      // ---- jobs ----
      async loadJobs() {
        const q = new URLSearchParams({ status: this.jf.status, kind: this.jf.kind, limit: 100 });
        try { this.jobsData = await api("GET", "/api/jobs?" + q); } catch (e) { if (!this.jobsData) this.fail(e); }
      },
      async loadJob(id, quiet) {
        try {
          const log = this.$refs.log, atEnd = log && log.scrollTop + log.clientHeight >= log.scrollHeight - 24;
          this.jobDetail = await api("GET", "/api/jobs/" + id);
          this.$nextTick(() => { const el = this.$refs.log; if (el && (atEnd || !quiet)) el.scrollTop = el.scrollHeight; });
        } catch (e) { if (!quiet) this.fail(e); }
      },
      async cancelJob(id) {
        if (!confirm(`Cancel job ${id}?`)) return;
        if (await this.post(`/api/jobs/${id}/cancel`, {})) { this.loadJobs(); this.loadJob(id, true); }
      },

      // ---- monitoring ----
      async loadMonitor() { try { this.mon = await api("GET", "/api/monitor?minutes=" + this.minutes); } catch (_) { /* the next tick tries again */ } },
      spark(points, value, max, color, unit) {
        // a time series: three grid lines with their values, the area and the line of the series
        const since = this.mon.now - this.minutes * 60, W = 260, H = 88, top = 4, bottom = H - 4, left = 30;
        const usable = points.map((p) => [p.at, value(p)]).filter((p) => p[1] !== null && p[1] !== undefined);
        const ceiling = max || Math.max(1, ...usable.map((p) => p[1])) * 1.2;
        const y = (v) => bottom - (Math.min(v, ceiling) / ceiling) * (bottom - top);
        const x = (at) => left + Math.max(0, ((at - since) / (this.minutes * 60)) * (W - left));
        const label = (v) => (unit === "%" ? Math.round(v) + "%" : unit === "GB" ? v.toFixed(1) : (v >= 10 ? Math.round(v) : v.toFixed(1)));
        let svg = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" role="img">`;
        for (const f of [0, 0.5, 1]) {
          const v = ceiling * f, yy = y(v).toFixed(1);
          svg += `<line class="grid-line" x1="${left}" y1="${yy}" x2="${W}" y2="${yy}" vector-effect="non-scaling-stroke"/>` +
            `<text x="${left - 4}" y="${(+yy + 3).toFixed(1)}" text-anchor="end">${label(v)}</text>`;
        }
        if (!usable.length) return svg + `<text class="no-data" x="${(left + W) / 2}" y="${H / 2}" text-anchor="middle">no data</text></svg>`;
        const xy = usable.map(([at, v]) => [x(at), y(v)]);
        const path = xy.map((p, i) => (i ? "L" : "M") + p[0].toFixed(1) + " " + p[1].toFixed(1)).join(" ");
        return svg + `<path d="${path} L${xy[xy.length - 1][0].toFixed(1)} ${bottom} L${xy[0][0].toFixed(1)} ${bottom} Z" style="fill:var(${color});fill-opacity:.14"/>` +
          `<path d="${path}" style="fill:none;stroke:var(${color});stroke-width:1.75" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>`;
      },
      tiles() {
        // memory, disk and GPU memory are in GiB in the samples
        const points = this.mon.samples, last = points[points.length - 1] || {};
        const gpu = (last.gpus && last.gpus[0]) || null;
        const g = (key) => (p) => (p.gpus && p.gpus[0] ? p.gpus[0][key] : null);
        const val = (v, unit) => (v === null || v === undefined ? "—" : v + (unit ? ` <small>${unit}</small>` : ""));
        const f1 = (v) => (v === null || v === undefined ? null : Number(v).toFixed(1));
        const tiles = [
          { key: "cpu", title: "CPU", value: val(last.cpu === null || last.cpu === undefined ? null : Math.round(last.cpu), "%"),
            chart: this.spark(points, (p) => p.cpu, 100, "--chart-1", "%"), note: `${this.mon.cpu_count} threads` },
          { key: "mem", title: "Memory", value: last.memory_used ? val(f1(last.memory_used), "/ " + f1(last.memory_total) + " GB") : "—",
            chart: this.spark(points, (p) => p.memory_used, last.memory_total, "--chart-2", "GB"),
            note: last.process_memory ? `this app: ${f1(last.process_memory)} GB` : "" },
          { key: "gpu", title: "GPU", value: val(gpu && gpu.util !== null ? Math.round(gpu.util) : null, "%"),
            chart: this.spark(points, g("util"), 100, "--chart-3", "%"), note: gpu ? gpu.name : "no GPU seen (NVML)" },
          { key: "gmem", title: "GPU memory", value: gpu && gpu.memory_used !== null ? val(f1(gpu.memory_used), "/ " + f1(gpu.memory_total) + " GB") : "—",
            chart: this.spark(points, g("memory_used"), gpu ? gpu.memory_total : null, "--chart-4", "GB"), note: last.running_job ? "job " + last.running_job : "no job running" },
          { key: "temp", title: "GPU temperature", value: val(gpu && gpu.temperature !== null ? Math.round(gpu.temperature) : null, "°C"),
            chart: this.spark(points, g("temperature"), 100, "--chart-5"), note: "" },
          { key: "queue", title: "Jobs queued", value: val(last.queued ?? null),
            chart: this.spark(points, (p) => p.queued, null, "--chart-6"),
            note: `${last.running ?? 0} running · ${last.workers_alive ?? 0} workers alive` },
          { key: "disk", title: "Data disk", value: last.disk_used ? val(Math.round(last.disk_used), "/ " + Math.round(last.disk_total) + " GB") : "—",
            chart: this.spark(points, (p) => p.disk_used, last.disk_total, "--chart-2", "GB"),
            note: last.disk_total ? `${Math.round(last.disk_total - last.disk_used)} GB free` : "" },
        ];
        return gpu ? tiles : tiles.filter((tile) => tile.key !== "gmem" && tile.key !== "temp");
      },
    };
  };
})();
