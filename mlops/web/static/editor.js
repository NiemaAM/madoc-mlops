/* The page with its lines, drawn in SVG: a viewer (hover, click) and an editor (select, move a line or its
   points, add / remove points, draw a new line, delete, undo; "crop": a box given to onCrop).
   The same file as MaDoc's (src/madoc_web/static/editor.js), with the crop mode.

   new LineEditor(host, {onChange(lines), onSelect(id), onHover(id), onState(), onCrop([left, top, right, bottom])})
   A line is {id, polygon: [[x, y]...], baseline: [[x, y]...], text, confidence, dirty}; coordinates are pixels of
   the scan. The editor works on its own copy: setLines() replaces it, onChange() gives it back after an edit. */
(function () {
  const NS = "http://www.w3.org/2000/svg";
  const el = (name, attrs) => {
    const node = document.createElementNS(NS, name);
    for (const key in attrs) node.setAttribute(key, attrs[key]);
    return node;
  };
  const clone = (lines) => lines.map((l) => ({ ...l, polygon: l.polygon.map((p) => [p[0], p[1]]), baseline: l.baseline.map((p) => [p[0], p[1]]) }));
  const meanY = (points) => points.reduce((sum, p) => sum + p[1], 0) / points.length;

  class LineEditor {
    constructor(host, callbacks) {
      this.host = host;
      this.cb = callbacks || {};
      this.lines = [];
      this.width = 1; this.height = 1;
      this.mode = "view";          // "view" | "select" | "draw" | "crop"
      this.selected = null; this.hovered = null; this.highlight = [];
      this.threshold = 0.8; this.zoomLevel = 1;
      this.undoStack = [];
      this.drag = null;
      this.svg = el("svg", {});
      this.image = el("image", { x: 0, y: 0 });
      this.layer = el("g", {});
      this.svg.append(this.image, this.layer);
      host.replaceChildren(this.svg);
      this.svg.addEventListener("pointerdown", (e) => this.down(e));
      this.svg.addEventListener("pointermove", (e) => this.move(e));
      this.svg.addEventListener("pointerup", (e) => this.up(e));
      this.svg.addEventListener("pointercancel", (e) => this.up(e));
      this.svg.addEventListener("dblclick", (e) => this.double(e));
      this.svg.addEventListener("pointerover", (e) => this.over(e, true));
      this.svg.addEventListener("pointerout", (e) => this.over(e, false));
    }

    // ---- what the page gives ----
    setImage(url, width, height) {
      if (this.url !== url) { this.url = url; this.image.setAttribute("href", url); }
      this.width = width || 1; this.height = height || 1;
      this.image.setAttribute("width", this.width); this.image.setAttribute("height", this.height);
      this.svg.setAttribute("viewBox", `0 0 ${this.width} ${this.height}`);
      this.applyZoom();
    }
    setLines(lines, keepUndo) {
      if (this.drag) return;       // not while the person drags: the server's answer comes after
      this.lines = clone(lines);
      if (!keepUndo) this.undoStack = [];
      if (this.selected && !this.lines.some((l) => l.id === this.selected)) this.selected = null;
      this.render(); this.state();
    }
    setMode(mode) {
      this.mode = mode;
      this.svg.classList.toggle("draw", mode === "draw" || mode === "crop");
      if (mode === "view") this.drag = null;
      this.render();
    }
    setSelected(id, scroll) {
      this.selected = id; this.render(); this.state();
      if (scroll && id) {
        const node = this.layer.querySelector(`[data-id="${id}"]`);
        if (node && node.scrollIntoView) node.scrollIntoView({ block: "nearest", inline: "nearest" });
      }
    }
    setHover(id) { this.hovered = id; this.paint(); }
    setHighlight(ids) { this.highlight = ids || []; this.paint(); }
    zoom(step) {
      const levels = [0.5, 0.75, 1, 1.5, 2, 3];
      const at = levels.reduce((best, v, i) => (Math.abs(v - this.zoomLevel) < Math.abs(levels[best] - this.zoomLevel) ? i : best), 0);
      this.zoomLevel = levels[Math.max(0, Math.min(levels.length - 1, at + step))];
      this.applyZoom(); this.render();
    }
    applyZoom() { this.svg.style.width = this.zoomLevel * 100 + "%"; }
    canUndo() { return this.undoStack.length > 0; }
    state() { if (this.cb.onState) this.cb.onState(); }

    // ---- drawing ----
    unit() {                        // one screen pixel in scan pixels
      const shown = this.svg.getBoundingClientRect().width || 1;
      return this.width / shown;
    }
    palette() {             // the colours of the theme (app.css: --poly-*), read at every drawing
      const css = getComputedStyle(this.host);
      const get = (name, fallback) => css.getPropertyValue(name).trim() || fallback;
      return { ok: get("--poly-ok", "#22a06b"), low: get("--poly-low", "#c9372c"), unread: get("--poly-unread", "#2b3db5"),
        dirty: get("--poly-dirty", "#d97008"), select: get("--poly-select", "#2b3db5"), label: get("--poly-label", "#44546f"),
        handle: get("--surface", "#ffffff") };
    }
    color(line) {
      const c = this.pal || this.palette();
      if (line.dirty) return c.dirty;
      if (line.confidence === null || line.confidence === undefined) return c.unread;
      return line.confidence < this.threshold ? c.low : c.ok;
    }
    render() {
      this.pal = this.palette();
      const c = this.pal;
      const u = this.unit();
      const nodes = [];
      this.lines.forEach((line, index) => {
        const points = line.polygon.map((p) => p.join(",")).join(" ");
        nodes.push(el("polygon", { points, "data-id": line.id, "class": "poly" }));
        nodes.push(el("polyline", { points: line.baseline.map((p) => p.join(",")).join(" "), "data-base": line.id, fill: "none", "pointer-events": "none" }));
        const right = line.polygon.reduce((best, p) => (p[0] > best[0] ? p : best), line.polygon[0]);
        const label = el("text", { x: Math.min(right[0] + 4 * u, this.width - 14 * u), y: meanY(line.polygon) + 4 * u, "font-size": 11 * u,
          "pointer-events": "none", "data-label": line.id, "font-family": "system-ui, sans-serif" });
        label.textContent = String(index + 1);
        nodes.push(label);
      });
      const selected = this.mode !== "view" && this.lines.find((l) => l.id === this.selected);
      if (selected) {
        selected.polygon.forEach((p, i) => nodes.push(el("circle", { cx: p[0], cy: p[1], r: 4.5 * u, "data-handle": "v", "data-index": i,
          fill: c.handle, stroke: c.select, "stroke-width": 1.5 * u, style: "cursor:move; touch-action:none" })));
        selected.baseline.forEach((p, i) => nodes.push(el("rect", { x: p[0] - 4 * u, y: p[1] - 4 * u, width: 8 * u, height: 8 * u, "data-handle": "b",
          "data-index": i, fill: c.dirty, stroke: c.handle, "stroke-width": 1.2 * u, style: "cursor:move; touch-action:none" })));
      }
      if (this.drag && (this.drag.kind === "draw" || this.drag.kind === "crop")) {
        const d = this.drag;
        nodes.push(el("rect", { x: Math.min(d.x0, d.x1), y: Math.min(d.y0, d.y1), width: Math.abs(d.x1 - d.x0), height: Math.abs(d.y1 - d.y0),
          fill: c.select, "fill-opacity": 0.18, stroke: c.select, "stroke-width": 1.5 * u, "stroke-dasharray": `${4 * u} ${3 * u}`, "pointer-events": "none" }));
      }
      this.layer.replaceChildren(...nodes);
      this.paint();
    }
    paint() {                       // colours only: cheap, done at every hover
      const c = this.pal || this.palette();
      const u = this.unit();
      for (const line of this.lines) {
        const node = this.layer.querySelector(`polygon[data-id="${line.id}"]`);
        if (!node) continue;
        const color = this.color(line);
        const strong = line.id === this.selected || line.id === this.hovered || this.highlight.includes(line.id);
        node.setAttribute("fill", color); node.setAttribute("stroke", line.id === this.selected ? c.select : color);
        node.setAttribute("fill-opacity", strong ? 0.45 : 0.14);
        node.setAttribute("stroke-width", (strong ? 2.5 : 1.2) * u);
        if (line.dirty) node.setAttribute("stroke-dasharray", `${5 * u} ${3 * u}`); else node.removeAttribute("stroke-dasharray");
        node.style.cursor = this.mode === "select" && line.id === this.selected ? "move" : (this.mode === "draw" ? "crosshair" : "pointer");
        if (this.mode === "select" && line.id === this.selected) node.style.touchAction = "none"; else node.style.touchAction = "";
        const base = this.layer.querySelector(`polyline[data-base="${line.id}"]`);
        base.setAttribute("stroke", c.label); base.setAttribute("stroke-width", 1 * u);
        base.setAttribute("stroke-opacity", this.mode === "view" ? 0 : 0.7);
        const label = this.layer.querySelector(`text[data-label="${line.id}"]`);
        label.setAttribute("fill", strong ? c.select : c.label); label.setAttribute("font-weight", strong ? "700" : "400");
      }
    }

    // ---- pointer ----
    point(e) {
      const m = this.svg.getScreenCTM();
      if (!m) return [0, 0];
      const p = new DOMPoint(e.clientX, e.clientY).matrixTransform(m.inverse());
      return [Math.max(0, Math.min(this.width, p.x)), Math.max(0, Math.min(this.height, p.y))];
    }
    line(id) { return this.lines.find((l) => l.id === id); }
    over(e, inside) {
      const id = e.target.getAttribute && e.target.getAttribute("data-id");
      if (!id || this.drag) return;
      this.hovered = inside ? id : null; this.paint();
      if (this.cb.onHover) this.cb.onHover(this.hovered);
    }
    down(e) {
      if (e.button !== undefined && e.button !== 0) return;
      const target = e.target;
      const [x, y] = this.point(e);
      const id = target.getAttribute("data-id");
      if (this.mode === "view") {
        if (id) { this.selected = id; this.paint(); if (this.cb.onSelect) this.cb.onSelect(id); }
        return;
      }
      const handle = target.getAttribute("data-handle");
      if (this.mode === "select" && handle && this.selected) {
        this.drag = { kind: "point", which: handle, index: Number(target.getAttribute("data-index")), before: clone(this.lines), moved: false };
      } else if (this.mode === "select" && id) {
        const already = this.selected === id;
        this.selected = id;
        if (this.cb.onSelect) this.cb.onSelect(id);
        this.drag = already ? { kind: "line", x, y, before: clone(this.lines), moved: false } : null;
        this.render(); this.state();
      } else if (this.mode === "draw" || this.mode === "crop") {
        this.drag = { kind: this.mode, x0: x, y0: y, x1: x, y1: y };
      } else {
        this.selected = null; this.render(); this.state();
        if (this.cb.onSelect) this.cb.onSelect(null);
      }
      if (this.drag) { try { this.svg.setPointerCapture(e.pointerId); } catch (_) { /* a synthetic event */ } e.preventDefault(); }
    }
    move(e) {
      const d = this.drag;
      if (!d) return;
      const [x, y] = this.point(e);
      if (d.kind === "draw" || d.kind === "crop") { d.x1 = x; d.y1 = y; }
      const line = this.line(this.selected);
      if (d.kind === "point" && line) {
        (d.which === "v" ? line.polygon : line.baseline)[d.index] = [Math.round(x), Math.round(y)];
        d.moved = true;
      }
      if (d.kind === "line" && line) {
        let dx = Math.round(x - d.x), dy = Math.round(y - d.y);
        const all = line.polygon.concat(line.baseline);
        dx = Math.max(-Math.min(...all.map((p) => p[0])), Math.min(this.width - Math.max(...all.map((p) => p[0])), dx));
        dy = Math.max(-Math.min(...all.map((p) => p[1])), Math.min(this.height - Math.max(...all.map((p) => p[1])), dy));
        if (dx || dy) {
          all.forEach((p) => { p[0] += dx; p[1] += dy; });
          d.x += dx; d.y += dy; d.moved = true;
        }
      }
      if (!this.frame) this.frame = requestAnimationFrame(() => { this.frame = null; this.render(); });
    }
    up(e) {
      const d = this.drag;
      if (!d) return;
      this.drag = null;
      try { this.svg.releasePointerCapture(e.pointerId); } catch (_) { /* not captured */ }
      if (d.kind === "crop") {
        const box = [Math.round(Math.min(d.x0, d.x1)), Math.round(Math.min(d.y0, d.y1)), Math.round(Math.max(d.x0, d.x1)), Math.round(Math.max(d.y0, d.y1))];
        this.render();
        if (this.cb.onCrop && box[2] - box[0] > 32 && box[3] - box[1] > 32) this.cb.onCrop(box);
        return;
      }
      if (d.kind === "draw") {
        const u = this.unit();
        const left = Math.round(Math.min(d.x0, d.x1)), right = Math.round(Math.max(d.x0, d.x1));
        const top = Math.round(Math.min(d.y0, d.y1)), bottom = Math.round(Math.max(d.y0, d.y1));
        if (right - left > 12 * u && bottom - top > 6 * u) this.addBox(left, top, right, bottom);
        else this.render();
        return;
      }
      if (d.moved) this.commit(d.before); else this.render();
    }
    double(e) {
      if (this.mode !== "select") return;
      const line = this.line(this.selected);
      if (!line) return;
      const target = e.target;
      if (target.getAttribute("data-handle") === "v" && line.polygon.length > 3) {
        const before = clone(this.lines);
        line.polygon.splice(Number(target.getAttribute("data-index")), 1);
        this.commit(before);
      } else if (target.getAttribute("data-id") === line.id) {
        const [x, y] = this.point(e);
        let best = 0, bestDistance = Infinity;
        line.polygon.forEach((a, i) => {
          const b = line.polygon[(i + 1) % line.polygon.length];
          const length = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
          const t = Math.max(0, Math.min(1, ((x - a[0]) * (b[0] - a[0]) + (y - a[1]) * (b[1] - a[1])) / (length * length)));
          const distance = Math.hypot(x - (a[0] + t * (b[0] - a[0])), y - (a[1] + t * (b[1] - a[1])));
          if (distance < bestDistance) { bestDistance = distance; best = i; }
        });
        const before = clone(this.lines);
        line.polygon.splice(best + 1, 0, [Math.round(x), Math.round(y)]);
        this.commit(before);
      }
    }
    key(e) {
      if (this.mode === "view") return;
      if ((e.key === "Delete" || e.key === "Backspace") && this.selected) { e.preventDefault(); this.deleteSelected(); }
      else if (e.key === "Escape") { this.drag = null; this.selected = null; this.render(); this.state(); }
      else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "z") { e.preventDefault(); this.undo(); }
    }

    // ---- edits ----
    commit(before) {
      this.undoStack.push(before);
      if (this.undoStack.length > 40) this.undoStack.shift();
      this.render(); this.state();
      if (this.cb.onChange) this.cb.onChange(clone(this.lines));
    }
    addBox(left, top, right, bottom) {
      const before = clone(this.lines);
      const y = Math.round(top + (bottom - top) * 0.75);       // the baseline: under the letters, above the descenders
      const line = { id: "new-" + Date.now().toString(36), polygon: [[left, top], [right, top], [right, bottom], [left, bottom]],
        baseline: [[left, y], [right, y]], text: "", confidence: null, dirty: true };
      const at = this.lines.findIndex((l) => meanY(l.baseline) > y);
      if (at < 0) this.lines.push(line); else this.lines.splice(at, 0, line);
      this.selected = line.id;
      this.selectAfter = this.lines.indexOf(line);            // the server gives the line another id
      this.commit(before);
    }
    boxSelected() {
      const line = this.line(this.selected);
      if (!line) return;
      const before = clone(this.lines);
      const xs = line.polygon.map((p) => p[0]), ys = line.polygon.map((p) => p[1]);
      const left = Math.min(...xs), right = Math.max(...xs), top = Math.min(...ys), bottom = Math.max(...ys);
      const y = Math.round(Math.min(bottom, Math.max(top, meanY(line.baseline))));
      line.polygon = [[left, top], [right, top], [right, bottom], [left, bottom]];
      line.baseline = [[left, y], [right, y]];
      this.commit(before);
    }
    deleteLine(id) {
      if (!this.line(id)) return;
      const before = clone(this.lines);
      this.lines = this.lines.filter((l) => l.id !== id);
      if (this.selected === id) this.selected = null;
      this.commit(before);
    }
    deleteSelected() { if (this.selected) this.deleteLine(this.selected); }
    undo() {
      const before = this.undoStack.pop();
      if (!before) return;
      this.lines = before;
      if (this.selected && !this.line(this.selected)) this.selected = null;
      this.render(); this.state();
      if (this.cb.onChange) this.cb.onChange(clone(this.lines));
    }
  }
  window.LineEditor = LineEditor;
})();
