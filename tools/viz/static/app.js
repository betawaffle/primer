// Run explorer page: run list, one run's curves, and run comparison.
// State lives in the URL hash (#run=<name>&metric=<m>&cmp=<a>,<b>&set=test&x=step)
// so a view can be bookmarked or reloaded.

const REFRESH_MS = 10000;
const METRICS = [
  ["exact_acc", "exact"], ["answer_acc", "answer"], ["pointer_acc", "pointer"], ["token_acc", "token"],
  ["value_acc", "value"], ["constrained_acc", "constrained"], ["leak_rate", "leak"], ["loss", "loss"],
  ["token", "token loss"], ["pointer", "pointer loss"], ["payload", "payload loss"],
];
const SETS = [["test", "test (held-out combinations)"], ["val", "val (unseen, same distribution)"], ["train", "train (fixed sample, eval mode)"]];

const cache = {};            // name -> {meta, entries, checkpoints, ...}
let runs = [];
let state = readHash();
let gateIndex = null;        // eval index shown in the gate heatmap; null = latest

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return {
    run: p.get("run"), metric: p.get("metric") || "exact_acc", set: p.get("set") || "test",
    x: p.get("x") || "step", cmp: (p.get("cmp") || "").split(",").filter(Boolean),
  };
}

function writeHash() {
  const p = new URLSearchParams();
  if (state.run) p.set("run", state.run);
  p.set("metric", state.metric);
  if (state.cmp.length) p.set("cmp", state.cmp.join(","));
  if (state.set !== "test") p.set("set", state.set);
  if (state.x !== "step") p.set("x", state.x);
  history.replaceState(null, "", "#" + p.toString());
}

const css = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const slot = i => css(`--s${(i % 8) + 1}`);
const isRate = m => m.endsWith("_acc") || m.endsWith("_rate");
const pct = v => v == null ? "–" : (v * 100).toFixed(Math.abs(v) < 0.1 ? 2 : 1) + "%";
const fmtFor = m => isRate(m) ? pct : fmtNum;
const metricName = m => (METRICS.find(x => x[0] === m) || [m, m])[1];
const nKey = m => m.replace(/_(acc|rate)$/, "_n");

function dur(s) {
  if (s == null) return "–";
  s = Math.round(s);
  const h = Math.floor(s / 3600), mi = Math.floor(s % 3600 / 60);
  return h ? `${h}h ${mi}m` : mi ? `${mi}m ${s % 60}s` : `${s}s`;
}

function ago(t) { return dur(Date.now() / 1000 - t) + " ago"; }

async function getJSON(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: ${r.status}`);
  return r.json();
}

async function loadRun(name) {
  const c = cache[name];
  const since = c && c.entries.length ? c.entries[c.entries.length - 1].step : -1;
  const r = await getJSON(`/api/runs/${encodeURIComponent(name)}?since=${since}`);
  if (c && since >= 0) {
    c.entries.push(...r.entries);
    Object.assign(c, { meta: r.meta, checkpoints: r.checkpoints, best: r.best, updated: r.updated });
  } else {
    cache[name] = r;
  }
  return cache[name];
}

// --- series helpers ---------------------------------------------------------

function xOf(e) { return state.x === "tokens" ? e.tokens : e.step; }

// 95% normal-approximation interval for a rate out of n tokens. Tokens within one
// sequence are correlated, so the true interval is wider; treat it as a floor.
function band(entries, get, m) {
  if (!isRate(m)) return [];
  return entries.map(e => {
    const d = get(e), p = d && d[m], n = d && d[nKey(m)];
    if (p == null || !n) return [xOf(e), null, null];
    const h = 1.96 * Math.sqrt(p * (1 - p) / n);
    return [xOf(e), Math.max(0, p - h), Math.min(1, p + h)];
  });
}

function series(name, color, entries, get, m, extra) {
  return {
    name, color, ...extra,
    points: entries.map(e => { const d = get(e); return [xOf(e), d && d[m] != null ? d[m] : null]; }),
    band: band(entries, get, m),
  };
}

// --- rendering ---------------------------------------------------------------

function renderRuns() {
  const nav = document.getElementById("runs");
  const head = `<div class="navhead"><span>${runs.length} runs</span><span>tick to compare</span></div>`;
  nav.innerHTML = head + runs.map(r => {
    const t = r.latest.test || {};
    const prog = r.steps ? `${(r.step || 0).toLocaleString()} / ${r.steps.toLocaleString()}` : `${(r.step || 0).toLocaleString()} steps`;
    return `<div class="run ${r.name === state.run ? "sel" : ""}" data-name="${esc(r.name)}">
      <input type="checkbox" aria-label="compare ${esc(r.name)}" ${state.cmp.includes(r.name) ? "checked" : ""}>
      <div class="name"><span class="dot ${r.live ? "live" : ""}" title="${r.live ? "running" : "not running"}"></span>${esc(r.name)}</div>
      <div class="meta">${prog} · test exact ${pct(t.exact_acc)} · ${ago(r.updated)}</div>
    </div>`;
  }).join("");
  nav.querySelectorAll(".run").forEach(el => {
    const name = el.dataset.name;
    el.querySelector("input").addEventListener("click", ev => {
      ev.stopPropagation();
      state.cmp = ev.target.checked ? [...state.cmp, name] : state.cmp.filter(n => n !== name);
      writeHash(); render();
    });
    el.addEventListener("click", () => { state.run = name; gateIndex = null; writeHash(); render(); });
  });
}

function seg(label, options, current, onPick) {
  const wrap = document.createElement("span");
  wrap.innerHTML = `<span class="label">${label}</span>`;
  const s = document.createElement("span");
  s.className = "seg";
  for (const [v, text] of options) {
    const b = document.createElement("button");
    b.textContent = text;
    b.className = v === current ? "on" : "";
    b.setAttribute("aria-pressed", v === current);
    b.addEventListener("click", () => onPick(v));
    s.appendChild(b);
  }
  wrap.appendChild(s);
  return wrap;
}

function card(parent, title, note) {
  const c = document.createElement("section");
  c.className = "card";
  c.innerHTML = `<div class="cardhead"><h2>${title}</h2>${note ? `<span class="muted">${note}</span>` : ""}</div>`;
  parent.appendChild(c);
  return c;
}

function chartBlock(parent, ser, opt) {
  const lg = document.createElement("div");
  if (ser.length > 1) { legend(lg, ser); parent.appendChild(lg); }
  const el = document.createElement("div");
  parent.appendChild(el);
  lineChart(el, { series: ser, xLabel: state.x, ...opt });
  return el;
}

function controls(main, availableMetrics) {
  const row = document.createElement("div");
  row.className = "controls";
  const ms = METRICS.filter(([m]) => availableMetrics.has(m));
  row.appendChild(seg("metric", ms, state.metric, v => { state.metric = v; writeHash(); render(); }));
  row.appendChild(seg("x", [["step", "step"], ["tokens", "tokens"]], state.x, v => { state.x = v; writeHash(); render(); }));
  main.appendChild(row);
  return row;
}

function metricsIn(entries) {
  const s = new Set();
  for (const e of entries) for (const k of ["test", "val", "train", "train_avg"]) if (e[k]) Object.keys(e[k]).forEach(m => s.add(m));
  return s;
}

function renderHeader(main, r) {
  const meta = r.meta || {};
  const last = r.entries[r.entries.length - 1] || {};
  const steps = (meta.options || {}).steps;
  const summary = runs.find(x => x.name === r.name) || {};
  const c = card(main, esc(r.name), summary.live ? `<span class="dot live"></span>running` : "");
  const frac = steps ? Math.min(1, (last.step || 0) / steps) : null;
  const rate = last.seconds ? last.step / last.seconds : null;
  const eta = frac != null && frac < 1 && rate ? (steps - last.step) / rate : null;
  const commit = meta.commit ? `<code>${esc(meta.commit.slice(0, 10))}</code>${meta.dirty ? " (uncommitted changes)" : meta.dirty === null ? " (dirty unknown)" : ""}` : "<span class='muted'>not recorded</span>";
  const data = meta.data ? Object.entries(meta.data).map(([k, d]) => `${k} ${(d.sequences || 0).toLocaleString()}`).join(", ") : "–";
  c.insertAdjacentHTML("beforeend", `
    ${frac != null ? `<div class="progress" role="progressbar" aria-valuenow="${Math.round(frac * 100)}" aria-valuemin="0" aria-valuemax="100"><div style="width:${frac * 100}%"></div></div>` : ""}
    <div class="facts">
      <span>step <b>${(last.step || 0).toLocaleString()}</b>${steps ? ` of ${steps.toLocaleString()}` : ""}</span>
      <span>elapsed <b>${dur(last.seconds)}</b></span>
      ${eta != null ? `<span>left <b>~${dur(eta)}</b></span>` : ""}
      ${rate ? `<span><b>${rate.toFixed(1)}</b> steps/s</span>` : ""}
      <span>tokens <b>${fmtSI(last.tokens || 0)}</b></span>
      <span>lr <b>${last.lr != null ? last.lr.toExponential(2) : "–"}</b></span>
      <span>checkpoints <b>${r.checkpoints.length}</b>${r.best ? " + best" : ""}</span>
      <span>last write <b>${ago(r.updated)}</b></span>
    </div>
    <div class="facts" style="margin-top:4px"><span>commit ${commit}</span><span>sequences: ${esc(data)}</span>
      ${meta.started ? `<span>started ${esc(meta.started)}</span>` : ""}</div>
    ${meta.argv ? `<details><summary>command and options</summary><pre class="mono">${esc(meta.argv.join(" "))}\n\n${esc(JSON.stringify(meta.options, null, 1))}</pre></details>` :
      `<p class="note">No run.json: this run predates the richer logging (primer#21), so only training averages and test results are available.</p>`}`);
}

function renderOverall(main, r) {
  const m = state.metric, E = r.entries;
  const ser = [
    series("test", slot(0), E, e => e.test, m),
    series("val", slot(1), E, e => e.val, m),
    series("train sample (eval)", slot(2), E, e => e.train, m),
    series("training batches (running avg)", css("--muted"), E, e => e.train_avg, m, { dash: true }),
  ].filter(s => s.points.some(p => p[1] != null));
  const c = card(main, `${metricName(m)}${isRate(m) ? "" : ""}`, isRate(m) ? "shaded: approx. 95% interval from token counts" : "");
  chartBlock(c, ser, { height: 260, yFmt: fmtFor(m), title: metricName(m) });
  // table view of the latest values, so nothing depends on color alone
  const last = E[E.length - 1] || {};
  const rows = [["test", last.test], ["val", last.val], ["train sample", last.train], ["training batches", last.train_avg]]
    .filter(([, d]) => d && d[m] != null);
  if (rows.length) {
    const best = s => { let b = null; for (const e of E) { const v = e[s] && e[s][m]; if (v != null && (b == null || (m.endsWith("_acc") ? v > b : v < b))) b = v; } return b; };
    c.insertAdjacentHTML("beforeend", `<div class="tablewrap" style="margin-top:8px"><table>
      <tr><th>set</th><th>latest</th><th>${m.endsWith("_acc") ? "best" : "lowest"}</th><th>out of</th></tr>
      ${rows.map(([n, d]) => { const key = { "train sample": "train", "training batches": "train_avg" }[n] || n;
        return `<tr><td>${n}</td><td>${fmtFor(m)(d[m])}</td><td>${fmtFor(m)(best(key))}</td><td>${isRate(m) && d[nKey(m)] ? Math.round(d[nKey(m)]).toLocaleString() : "–"}</td></tr>`; }).join("")}
    </table></div>`);
  }
}

function renderBreakdowns(main, r) {
  const m = state.metric, E = r.entries;
  const keys = k => { const s = new Set(); for (const e of E) if (e[k]) Object.keys(e[k]).forEach(x => s.add(x)); return [...s].sort(); };
  const all = [];
  const collect = k => { for (const e of E) for (const d of Object.values(e[k] || {})) if (d && d[m] != null) all.push(d[m]); };
  ["test_by", "val_by", "train_by"].forEach(collect);
  if (!all.length) return;
  let lo = Math.min(...all), hi = Math.max(...all);
  if (isRate(m)) { lo = Math.max(0, lo - 0.03); hi = Math.min(1, hi + 0.03); }
  const dom = [lo, hi];

  const held = keys("test_by");
  if (held.length) {
    const c = card(main, "Test, per held-out combination", "same y-scale in every panel");
    const g = document.createElement("div"); g.className = "multiples"; c.appendChild(g);
    for (const k of held) {
      const p = document.createElement("div");
      const latest = (E[E.length - 1].test_by || {})[k];
      p.innerHTML = `<h3>${esc(k)} <span class="muted">${latest ? fmtFor(m)(latest[m]) : ""}</span></h3>`;
      g.appendChild(p);
      const el = document.createElement("div"); p.appendChild(el);
      lineChart(el, { series: [series("test", slot(0), E, e => (e.test_by || {})[k], m)], height: 130, compact: true, yDomain: dom, yFmt: fmtFor(m), xLabel: state.x });
    }
  }
  const groups = [...new Set([...keys("val_by"), ...keys("train_by")])].sort();
  if (groups.length) {
    const c = card(main, "Per generator group: unseen vs. trained-on", "the gap between the lines is memorization");
    const lg = document.createElement("div"); c.appendChild(lg);
    legend(lg, [{ name: "val", color: slot(1) }, { name: "train sample", color: slot(2) }]);
    const g = document.createElement("div"); g.className = "multiples"; c.appendChild(g);
    for (const k of groups) {
      const p = document.createElement("div");
      p.innerHTML = `<h3>${esc(k)}</h3>`;
      g.appendChild(p);
      const el = document.createElement("div"); p.appendChild(el);
      lineChart(el, {
        series: [series("val", slot(1), E, e => (e.val_by || {})[k], m), series("train sample", slot(2), E, e => (e.train_by || {})[k], m)],
        height: 130, compact: true, yDomain: dom, yFmt: fmtFor(m), xLabel: state.x,
      });
    }
  }
}

function renderOptim(main, r) {
  const E = r.entries;
  const c = card(main, "Optimization");
  const g = document.createElement("div"); g.className = "trio"; c.appendChild(g);
  // lay out every panel before drawing any: the grid sizes its columns from how many there are
  const panels = [
    ["learning rate", e => e.lr, v => v.toExponential(1)],
    ["gradient norm (running avg, before clipping)", e => e.train_avg && e.train_avg.grad_norm, fmtNum],
    ["weight norm (decayed matrices)", e => e.weight_norm, fmtNum],
    ["training loss (running avg)", e => e.train_avg && e.train_avg.loss, fmtNum],
  ].map(([title, get, fmt]) => {
    const p = document.createElement("div");
    p.innerHTML = `<h3>${title}</h3>`; g.appendChild(p);
    const el = document.createElement("div"); p.appendChild(el);
    return [el, title, get, fmt];
  });
  for (const [el, title, get, fmt] of panels)
    lineChart(el, { series: [{ name: title, color: slot(0), points: E.map(e => [xOf(e), get(e) ?? null]) }], height: 150, compact: true, yFmt: fmt, xLabel: state.x });
}

function renderGates(main, r) {
  const E = r.entries.filter(e => Array.isArray(e.gates));
  if (!E.length) return;
  const i = gateIndex == null ? E.length - 1 : Math.min(gateIndex, E.length - 1);
  const G = E[i].gates, L = G.length, Hh = G[0].length;
  const c = card(main, "RoPE gates per layer and head", "mean over planes; 1 = rotated (uses position), 0 = plain");
  const lo = css("--seq-lo"), hi = css("--seq-hi");
  const mix = v => `color-mix(in oklab, ${hi} ${Math.round(Math.max(0, Math.min(1, v)) * 100)}%, ${lo})`;
  let html = `<div class="heat" style="grid-template-columns: 56px repeat(${Hh}, minmax(28px, 1fr))"><span></span>`;
  for (let h = 0; h < Hh; h++) html += `<span style="text-align:center">head ${h}</span>`;
  for (let l = 0; l < L; l++) {
    html += `<span>layer ${l}</span>`;
    for (let h = 0; h < Hh; h++) {
      const v = G[l][h];
      html += `<div class="cell" style="background:${mix(v)};color:${v > 0.55 ? "var(--surface)" : "var(--ink)"}" title="layer ${l}, head ${h}: ${v.toFixed(3)}">${v.toFixed(2)}</div>`;
    }
  }
  html += `</div><div style="margin-top:10px"><label class="label" for="gstep">step ${E[i].step.toLocaleString()}</label><br>
    <input id="gstep" type="range" min="0" max="${E.length - 1}" value="${i}"></div>`;
  c.insertAdjacentHTML("beforeend", html);
  c.querySelector("#gstep").addEventListener("input", ev => { gateIndex = +ev.target.value; render(); });
}

async function renderCompare(main) {
  const names = state.cmp.slice(0, 8);
  const rs = (await Promise.all(names.map(n => loadRun(n).catch(() => null)))).filter(Boolean);
  const m = state.metric;
  const c = card(main, `Compare: ${metricName(m)}`, state.cmp.length > 8 ? "showing the first 8 ticked runs" : "");
  const row = document.createElement("div"); row.className = "controls"; c.appendChild(row);
  row.appendChild(seg("set", SETS.map(([k, t]) => [k, k]).concat([["train_avg", "training batches"]]), state.set, v => { state.set = v; writeHash(); render(); }));
  const ser = rs.map((r, i) => series(r.name, slot(i), r.entries, e => e[state.set], m));
  chartBlock(c, ser, { height: 280, yFmt: fmtFor(m), title: "compare" });
}

let rendering = false;
async function render() {
  if (rendering) return;
  rendering = true;
  try {
    renderRuns();
    const main = document.getElementById("main");
    const frag = document.createElement("div");
    let r = null;
    if (state.run) {
      try { r = await loadRun(state.run); } catch (e) { r = null; }
    }
    const entries = [...(r ? r.entries : []), ...state.cmp.flatMap(n => (cache[n] || { entries: [] }).entries)];
    const avail = metricsIn(entries);
    if (!avail.has(state.metric) && avail.size) state.metric = METRICS.find(([k]) => avail.has(k))[0];
    main.replaceChildren(frag);
    if (r || state.cmp.length) controls(frag, avail.size ? avail : new Set(METRICS.map(x => x[0])));
    if (state.cmp.length >= 1) await renderCompare(frag);
    if (r) {
      renderHeader(frag, r);
      renderOverall(frag, r);
      renderBreakdowns(frag, r);
      renderOptim(frag, r);
      renderGates(frag, r);
    } else if (!state.cmp.length) {
      frag.innerHTML = `<p class="muted">${runs.length ? "Pick a run on the left." : "No runs found."}</p>`;
    }
  } finally {
    rendering = false;
  }
}

async function refresh() {
  try {
    runs = await getJSON("/api/runs");
    if (!state.run && runs.length) { state.run = runs[0].name; writeHash(); }
    await render();
    document.getElementById("refresh").textContent = `updated ${new Date().toLocaleTimeString()} · every ${REFRESH_MS / 1000}s`;
  } catch (e) {
    document.getElementById("refresh").textContent = `can't reach the server: ${e.message}`;
  }
}

window.addEventListener("hashchange", () => { state = readHash(); render(); });
window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", render);
let resizeTimer;
window.addEventListener("resize", () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(render, 150); });
refresh();
setInterval(refresh, REFRESH_MS);
