// A small SVG line chart with a crosshair tooltip. No dependencies.
//
// lineChart(el, {series, height, yFmt, xFmt, yDomain, compact})
//   series: [{name, color, dash, points: [[x, y], ...], band: [[x, lo, hi], ...]}]
// Points with a null y are gaps. Shared x values line up in the tooltip.

const SVGNS = "http://www.w3.org/2000/svg";

function svg(tag, attrs, parent) {
  const e = document.createElementNS(SVGNS, tag);
  for (const k in attrs) e.setAttribute(k, attrs[k]);
  if (parent) parent.appendChild(e);
  return e;
}

function niceTicks(lo, hi, n) {
  if (!(hi > lo)) { hi = lo + 1; }
  const step0 = (hi - lo) / Math.max(1, n);
  const mag = Math.pow(10, Math.floor(Math.log10(step0)));
  const step = [1, 2, 2.5, 5, 10].map(m => m * mag).find(s => s >= step0) || 10 * mag;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
  return out;
}

function fmtSI(v) {
  const a = Math.abs(v);
  if (a >= 1e9) return +(v / 1e9).toPrecision(3) + "G";
  if (a >= 1e6) return +(v / 1e6).toPrecision(3) + "M";
  if (a >= 1e3) return +(v / 1e3).toPrecision(3) + "k";
  return String(+v.toPrecision(3));
}

function fmtNum(v) {
  if (v == null) return "–";
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return String(+v.toPrecision(4));
}

const tipEl = () => document.getElementById("tip");

function showTip(html, ev) {
  const t = tipEl();
  t.innerHTML = html;
  t.hidden = false;
  const r = t.getBoundingClientRect();
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + r.width > window.innerWidth - 8) x = ev.clientX - r.width - 14;
  if (y + r.height > window.innerHeight - 8) y = ev.clientY - r.height - 14;
  t.style.left = Math.max(8, x) + "px";
  t.style.top = Math.max(8, y) + "px";
}

function hideTip() { tipEl().hidden = true; }

function esc(s) {
  return String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
}

function lineChart(el, opt) {
  el.innerHTML = "";
  el.classList.add("chart");
  const series = opt.series.filter(s => s.points.some(p => p[1] != null));
  const W = Math.max(200, el.clientWidth || 600);
  const H = opt.height || 220;
  const m = { l: 44, r: 10, t: 8, b: 22 };
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const root = svg("svg", { viewBox: `0 0 ${W} ${H}`, width: W, height: H, role: "img", "aria-label": opt.title || "chart" }, el);
  if (!series.length) {
    const t = svg("text", { x: W / 2, y: H / 2, "text-anchor": "middle" }, root);
    t.textContent = "no data";
    return;
  }
  let xs = [], ys = [];
  for (const s of series) {
    for (const [x, y] of s.points) { xs.push(x); if (y != null) ys.push(y); }
    for (const [, lo, hi] of s.band || []) { if (lo != null) ys.push(lo); if (hi != null) ys.push(hi); }
  }
  let x0 = Math.min(...xs), x1 = Math.max(...xs);
  if (x1 === x0) { x1 = x0 + 1; }
  let [y0, y1] = opt.yDomain || [Math.min(...ys), Math.max(...ys)];
  if (!opt.yDomain) { const pad = (y1 - y0) * 0.06 || Math.abs(y1) * 0.05 || 0.05; y0 -= pad; y1 += pad; }
  if (opt.zeroFloor && y0 < 0 && Math.min(...ys) >= 0) y0 = 0;
  const X = x => m.l + (x - x0) / (x1 - x0) * iw;
  const Y = y => m.t + (1 - (y - y0) / (y1 - y0)) * ih;
  const yFmt = opt.yFmt || fmtNum, xFmt = opt.xFmt || fmtSI;

  const g = svg("g", { class: "grid" }, root);
  const ax = svg("g", {}, root);
  for (const v of niceTicks(y0, y1, opt.compact ? 4 : 5)) {
    if (v < y0 || v > y1) continue;
    svg("line", { x1: m.l, x2: W - m.r, y1: Y(v), y2: Y(v) }, g);
    const t = svg("text", { x: m.l - 6, y: Y(v) + 3.5, "text-anchor": "end" }, ax);
    t.textContent = yFmt(v);
  }
  for (const v of niceTicks(x0, x1, Math.max(2, Math.floor(iw / 90)))) {
    if (v < x0 || v > x1) continue;
    const t = svg("text", { x: X(v), y: H - 6, "text-anchor": "middle" }, ax);
    t.textContent = xFmt(v);
  }
  svg("line", { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih }, svg("g", { class: "axis" }, root));

  for (const s of series) {
    const band = (s.band || []).filter(b => b[1] != null && b[2] != null);
    if (band.length > 1) {
      const d = band.map((b, i) => `${i ? "L" : "M"}${X(b[0])},${Y(b[2])}`).join("") +
        band.slice().reverse().map(b => `L${X(b[0])},${Y(b[1])}`).join("") + "Z";
      svg("path", { d, class: "band", fill: s.color }, root);
    }
  }
  for (const s of series) {
    let d = "", pen = false;
    for (const [x, y] of s.points) {
      if (y == null) { pen = false; continue; }
      d += `${pen ? "L" : "M"}${X(x)},${Y(y)}`;
      pen = true;
    }
    const attrs = { d, class: "series", stroke: s.color };
    if (s.dash) attrs["stroke-dasharray"] = "5 4";
    if (s.width) attrs["stroke-width"] = s.width;
    svg("path", attrs, root);
    const pts = s.points.filter(p => p[1] != null);
    if (pts.length === 1) svg("circle", { cx: X(pts[0][0]), cy: Y(pts[0][1]), r: 4, fill: s.color, class: "pt" }, root);
  }

  // hover: snap to the nearest logged x
  const allX = [...new Set(xs)].sort((a, b) => a - b);
  const cross = svg("line", { class: "cross", y1: m.t, y2: m.t + ih, visibility: "hidden" }, root);
  const dots = series.map(s => svg("circle", { r: 4, fill: s.color, class: "pt", visibility: "hidden" }, root));
  const hit = svg("rect", { x: m.l, y: 0, width: iw, height: H, fill: "transparent" }, root);
  hit.addEventListener("mousemove", ev => {
    const rect = root.getBoundingClientRect();
    const px = (ev.clientX - rect.left) * (W / rect.width);
    const xv = x0 + (px - m.l) / iw * (x1 - x0);
    let best = allX[0];
    for (const x of allX) if (Math.abs(x - xv) < Math.abs(best - xv)) best = x;
    cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best));
    cross.setAttribute("visibility", "visible");
    let rows = "";
    series.forEach((s, i) => {
      const p = s.points.find(p => p[0] === best);
      if (!p || p[1] == null) { dots[i].setAttribute("visibility", "hidden"); return; }
      dots[i].setAttribute("cx", X(best)); dots[i].setAttribute("cy", Y(p[1]));
      dots[i].setAttribute("visibility", "visible");
      const b = (s.band || []).find(b => b[0] === best);
      const ci = b && b[1] != null ? ` <span class="muted">±${yFmt((b[2] - b[1]) / 2)}</span>` : "";
      rows += `<div class="row"><span><i style="background:${s.color}"></i>${esc(s.name)}</span><b>${yFmt(p[1])}${ci}</b></div>`;
    });
    showTip(`<div class="muted">${esc(opt.xLabel || "step")} ${xFmt === fmtSI ? best.toLocaleString() : xFmt(best)}</div>${rows}`, ev);
  });
  hit.addEventListener("mouseleave", () => {
    cross.setAttribute("visibility", "hidden");
    dots.forEach(d => d.setAttribute("visibility", "hidden"));
    hideTip();
  });
}

function legend(el, series) {
  el.className = "legend";
  el.innerHTML = series.map(s =>
    `<span><i class="${s.dash ? "dash" : ""}" style="background:${s.color};border-color:${s.color}"></i>${esc(s.name)}</span>`).join("");
}
