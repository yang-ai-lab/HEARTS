// Findings evidence modal — each "What we learned" card opens a dialog that
// redraws the paper's supporting figure in the site's own visual language
// (warm Morandi palette, inline themed SVG — no bitmap screenshots).
//
// Every chart is driven by data recomputed straight from the paper's own
// figure pipelines (see js/findings-data.js): the intelligence-index scatter
// (Fig. 3), perception clear/unclear strip (Fig. 12), sequence-length &
// sampling-frequency rank-rank plots (Fig. 5), model×domain Kappa heatmap
// (Fig. 7), family-consistency scatter (Fig. 8) and input-format KDEs (Fig. 9).

import { FAM_PTS, INTEL, PERCEPTION, SEQ, FREQ, HEAT, FORMAT } from './findings-data.js';

/* ============================== palette helper ============================= */
// Charts must stay in light styling regardless of theme, so the palette is read
// from a hidden probe carrying .fm-lightscope (the same light tokens the plot
// panel forces in CSS) rather than from the (possibly dark) document root.
let _lightProbe = null;
function lightProbe() {
  if (_lightProbe && _lightProbe.isConnected) return _lightProbe;
  _lightProbe = document.createElement('div');
  _lightProbe.className = 'fm-lightscope';
  _lightProbe.style.cssText = 'position:absolute;width:0;height:0;overflow:hidden;pointer-events:none;';
  document.body.appendChild(_lightProbe);
  return _lightProbe;
}
function palette() {
  const s = getComputedStyle(lightProbe());
  const g = (n) => s.getPropertyValue(n).trim();
  return {
    accent: g('--accent'), clay: g('--m-clay'), sage: g('--m-sage'),
    blue: g('--m-blue'), ochre: g('--m-ochre'),
    text: g('--text'), text3: g('--text-3'), text4: g('--text-4'),
    border: g('--border'), grid: g('--border-soft'), surface2: g('--surface-2'),
  };
}

/* ================================ svg helpers ============================== */
const f1 = (n) => (Math.round(n * 10) / 10).toString();
const gauss = (x, m, s) => Math.exp(-0.5 * ((x - m) / s) ** 2);
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));

function svgWrap(w, h, inner) {
  return `<svg viewBox="0 0 ${w} ${h}" class="fm-svg" preserveAspectRatio="xMidYMid meet" role="img">${inner}</svg>`;
}

function hexToRgb(h) {
  h = h.replace('#', '');
  if (h.length === 3) h = h.split('').map((c) => c + c).join('');
  const n = parseInt(h, 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}
function mix(c1, c2, t) {
  const a = hexToRgb(c1), b = hexToRgb(c2);
  const r = a.map((v, i) => Math.round(v + (b[i] - v) * t));
  return `rgb(${r[0]},${r[1]},${r[2]})`;
}
function rgbToArr(c) {
  if (c.startsWith('#')) return hexToRgb(c);
  return c.match(/\d+/g).slice(0, 3).map(Number);
}
function lum(c) { const [r, g, b] = rgbToArr(c); return (0.299 * r + 0.587 * g + 0.114 * b) / 255; }

function linfit(pts) {
  const n = pts.length;
  const sx = pts.reduce((a, p) => a + p[0], 0), sy = pts.reduce((a, p) => a + p[1], 0);
  const sxx = pts.reduce((a, p) => a + p[0] * p[0], 0), sxy = pts.reduce((a, p) => a + p[0] * p[1], 0);
  const m = (n * sxy - sx * sy) / (n * sxx - sx * sx);
  return { m, b: (sy - m * sx) / n };
}
// stable pseudo-jitter in [-1,1] from an integer index (no Math.random → identical each render)
function jit(i) { const t = Math.sin(i * 12.9898) * 43758.5453; return (t - Math.floor(t)) * 2 - 1; }

/* =============================== chart: dists ============================== */
// Overlaid Gaussian densities on a [0,1] value axis with dashed mean markers.
function distChart(curves, o = {}) {
  const P = palette();
  const W = 560, H = o.height || 220, mL = 16, mR = 16, mT = 26, mB = 36;
  const x0 = mL, x1 = W - mR, yB = H - mB, yT = mT;
  const [d0, d1] = o.domain || [0, 1];
  const sx = (v) => x0 + ((v - d0) / (d1 - d0)) * (x1 - x0);
  const N = 170;
  let ymax = 0;
  const built = curves.map((c) => {
    const pts = [];
    for (let i = 0; i <= N; i++) { const x = d0 + ((d1 - d0) * i) / N; const y = gauss(x, c.mean, c.sigma); pts.push([x, y]); if (y > ymax) ymax = y; }
    return { ...c, pts };
  });
  const sy = (v) => yB - (v / ymax) * (yB - yT) * 0.9;
  const body = built.map((c) => {
    let area = `M ${f1(sx(d0))} ${f1(yB)}`, line = '';
    c.pts.forEach((p, i) => { const X = f1(sx(p[0])), Y = f1(sy(p[1])); area += ` L ${X} ${Y}`; line += `${i ? 'L' : 'M'} ${X} ${Y} `; });
    area += ` L ${f1(sx(d1))} ${f1(yB)} Z`;
    return `<path d="${area}" fill="${c.color}" fill-opacity="${c.fill ?? 0.2}"/><path d="${line}" fill="none" stroke="${c.color}" stroke-width="2.2" stroke-linejoin="round"/>`;
  }).join('');
  const means = built.map((c) => {
    const mx = f1(sx(c.mean));
    return `<line x1="${mx}" y1="${yT - 1}" x2="${mx}" y2="${yB}" stroke="${c.color}" stroke-width="1.4" stroke-dasharray="3 3" opacity="0.85"/><text x="${mx}" y="${yT - 7}" text-anchor="middle" class="fm-num" fill="${c.color}">${c.mean.toFixed(2)}</text>`;
  }).join('');
  const ticks = (o.ticks || [0, 0.2, 0.4, 0.6, 0.8, 1]).map((t) => `<text x="${f1(sx(t))}" y="${yB + 15}" text-anchor="middle" class="fm-tick">${t}</text>`).join('');
  const axis = `<line x1="${x0}" y1="${yB}" x2="${x1}" y2="${yB}" stroke="${P.border}" stroke-width="1"/>`;
  const lab = `<text x="${W / 2}" y="${H - 3}" text-anchor="middle" class="fm-axis">${o.axisLabel || ''}</text>`;
  return svgWrap(W, H, body + means + axis + ticks + lab);
}

/* ============================== chart: scatter ============================= */
// pts: [[x,y]] (single ptColor) or [{x,y,color,name}]. Supports a fixed fit
// line {m,b,x0,x1}, a horizontal baseline, custom x ticks, a "higher is
// better" arrow and a boxed annotation — covers Figs 3 & 5 and the families.
function scatterChart(o) {
  const P = palette();
  const W = o.W || 560, H = o.height || 240, mL = o.mL ?? 48, mR = o.mR ?? 14, mT = o.mT ?? 16, mB = 40;
  const x0 = mL, x1 = W - mR, yB = H - mB, yT = mT;
  const [xa, xb] = o.xdom, [ya, yb] = o.ydom;
  const sx = (v) => x0 + ((v - xa) / (xb - xa)) * (x1 - x0);
  const sy = (v) => yB - ((v - ya) / (yb - ya)) * (yB - yT);
  const pts = o.pts.map((p) => Array.isArray(p) ? { x: p[0], y: p[1], color: o.ptColor } : p);
  let g = '';
  (o.yticks || []).forEach((t) => {
    g += `<line x1="${x0}" y1="${f1(sy(t))}" x2="${x1}" y2="${f1(sy(t))}" stroke="${P.grid}" stroke-width="1"/><text x="${x0 - 8}" y="${f1(sy(t) + 3)}" text-anchor="end" class="fm-tick">${t}</text>`;
  });
  const xticks = o.xticksCustom || (o.xticks || []).map((t) => ({ x: t, label: o.xfmt ? o.xfmt(t) : t }));
  xticks.forEach((t) => { g += `<text x="${f1(sx(t.x))}" y="${yB + 15}" text-anchor="middle" class="fm-tick">${t.label}</text>`; });
  if (o.diagonal) { const lo = Math.max(xa, ya), hi = Math.min(xb, yb); g += `<line x1="${f1(sx(lo))}" y1="${f1(sy(lo))}" x2="${f1(sx(hi))}" y2="${f1(sy(hi))}" stroke="${P.text4}" stroke-width="1.3" stroke-dasharray="4 4"/>`; }
  if (o.baseline != null) g += `<line x1="${x0}" y1="${f1(sy(o.baseline))}" x2="${x1}" y2="${f1(sy(o.baseline))}" stroke="${P.text4}" stroke-width="1.4" stroke-dasharray="5 4"/>`;
  if (o.fit) {
    const fit = o.fit === 'auto' ? linfit(o.pts) : o.fit;
    const fx0 = fit.x0 ?? xa, fx1 = fit.x1 ?? xb;
    const fy = (x) => fit.m * x + fit.b;
    g += `<line x1="${f1(sx(fx0))}" y1="${f1(sy(fy(fx0)))}" x2="${f1(sx(fx1))}" y2="${f1(sy(fy(fx1)))}" stroke="${o.fitColor || P.accent}" stroke-width="2.4"/>`;
  }
  const r = o.r || 4;
  g += pts.map((p) =>
    `<circle cx="${f1(sx(p.x))}" cy="${f1(sy(p.y))}" r="${r}" fill="${p.color}" fill-opacity="0.6" stroke="${p.color}" stroke-width="1">${p.name ? `<title>${p.name}</title>` : ''}</circle>`
  ).join('');
  g += `<line x1="${x0}" y1="${yT}" x2="${x0}" y2="${yB}" stroke="${P.border}" stroke-width="1"/><line x1="${x0}" y1="${yB}" x2="${x1}" y2="${yB}" stroke="${P.border}" stroke-width="1"/>`;
  // up-arrow at the y-axis top + label, both in the top margin so data points never overlap them
  if (o.higherBetter) g += `<path d="M ${x0} ${yT - 9} l -4 7 l 8 0 z" fill="${P.text3}"/><text x="${x0 + 8}" y="${yT - 4}" text-anchor="start" class="fm-tick" fill="${P.text3}">Higher is better</text>`;
  if (o.xlab) g += `<text x="${(x0 + x1) / 2}" y="${H - 3}" text-anchor="middle" class="fm-axis">${o.xlab}</text>`;
  if (o.ylab) g += `<text x="11" y="${(yT + yB) / 2}" text-anchor="middle" transform="rotate(-90 11 ${(yT + yB) / 2})" class="fm-axis">${o.ylab}</text>`;
  if (o.annot) {
    const left = o.annotLeft;
    const ax = left ? x0 + 7 : x1 - 6, anchor = left ? 'start' : 'end';
    if (o.annotBox) {
      const lines = o.annot.split('\n');
      const bw = Math.max(...lines.map((l) => l.length)) * 6.5 + 18;
      const bx = x1 - 4 - bw, by = 2;   // raised flush to the very top-right corner
      // the patch is always a light cream, so the text + border stay dark in both themes
      g += `<rect x="${bx}" y="${by}" width="${bw}" height="${8 + lines.length * 13}" rx="5" fill="${mix(P.ochre, '#ffffff', 0.78)}" stroke="rgba(70,50,30,0.35)" stroke-width="1"/>`;
      lines.forEach((ln, i) => { g += `<text x="${bx + bw / 2}" y="${by + 15 + i * 13}" text-anchor="middle" class="fm-annot" fill="#3a2a1c">${ln}</text>`; });
    } else {
      g += `<text x="${ax}" y="${yT + 14}" text-anchor="${anchor}" class="fm-annot" fill="${o.annotColor || P.accent}">${o.annot}</text>`;
    }
  }
  return svgWrap(W, H, g);
}

/* ============================== chart: beeswarm =========================== */
// Two-row strip plot (Fig. 12): clear vs unclear instruction, each dot one
// (model, task) score; dashed group means + a solid overall mean line.
function beeswarm(o) {
  const P = palette();
  const W = 560, H = 250, mL = 122, mR = 20, mT = 18, mB = 46;
  const x0 = mL, x1 = W - mR, yB = H - mB, yT = mT;
  const sx = (v) => x0 + v * (x1 - x0);
  const bandH = (yB - yT) / 2;
  const cyTop = yT + bandH * 0.5, cyBot = yT + bandH * 1.5;
  const spread = bandH * 0.66;
  const row = (vals, cy, color, seed) => vals.map((v, i) =>
    `<circle cx="${f1(sx(v))}" cy="${f1(cy + jit(i + seed) * spread * 0.5)}" r="3.2" fill="${color}" fill-opacity="0.62" stroke="#fff" stroke-width="0.6"/>`).join('');
  let g = '';
  // gridlines
  [0, 0.2, 0.4, 0.6, 0.8, 1].forEach((t) => { g += `<line x1="${f1(sx(t))}" y1="${yT}" x2="${f1(sx(t))}" y2="${yB}" stroke="${P.grid}" stroke-width="1" stroke-dasharray="3 3"/><text x="${f1(sx(t))}" y="${yB + 16}" text-anchor="middle" class="fm-tick">${t}</text>`; });
  g += row(o.top.vals, cyTop, o.top.color, 1) + row(o.bottom.vals, cyBot, o.bottom.color, 9000);
  // mean lines (group means span their own band; overall spans full height)
  g += `<line x1="${f1(sx(o.top.mean))}" y1="${yT}" x2="${f1(sx(o.top.mean))}" y2="${f1(cyTop + spread * 0.5)}" stroke="${o.top.color}" stroke-width="2.2" stroke-dasharray="5 3"/>`;
  g += `<line x1="${f1(sx(o.bottom.mean))}" y1="${f1(cyBot - spread * 0.5)}" x2="${f1(sx(o.bottom.mean))}" y2="${yB}" stroke="${o.bottom.color}" stroke-width="2.2" stroke-dasharray="5 3"/>`;
  g += `<line x1="${f1(sx(o.overall.value))}" y1="${yT}" x2="${f1(sx(o.overall.value))}" y2="${yB}" stroke="${o.overall.color}" stroke-width="2.4"/>`;
  // row labels
  const lbl = (txt, cy) => txt.split('\n').map((ln, k, a) => `<text x="${mL - 12}" y="${f1(cy + (k - (a.length - 1) / 2) * 15 + 4)}" text-anchor="end" class="fm-rowlab">${ln}</text>`).join('');
  g += lbl(o.top.label, cyTop) + lbl(o.bottom.label, cyBot);
  g += `<line x1="${x0}" y1="${yB}" x2="${x1}" y2="${yB}" stroke="${P.border}" stroke-width="1"/>`;
  g += `<text x="${(x0 + x1) / 2}" y="${H - 4}" text-anchor="middle" class="fm-axis">${o.axisLabel}</text>`;
  return svgWrap(W, H, g);
}

/* ================================ chart: KDE ============================== */
// Gaussian KDE (scipy/seaborn 'scott' bandwidth) over real per-format scores,
// clipped to [0,1] — Fig. 9 input-format densities.
function kdeChart(series, o = {}) {
  const P = palette();
  const W = o.W || 280, H = o.height || 210, mL = o.mL ?? 20, mR = o.mR ?? 20, mT = 14, mB = 38;
  const x0 = mL, x1 = W - mR, yB = H - mB, yT = mT;
  const sx = (v) => x0 + v * (x1 - x0);
  const N = 140;
  const stdev = (a) => { const m = a.reduce((s, v) => s + v, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / (a.length - 1)); };
  let ymax = 0;
  const built = series.map((s) => {
    const bw = Math.max(0.04, stdev(s.vals) * Math.pow(s.vals.length, -0.2));
    const dens = [];
    for (let i = 0; i <= N; i++) {
      const x = i / N;
      let d = 0; s.vals.forEach((xi) => { d += gauss(x, xi, bw); });
      d /= (s.vals.length * bw * Math.sqrt(2 * Math.PI));
      dens.push([x, d]); if (d > ymax) ymax = d;
    }
    return { ...s, dens };
  });
  const sy = (v) => yB - (v / ymax) * (yB - yT) * 0.92;
  let g = '';
  [0, 0.5, 1].forEach((t) => { g += `<line x1="${f1(sx(t))}" y1="${yT}" x2="${f1(sx(t))}" y2="${yB}" stroke="${P.grid}" stroke-width="1" stroke-dasharray="3 3"/><text x="${f1(sx(t))}" y="${yB + 15}" text-anchor="middle" class="fm-tick">${t}</text>`; });
  built.forEach((s) => {
    let area = `M ${f1(sx(0))} ${f1(yB)}`, line = '';
    s.dens.forEach((p, i) => { const X = f1(sx(p[0])), Y = f1(sy(p[1])); area += ` L ${X} ${Y}`; line += `${i ? 'L' : 'M'} ${X} ${Y} `; });
    area += ` L ${f1(sx(1))} ${f1(yB)} Z`;
    g += `<path d="${area}" fill="${s.color}" fill-opacity="0.3"/><path d="${line}" fill="none" stroke="${s.color}" stroke-width="2.2" stroke-linejoin="round"/>`;
    g += `<line x1="${f1(sx(s.mean))}" y1="${yT}" x2="${f1(sx(s.mean))}" y2="${yB}" stroke="${s.color}" stroke-width="1.8" stroke-dasharray="4 3" opacity="0.9"/>`;
  });
  g += `<line x1="${x0}" y1="${yB}" x2="${x1}" y2="${yB}" stroke="${P.border}" stroke-width="1"/>`;
  g += `<text x="${(x0 + x1) / 2}" y="${H - 3}" text-anchor="middle" class="fm-axis">${o.axisLabel || 'Score'}</text>`;
  if (o.title) g += `<text x="${(x0 + x1) / 2}" y="${yT - 2}" text-anchor="middle" class="fm-cellt-svg">${o.title}</text>`;
  return svgWrap(W, H, g);
}

/* ============================== chart: heatGrid =========================== */
// Diverging RdBu (centered at 0) — model rows × domain cols, mean Kappa (Fig. 7).
// Domain labels sit BELOW the grid (rotated 45°), matching the paper figure.
function heatGrid(rows, cols, vals, o = {}) {
  const P = palette();
  const vmax = o.vmax || 0.5;
  const colW = o.colW || 40, rowH = o.rowH || 21, mL = o.mL || 116, mT = o.mT || 10, mB = o.mB || 62;
  const gridBottom = mT + rows.length * rowH;
  const W = mL + cols.length * colW + 6, H = gridBottom + mB;
  const color = (v) => { const t = clamp(v / vmax, -1, 1); return t >= 0 ? mix(P.surface2, P.accent, t) : mix(P.surface2, P.blue, -t); };
  let g = '';
  rows.forEach((rname, i) => {
    const y = mT + rowH * i;
    g += `<text x="${mL - 8}" y="${f1(y + rowH / 2 + 3)}" text-anchor="end" class="fm-heatrow">${rname}</text>`;
    vals[i].forEach((v, j) => {
      const x = mL + colW * j, col = color(v), txt = lum(col) > 0.55 ? P.text : '#fff';
      g += `<rect x="${f1(x + 0.6)}" y="${f1(y + 0.6)}" width="${colW - 1.2}" height="${rowH - 1.2}" rx="2" fill="${col}"/><text x="${f1(x + colW / 2)}" y="${f1(y + rowH / 2 + 3)}" text-anchor="middle" class="fm-heatval" fill="${txt}">${v.toFixed(2)}</text>`;
    });
  });
  // domain labels below the grid, rotated 45° (label end at its column)
  const ly = gridBottom + 7;
  cols.forEach((c, j) => { const cx = mL + colW * (j + 0.5); g += `<text x="${f1(cx)}" y="${ly}" text-anchor="end" class="fm-heatcol" transform="rotate(-45 ${f1(cx)} ${ly})">${c}</text>`; });
  return svgWrap(W, H, g);
}

/* =============================== layout (html) ============================ */
function legend(items) {
  return `<div class="fmfig__legend">${items.map((l) => `<span class="fmfig__lg"><i style="background:${l.color}"></i>${l.label}</span>`).join('')}</div>`;
}
function figure({ title, legend: lg, svg, note }) {
  // The plot panel carries fm-lightscope so the chart stays light even in dark mode.
  return `<figure class="fmfig"><div class="fmfig__top"><figcaption class="fmfig__cap">${title}</figcaption>${lg || ''}</div><div class="fmfig__plot fm-lightscope">${svg}</div>${note ? `<p class="fmfig__note">${note}</p>` : ''}</figure>`;
}
// per-panel header: model/family icon + label, centered above the chart
function cellHeader(iconFile, label, large) {
  return `<span class="fmfig__cellt${large ? ' fmfig__cellt--lg' : ''}"><img class="fmfig__icon" src="llm_icons/${iconFile}" alt=""/>${label}</span>`;
}

/* ================================ findings ================================ */
const FINDINGS = {
  1: {
    tag: 'Finding 1', ref: 'Figures 3 & 4',
    title: 'Far behind specialists',
    statement: 'LLMs lag substantially behind specialized models, and their performance on health time series is only weakly related to their general reasoning capabilities.',
    lead: 'On a curated subset of 32 inference & deduction tasks, the best-of-14 LLM still sits well below published task-specific models — and HeaRTS score barely tracks the general-intelligence index.',
    body() {
      const P = palette();
      const gap = distChart([
        { label: 'Avg across LLMs', color: P.blue, mean: 0.46, sigma: 0.14 },
        { label: 'Best across LLMs', color: P.sage, mean: 0.54, sigma: 0.13 },
        { label: 'Specialized ML models', color: P.accent, mean: 0.84, sigma: 0.10 },
      ], { axisLabel: 'Task accuracy' });
      const pts = INTEL.pts.map((p) => ({ x: p.x, y: p.y, name: p.name, color: p.above ? P.accent : P.blue }));
      const sc = scatterChart({
        pts, xdom: [40, 72], ydom: [0.44, 0.67],
        xticks: [45, 50, 55, 60, 65, 70], yticks: [0.45, 0.50, 0.55, 0.60, 0.65],
        xlab: 'Intelligence index', ylab: 'HeaRTS overall score',
        baseline: INTEL.baseline,
        fit: { m: INTEL.fit.m, b: INTEL.fit.b, x0: INTEL.fit.x0, x1: INTEL.fit.x1 }, fitColor: P.accent,
        annot: `R² = ${INTEL.fit.r2.toFixed(3)}`, annotLeft: true, annotColor: P.accent, r: 5,
      });
      return figure({
        title: 'The specialist gap',
        legend: legend([
          { label: 'Avg across LLMs · 0.46', color: P.blue },
          { label: 'Best across LLMs · 0.54', color: P.sage },
          { label: 'Specialized ML · 0.84', color: P.accent },
        ]),
        svg: gap,
        note: 'Specialized models concentrate in a high-accuracy regime; even the best LLM, hand-picked per task, never closes the gap.',
      }) + figure({
        title: 'Weakly tied to general reasoning',
        legend: legend([
          { label: 'above naive baseline', color: P.accent },
          { label: 'below baseline (0.61)', color: P.blue },
        ]),
        svg: sc,
        note: 'Each dot is one of 12 models (hover for names). Among those above the naive baseline (dashed), a higher general-intelligence index buys almost no HeaRTS gain — the fit is nearly flat (R² = 0.117).',
      });
    },
  },

  2: {
    tag: 'Finding 2', ref: '§4.2 — category-level behavior',
    title: 'Shortcuts over reasoning',
    statement: 'LLMs rely heavily on low-complexity heuristics and explicit priors, and show limited deep temporal reasoning across task categories.',
    lead: 'Looking across the four reasoning categories, a single pattern recurs: models do well exactly where a rule, threshold or prior carries the answer, and fall back on mechanical shortcuts wherever genuine temporal reasoning is required.',
    body() {
      const P = palette();
      const cats = [
        ['Perception', 'Robust in aggregate, and strongest on tasks with a well-defined algorithm or an invokable function to call.'],
        ['Inference', 'Polarized — succeeds with explicit thresholds (hypotension < 65) or clear priors (reduced COVID mobility), but slides to near-random on fine-grained distinctions and raw-signal feature extraction.'],
        ['Generation', 'Defaults to low-complexity heuristics: copy-paste with noise, linear interpolation, averaging, or peak-finding — mechanical transformation, not semantic reconstruction.'],
        ['Deduction', 'Mirrors inference: leans on matching salient features to priors (delta waves → early sleep) instead of modeling temporal causality and signal continuity.'],
      ];
      const list = `<div class="fm-cats">${cats.map(([k, v]) =>
        `<div class="fm-cat"><span class="fm-cat__k">${k}</span><span class="fm-cat__v">${v}</span></div>`).join('')}</div>`;
      const bee = beeswarm({
        top: { label: 'Clear\nInstructions', color: P.blue, vals: PERCEPTION.clear, mean: PERCEPTION.clear_mean },
        bottom: { label: 'Unclear\nInstructions', color: P.clay, vals: PERCEPTION.unclear, mean: PERCEPTION.unclear_mean },
        overall: { value: PERCEPTION.overall_mean, color: P.sage },
        axisLabel: 'Model performance',
      });
      return list + figure({
        title: 'Perception tasks: clear vs. unclear instructions',
        legend: legend([
          { label: `Clear · mean ${PERCEPTION.clear_mean.toFixed(2)}`, color: P.blue },
          { label: `Unclear · mean ${PERCEPTION.unclear_mean.toFixed(2)}`, color: P.clay },
          { label: `Overall ${PERCEPTION.overall_mean.toFixed(2)}`, color: P.sage },
        ]),
        svg: bee,
        note: 'Every dot is one model on one perception task. With an explicit, checkable instruction, scores cluster near the top; strip the rule away — features that need domain extraction without a reference — and the same models drop sharply toward chance.',
      });
    },
  },

  3: {
    tag: 'Finding 3', ref: 'Figures 5 & 7',
    title: 'Complexity breaks them',
    statement: 'Performance degrades with temporal complexity, with a shared, model-agnostic difficulty ordering across domains and input modalities.',
    lead: 'Longer inputs and higher sampling rates reliably cost accuracy — and every model walks the same easy-to-hard path across domains, so whatever one model finds tractable, the whole cohort does too.',
    body() {
      const P = palette();
      const seq = scatterChart({
        W: 560, height: 208, mT: 20, pts: SEQ.pts, ptColor: P.blue,
        xdom: [0.4, 6.6], ydom: [0, SEQ.nmax + 3], xticksCustom: SEQ.ticks.map((t) => ({ x: t.x, label: t.label })),
        yticks: [0, 20, 40, 60], xlab: 'Sequence length', ylab: 'Performance rank',
        fit: 'auto', fitColor: P.text4, higherBetter: true,
        annot: `Spearman ρ = ${SEQ.rho.toFixed(2)}\np = ${SEQ.p.toExponential(1)}`, annotBox: true, r: 3.4,
      });
      const freq = scatterChart({
        W: 560, height: 208, mT: 20, pts: FREQ.pts, ptColor: P.blue,
        xdom: [0, FREQ.nmax + 2], ydom: [0, FREQ.nmax + 3], xticksCustom: FREQ.ticks.map((t) => ({ x: t.x, label: t.label })),
        yticks: [0, 20, 40, 60], xlab: 'Sampling frequency', ylab: 'Performance rank',
        fit: 'auto', fitColor: P.text4, higherBetter: true,
        annot: `Spearman ρ = ${FREQ.rho.toFixed(2)}\np = ${FREQ.p.toExponential(1)}`, annotBox: true, r: 3.4,
      });
      const heat = heatGrid(HEAT.rows, HEAT.cols, HEAT.vals, { vmax: HEAT.vmax });
      return figure({
        title: 'Longer inputs → lower performance rank',
        svg: seq,
        note: 'Rank-rank plot after removing 6 confounded tasks: longer sequences correlate significantly with lower performance (Spearman ρ = −0.55).',
      }) + figure({
        title: 'Higher sampling frequency → lower performance rank',
        svg: freq,
        note: 'The same holds for sampling frequency (Spearman ρ = −0.53) — denser signals are harder for every model.',
      }) + figure({
        title: 'One difficulty order, shared by every model',
        legend: legend([{ label: 'κ < 0 (below naive)', color: P.blue }, { label: 'κ > 0 (above naive)', color: P.accent }]),
        svg: heat,
        note: 'Per-domain Kappa (improvement over the naive baseline) for all 16 models, domains sorted easiest→hardest. The warm→cool gradient runs the same way across every row — a difficulty hierarchy intrinsic to the signals, not to any one architecture.',
      });
    },
  },

  4: {
    tag: 'Finding 4', ref: 'Figure 8',
    title: 'Scaling isn\'t enough',
    statement: 'Models within the same architectural family exhibit highly consistent behavioral patterns, while scaling does not necessarily yield improved proficiency.',
    lead: 'Plot a small model against its larger, newer sibling task-by-task and the points hug the y = x line: the pair succeeds and fails on exactly the same tasks. Family decides the difficulty profile; scale barely moves it.',
    body() {
      const P = palette();
      const fams = [
        { name: 'GPT', x: 'GPT 4.1 Mini', y: 'GPT 5 Mini', r: 0.96, pts: FAM_PTS.GPT, col: P.blue, icon: 'openai.png' },
        { name: 'Gemini', x: 'Gemini 2.5 Flash', y: 'Gemini 2.5 Pro', r: 0.84, pts: FAM_PTS.Gemini, col: P.sage, icon: 'gemini-color.png' },
        { name: 'Qwen', x: 'Qwen3 235B', y: 'Qwen3 Coder', r: 0.96, pts: FAM_PTS.Qwen, col: P.ochre, icon: 'qwen-color.png' },
      ];
      const panels = fams.map((fm) => {
        const sc = scatterChart({
          W: 250, height: 250, mL: 40, mR: 40, pts: fm.pts, ptColor: fm.col,
          xdom: [0, 1], ydom: [0, 1], xticks: [0, 0.5, 1], yticks: [0, 0.5, 1],
          xlab: fm.x, ylab: fm.y, diagonal: true, annot: `r = ${fm.r.toFixed(2)}`, annotColor: fm.col, annotLeft: true, r: 3,
        });
        return `<div class="fmfig__cell">${cellHeader(fm.icon, fm.name + ' family')}${sc}</div>`;
      }).join('');
      return figure({
        title: 'A model and its bigger sibling solve the same tasks',
        svg: `<div class="fmfig__row">${panels}</div>`,
        note: 'Each dot is one of the 110 tasks; axes are the two models\' scores. Points sit on the diagonal (dashed) with correlations up to r = 0.96 — the larger / newer model is rarely better, just bigger.',
      });
    },
  },

  5: {
    tag: 'Finding 5', ref: 'Figure 9',
    title: 'Format shifts the level, not the order',
    statement: 'Input format mainly shifts absolute performance, while relative task difficulty remains consistent across formats.',
    lead: 'Hand the same short-sequence tasks to a model as a file, as text, or as an image. File and text land on top of each other; visual ingestion loses fidelity and drops the absolute score — but the shape of the distribution, and so the ordering of which tasks are hard, is unchanged.',
    body() {
      const P = palette();
      const fmt = FORMAT;
      const panel = (block) => kdeChart([
        { label: 'File', color: P.blue, vals: block.File.vals, mean: block.File.mean },
        { label: 'Text', color: P.sage, vals: block.Text.vals, mean: block.Text.mean },
        { label: 'Image', color: P.accent, vals: block.Image.vals, mean: block.Image.mean },
      ], { axisLabel: 'Score' });
      const grok = fmt.grok, llama = fmt.llama;
      return figure({
        title: 'Same model, three input formats',
        legend: legend([{ label: 'File', color: P.blue }, { label: 'Text', color: P.sage }, { label: 'Image', color: P.accent }]),
        svg: `<div class="fmfig__row fmfig__row--2">`
          + `<div class="fmfig__cell">${cellHeader('grok.png', 'Grok 4.1 Fast', true)}${panel(grok)}</div>`
          + `<div class="fmfig__cell">${cellHeader('meta-color.png', 'Llama 4 Maverick', true)}${panel(llama)}</div></div>`,
        note: `Densities over ${fmt.n} short-sequence tasks, with dashed format means. For Grok, file and text are at parity (${grok.File.mean.toFixed(2)} / ${grok.Text.mean.toFixed(2)}) while image drops to ${grok.Image.mean.toFixed(2)}; Llama shows the same ordering with a smaller gap (${llama.File.mean.toFixed(2)} / ${llama.Text.mean.toFixed(2)} / ${llama.Image.mean.toFixed(2)}). The curve shapes stay the same across formats.`,
      });
    },
  },
};

/* ================================== modal ================================= */
let refs = null, lastOpener = null, openId = null;

function buildRefs() {
  const root = document.getElementById('finding-modal');
  if (!root) return null;
  return {
    root, panel: root.querySelector('.fmodal__panel'),
    tag: root.querySelector('.fmodal__tag'), title: root.querySelector('.fmodal__title'),
    statement: root.querySelector('.fmodal__statement'), body: root.querySelector('.fmodal__body'),
    ref: root.querySelector('.fmodal__ref'), closeBtn: root.querySelector('.fmodal__close'),
  };
}

function renderBody(id) {
  const fd = FINDINGS[id];
  if (!fd || !refs) return;
  refs.tag.textContent = fd.tag;
  refs.title.textContent = fd.title;
  refs.statement.textContent = `“${fd.statement}”`;
  refs.ref.textContent = fd.ref;
  refs.body.innerHTML = (fd.lead ? `<p class="fmodal__lead">${fd.lead}</p>` : '') + fd.body();
}

function focusable() {
  return [...refs.panel.querySelectorAll('button, a[href], [tabindex]:not([tabindex="-1"])')].filter((el) => !el.hasAttribute('disabled'));
}
function onKey(e) {
  if (e.key === 'Escape') { close(); return; }
  if (e.key !== 'Tab') return;
  const f = focusable(); if (!f.length) return;
  const first = f[0], last = f[f.length - 1];
  if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
  else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
}

function open(id) {
  if (!refs) return;
  openId = id; lastOpener = document.activeElement;
  renderBody(id);
  refs.root.hidden = false; refs.body.scrollTop = 0;
  document.body.classList.add('fmodal-open');
  requestAnimationFrame(() => { refs.root.classList.add('is-open'); refs.closeBtn.focus(); });
  document.addEventListener('keydown', onKey, true);
}
function close() {
  if (!refs || refs.root.hidden) return;
  refs.root.classList.remove('is-open');
  document.removeEventListener('keydown', onKey, true);
  document.body.classList.remove('fmodal-open');
  const done = () => { refs.root.hidden = true; refs.panel.removeEventListener('transitionend', done); };
  refs.panel.addEventListener('transitionend', done);
  setTimeout(done, 320);
  openId = null;
  if (lastOpener && lastOpener.focus) lastOpener.focus();
}

export function initFindingsModal() {
  refs = buildRefs();
  if (!refs) return;
  document.querySelectorAll('.finding[data-finding]').forEach((card) => {
    card.addEventListener('click', () => open(card.dataset.finding));
  });
  refs.root.querySelectorAll('[data-fmodal-close]').forEach((el) => el.addEventListener('click', close));
  window.addEventListener('themechange', () => { if (openId) renderBody(openId); });
}
