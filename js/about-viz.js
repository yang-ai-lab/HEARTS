// About-section visualizations:
//  • renderDomainDonut() — interactive SVG donut of the 12 health domains.
//  • renderModalities()  — the 20 signal modalities, each with its real signal icon
//    (traced from the paper's hero figure, tinted to the modality palette).
//  These two share one highlight controller: hovering a domain (slice or legend row)
//  lights up that domain AND every modality it records; hovering a modality lights up
//  that modality AND every domain that uses it — dimming everything else, pie included.
//  • renderFindingsViz(cats) — horizontal bars of average LLM score per reasoning level.
import { MOD_ICONS } from './icon-paths.js';

// [name, share %, color, icon-id (sprite symbol from the paper's domain icons)]
const DOMAINS = [
  ['Sleep', 21.8, '#B26248', 'dom-sleep'], ['Metabolic', 14.5, '#C2954A', 'dom-metabolic'],
  ['Respiration', 10.0, '#7C9479', 'dom-respiration'], ['Emotion', 9.1, '#6E8AA6', 'dom-emotion'],
  ['Motion', 9.1, '#A8728C', 'dom-motion'], ['Surgery', 8.2, '#C98A5B', 'dom-surgery'],
  ['Speech', 7.3, '#8B9D6B', 'dom-speech'], ['Behavior', 5.4, '#6F9B9B', 'dom-behavior'],
  ['Ophthalmology', 4.5, '#B4848B', 'dom-ophthalmology'], ['Eye Movement', 4.5, '#9788A8', 'dom-eye-movement'],
  ['Covid Cough', 2.8, '#C7A36B', 'dom-covid-cough'], ['Gesture', 2.8, '#7E8CA6', 'dom-gesture'],
];

// The 20 signal modalities, in display order. Colours cycle the modality palette.
const MODS = [
  'EEG', 'ECG', 'PPG', 'SpO₂', 'Blood Pressure', 'Heart Rate', 'EMG', 'IMU', 'Temperature', 'Eye Tracking',
  'EOG', 'PERG', 'CGM', 'EDA', 'Thoracic', 'Respiration', 'Air Flow', 'Audio', 'Annotation', 'Aggregated',
];
const WAVE_COLORS = ['#B26248', '#C2954A', '#7C9479', '#6E8AA6', '#A8728C', '#6F9B9B'];

// Domain → modality indices it records (HEARTS paper, Table 3). MOD_DOMAINS is the
// inverse. BVP (Emotion) is folded into PPG; both are photoplethysmography signals.
const DOMAIN_MODS = [
  [0, 1, 5, 10, 14, 16, 18], // Sleep:    ECG, EEG, EOG, Air Flow, Thoracic, HR, Annotation
  [5, 12, 18],               // Metabolic: CGM, HR, Annotation
  [3, 5, 15],                // Respiration: SpO₂, HR, Respiration
  [2, 5, 8, 13],             // Emotion:  PPG(BVP), HR, Temperature, EDA
  [7],                       // Motion:   IMU
  [0, 1, 2, 4, 18],          // Surgery:  EEG, ECG, PPG, Blood Pressure, Annotation
  [17],                      // Speech:   Audio
  [19],                      // Behavior: Aggregated
  [11],                      // Ophthalmology: PERG
  [9],                       // Eye Movement:  Eye Tracking
  [17],                      // Covid Cough:   Audio
  [6],                       // Gesture:  EMG
];
const MOD_DOMAINS = MODS.map((_, j) => DOMAIN_MODS.reduce((a, mods, d) => (mods.includes(j) ? a.concat(d) : a), []));

// ---- shared highlight controller (donut legend + pie + modality cloud) ----
let donutRefs = null;   // { mount, legend, segEls, liEls }
let modRefs = null;     // { mount, modEls }

function setHighlight(domSet, modSet) {
  const dOn = domSet && domSet.size > 0;
  const mOn = modSet && modSet.size > 0;
  if (donutRefs) {
    donutRefs.mount.classList.toggle('pie--active', !!dOn);
    donutRefs.legend.classList.toggle('legend--active', !!dOn);
    donutRefs.segEls.forEach((s) => s.classList.toggle('is-on', !!dOn && domSet.has(Number(s.dataset.i))));
    donutRefs.liEls.forEach((l) => l.classList.toggle('is-on', !!dOn && domSet.has(Number(l.dataset.i))));
  }
  if (modRefs) {
    modRefs.mount.classList.toggle('mods--active', !!mOn);
    modRefs.modEls.forEach((m) => m.classList.toggle('is-on', !!mOn && modSet.has(Number(m.dataset.i))));
  }
}
const clearHighlight = () => setHighlight(null, null);

const prefersReduced = window.matchMedia?.('(prefers-reduced-motion: reduce)').matches || false;

// Replay an entrance each time `el` enters the viewport: on(true) when visible, on(false)
// (instantly) when it leaves so the animation can re-run on the next scroll-in.
function onReveal(el, on, opts = {}) {
  if (prefersReduced || !('IntersectionObserver' in window)) { on(true); return; }
  const io = new IntersectionObserver((entries) => {
    entries.forEach((e) => on(e.isIntersecting));
  }, { threshold: opts.threshold || 0.3 });
  io.observe(el);
}

export function renderDomainDonut() {
  const mount = document.getElementById('cov-donut');
  const legend = document.getElementById('cov-legend');
  if (!mount || !legend) return;

  const total = DOMAINS.reduce((a, d) => a + d[1], 0);
  let acc = 0;
  // Each slice is a dashed ring. The circle is rotated -90° so the path's open/close
  // seam (natively at 3 o'clock) sits at 12 o'clock — a real slice boundary — instead
  // of bisecting a slice, which otherwise leaves a hairline break (was visible on
  // 'Metabolic'). Offset −acc then starts each slice at its cumulative position.
  const segs = DOMAINS.map((d, i) => {
    const pct = (d[1] / total) * 100;
    const dash = `${pct.toFixed(3)} 100`;   // gap ≥ remaining circumference → no repeat
    const off = (-acc).toFixed(3);
    acc += pct;
    // Rendered collapsed (0 100); the draw-in below sweeps each slice out clockwise,
    // staggered by --draw-d so the ring appears to grow round the circle.
    return `<circle class="pie-seg" data-i="${i}" data-dash="${dash}" style="--draw-d:calc(${i} * 55ms)"
      r="15.91549" cx="21" cy="21" fill="none" stroke="${d[2]}" stroke-width="8.5"
      stroke-dasharray="0 100" stroke-dashoffset="${off}" transform="rotate(-90 21 21)"></circle>`;
  }).join('');

  mount.innerHTML =
    `<svg viewBox="0 0 42 42" class="pie-svg" aria-hidden="true">${segs}</svg>` +
    `<span class="pie-center"><b>12</b><i>domains</i></span>`;
  legend.innerHTML = DOMAINS.map((d, i) =>
    `<li data-i="${i}"><svg class="dom-ic" style="color:${d[2]}" aria-hidden="true"><use href="#${d[3]}"/></svg><span>${d[0]}</span></li>`
  ).join('');

  donutRefs = {
    mount, legend,
    segEls: [...mount.querySelectorAll('.pie-seg')],
    liEls: [...legend.querySelectorAll('li')],
  };
  const wire = (el) => {
    const i = Number(el.dataset.i);
    el.addEventListener('mouseenter', () => setHighlight(new Set([i]), new Set(DOMAIN_MODS[i])));
    el.addEventListener('mouseleave', clearHighlight);
  };
  donutRefs.segEls.forEach(wire);
  donutRefs.liEls.forEach(wire);

  // Draw-in: sweep the slices out (and the centre label fades) on each scroll-in.
  const draw = (on) => {
    if (!on) {
      mount.classList.add('redraw-instant');
      mount.classList.remove('is-drawn');
      donutRefs.segEls.forEach((s) => s.setAttribute('stroke-dasharray', '0 100'));
      void mount.offsetWidth;                 // commit the reset with no transition
      mount.classList.remove('redraw-instant');
      return;
    }
    mount.classList.add('is-drawn');
    donutRefs.segEls.forEach((s) => s.setAttribute('stroke-dasharray', s.dataset.dash));
  };
  onReveal(mount, draw, { threshold: 0.35 });
}

// Build the inline SVG for one modality icon, tinted with `col`. Cut-out paths use the
// chip background so they read as negative space in both themes.
function modIconSVG(name, col) {
  const ic = MOD_ICONS[name];
  if (!ic) return '';
  const strokes = (ic.st || []).map((s) =>
    `<path d="${s.d}" fill="none" stroke="${s.cls === 'cut' ? 'var(--surface-2)' : 'currentColor'}" stroke-width="${s.w}" stroke-linejoin="round" stroke-linecap="round"/>`
  ).join('');
  return `<svg class="mod-ic" viewBox="${ic.vb}" style="color:${col}" aria-hidden="true">` +
    (ic.ink ? `<path d="${ic.ink}" fill="currentColor"/>` : '') +
    (ic.cut ? `<path d="${ic.cut}" fill="var(--surface-2)"/>` : '') +
    strokes + '</svg>';
}

export function renderModalities() {
  const mount = document.getElementById('mod-cloud');
  if (!mount) return;
  mount.innerHTML = MODS.map((name, i) => {
    const col = WAVE_COLORS[i % WAVE_COLORS.length];
    return `<span class="mod" data-i="${i}" style="--i:${i}">${modIconSVG(name, col)}<span class="mod-name">${name}</span></span>`;
  }).join('');

  modRefs = { mount, modEls: [...mount.querySelectorAll('.mod')] };
  modRefs.modEls.forEach((el) => {
    const j = Number(el.dataset.i);
    el.addEventListener('mouseenter', () => setHighlight(new Set(MOD_DOMAINS[j]), new Set([j])));
    el.addEventListener('mouseleave', clearHighlight);
  });

  // Staggered blur-in: chips de-blur one after another whenever the cloud scrolls in.
  onReveal(mount, (on) => mount.classList.toggle('is-in', on), { threshold: 0.2 });
}

// cats: [{ label, color, value, naive }] — value = avg LLM score, naive =
// naive-baseline score for that reasoning level. Horizontal bars (avg LLM) with
// the per-level naive baseline overlaid as a goalpost tick, so the gap between
// the bar end and the tick reads at a glance (e.g. perception's naive ≈ 1.0).
export function renderFindingsViz(cats, opts = {}) {
  const mount = document.getElementById('findings-viz');
  if (!mount || !cats || !cats.length) return;
  const maxScale = 1.0;          // perception's naive baseline is 1.0
  const W = 540, padT = 22, padB = 34, rowH = 44, mL = 100, th = 17;
  const x0 = mL, x1 = W - 52, valX = W - 8;
  const H = padT + cats.length * rowH + padB;
  const sx = (v) => x0 + (Math.max(0, Math.min(v, maxScale)) / maxScale) * (x1 - x0);

  const bars = cats.map((c, i) => {
    const cy = padT + i * rowH + rowH / 2, by = cy - th / 2, bx = sx(c.value), nx = sx(c.naive);
    return `<rect class="fv-track-bg" x="${x0}" y="${by}" width="${x1 - x0}" height="${th}" rx="${th / 2}"/>`
      + `<rect x="${x0}" y="${by}" width="${(bx - x0).toFixed(1)}" height="${th}" rx="${th / 2}" fill="${c.color}"/>`
      // naive baseline goalpost for this level + its value just above the tick
      + `<line class="fv-baseline" x1="${nx.toFixed(1)}" y1="${by - 5}" x2="${nx.toFixed(1)}" y2="${by + th + 5}" stroke-dasharray="3 2" stroke-width="2"/>`
      + `<text class="fv-base" x="${nx.toFixed(1)}" y="${by - 8}" text-anchor="middle">${c.naive.toFixed(2)}</text>`
      + `<text class="fv-cat" x="${x0 - 12}" y="${cy + 4}" text-anchor="end">${c.label}</text>`
      + `<text class="fv-num tabular" x="${valX}" y="${cy + 4}" text-anchor="end">${c.value.toFixed(2)}</text>`;
  }).join('');

  // legend: a goalpost glyph + label, centered under the bars
  const ly = H - 10, lcx = W / 2 - 105;
  const legend = `<line class="fv-baseline" x1="${lcx}" y1="${ly - 7}" x2="${lcx}" y2="${ly + 2}" stroke-dasharray="3 2" stroke-width="2"/>`
    + `<text class="fv-base" x="${lcx + 8}" y="${ly}" text-anchor="start">Naive baseline (per level)</text>`;

  mount.innerHTML = `<svg viewBox="0 0 ${W} ${H}" class="fv-svg" role="img" aria-label="Average LLM score by reasoning level with the naive baseline overlaid per level">${bars}${legend}</svg>`;
}
