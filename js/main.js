// Site orchestration: theme toggle, navbar, mobile menu, ambient live rendering
// (scroll progress, reveal-on-scroll, count-up), citation copy, and the
// leaderboard (data load + chart/table view toggle, theme-aware re-render).
import { loadLeaderboard, CATEGORY_ORDER, INTERM_CAT_ORDER, CATEGORY_LABELS } from './leaderboard-data.js';
import { loadLightening } from './leaderboard-lightening.js';
import { renderChart } from './leaderboard-chart.js';
import { mountTable } from './leaderboard-table.js';
import { renderDomainDonut, renderModalities } from './about-viz.js';
import { initFindingsModal } from './findings-modal.js';

const prefersReduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Track scroll direction so entrance animations replay only on downward scroll.
let scrollDir = 'down';
function initScrollDir() {
  let lastY = window.scrollY || 0;
  window.addEventListener('scroll', () => {
    const y = window.scrollY || 0;
    if (Math.abs(y - lastY) > 2) { scrollDir = y > lastY ? 'down' : 'up'; lastY = y; }
  }, { passive: true });
}

/* ---------- Dark / light theme ---------- */
function applyThemeIcon() {
  // Icons are visibility-swapped by CSS via html.dark; nothing to do here beyond
  // keeping aria state in sync.
  const btn = document.getElementById('theme-toggle');
  if (btn) btn.setAttribute('aria-pressed', String(document.documentElement.classList.contains('dark')));
}

function setTheme(dark, persist) {
  document.documentElement.classList.toggle('dark', dark);
  if (persist) {
    try { localStorage.setItem('hearts-theme', dark ? 'dark' : 'light'); } catch { /* ignore */ }
  }
  const meta = document.querySelector('meta[name="theme-color"]:not([media])');
  if (meta) meta.setAttribute('content', dark ? '#0a0d13' : '#ffffff');
  applyThemeIcon();
  // Let theme-sensitive widgets (the canvas chart) repaint.
  window.dispatchEvent(new CustomEvent('themechange', { detail: { dark } }));
}

function initTheme() {
  applyThemeIcon();
  const btn = document.getElementById('theme-toggle');
  if (btn) {
    btn.addEventListener('click', () =>
      setTheme(!document.documentElement.classList.contains('dark'), true)
    );
  }
  // Follow the OS only while the user hasn't made an explicit choice.
  const mq = window.matchMedia('(prefers-color-scheme: dark)');
  mq.addEventListener?.('change', (e) => {
    let saved = null;
    try { saved = localStorage.getItem('hearts-theme'); } catch { /* ignore */ }
    if (!saved) setTheme(e.matches, false);
  });
}

/* ---------- Navbar: background on scroll ---------- */
function initNavbar() {
  const nav = document.getElementById('navbar');
  if (!nav) return;
  const SCROLLED = ['bg-white/80', 'backdrop-blur-md', 'shadow-sm', 'border-b', 'border-gray-100'];
  const onScroll = () => {
    if (window.scrollY > 10) {
      nav.classList.add(...SCROLLED);
      nav.classList.remove('bg-transparent');
    } else {
      nav.classList.remove(...SCROLLED);
      nav.classList.add('bg-transparent');
    }
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
}

/* ---------- Mobile menu ---------- */
function initMobileMenu() {
  const btn = document.getElementById('mobile-menu-btn');
  const menu = document.getElementById('mobile-menu');
  if (!btn || !menu) return;
  const open = document.getElementById('icon-menu');
  const close = document.getElementById('icon-close');
  const setOpen = (isOpen) => {
    menu.classList.toggle('hidden', !isOpen);
    open?.classList.toggle('hidden', isOpen);
    close?.classList.toggle('hidden', !isOpen);
  };
  btn.addEventListener('click', () => setOpen(menu.classList.contains('hidden')));
  menu.querySelectorAll('a[href^="#"]').forEach((a) =>
    a.addEventListener('click', () => setOpen(false))
  );
}

/* ---------- Active nav link on scroll ---------- */
function initActiveLink() {
  const links = [...document.querySelectorAll('[data-navlink]')];
  if (!links.length) return;
  const byId = new Map(links.map((l) => [l.getAttribute('data-navlink'), l]));
  const ACTIVE = ['text-gray-900', 'font-semibold'];
  const observer = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        links.forEach((l) => l.classList.remove(...ACTIVE));
        byId.get(e.target.id)?.classList.add(...ACTIVE);
      });
    },
    { rootMargin: '-45% 0px -50% 0px' }
  );
  ['home', 'leaderboard', 'about', 'submit', 'contact', 'citation'].forEach((id) => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}

/* ---------- Hero GIF: play once, then settle on the static final figure ---------- */
function initHeroGif() {
  const gif = document.getElementById('hero-gif');
  const still = document.getElementById('hero-still');
  if (!gif || !still) return;
  const DURATION_MS = 13400; // one full loop of hero-icon.gif (239 frames)
  let settled = false;
  const settle = () => { if (settled) return; settled = true; gif.classList.add('is-hidden'); };
  const arm = () => setTimeout(settle, DURATION_MS);
  if (gif.complete && gif.naturalWidth) arm();
  else gif.addEventListener('load', arm, { once: true });
  gif.addEventListener('error', settle);
}

/* ---------- Scroll progress bar (live render) ---------- */
function initScrollProgress() {
  const bar = document.getElementById('scroll-progress');
  if (!bar) return;
  let ticking = false;
  const update = () => {
    const doc = document.documentElement;
    const max = doc.scrollHeight - doc.clientHeight;
    const ratio = max > 0 ? Math.min(1, doc.scrollTop / max) : 0;
    bar.style.transform = `scaleX(${ratio})`;
    ticking = false;
  };
  const onScroll = () => {
    if (!ticking) { ticking = true; requestAnimationFrame(update); }
  };
  window.addEventListener('scroll', onScroll, { passive: true });
  window.addEventListener('resize', onScroll, { passive: true });
  update();
}

/* ---------- Reveal-on-scroll: tag top-level blocks, fade them up ---------- */
function initReveal() {
  if (prefersReduced || !('IntersectionObserver' in window)) return;

  // Collect the first "content row" inside each section (descend single-div wrappers).
  const blocks = [];
  document.querySelectorAll('main > section').forEach((sec) => {
    let host = sec;
    let guard = 0;
    while (host.children.length === 1 && host.firstElementChild.tagName === 'DIV' && guard++ < 4) {
      host = host.firstElementChild;
    }
    [...host.children].forEach((el) => blocks.push(el));
  });

  // The leaderboard panels manage their own swap animation, so keep them out of the
  // scroll-reveal system (otherwise reveal's opacity would fight the swap).
  const SKIP = new Set(['lb-chart-panel', 'lb-table-panel', 'lb-status']);
  document.documentElement.classList.add('reveal-on');
  blocks.filter((el) => !SKIP.has(el.id)).forEach((el, i) => {
    el.setAttribute('data-reveal', '');
    el.style.setProperty('--reveal-delay', `${Math.min(i % 6, 5) * 70}ms`);
  });

  // Observe EVERY [data-reveal] — auto-tagged blocks AND ones hand-placed in the
  // markup (e.g. nested finding cards), so nothing stays stuck at opacity 0.
  // Replay the entrance only when scrolling DOWN; on upward scroll elements (re)appear
  // instantly without animating, and leaving resets them (instantly) for a later replay.
  const all = [...document.querySelectorAll('[data-reveal]')];
  const setInstant = (el, visible) => {
    el.classList.add('reveal-instant');                 // disable transitions on el + children
    el.classList.toggle('is-visible', visible);
    void el.offsetWidth;                                // commit the change without a transition
    requestAnimationFrame(() => el.classList.remove('reveal-instant'));
  };
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (e.isIntersecting) {
          if (scrollDir === 'down') e.target.classList.add('is-visible'); // animate in
          else setInstant(e.target, true);                                 // upward: no replay
        } else {
          setInstant(e.target, false);                                     // reset for next replay
        }
      });
    },
    { rootMargin: '0px 0px -10% 0px', threshold: 0.08 }
  );
  all.forEach((el) => io.observe(el));
}

/* ---------- Count-up for stat numbers ---------- */
function initCountUp() {
  const els = [...document.querySelectorAll('[data-count]')];
  if (!els.length) return;
  if (prefersReduced || !('IntersectionObserver' in window)) return;

  const animate = (el) => {
    const target = parseFloat(el.getAttribute('data-count'));
    if (!Number.isFinite(target)) return;
    const gen = (el._cuGen = (el._cuGen || 0) + 1);   // cancel any in-flight run
    const dur = 1100;
    let start = null;
    const tick = (ts) => {
      if (el._cuGen !== gen) return;                  // superseded — stop
      if (start === null) start = ts;
      const p = Math.min(1, (ts - start) / dur);
      const eased = 1 - Math.pow(1 - p, 3);
      el.textContent = String(Math.round(target * eased));
      if (p < 1) requestAnimationFrame(tick);
      else el.textContent = String(target);
    };
    requestAnimationFrame(tick);
  };

  // Re-count only when the number re-enters while scrolling DOWN; on upward scroll it
  // simply shows the final value without animating.
  const io = new IntersectionObserver(
    (entries) => {
      entries.forEach((e) => {
        if (!e.isIntersecting) return;
        if (scrollDir === 'down') animate(e.target);
        else { e.target._cuGen = (e.target._cuGen || 0) + 1; e.target.textContent = e.target.getAttribute('data-count'); }
      });
    },
    { threshold: 0.6 }
  );
  els.forEach((el) => io.observe(el));
}

/* ---------- Cursor-follow spotlight on cards ---------- */
function initSpotlight() {
  if (prefersReduced) return;
  const cards = document.querySelectorAll(
    '.lb-leader-card, .tax-card, .step-card, .statcard, .cov-card, .fv-card, .finding'
  );
  cards.forEach((card) => {
    card.classList.add('spot');
    const ring = document.createElement('span');
    ring.className = 'spot-ring';
    ring.setAttribute('aria-hidden', 'true');
    card.appendChild(ring);
    card.addEventListener('pointermove', (e) => {
      const r = card.getBoundingClientRect();
      card.style.setProperty('--mx', `${((e.clientX - r.left) / r.width) * 100}%`);
      card.style.setProperty('--my', `${((e.clientY - r.top) / r.height) * 100}%`);
    });
  });
}

/* ---------- Citation copy button ---------- */
function initCopyButtons() {
  document.querySelectorAll('[data-copy-target]').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const target = document.getElementById(btn.getAttribute('data-copy-target'));
      if (!target) return;
      try {
        await navigator.clipboard.writeText(target.textContent.trim());
        const done = btn.querySelector('[data-copy-done]');
        const idle = btn.querySelector('[data-copy-idle]');
        if (done && idle) {
          idle.classList.add('hidden');
          done.classList.remove('hidden');
          setTimeout(() => { done.classList.add('hidden'); idle.classList.remove('hidden'); }, 2000);
        }
      } catch { /* clipboard unavailable */ }
    });
  });
}

/* ---------- Leaderboard ---------- */
const CHART_ACTIVE = ['bg-[var(--emphasis-color)]', 'text-white', 'shadow-sm'];
const TABLE_ACTIVE = ['bg-white', 'text-gray-900', 'shadow-sm'];
const INACTIVE = ['text-gray-500', 'hover:text-gray-900'];

let lbData = null;      // active benchmark's high-level rows (theme-aware chart re-render)
let chartVisible = false;
let chartReady = false; // chart is rendered lazily on first switch to the chart view

// Loaded { data, intermCatData } per benchmark track, plus which one is showing.
const benchmarks = { full: null, lightening: null };
let activeBenchmark = 'lightening';

function ensureChart() {
  if (chartReady || !lbData) return;
  const canvas = document.getElementById('lb-canvas');
  if (canvas) { renderChart(canvas, lbData); chartReady = true; }
}

function setView(view) {
  const chartPanel = document.getElementById('lb-chart-panel');
  const tablePanel = document.getElementById('lb-table-panel');
  const chartBtn = document.getElementById('lb-view-chart');
  const tableBtn = document.getElementById('lb-view-table');
  const showChart = view === 'chart';
  chartVisible = showChart;
  chartPanel.classList.toggle('hidden', !showChart);
  tablePanel.classList.toggle('hidden', showChart);
  chartBtn.classList.remove(...CHART_ACTIVE, ...INACTIVE);
  tableBtn.classList.remove(...TABLE_ACTIVE, ...INACTIVE);
  chartBtn.classList.add(...(showChart ? CHART_ACTIVE : INACTIVE));
  tableBtn.classList.add(...(showChart ? INACTIVE : TABLE_ACTIVE));
  // Replay the entrance animation on the panel being shown (every switch).
  replayLbAnim(showChart ? chartPanel : tablePanel);
}

function replayLbAnim(el) {
  if (!el || prefersReduced) return;
  el.classList.remove('lb-anim');
  void el.offsetWidth;        // restart the CSS animation
  el.classList.add('lb-anim');
}

function fillSummary(data) {
  const top = data[0];
  if (!top) return;
  const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
  set('lb-leader-name', top.Model);
  set('lb-leader-score', top['Overall Score']);
  const logo = document.getElementById('lb-leader-logo');
  if (logo && top.Icon) { logo.src = `llm_icons/${top.Icon}`; logo.alt = top.Model; }
}

// Copy the FULL track's naive baseline (high-level + task-type rows) into another
// track that lacks one, re-sorting each table by overall score (desc).
function injectNaiveBaseline(target, source) {
  if (!target || !source) return;
  const byScore = (a, b) => parseFloat(b['Overall Score']) - parseFloat(a['Overall Score']);
  const clone = (r) => ({ ...r, categories: { ...r.categories } });
  for (const key of ['data', 'intermCatData']) {
    if (target[key].some((r) => r.Type === 'Baseline')) continue;     // already present
    const naive = source[key].find((r) => r.Type === 'Baseline');
    if (naive) target[key] = [...target[key], clone(naive)].sort(byScore);
  }
}

// Track name shown on the champion-card tag.
const BENCH_META = {
  full:       { name: 'HEARTS-FULL' },
  lightening: { name: 'HEARTS-LIGHTENING' },
};

// (Re)render the champion card + table + chart for the given benchmark track.
function applyBenchmark(key) {
  const bm = benchmarks[key];
  if (!bm) return;
  activeBenchmark = key;
  lbData = bm.data;
  fillSummary(bm.data);
  // Reinforce which board is on screen via a tag on the champion card.
  const meta = BENCH_META[key];
  const tag = document.getElementById('lb-leader-track');
  if (tag && meta) tag.textContent = meta.name;
  mountTable(document.getElementById('lb-table-panel'), {
    data: bm.data,
    intermCatData: bm.intermCatData,
    categoryOrder: CATEGORY_ORDER,
    intermCatOrder: INTERM_CAT_ORDER,
    categoryLabels: CATEGORY_LABELS,
  });
  // The chart is per-benchmark — rebuild it (now if visible, else lazily on next view).
  chartReady = false;
  if (chartVisible) ensureChart();
  // Replay the entrance on whichever panel is showing so the swap reads as a change.
  replayLbAnim(document.getElementById(chartVisible ? 'lb-chart-panel' : 'lb-table-panel'));
}

// Active state is fully driven by aria-pressed (styled in theme.css).
function syncBenchButtons() {
  document.querySelectorAll('[data-benchmark]').forEach((b) =>
    b.setAttribute('aria-pressed', String(b.getAttribute('data-benchmark') === activeBenchmark))
  );
}

// Switch tracks + keep the toggle in sync. Returns false if the track is unavailable.
function setBenchmark(key) {
  if (key === activeBenchmark || !benchmarks[key]) return false;
  applyBenchmark(key);
  syncBenchButtons();
  return true;
}

function initBenchmarkToggle() {
  const buttons = [...document.querySelectorAll('[data-benchmark]')];
  if (!buttons.length) return;
  // A track that failed to load gets disabled rather than silently inert.
  buttons.forEach((b) => {
    if (!benchmarks[b.getAttribute('data-benchmark')]) b.disabled = true;
  });
  buttons.forEach((b) => b.addEventListener('click', () => setBenchmark(b.getAttribute('data-benchmark'))));
  syncBenchButtons();
}

// News deep-links that should land on the LIGHTENING reasoning-taxonomy table.
// (The anchor href handles the scroll; we switch the board + view.)
function initLeaderboardDeepLinks() {
  document.querySelectorAll('[data-go-lightening]').forEach((a) =>
    a.addEventListener('click', () => {
      if (!benchmarks.lightening) return;     // unavailable → fall back to the plain #leaderboard scroll
      applyBenchmark('lightening');           // re-mount forces the default reasoning-taxonomy breakdown
      syncBenchButtons();
      setView('table');
    })
  );
}

async function initLeaderboard() {
  const status = document.getElementById('lb-status');
  try {
    const [full, lightening] = await Promise.all([
      loadLeaderboard(),
      loadLightening().catch((e) => { console.error(e); return null; }),
    ]);
    benchmarks.full = full;
    benchmarks.lightening = lightening;
    // LIGHTENING ships without a naive row; reuse FULL's so both boards share the
    // identical baseline (same numbers), inserted in rank order.
    injectNaiveBaseline(benchmarks.lightening, benchmarks.full);
    // Lazy-render the chart only when the user first switches to it (table is default).
    document.getElementById('lb-view-chart')?.addEventListener('click', () => { setView('chart'); ensureChart(); });
    document.getElementById('lb-view-table')?.addEventListener('click', () => setView('table'));
    initBenchmarkToggle();
    initLeaderboardDeepLinks();
    // Default to the LIGHTENING track + table (reasoning-taxonomy breakdown),
    // falling back to FULL if LIGHTENING failed to load.
    status?.remove();
    applyBenchmark(benchmarks.lightening ? 'lightening' : 'full');
    syncBenchButtons();
    setView('table');
  } catch (err) {
    if (status) {
      status.className = 'lb-status-msg';
      status.removeAttribute('aria-busy');
      status.textContent = 'Failed to load leaderboard data.';
    }
    console.error(err);
  }

  // Repaint the canvas chart with theme-appropriate axis/grid colors (if it's been built).
  window.addEventListener('themechange', () => {
    if (chartReady && chartVisible) {
      const canvas = document.getElementById('lb-canvas');
      if (canvas) renderChart(canvas, lbData);
    }
  });
}

/* ---------- HEARTS-LIGHTENING card: open + scroll when linked from the news ---------- */
function initLightening() {
  const det = document.getElementById('hearts-lightening');
  if (!det) return;
  const reveal = () => {
    det.open = true;
    // Let the open state lay out before scrolling so we land on the expanded card.
    requestAnimationFrame(() => det.scrollIntoView({ behavior: 'smooth', block: 'start' }));
  };
  // Any link targeting the card opens + scrolls — even if the hash is already set.
  document.querySelectorAll('a[href="#hearts-lightening"]').forEach((a) =>
    a.addEventListener('click', (e) => {
      e.preventDefault();
      history.pushState(null, '', '#hearts-lightening');
      reveal();
    })
  );
  window.addEventListener('hashchange', () => {
    if (location.hash === '#hearts-lightening') reveal();
  });
  if (location.hash === '#hearts-lightening') reveal();
}

/* ---------- Boot ---------- */
function boot() {
  initTheme();
  initNavbar();
  initMobileMenu();
  initActiveLink();
  initHeroGif();
  initScrollDir();
  initScrollProgress();
  initReveal();
  initCountUp();
  initCopyButtons();
  initSpotlight();
  renderDomainDonut();
  renderModalities();
  initFindingsModal();
  initLightening();
  initLeaderboard();
  if (window.lucide) window.lucide.createIcons();
}

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', boot);
} else {
  boot();
}
