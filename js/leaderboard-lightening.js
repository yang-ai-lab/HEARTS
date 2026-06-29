// HEARTS-LIGHTENING data pipeline.
// Unlike the FULL track (which aggregates per-task scores from merged_metric_results.csv
// at runtime), the LIGHTENING scores arrive PRE-AGGREGATED in two CSVs:
//   • high-level CSV    — rows = high-level categories + an "Overall" row; cols = models
//   • category CSV      — rows = the 10 task-type "<name>-Overall" categories; cols = models
// We reshape both into the same { data, intermCatData } structure the table/chart consume.
import { MODELS } from './models.js';

const HIGH_LEVEL_URL = 'data/lightening_high_level.csv';
const CATEGORY_URL = 'data/lightening_category_overall.csv';

// Models present in the LIGHTENING CSVs that should NOT appear on the board.
const EXCLUDE = new Set(['grok_4_3']);

// CSV high-level row label -> internal category key used across the leaderboard.
const HIGH_LEVEL_KEY = {
  Perception: 'perception',
  Inference: 'inference',
  Generation: 'generation',
  Deduction: 'deduction',
};

function parseCsv(url) {
  return fetch(url)
    .then((r) => {
      if (!r.ok) throw new Error(`Failed to load ${url}: ${r.status}`);
      return r.text();
    })
    .then((text) => window.Papa.parse(text, { header: true, skipEmptyLines: true }).data);
}

// Column keys (model identifiers) shared by both CSVs, minus the leading label column
// and the excluded models, keeping only those we have display config for.
function modelColumns(rows, labelCol) {
  if (!rows.length) return [];
  return Object.keys(rows[0]).filter(
    (k) => k !== labelCol && !EXCLUDE.has(k) && MODELS[k]
  );
}

function num(value) {
  const n = parseFloat(String(value));
  return Number.isNaN(n) ? null : n;
}

function toRow(model, total, categories) {
  const cfg = MODELS[model];
  return {
    Model: cfg.displayName,
    'Overall Score': total.toFixed(2),
    Type: cfg.type,
    Source: cfg.source,
    Icon: cfg.icon,
    categories,
  };
}

// Fetch + reshape both CSVs. Returns { data, intermCatData } sorted by overall (desc).
export async function loadLightening() {
  const [highRows, catRows] = await Promise.all([
    parseCsv(HIGH_LEVEL_URL),
    parseCsv(CATEGORY_URL),
  ]);

  const models = modelColumns(highRows, 'High-Level-Category');

  // Index high-level rows by their internal key + grab the Overall row for the total.
  const highByCat = new Map();          // categoryKey -> CSV row
  let overallRow = null;
  for (const row of highRows) {
    const label = row['High-Level-Category'];
    if (label === 'Overall') overallRow = row;
    else if (HIGH_LEVEL_KEY[label]) highByCat.set(HIGH_LEVEL_KEY[label], row);
  }

  // Per-model overall total = the CSV "Overall" row (mean of the 4 high-level categories).
  const totals = new Map();
  for (const m of models) {
    const t = overallRow ? num(overallRow[m]) : null;
    if (t !== null) totals.set(m, t);
  }

  // High-level breakdown rows.
  const data = [];
  for (const m of models) {
    if (!totals.has(m)) continue;
    const categories = {};
    for (const [key, row] of highByCat) {
      const v = num(row[m]);
      if (v !== null) categories[key] = v;
    }
    data.push(toRow(m, totals.get(m), categories));
  }

  // Task-type breakdown rows — same overall total as the high-level board (matches FULL).
  const intermCatData = [];
  for (const m of models) {
    if (!totals.has(m)) continue;
    const categories = {};
    for (const row of catRows) {
      const label = String(row.Category || '').replace(/-Overall$/, '').trim();
      if (!label) continue;
      const v = num(row[m]);
      if (v !== null) categories[label] = v;
    }
    intermCatData.push(toRow(m, totals.get(m), categories));
  }

  const byScore = (a, b) => parseFloat(b['Overall Score']) - parseFloat(a['Overall Score']);
  return { data: data.sort(byScore), intermCatData: intermCatData.sort(byScore) };
}
