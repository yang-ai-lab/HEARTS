// Leaderboard data pipeline — faithful port of src/components/leaderboardData.ts.
// Difference from the React source: the merged CSV is fetched at runtime (was a
// build-time `?raw` import). PapaParse is the vendored UMD global (window.Papa).
import { TARGET_MODELS, MODELS } from './models.js';

const CSV_URL = 'data/merged_metric_results.csv';

export const CATEGORY_ORDER = ['perception', 'inference', 'generation', 'deduction'];

export const INTERM_CAT_ORDER = [
  'Stat Calculation',
  'Feature Extraction',
  'Localization',
  'Physiology Classification',
  'Individual-level Analysis',
  'Imputation',
  'Forecasting',
  'Translation',
  'Time-irreversibility',
  'Cross-visit Comparison',
];

// Display labels for the task-type column headers (paper Table 2 naming).
// Keys are the internal interm_cat identifiers; values are what the header shows.
// Categories not listed here fall back to their key (e.g. the high-level columns).
export const CATEGORY_LABELS = {
  'Stat Calculation': 'Stat Calculation',
  'Feature Extraction': 'Feature Extraction',
  'Localization': 'Event Localization',
  'Physiology Classification': 'Physiological Classification',
  'Individual-level Analysis': 'Subject Profiling',
  'Forecasting': 'Future Forecasting',
  'Imputation': 'Signal Imputation',
  'Translation': 'Cross-modal Translation',
  'Time-irreversibility': 'Temporal Ordering',
  'Cross-visit Comparison': 'Trajectory Analysis',
};

const INTERM_CAT_TO_HIGH_LEVEL = {
  'Stat Calculation': 'perception',
  'Feature Extraction': 'perception',
  'Physiology Classification': 'inference',
  'Individual-level Analysis': 'inference',
  'Imputation': 'generation',
  'Forecasting': 'generation',
  'Translation': 'generation',
  'Time-irreversibility': 'deduction',
  'Localization': 'inference',
  'Cross-visit Comparison': 'deduction',
};

function cleanIntermCat(value) {
  try {
    const parsed = JSON.parse(String(value).replace(/'/g, '"'));
    if (Array.isArray(parsed) && parsed.length > 0) {
      return String(parsed[0]);
    }
  } catch {
    // Fall through to cleanup.
  }
  return String(value).replace(/[[\]'"]/g, '').trim();
}

function addScore(store, model, category, score) {
  if (!store.has(model)) store.set(model, new Map());
  const modelCategories = store.get(model);
  if (!modelCategories.has(category)) modelCategories.set(category, []);
  modelCategories.get(category).push(score);
}

function finalizeScores(scoreMap) {
  const categories = new Set();
  for (const [, modelCategories] of scoreMap) {
    for (const category of modelCategories.keys()) categories.add(category);
  }
  const categoryCount = categories.size;
  const finalScores = new Map();

  for (const [model, modelCategories] of scoreMap) {
    const categoryMeans = {};
    let totalScore = 0;
    for (const [categoryName, scores] of modelCategories) {
      const mean = scores.reduce((sum, s) => sum + s, 0) / scores.length;
      categoryMeans[categoryName] = mean;
      totalScore += mean / categoryCount;
    }
    finalScores.set(model, { total: totalScore, categories: categoryMeans });
  }
  return finalScores;
}

function toBenchmarkData(scores, overallScores) {
  const rows = [];
  for (const [modelName, scoreData] of scores) {
    const modelConfig = MODELS[modelName];
    if (!modelConfig) continue;
    const overallScore = (overallScores && overallScores.get(modelName)?.total) ?? scoreData.total;
    rows.push({
      Model: modelConfig.displayName,
      'Overall Score': overallScore.toFixed(2),
      Type: modelConfig.type,
      Source: modelConfig.source,
      Icon: modelConfig.icon,
      categories: scoreData.categories,
    });
  }
  return rows.sort((a, b) => parseFloat(b['Overall Score']) - parseFloat(a['Overall Score']));
}

function buildLeaderboardData(rows) {
  const categoryScores = new Map();
  const intermCategoryScores = new Map();

  for (const row of rows) {
    const intermCategory = cleanIntermCat(row.interm_cat);
    const highLevelCategory = INTERM_CAT_TO_HIGH_LEVEL[intermCategory];

    for (const model of TARGET_MODELS) {
      let score = parseFloat(String(row[model]));
      if (Number.isNaN(score)) {
        if (model !== 'naive') continue;
        score = 1;
      }
      addScore(intermCategoryScores, model, intermCategory, score);
      if (highLevelCategory) addScore(categoryScores, model, highLevelCategory, score);
    }
  }

  const highLevelScores = finalizeScores(categoryScores);
  const intermScores = finalizeScores(intermCategoryScores);

  return {
    data: toBenchmarkData(highLevelScores),
    intermCatData: toBenchmarkData(intermScores, highLevelScores),
  };
}

// Fetch + parse + aggregate. Returns { data, intermCatData }.
export async function loadLeaderboard() {
  const text = await fetch(CSV_URL).then((r) => {
    if (!r.ok) throw new Error(`Failed to load ${CSV_URL}: ${r.status}`);
    return r.text();
  });
  const parsed = window.Papa.parse(text, { header: true, skipEmptyLines: true });
  return buildLeaderboardData(parsed.data);
}
