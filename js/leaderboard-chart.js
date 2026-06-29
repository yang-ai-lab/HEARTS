// Leaderboard stacked-bar chart — port of src/components/LeaderboardChart.tsx.
// Uses the vendored Chart.js UMD global (window.Chart), which auto-registers all
// controllers/elements. The icon plugin is passed per-chart via config.plugins.

// Category display order with alpha values (matching the Python/React source)
const CATEGORY_ORDER = ['deduction', 'generation', 'inference', 'perception'];
const CATEGORY_ALPHAS = {
  perception: 1.0,   // Bottom - highest opacity
  inference: 0.8,
  generation: 0.6,
  deduction: 0.4,    // Top - lowest opacity
};

// Gradient colors from blue (low score) to red (high score)
const GRADIENT_COLORS = [
  '#4575B4', '#74ADD1', '#ABD9E9', '#E0F3F8', '#FFFFBF',
  '#FEE090', '#FDAE61', '#F46D43', '#D73027',
];

function isNaiveModel(model) {
  return model.Type === 'Baseline' || model.Source === 'Reference';
}

function getModelColor(modelIndex, totalModels, isNaive) {
  if (isNaive) return '#9CA3AF'; // Gray for naive baseline
  const ratio = totalModels > 1 ? 1 - modelIndex / (totalModels - 1) : 0;
  const colorIndex = Math.round(ratio * (GRADIENT_COLORS.length - 1));
  return GRADIENT_COLORS[colorIndex];
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

const iconCache = new Map();
function getCachedIcon(iconName) {
  const cached = iconCache.get(iconName);
  if (cached) return cached;
  const image = new Image();
  image.src = `llm_icons/${iconName}`;
  iconCache.set(iconName, image);
  return image;
}

// Plugin to draw model icons on top of the stacked bars.
// Preserves the source's async guard: if an icon image hasn't finished loading on the
// first paint, draw it once it loads via chart.draw() (Gotcha #11 — icons on first paint).
const iconPlugin = {
  id: 'iconPlugin',
  afterDatasetDraw(chart) {
    const { ctx, data } = chart;
    const lastRealDatasetIndex = chart.data.datasets.length - 2; // before the Icons dataset
    const lastMeta = chart.getDatasetMeta(lastRealDatasetIndex);
    const icons = data.datasets[data.datasets.length - 1].icons;
    if (!icons) return;

    lastMeta.data.forEach((bar, index) => {
      const iconSrc = icons[index];
      if (!iconSrc) return;
      const img = getCachedIcon(iconSrc);
      const size = 32;
      const x = bar.x - size / 2;
      const y = bar.y - size - 10; // above the top of the stacked bar
      const drawIcon = () => {
        // light chip behind the logo so dark/transparent marks stay legible on the
        // dark-themed canvas (and adds a subtle lift in light mode too).
        ctx.save();
        ctx.beginPath();
        ctx.arc(bar.x, y + size / 2, size / 2 + 4, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.shadowColor = 'rgba(0, 0, 0, 0.18)';
        ctx.shadowBlur = 6;
        ctx.shadowOffsetY = 1;
        ctx.fill();
        ctx.restore();
        ctx.drawImage(img, x, y, size, size);
      };
      if (img.complete) {
        drawIcon();
      } else {
        img.onload = () => chart.draw();
      }
    });
  },
};

// Axis/grid colors that follow the active light/dark theme.
function themePalette() {
  const dark = document.documentElement.classList.contains('dark');
  return dark
    ? { text: '#c4cdda', title: '#e8edf4', grid: 'rgba(255, 255, 255, 0.08)' }
    : { text: '#374151', title: '#111827', grid: '#e5e7eb' };
}

function buildConfig(data) {
  const pal = themePalette();
  // Normalize category scores so they sum to the overall score.
  const normalizedData = data.map((d) => {
    const sum = Object.values(d.categories).reduce((a, b) => a + b, 0);
    const overallScore = parseFloat(d['Overall Score']);
    const normalizedCategories = {};
    for (const [cat, value] of Object.entries(d.categories)) {
      normalizedCategories[cat] = sum === 0 ? 0 : (value / sum) * overallScore;
    }
    return { ...d, categories: normalizedCategories };
  });

  // Models are sorted by total score (desc), so index 0 is highest (red/warm).
  const modelColors = normalizedData.map((d, i) =>
    getModelColor(i, normalizedData.length, isNaiveModel(d))
  );

  const chartData = {
    labels: normalizedData.map((d) => d.Model),
    datasets: [
      ...CATEGORY_ORDER.slice().reverse().map((cat) => ({
        label: cat.charAt(0).toUpperCase() + cat.slice(1),
        data: normalizedData.map((d) => d.categories[cat] || 0),
        backgroundColor: normalizedData.map((_, i) =>
          hexToRgba(modelColors[i], CATEGORY_ALPHAS[cat])
        ),
        barPercentage: 0.8,
        categoryPercentage: 0.9,
      })),
      {
        label: 'Icons',
        data: normalizedData.map(() => 0),
        backgroundColor: 'transparent',
        icons: normalizedData.map((d) => d.Icon),
      },
    ],
  };

  const options = {
    responsive: true,
    maintainAspectRatio: false,
    layout: { padding: { top: 50 } },
    plugins: {
      legend: { display: false },
      tooltip: {
        callbacks: {
          // Show the ORIGINAL (un-normalized) category value.
          label: (context) => {
            const label = context.dataset.label || '';
            const cat = label.toLowerCase();
            const originalValue = data[context.dataIndex].categories[cat] || 0;
            return `${label}: ${originalValue.toFixed(2)}`;
          },
          footer: (items) => {
            const modelIndex = items[0].dataIndex;
            const values = Object.values(data[modelIndex].categories);
            const total = values.reduce((a, b) => a + b, 0) / values.length;
            return `Total: ${total.toFixed(2)}`;
          },
        },
      },
    },
    scales: {
      y: {
        min: 0,
        max: 0.8,
        stacked: true,
        grid: { color: pal.grid, tickBorderDash: [5, 5] },
        ticks: { font: { size: 14, weight: 'bold' }, color: pal.text },
        title: { display: true, text: 'Overall Score', font: { size: 16, weight: 'bold' }, color: pal.title },
      },
      x: {
        stacked: true,
        ticks: { autoSkip: false, maxRotation: 45, minRotation: 45, font: { size: 12, weight: 'bold' }, color: pal.text },
        grid: { display: false },
      },
    },
  };

  return { type: 'bar', data: chartData, options, plugins: [iconPlugin] };
}

// Render (or re-render) the chart into a <canvas>. Destroys any previous instance.
let chartInstance = null;
export function renderChart(canvas, data) {
  if (chartInstance) {
    chartInstance.destroy();
    chartInstance = null;
  }
  chartInstance = new window.Chart(canvas, buildConfig(data));
  return chartInstance;
}
