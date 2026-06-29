// Leaderboard table — teambench-inspired presentation.
// Renders into a container element and manages its own sort + breakdown state.
// Visuals (.lb-*) live in theme.css and follow the active light/dark theme.

const TYPE_BADGE = {
  Reasoning: 'bg-purple-100 text-purple-800',
  Baseline: 'bg-yellow-100 text-yellow-800',
  _default: 'bg-green-100 text-green-800',
};
const SOURCE_BADGE = {
  'Open-source': 'bg-[var(--emphasis-soft-bg)] text-[var(--emphasis-color)]',
  Reference: 'bg-orange-100 text-orange-800',
  _default: 'bg-gray-100 text-gray-800',
};

export function mountTable(container, { data, intermCatData, categoryOrder, intermCatOrder, categoryLabels = {} }) {
  let breakdownMode = 'high';        // 'high' | 'interm'
  let sortColumn = 'total';
  let sortDirection = 'desc';        // 'asc' | 'desc'

  function currentData() {
    return breakdownMode === 'high' ? data : intermCatData;
  }
  function currentCategories() {
    return breakdownMode === 'high' ? categoryOrder : intermCatOrder;
  }

  function sortedRows() {
    return [...currentData()].sort((a, b) => {
      const va = sortColumn === 'total' ? parseFloat(a['Overall Score']) : (a.categories[sortColumn] || 0);
      const vb = sortColumn === 'total' ? parseFloat(b['Overall Score']) : (b.categories[sortColumn] || 0);
      return sortDirection === 'desc' ? vb - va : va - vb;
    });
  }

  function sortIcon(column) {
    if (sortColumn !== column) return '<i data-lucide="chevrons-up-down" class="w-3.5 h-3.5 lb-sorticon"></i>';
    const name = sortDirection === 'desc' ? 'chevron-down' : 'chevron-up';
    return `<i data-lucide="${name}" class="w-3.5 h-3.5"></i>`;
  }

  function headCell(column, label) {
    const active = sortColumn === column ? ' is-active' : '';
    return `<th scope="col" data-sort="${column}" class="is-sortable${active}">
      <span class="lb-th-inner">${label}${sortIcon(column)}</span>
    </th>`;
  }

  function rankCell(index) {
    const r = index + 1;
    const medal = r <= 3 ? ` lb-rank--${r}` : '';
    return `<span class="lb-rank${medal}">${r}</span>`;
  }

  function toggleBtn(mode, label) {
    const active = breakdownMode === mode;
    const cls = active
      ? 'bg-[var(--emphasis-color)] text-white shadow-sm'
      : 'text-gray-500 hover:text-gray-900';
    return `<button data-mode="${mode}" class="px-4 py-2 rounded-md text-sm font-medium transition-all ${cls}">${label}</button>`;
  }

  function render() {
    const cats = currentCategories();
    const rows = sortedRows();

    // Only the Total Score column highlights its best (bold + red).
    const totalMax = Math.max(...rows.map((r) => parseFloat(r['Overall Score'])));

    const catHead = cats.map((c) => headCell(c, categoryLabels[c] || c)).join('');

    const body = rows.map((row, index) => {
      const typeCls = TYPE_BADGE[row.Type] || TYPE_BADGE._default;
      const srcCls = SOURCE_BADGE[row.Source] || SOURCE_BADGE._default;
      const total = parseFloat(row['Overall Score']);

      const catCells = cats.map((c) => `<td>${(row.categories[c] || 0).toFixed(2)}</td>`).join('');

      const totalBest = total > 0 && Math.abs(total - totalMax) < 1e-9 ? ' lb-best' : '';

      return `<tr class="${index === 0 ? 'is-leader' : ''}">
        <td>${rankCell(index)}</td>
        <td>
          <div class="lb-model">
            <img class="lb-model__logo" src="llm_icons/${row.Icon}" alt="" loading="lazy">
            <span class="lb-model__name">${row.Model}</span>
          </div>
        </td>
        <td class="lb-score${totalBest}">${row['Overall Score']}</td>
        ${catCells}
        <td><span class="lb-badge ${typeCls}">${row.Type}</span></td>
        <td><span class="lb-badge ${srcCls}">${row.Source}</span></td>
      </tr>`;
    }).join('');

    container.innerHTML = `
      <div class="flex flex-col items-center gap-2 mb-6">
        <span class="lb-viewcap">Group categories by</span>
        <div class="seg">
          ${toggleBtn('high', 'Reasoning Taxonomy')}
          ${toggleBtn('interm', 'Task Type')}
        </div>
      </div>
      <div class="lb-card">
        <div class="lb-scroll">
          <table class="lb-table">
            <thead>
              <tr>
                <th scope="col">Rank</th>
                <th scope="col">Model</th>
                ${headCell('total', 'Total Score')}
                ${catHead}
                <th scope="col">Type</th>
                <th scope="col">Source</th>
              </tr>
            </thead>
            <tbody>${body}</tbody>
          </table>
        </div>
        <div class="lb-note">
          The leading Total Score is highlighted. Category scores are the mean across all tasks in that
          category; Total Score is the mean of the category scores.
        </div>
      </div>`;

    // Wire interactions
    container.querySelectorAll('[data-sort]').forEach((th) => {
      th.addEventListener('click', () => {
        const column = th.getAttribute('data-sort');
        if (sortColumn === column) {
          sortDirection = sortDirection === 'desc' ? 'asc' : 'desc';
        } else {
          sortColumn = column;
          sortDirection = 'desc';
        }
        render();
      });
    });
    container.querySelectorAll('[data-mode]').forEach((btn) => {
      btn.addEventListener('click', () => {
        breakdownMode = btn.getAttribute('data-mode');
        sortColumn = 'total';
        sortDirection = 'desc';
        render();
      });
    });

    // Re-render Lucide icons for the freshly-inserted nodes.
    if (window.lucide) window.lucide.createIcons();
  }

  render();
}
