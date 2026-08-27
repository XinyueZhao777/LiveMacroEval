/* Renders every number on the page from data/leaderboard.json.
   To refresh the site, edit that JSON — do not edit index.html. */

const fmt = (v, d = 3) => (v >= 0 ? '+' : '−') + Math.abs(v).toFixed(d);

function scoreCell(v) {
  const cls = v > 0 ? 'pos' : v < 0 ? 'neg' : '';
  return `<span class="score ${cls}">${fmt(v)}</span>`;
}

function bar(v, max) {
  const pct = (Math.abs(v) / max) * 50;
  const side = v >= 0
    ? `left:50%;width:${pct}%`
    : `right:50%;width:${pct}%`;
  return `<div class="bartrack"><div class="barzero" style="left:50%"></div>
          <div class="bar ${v >= 0 ? 'pos' : 'neg'}" style="${side}"></div></div>`;
}

function renderHeadline(h) {
  const max = Math.max(...h.rows.map(r => Math.abs(r.score))) || 1;
  let rank = 0;
  const body = h.rows.map(r => {
    const isRef = r.ci === null && r.score === 0;
    const isLead = r.kind === 'llm' && r.score === Math.max(...h.rows.filter(x => x.kind === 'llm').map(x => x.score));
    if (!isRef) rank++;
    const kindLabel = { llm: 'LLM', human: 'Human', econ: 'Econ' }[r.kind] || '';
    return `<tr class="${isRef ? 'ref' : isLead ? 'lead' : ''}">
      <td class="rank">${isRef ? '—' : rank}</td>
      <td><span class="rowname">${r.name}</span><span class="kind ${r.kind}">${kindLabel}</span>
          ${r.note ? `<span class="rownote">${r.note}</span>` : ''}</td>
      <td class="num">${scoreCell(r.score)}</td>
      <td class="num ci">${r.ci ? `[${fmt(r.ci[0])}, ${fmt(r.ci[1])}]` : '—'}</td>
      <td class="num ci">${r.events ?? '—'}</td>
      <td class="barcell">${bar(r.score, max)}</td>
    </tr>`;
  }).join('');

  document.getElementById('lb-window').textContent = h.window;
  document.getElementById('lb-note').textContent = h.note;
  document.getElementById('lb-body').innerHTML = body;
}

function renderAgentDesign(a) {
  document.getElementById('ad-window').textContent = a.window;
  document.getElementById('ad-note').textContent = a.note;
  document.getElementById('ad-body').innerHTML = a.rows.map(r => `
    <tr class="${r.best ? 'lead' : ''}">
      <td><span class="rowname">${r.name}</span>${r.note ? `<span class="rownote">${r.note}</span>` : ''}</td>
      <td class="num">${scoreCell(r.score)}</td>
    </tr>`).join('');
}

function renderIndicators(list) {
  const total = list.reduce((n, t) => n + t.items.length, 0);
  document.getElementById('ind-count').textContent = total;
  document.getElementById('ind-grid').innerHTML = list.map(t => `
    <div class="card">
      <span class="tag">${t.items.length} indicators</span>
      <h4>${t.theme}</h4>
      <p class="blurb">${t.blurb}</p>
      <ul>${t.items.map(i => `<li>${i}</li>`).join('')}</ul>
    </div>`).join('');
}

function renderFed(list) {
  document.getElementById('fed-list').innerHTML =
    list.map(f => `<li>${f.name} <span class="ci">— ${f.target}</span></li>`).join('');
}

fetch('data/leaderboard.json?v=' + Date.now())
  .then(r => r.json())
  .then(d => {
    document.getElementById('last-updated').textContent = d.last_updated;
    document.getElementById('next-update').textContent = d.next_update;
    renderHeadline(d.headline);
    renderAgentDesign(d.agent_design);
    renderIndicators(d.indicators);
    renderFed(d.comparators.fed);
  })
  .catch(e => {
    console.error(e);
    document.getElementById('lb-body').innerHTML =
      '<tr><td colspan="6">Could not load data/leaderboard.json. If you opened this file directly with file://, serve it instead: <code>python3 -m http.server</code></td></tr>';
  });
