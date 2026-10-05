/**
 * app.js — Smart Vacuum RL Frontend
 * Communicates with FastAPI backend; handles:
 *   - Environment loading (presets, random)
 *   - Training (POST /api/train)
 *   - Agent evaluation (random & trained)
 *   - Step-by-step animation playback
 *   - Chart rendering (Chart.js)
 *   - Q-table inspection
 *   - Comparison table
 */

/* ═══════════════════════════════════════════════════════════════════════
   CONSTANTS & STATE
   ═══════════════════════════════════════════════════════════════════════ */
const API = '';          // Same-origin; FastAPI serves frontend at /
const ACTIONS = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'CLEAN'];

const state = {
  envConfig:       null,   // current grid config from server
  trainingResults: null,   // results from /api/train
  randomResult:    null,   // from /api/test/random
  trainedResult:   null,   // from /api/test/trained
  animTimer:       null,   // setInterval handle
  animTrace:       [],     // step-by-step frames
  animIndex:       0,
  animPaused:      false,
  isTrained:       false,
  chart:           null,
  activeChart:     'reward',
};

/* ═══════════════════════════════════════════════════════════════════════
   INIT
   ═══════════════════════════════════════════════════════════════════════ */
document.addEventListener('DOMContentLoaded', async () => {
  setupSpeedSlider();
  await loadPreset('simple');   // default room on load
});

/* ═══════════════════════════════════════════════════════════════════════
   ENVIRONMENT
   ═══════════════════════════════════════════════════════════════════════ */
async function loadPreset(name) {
  try {
    const res  = await apiFetch(`/api/environment/preset/${name}`);
    state.envConfig = res.config;
    renderGrid(res.env.grid, res.env.vacuum_pos);
    updatePositionLabel(res.env.vacuum_pos);
    updateActionLabel(null);
    clearStats();
    toast(`Loaded: ${res.preset.name}`, 'info');
    // Highlight active preset button
    ['simple','medium','complex'].forEach(p => {
      document.getElementById(`btn-preset-${p}`)?.classList.toggle('active', p === name);
    });
  } catch(e) {
    toast(`Failed to load preset: ${e.message}`, 'error');
  }
}

async function generateRandom() {
  const size     = intVal('cfg-size',  5);
  const nDirty   = intVal('cfg-dirty', 5);
  const nObs     = intVal('cfg-obs',   3);

  try {
    const res = await apiFetch('/api/environment/random', 'POST', {
      size, n_dirty: nDirty, n_obstacles: nObs
    });
    state.envConfig = res.config;
    renderGrid(res.env.grid, res.env.vacuum_pos);
    updatePositionLabel(res.env.vacuum_pos);
    updateActionLabel(null);
    clearStats();
    toast('Random room generated!', 'success');
    ['simple','medium','complex'].forEach(p =>
      document.getElementById(`btn-preset-${p}`)?.classList.remove('active'));
  } catch(e) {
    toast(`Error: ${e.message}`, 'error');
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   TRAINING
   ═══════════════════════════════════════════════════════════════════════ */
async function trainAgent() {
  const btn = document.getElementById('btn-train');
  btn.disabled = true;
  btn.textContent = '⏳ Training…';
  showLog(true);

  const payload = {
    env_config:      state.envConfig,
    episodes:        intVal('hp-episodes',  3000),
    learning_rate:   floatVal('hp-lr',      0.1),
    discount_factor: floatVal('hp-gamma',   0.9),
    epsilon:         floatVal('hp-eps',     1.0),
    epsilon_min:     floatVal('hp-eps-min', 0.01),
    epsilon_decay:   floatVal('hp-decay',   0.995),
    max_steps:       intVal('hp-maxsteps',  200),
  };

  appendLog(`▶ Starting training for ${payload.episodes} episodes…`);

  try {
    const res = await apiFetch('/api/train', 'POST', payload);
    appendLog(`✓ Training complete!`);
    appendLog(`  Best reward: ${res.best_reward} at episode ${res.best_episode}`);
    appendLog(`  Final ε: ${res.final_epsilon}`);
    appendLog(`  Q-table states: ${res.q_table_size}`);
    appendLog(`  Total Q-updates: ${res.total_q_updates}`);

    // Fetch full results for charts
    const fullRes = await apiFetch('/api/training/results');
    state.trainingResults = fullRes;
    state.isTrained = true;

    updateTrainedBadge(true);
    document.getElementById('btn-run-trained').disabled = false;

    // Update stat cards with final-episode info
    setStatCard('stat-episode', res.episodes);
    setStatCard('stat-epsilon', res.final_epsilon);
    setStatCard('stat-best-reward', res.best_reward);
    const lastReward = fullRes.episode_rewards.at(-1);
    setStatCard('stat-reward', lastReward);

    // Charts
    drawChart(state.activeChart);
    hidePlaceholder();

    // Q-table summary
    await refreshQTableSummary();

    toast(`Training done! ${res.episodes} episodes · ${res.q_table_size} states learned`, 'success');
  } catch(e) {
    appendLog(`✗ Error: ${e.message}`);
    toast(`Training failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '🧠 Train Agent';
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   EVALUATION — Random Agent
   ═══════════════════════════════════════════════════════════════════════ */
async function runRandom() {
  if (!state.envConfig) { toast('Load a room first.', 'error'); return; }
  stopAnimation();

  const btn = document.getElementById('btn-run-random');
  btn.disabled = true; btn.textContent = '⏳ Running…';

  try {
    const res = await apiFetch('/api/test/random', 'POST', {
      env_config: state.envConfig,
      max_steps:  intVal('hp-maxsteps', 200),
    });
    state.randomResult = res;
    toast(`Random agent: ${res.completion_pct}% cleaned in ${res.steps} steps`, 'info');
    playTrace(res.steps_trace, 'Random Agent');
    tryShowComparison();
  } catch(e) {
    toast(`Error: ${e.message}`, 'error');
  } finally {
    btn.disabled = false; btn.textContent = '🎲 Run Random Agent';
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   EVALUATION — Trained Agent
   ═══════════════════════════════════════════════════════════════════════ */
async function runTrained() {
  if (!state.isTrained) { toast('Train the agent first!', 'error'); return; }
  stopAnimation();

  const btn = document.getElementById('btn-run-trained');
  btn.disabled = true; btn.textContent = '⏳ Running…';

  try {
    const res = await apiFetch('/api/test/trained', 'POST', {
      env_config: state.envConfig,
      max_steps:  intVal('hp-maxsteps', 200),
    });
    state.trainedResult = res;
    toast(`Trained agent: ${res.completion_pct}% cleaned in ${res.steps} steps`, 'success');
    playTrace(res.steps_trace, 'Trained Agent');
    tryShowComparison();
  } catch(e) {
    toast(`Error: ${e.message}`, 'error');
  } finally {
    btn.disabled = false; btn.textContent = '▶ Run Trained Agent';
  }
}

/* ═══════════════════════════════════════════════════════════════════════
   ANIMATION
   ═══════════════════════════════════════════════════════════════════════ */
function playTrace(trace, label) {
  stopAnimation();
  state.animTrace  = trace;
  state.animIndex  = 0;
  state.animPaused = false;
  appendLog(`\n▶ Playing ${label} trace (${trace.length} steps)…`);
  showLog(true);

  const delay = () => parseInt(document.getElementById('anim-speed').value) || 500;

  function step() {
    if (state.animPaused || state.animIndex >= state.animTrace.length) {
      if (state.animIndex >= state.animTrace.length) {
        const last = state.animTrace.at(-1);
        appendLog(`✓ Done — Completion: ${last.completion_pct}% | Collisions: ${last.collisions_so_far}`);
      }
      return;
    }

    const frame = state.animTrace[state.animIndex];
    renderGridFromFrame(frame);
    updateStatsFromFrame(frame);
    if (frame.action_name) updateActionLabel(frame.action_name);

    // Q-table for current state if trained
    if (state.isTrained && frame.step > 0) {
      fetchQValues(frame.vacuum_pos[0], frame.vacuum_pos[1], frame.dirty_cells);
    }

    state.animIndex++;
    state.animTimer = setTimeout(step, delay());
  }

  step();
  document.getElementById('btn-stop').disabled   = false;
  document.getElementById('btn-resume').disabled  = true;
}

function stopAnimation() {
  if (state.animTimer) { clearTimeout(state.animTimer); state.animTimer = null; }
  state.animPaused = true;
  document.getElementById('btn-stop').disabled   = false;
  document.getElementById('btn-resume').disabled  = false;
}

function resumeAnimation() {
  if (!state.animTrace.length) return;
  state.animPaused = false;
  document.getElementById('btn-resume').disabled = true;
  // re-trigger from current index
  playTrace(state.animTrace.slice(state.animIndex), 'Resumed');
  state.animIndex = 0; // playTrace reset — we sliced already
}

/* ═══════════════════════════════════════════════════════════════════════
   GRID RENDERING
   ═══════════════════════════════════════════════════════════════════════ */
function renderGrid(grid, vacuumPos) {
  const container = document.getElementById('grid-container');
  const size = grid.length;
  container.style.gridTemplateColumns = `repeat(${size}, 52px)`;
  container.innerHTML = '';

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const cell    = document.createElement('div');
      const cellType = grid[r][c];
      cell.className = `grid-cell cell-${cellType}`;
      cell.id        = `cell-${r}-${c}`;
      cell.title     = `(${r},${c}) — ${cellType}`;
      cell.dataset.row = r;
      cell.dataset.col = c;

      // Cell icons
      if (cellType === 'vacuum')   cell.textContent = '🤖';
      else if (cellType === 'dirty')    cell.textContent = '🟫';
      else if (cellType === 'obstacle') cell.textContent = '⬛';
      else cell.textContent = '';

      // Click to inspect Q-values
      cell.addEventListener('click', () => {
        if (state.isTrained) {
          fetchQValues(r, c, getCurrentDirtyCells());
        }
      });

      container.appendChild(cell);
    }
  }
}

function renderGridFromFrame(frame) {
  const grid = frame.grid;
  const size  = grid.length;
  const container = document.getElementById('grid-container');

  if (!container.children.length || container.style.gridTemplateColumns !== `repeat(${size}, 52px)`) {
    renderGrid(grid, frame.vacuum_pos);
    return;
  }

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const cell     = document.getElementById(`cell-${r}-${c}`);
      if (!cell) continue;
      const cellType = grid[r][c];
      const prevType = cell.className.replace('grid-cell cell-', '').split(' ')[0];

      if (cellType !== prevType || frame.cleaned) {
        cell.className = `grid-cell cell-${cellType}`;

        // Animation effects
        if (cellType === 'vacuum' && frame.collision) {
          cell.classList.add('cell-collision-flash');
          setTimeout(() => cell.classList.remove('cell-collision-flash'), 500);
        }

        if (frame.cleaned && prevType === 'dirty' && cellType === 'vacuum') {
          cell.classList.add('just-cleaned');
          setTimeout(() => cell.classList.remove('just-cleaned'), 600);
        }

        if (prevType === 'dirty' && cellType === 'clean') {
          cell.classList.add('cell-cleaning-flash');
          setTimeout(() => cell.classList.remove('cell-cleaning-flash'), 500);
        }
      }

      // Icon
      if (cellType === 'vacuum')        cell.textContent = '🤖';
      else if (cellType === 'dirty')    cell.textContent = '🟫';
      else if (cellType === 'obstacle') cell.textContent = '⬛';
      else                              cell.textContent = '';
    }
  }
}

function getCurrentDirtyCells() {
  const cells = [];
  document.querySelectorAll('.grid-cell.cell-dirty').forEach(el => {
    cells.push([parseInt(el.dataset.row), parseInt(el.dataset.col)]);
  });
  return cells;
}

/* ═══════════════════════════════════════════════════════════════════════
   STATS
   ═══════════════════════════════════════════════════════════════════════ */
function updateStatsFromFrame(frame) {
  setStatCard('stat-steps',      frame.step);
  setStatCard('stat-reward',     frame.cumulative_reward);
  setStatCard('stat-collisions', frame.collisions_so_far);
  setStatCard('stat-completion', frame.completion_pct + '%');
  updateProgressBar(frame.completion_pct);
  updatePositionLabel(frame.vacuum_pos);
}

function clearStats() {
  ['stat-episode','stat-reward','stat-avg-reward','stat-best-reward',
   'stat-steps','stat-collisions','stat-epsilon','stat-completion'].forEach(id => {
    setStatCard(id, '—');
  });
  updateProgressBar(0);
  document.getElementById('action-label').textContent = 'Action: —';
  document.getElementById('pos-label').textContent    = 'Position: —';
}

function setStatCard(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}

function updateProgressBar(pct) {
  const bar = document.getElementById('progress-bar');
  if (bar) bar.style.width = Math.min(100, pct) + '%';
}

function updateActionLabel(action) {
  const el = document.getElementById('action-label');
  if (!el) return;
  if (!action) { el.textContent = 'Action: —'; return; }
  const icons = { UP:'↑', DOWN:'↓', LEFT:'←', RIGHT:'→', CLEAN:'🧹', START:'•' };
  el.textContent = `Action: ${icons[action] || ''} ${action}`;
}

function updatePositionLabel(pos) {
  const el = document.getElementById('pos-label');
  if (el && pos) el.textContent = `Position: (${pos[0]}, ${pos[1]})`;
}

/* ═══════════════════════════════════════════════════════════════════════
   CHARTS
   ═══════════════════════════════════════════════════════════════════════ */
function switchChart(type) {
  state.activeChart = type;
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.getElementById(`tab-${type}`)?.classList.add('active');
  if (state.trainingResults) drawChart(type);
}

function drawChart(type) {
  const res = state.trainingResults;
  if (!res) return;

  const canvas = document.getElementById('training-chart');
  const n      = res.episode_rewards.length;

  // Sample every K episodes for performance
  const K = Math.max(1, Math.floor(n / 300));

  let raw, avg, label, color;
  switch(type) {
    case 'reward':
      raw   = sample(res.episode_rewards,     K);
      avg   = sample(res.rolling_avg_rewards,  K);
      label = 'Episode Reward';
      color = '#6c63ff';
      break;
    case 'steps':
      raw   = sample(res.episode_steps,       K);
      avg   = sample(res.rolling_avg_steps,   K);
      label = 'Steps per Episode';
      color = '#38bdf8';
      break;
    case 'completion':
      raw   = sample(res.episode_completion,  K);
      avg   = movingAvg(sample(res.episode_completion, K), 20);
      label = 'Completion %';
      color = '#22d3a0';
      break;
    case 'epsilon':
      raw   = sample(res.episode_epsilons,    K);
      avg   = raw;
      label = 'Epsilon (ε)';
      color = '#fbbf24';
      break;
  }

  const labels = raw.map((_, i) => (i * K) + 1);

  if (state.chart) { state.chart.destroy(); }

  state.chart = new Chart(canvas, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label:           label,
          data:            raw,
          borderColor:     color + '60',
          backgroundColor: 'transparent',
          pointRadius:     0,
          borderWidth:     1,
          tension:         0.3,
        },
        {
          label:           `${label} (100-ep avg)`,
          data:            avg,
          borderColor:     color,
          backgroundColor: color + '18',
          pointRadius:     0,
          borderWidth:     2,
          tension:         0.4,
          fill:            true,
        },
      ]
    },
    options: {
      responsive:          true,
      maintainAspectRatio: false,
      animation:           { duration: 600 },
      plugins: {
        legend: { labels: { color: '#8892b0', font: { size: 11 } } },
        tooltip: {
          backgroundColor: '#141728',
          borderColor:     '#2a2f4a',
          borderWidth:     1,
          titleColor:      '#e8eaf6',
          bodyColor:       '#8892b0',
        }
      },
      scales: {
        x: {
          ticks: { color: '#4a5568', maxTicksLimit: 8 },
          grid:  { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: 'Episode', color: '#8892b0' }
        },
        y: {
          ticks: { color: '#4a5568' },
          grid:  { color: 'rgba(255,255,255,0.04)' },
          title: { display: true, text: label, color: '#8892b0' }
        }
      }
    }
  });
}

function hidePlaceholder() {
  const ph = document.getElementById('chart-placeholder');
  if (ph) ph.style.display = 'none';
}

/* ═══════════════════════════════════════════════════════════════════════
   COMPARISON
   ═══════════════════════════════════════════════════════════════════════ */
function tryShowComparison() {
  if (!state.randomResult || !state.trainedResult) return;

  const R = state.randomResult;
  const T = state.trainedResult;

  const tbody = document.getElementById('compare-tbody');
  tbody.innerHTML = '';

  const rows = [
    ['Total Reward',   R.total_reward,  T.total_reward,  false],
    ['Steps',          R.steps,         T.steps,         false],
    ['Completion %',   R.completion_pct, T.completion_pct, false],
    ['Collisions',     R.collisions,    T.collisions,    false],
    ['Efficiency',     calcEfficiency(R), calcEfficiency(T), false],
  ];

  rows.forEach(([metric, rVal, tVal]) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `<td>${metric}</td><td>${rVal}</td><td>${tVal}</td>`;
    tbody.appendChild(tr);
  });

  // Bar visualisation
  const barsEl = document.getElementById('compare-bars');
  barsEl.innerHTML = '';

  const barMetrics = [
    { label: 'Total Reward', r: R.total_reward, t: T.total_reward, higherBetter: true },
    { label: 'Completion %', r: R.completion_pct, t: T.completion_pct, higherBetter: true },
    { label: 'Collisions',   r: R.collisions,    t: T.collisions,    higherBetter: false },
  ];

  barMetrics.forEach(({ label, r, t, higherBetter }) => {
    const maxVal = Math.max(Math.abs(r), Math.abs(t), 1);
    const rPct   = Math.min(100, Math.max(0, (Math.abs(r) / maxVal) * 100));
    const tPct   = Math.min(100, Math.max(0, (Math.abs(t) / maxVal) * 100));

    const el = document.createElement('div');
    el.className = 'compare-bar-row';
    el.innerHTML = `
      <div class="compare-bar-label">
        <span>${label}</span>
        <span>🎲 ${r} &nbsp;|&nbsp; 🧠 ${t}</span>
      </div>
      <div class="compare-bar-track">
        <div class="compare-bar-fill bar-random"  style="width:${rPct}%"></div>
      </div>
      <div class="compare-bar-track">
        <div class="compare-bar-fill bar-trained" style="width:${tPct}%"></div>
      </div>
    `;
    barsEl.appendChild(el);
  });

  document.getElementById('comparison-placeholder').style.display = 'none';
  document.getElementById('comparison-table').style.display       = 'block';
}

function calcEfficiency(result) {
  return (result.completion_pct / Math.max(result.steps, 1)).toFixed(4);
}

/* ═══════════════════════════════════════════════════════════════════════
   Q-TABLE
   ═══════════════════════════════════════════════════════════════════════ */
async function refreshQTableSummary() {
  try {
    const res = await apiFetch('/api/q-table/summary');
    const el  = document.getElementById('qtable-summary');
    if (!el) return;
    el.innerHTML = `
      <strong>States learned:</strong> ${res.q_table_size}<br>
      <strong>Total Q-updates:</strong> ${res.total_updates}<br>
      <strong>Final ε:</strong> ${res.current_epsilon}<br>
      <strong>α / γ:</strong> ${res.hyperparams.learning_rate} / ${res.hyperparams.discount_factor}
    `;
    document.getElementById('qtable-placeholder').style.display = 'none';
    document.getElementById('qtable-viewer').style.display      = 'block';
  } catch(_) {}
}

async function fetchQValues(row, col, dirtyCells) {
  if (!state.isTrained) return;
  try {
    const res = await apiFetch('/api/q-table/state', 'POST', {
      row, col, dirty_cells: dirtyCells
    });
    renderQValues(res);
  } catch(_) {}
}

function renderQValues(info) {
  const container = document.getElementById('qtable-values');
  if (!container) return;

  const qVals   = info.q_values;
  const best    = info.best_action;
  const allVals = Object.values(qVals);
  const maxAbs  = Math.max(...allVals.map(Math.abs), 1);

  container.innerHTML = '';

  ACTIONS.forEach(action => {
    const val     = qVals[action] ?? 0;
    const pct     = Math.min(100, (Math.abs(val) / maxAbs) * 100);
    const isBest  = action === best;
    const isNeg   = val < 0;

    const row = document.createElement('div');
    row.className = 'qval-row';
    row.innerHTML = `
      <span class="qval-action ${isBest ? 'qval-best' : ''}">${action}</span>
      <div class="qval-bar-track">
        <div class="qval-bar-fill ${isNeg ? 'negative' : ''}" style="width:${pct}%"></div>
      </div>
      <span class="qval-num ${isBest ? 'qval-best' : ''}">${val.toFixed(3)}${isBest ? '<span class="best-action-tag">BEST</span>' : ''}</span>
    `;
    container.appendChild(row);
  });

  // Update hint
  const hint = document.getElementById('qtable-hint');
  if (hint) hint.textContent = `State: (${info.state}) → Best: ${best}`;
}

/* ═══════════════════════════════════════════════════════════════════════
   RESET
   ═══════════════════════════════════════════════════════════════════════ */
async function resetEnvironment() {
  stopAnimation();
  try {
    await apiFetch('/api/reset/environment', 'POST', {});
    await loadPreset('simple');
    clearStats();
    toast('Environment reset.', 'info');
  } catch(e) { toast(e.message, 'error'); }
}

async function resetTraining() {
  stopAnimation();
  try {
    await apiFetch('/api/reset/training', 'POST', {});
    state.trainingResults = null;
    state.randomResult    = null;
    state.trainedResult   = null;
    state.isTrained       = false;
    updateTrainedBadge(false);
    document.getElementById('btn-run-trained').disabled = true;
    document.getElementById('qtable-placeholder').style.display = 'block';
    document.getElementById('qtable-viewer').style.display      = 'none';
    document.getElementById('comparison-placeholder').style.display = 'block';
    document.getElementById('comparison-table').style.display       = 'none';
    document.getElementById('chart-placeholder').style.display      = 'flex';
    if (state.chart) { state.chart.destroy(); state.chart = null; }
    clearStats();
    if (state.trainingResults) showLog(false);
    toast('Training state cleared.', 'info');
  } catch(e) { toast(e.message, 'error'); }
}

/* ═══════════════════════════════════════════════════════════════════════
   SAVE / LOAD
   ═══════════════════════════════════════════════════════════════════════ */
async function saveQTable() {
  if (!state.isTrained) { toast('Train the agent first.', 'error'); return; }
  try {
    const res = await apiFetch('/api/q-table/save', 'POST', {});
    toast(`Q-table saved! ${res.message}`, 'success');
  } catch(e) { toast(e.message, 'error'); }
}

async function loadQTable() {
  try {
    const res = await apiFetch('/api/q-table/load', 'POST', {});
    state.isTrained = true;
    updateTrainedBadge(true);
    document.getElementById('btn-run-trained').disabled = false;
    await refreshQTableSummary();
    toast(`Q-table loaded — ${res.q_table_size} states`, 'success');
  } catch(e) { toast(`Could not load: ${e.message}`, 'error'); }
}

/* ═══════════════════════════════════════════════════════════════════════
   UI HELPERS
   ═══════════════════════════════════════════════════════════════════════ */
function setupSpeedSlider() {
  const slider = document.getElementById('anim-speed');
  const label  = document.getElementById('speed-label');
  slider.addEventListener('input', () => {
    label.textContent = slider.value + 'ms';
  });
}

function showLog(show) {
  const card = document.getElementById('training-log-card');
  if (card) card.style.display = show ? 'block' : 'none';
}

function appendLog(msg) {
  const box = document.getElementById('training-log');
  if (!box) return;
  box.textContent += msg + '\n';
  box.scrollTop = box.scrollHeight;
}

function updateTrainedBadge(trained) {
  const badge = document.getElementById('badge-trained');
  if (!badge) return;
  if (trained) {
    badge.textContent = '● Agent Trained';
    badge.className   = 'badge badge-accent';
  } else {
    badge.textContent = '○ Not Trained';
    badge.className   = 'badge badge-blue';
  }
}

function toast(msg, type = 'info') {
  const el = document.getElementById('toast');
  if (!el) return;
  el.textContent = msg;
  el.className   = `toast toast-${type} show`;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => el.classList.remove('show'), 3500);
}

/* ═══════════════════════════════════════════════════════════════════════
   API FETCH WRAPPER
   ═══════════════════════════════════════════════════════════════════════ */
async function apiFetch(endpoint, method = 'GET', body = null) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body !== null) opts.body = JSON.stringify(body);

  const res = await fetch(API + endpoint, opts);
  if (!res.ok) {
    let detail = `HTTP ${res.status}`;
    try {
      const err = await res.json();
      if (err.detail) {
        detail = typeof err.detail === 'string'
          ? err.detail
          : JSON.stringify(err.detail);
      }
    } catch(_) {}
    throw new Error(detail);
  }
  return res.json();
}

/* ═══════════════════════════════════════════════════════════════════════
   UTILITY
   ═══════════════════════════════════════════════════════════════════════ */
function intVal(id, def)   { return parseInt(document.getElementById(id)?.value) || def; }
function floatVal(id, def) { return parseFloat(document.getElementById(id)?.value) || def; }

function sample(arr, k) {
  if (k <= 1) return arr;
  return arr.filter((_, i) => i % k === 0);
}

function movingAvg(arr, w) {
  return arr.map((_, i) => {
    const sl = arr.slice(Math.max(0, i - w + 1), i + 1);
    return sl.reduce((a, b) => a + b, 0) / sl.length;
  });
}
