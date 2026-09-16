// Minimal App State
let presets = [];
let activePreset = null;

const $ = (id) => document.getElementById(id);

const els = {
  presetSelect: $('preset-select'),
  btnRun: $('btn-run'),
  summaryBar: $('summary-bar'),
  sumSpeedup: $('sum-speedup'),
  sumTimes: $('sum-times'),
  
  // Left: Parallel Constrained
  timerParallel: $('timer-parallel'),
  parallelOutput: $('parallel-output'),
  bodyParallel: $('body-parallel'),
  
  // Right: Naive
  timerNaive: $('timer-naive'),
  streamOutput: $('stream-output'),
  bodyNaive: $('body-naive'),
  badgeNaiveHallucinated: $('badge-naive-hallucinated'),
};

// Synchronized scrolling for side-by-side comparison
let isSyncingNaive = false;
let isSyncingParallel = false;

function setupScrollSync() {
  if (els.bodyNaive && els.bodyParallel) {
    els.bodyNaive.addEventListener('scroll', () => {
      if (isSyncingNaive) return;
      isSyncingParallel = true;
      els.bodyParallel.scrollTop = els.bodyNaive.scrollTop;
      els.bodyParallel.scrollLeft = els.bodyNaive.scrollLeft;
      requestAnimationFrame(() => { isSyncingParallel = false; });
    });
    
    els.bodyParallel.addEventListener('scroll', () => {
      if (isSyncingParallel) return;
      isSyncingNaive = true;
      els.bodyNaive.scrollTop = els.bodyParallel.scrollTop;
      els.bodyNaive.scrollLeft = els.bodyParallel.scrollLeft;
      requestAnimationFrame(() => { isSyncingNaive = false; });
    });
  }
}

function escapeHtml(str) {
  if (typeof str !== 'string') str = String(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function escapeAttr(str) {
  if (typeof str !== 'string') str = String(str);
  return str
    .replace(/&/g, '&amp;')
    .replace(/"/g, '&quot;');
}

function formatValue(val) {
  if (val === null || val === undefined) {
    return '<span class="json-val-null">null</span>';
  }
  if (typeof val === 'boolean') {
    return `<span class="json-val-bool bool-${val}">${val}</span>`;
  }
  if (typeof val === 'number') {
    return `<span class="json-val-num">${val}</span>`;
  }
  if (typeof val === 'string') {
    return `<span class="json-val-str">"${escapeHtml(val)}"</span>`;
  }
  if (Array.isArray(val)) {
    return `<span class="json-punct">[</span>${val.map(formatValue).join('<span class="json-punct">, </span>')}<span class="json-punct">]</span>`;
  }
  return `<span class="json-val-str">${escapeHtml(JSON.stringify(val))}</span>`;
}

// Single-line JSON formatters so keys line up line-for-line across both panels
function formatParallelJson(obj) {
  if (!obj || typeof obj !== 'object') return '<div class="json-row"><span class="json-punct">{ }</span></div>';
  const schemaKeys = activePreset && activePreset.schema ? Object.keys(activePreset.schema) : Object.keys(obj);
  const keys = schemaKeys.filter(k => k in obj);
  Object.keys(obj).forEach(k => {
    if (!keys.includes(k)) keys.push(k);
  });
  
  const lines = ['<div class="json-row"><span class="json-punct">{</span></div>'];
  keys.forEach((key, idx) => {
    const isLast = idx === keys.length - 1;
    const comma = isLast ? '' : '<span class="json-punct">,</span>';
    const val = obj[key];
    
    let inner = '';
    if (val && typeof val === 'object' && 'value' in val && 'prob' in val) {
      const valHtml = formatValue(val.value);
      const probNum = typeof val.prob === 'number' ? val.prob.toFixed(4) : escapeHtml(val.prob);
      inner = `  <span class="json-key">"${escapeHtml(key)}"</span><span class="json-punct">: { </span><span class="json-sub">"value"</span><span class="json-punct">: </span>${valHtml}<span class="json-punct">, </span><span class="json-sub">"prob"</span><span class="json-punct">: </span><span class="json-val-prob">${probNum}</span><span class="json-punct"> }</span>${comma}`;
    } else {
      inner = `  <span class="json-key">"${escapeHtml(key)}"</span><span class="json-punct">: </span>${formatValue(val)}${comma}`;
    }
    
    lines.push(`<div class="json-row" data-key="${escapeHtml(key)}">${inner}</div>`);
  });
  lines.push('<div class="json-row"><span class="json-punct">}</span></div>');
  return lines.join('');
}

function formatNaiveJson(obj) {
  if (!obj || typeof obj !== 'object') return '<div class="json-row"><span class="json-punct">{ }</span></div>';
  const schemaKeys = activePreset && activePreset.schema ? Object.keys(activePreset.schema) : Object.keys(obj);
  const keys = schemaKeys.filter(k => k in obj);
  Object.keys(obj).forEach(k => {
    if (!keys.includes(k)) keys.push(k);
  });
  
  const lines = ['<div class="json-row"><span class="json-punct">{</span></div>'];
  keys.forEach((key, idx) => {
    const isLast = idx === keys.length - 1;
    const comma = isLast ? '' : '<span class="json-punct">,</span>';
    const val = obj[key];
    const inner = `  <span class="json-key">"${escapeHtml(key)}"</span><span class="json-punct">: </span>${formatValue(val)}${comma}`;
    lines.push(`<div class="json-row" data-key="${escapeHtml(key)}">${inner}</div>`);
  });
  lines.push('<div class="json-row"><span class="json-punct">}</span></div>');
  return lines.join('');
}

function highlightStreamJson(text) {
  if (!text) return '';
  const lines = text.split('\n');
  const rendered = lines.map((line, idx) => {
    const isLastLine = idx === lines.length - 1;
    let lineHtml = escapeHtml(line);
    
    if (lineHtml.includes('":')) {
      lineHtml = lineHtml.replace(/"([^"]+)"(\s*:)/g, '<span class="json-key">"$1"</span>$2');
    } else if (isLastLine && /^(\s*)"([^"]*)$/.test(lineHtml)) {
      lineHtml = lineHtml.replace(/^(\s*)"([^"]*)$/, '$1<span class="json-key">"$2</span>');
    } else if (isLastLine && /^(\s*)"([^"]+)"(\s*)$/.test(lineHtml)) {
      lineHtml = lineHtml.replace(/^(\s*)"([^"]+)"(\s*)$/, '$1<span class="json-key">"$2"</span>$3');
    }
    
    return lineHtml;
  });
  
  return rendered.join('\n');
}

function setupRowHoverSync() {
  document.addEventListener('mouseover', (e) => {
    const row = e.target.closest('.json-row');
    if (row && row.dataset.key) {
      const key = row.dataset.key;
      document.querySelectorAll(`.json-row[data-key="${CSS.escape(key)}"]`).forEach(el => {
        el.classList.add('row-hover');
      });
    }
  });

  document.addEventListener('mouseout', (e) => {
    const row = e.target.closest('.json-row');
    if (row && row.dataset.key) {
      const key = row.dataset.key;
      document.querySelectorAll(`.json-row[data-key="${CSS.escape(key)}"]`).forEach(el => {
        el.classList.remove('row-hover');
      });
    }
  });
}

// Initialize
async function init() {
  setupScrollSync();
  setupRowHoverSync();
  try {
    const res = await fetch('/api/presets');
    presets = await res.json();
    
    els.presetSelect.innerHTML = '';
    presets.forEach((p) => {
      const opt = document.createElement('option');
      opt.value = p.id;
      opt.textContent = `${p.title}`;
      els.presetSelect.appendChild(opt);
    });
    
    if (presets.length > 0) {
      activePreset = presets[0];
    }
  } catch (e) {
    console.error('Error loading presets:', e);
  }

  els.presetSelect.addEventListener('change', (e) => {
    activePreset = presets.find(p => p.id === e.target.value);
    reset();
  });

  els.btnRun.addEventListener('click', runComparison);
}

function reset() {
  els.timerNaive.textContent = '0.0 ms';
  els.timerParallel.textContent = '0.0 ms';
  els.streamOutput.innerHTML = '<span class="placeholder-text">Click "Run Comparison" to start...</span>';
  els.parallelOutput.innerHTML = '<span class="placeholder-text">Click "Run Comparison" to start...</span>';
  els.summaryBar.classList.add('hidden');
  if (els.badgeNaiveHallucinated) {
    els.badgeNaiveHallucinated.classList.add('hidden');
    els.badgeNaiveHallucinated.textContent = '';
  }
}

// Execute Parallel Constrained Generation
async function runParallel(payload) {
  els.parallelOutput.innerHTML = '<span class="placeholder-text">// Evaluating parallel forward pass across all schema fields...</span>';
  
  const t0 = performance.now();
  let running = true;
  function tickParallel() {
    if (!running) return;
    const ms = performance.now() - t0;
    els.timerParallel.textContent = `${ms.toFixed(1)} ms`;
    requestAnimationFrame(tickParallel);
  }
  requestAnimationFrame(tickParallel);
  
  try {
    const res = await fetch('/api/run-parallel', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await res.json();
    running = false;
    const elapsed = data.elapsed_ms || (performance.now() - t0);
    els.timerParallel.textContent = `${elapsed.toFixed(1)} ms`;
    renderParallel(data);
    return data;
  } catch (err) {
    running = false;
    throw err;
  }
}

function renderParallel(data) {
  if (data.parsed_json) {
    els.parallelOutput.innerHTML = formatParallelJson(data.parsed_json);
  }
}

// Stream Naive
async function streamNaive(payload) {
  els.streamOutput.textContent = '';
  
  const t0 = performance.now();
  let running = true;
  function tickNaive() {
    if (!running) return;
    const ms = performance.now() - t0;
    els.timerNaive.textContent = `${ms.toFixed(1)} ms`;
    requestAnimationFrame(tickNaive);
  }
  requestAnimationFrame(tickNaive);
  
  let tokenCount = 0;
  let finalResult = null;
  
  try {
    const response = await fetch('/api/stream-naive', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let text = '';
    
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      
      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');
      
      for (const line of lines) {
        if (line.startsWith('data: ')) {
          try {
            const ev = JSON.parse(line.substring(6));
            if (ev.type === 'token') {
              text += ev.token;
              tokenCount = ev.token_count;
              els.streamOutput.innerHTML = highlightStreamJson(text);
              els.streamOutput.scrollTop = els.streamOutput.scrollHeight;
            } else if (ev.type === 'done') {
              finalResult = ev.result;
            }
          } catch (e) {}
        }
      }
    }
  } finally {
    running = false;
  }
  
  if (finalResult) {
    els.timerNaive.textContent = `${finalResult.elapsed_ms.toFixed(1)} ms`;
    let parsedObj = finalResult.parsed_json;
    if (parsedObj) {
      els.streamOutput.innerHTML = formatNaiveJson(parsedObj);
    } else {
      try {
        const cleaned = text.substring(text.indexOf('{'), text.lastIndexOf('}') + 1);
        parsedObj = JSON.parse(cleaned);
        els.streamOutput.innerHTML = formatNaiveJson(parsedObj);
      } catch (e) {
        els.streamOutput.innerHTML = highlightStreamJson(text);
      }
    }

    // Check for missing / hallucinated fields
    const expectedKeys = activePreset && activePreset.schema ? Object.keys(activePreset.schema) : [];
    let missingCount = 0;
    if (parsedObj && typeof parsedObj === 'object') {
      const parsedKeys = Object.keys(parsedObj);
      const missing = expectedKeys.filter(k => !parsedKeys.includes(k));
      const extra = parsedKeys.filter(k => !expectedKeys.includes(k));
      const invalidEnums = finalResult.invalid_enums || [];
      missingCount = missing.length + extra.length + invalidEnums.length;
    } else {
      missingCount = expectedKeys.length;
    }

    if (missingCount > 0 && els.badgeNaiveHallucinated) {
      els.badgeNaiveHallucinated.textContent = `${missingCount} field${missingCount > 1 ? 's' : ''} hallucinated`;
      els.badgeNaiveHallucinated.classList.remove('hidden');
    }
  }
  
  return finalResult;
}

// Side-by-Side Comparison Runner
async function runComparison() {
  if (!activePreset) return;
  
  reset();
  els.btnRun.disabled = true;
  els.btnRun.textContent = 'Running...';
  
  const payload = {
    context: activePreset.context,
    schema: activePreset.schema
  };
  
  try {
    // 1. Run Parallel Constrained inference first
    const parallelData = await runParallel(payload);
    
    // 2. Stream Naive right after
    const naiveData = await streamNaive(payload);
    
    // 3. Update summary pill
    if (parallelData && naiveData) {
      const speedup = (naiveData.elapsed_ms / Math.max(parallelData.elapsed_ms, 1.0)).toFixed(1);
      els.sumSpeedup.textContent = `${speedup}x FASTER`;
      els.sumTimes.textContent = `${parallelData.elapsed_ms.toFixed(1)} ms vs ${naiveData.elapsed_ms.toFixed(1)} ms`;
      els.summaryBar.classList.remove('hidden');
    }
  } catch (err) {
    console.error('Run failed:', err);
  } finally {
    els.btnRun.disabled = false;
    els.btnRun.innerHTML = '<span>⚡ Run Comparison</span>';
  }
}

document.addEventListener('DOMContentLoaded', init);
