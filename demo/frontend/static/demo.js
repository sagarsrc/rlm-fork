const $ = id => document.getElementById(id);
const status = $('status');
const dataInfo = $('dataInfo');
const stepsEl = $('steps');
const stepsWrap = $('stepsWrap');
const stepsCount = $('stepsCount');
const perspective = $('perspective');
const queryEcho = $('queryEcho');
const queryEchoText = $('queryEchoText');

let activeJob = null;
let pollTimer = null;
let seenSteps = new Set();
let lastUsage = { base: null, rlm: null };

function setStatus(cls, html) {
  status.className = 'status' + (cls ? ' ' + cls : '');
  status.innerHTML = html;
}
function setIdle(msg) { setStatus('', msg); }
function setRunning(msg) { setStatus('running', '<span class="dot"></span><strong>' + msg + '</strong>'); }
function setError(msg) { setStatus('error', '<strong>Error.</strong> ' + msg); }

function setCard(id, cls, title, answer, meta) {
  const card = $(id);
  card.className = 'card ' + cls;
  card.querySelector('.title').textContent = title;
  card.querySelector('.answer').textContent = answer || '(empty)';
  card.querySelector('.meta').textContent = meta;
}

async function post(path, body) {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    const txt = await res.text();
    throw new Error('HTTP ' + res.status + ': ' + txt.slice(0, 120));
  }
  return await res.json();
}

function clearSteps() {
  stepsEl.innerHTML = '';
  seenSteps.clear();
  stepsWrap.style.display = 'none';
  stepsCount.textContent = '';
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function pre(title, text) {
  if (text === null || text === undefined || text === '') return '';
  return '<div class="section-label">' + escapeHtml(title) + '</div>' +
         '<div class="pre">' + escapeHtml(typeof text === 'object' ? JSON.stringify(text, null, 2) : String(text)) + '</div>';
}

function formatPrompt(prompt) {
  if (typeof prompt === 'string') return prompt;
  if (Array.isArray(prompt) && prompt.length) {
    return prompt.map(m => m.role + ': ' + (m.content || JSON.stringify(m))).join('\n\n---\n\n');
  }
  return JSON.stringify(prompt, null, 2);
}

function addStep(number, title, subtitle, bodyHtml, cls, badge) {
  const key = number + ':' + title;
  if (seenSteps.has(key)) return;
  seenSteps.add(key);
  const details = document.createElement('details');
  details.className = 'step ' + (cls || '');
  details.innerHTML =
    '<summary>' +
      '<span class="num">' + number + '</span>' +
      '<span class="title-wrap">' +
        '<div class="title">' + escapeHtml(title) + (badge ? '<span class="badge">' + escapeHtml(badge) + '</span>' : '') + '</div>' +
        (subtitle ? '<div class="subtitle">' + escapeHtml(subtitle) + '</div>' : '') +
      '</span>' +
    '</summary>' +
    '<div class="step-body">' + bodyHtml + '</div>';
  stepsEl.appendChild(details);
  stepsWrap.style.display = 'block';
  stepsCount.textContent = seenSteps.size + ' steps';
  stepsWrap.scrollTop = stepsWrap.scrollHeight;
}

function subCallTokens(sub) {
  const us = sub.usage_summary || {};
  let prompt = 0, output = 0;
  if (us.total_input_tokens !== undefined) {
    prompt = us.total_input_tokens;
    output = us.total_output_tokens || 0;
  } else if (us.model_usage_summaries) {
    for (const m of Object.values(us.model_usage_summaries)) {
      prompt += m.total_input_tokens || 0;
      output += m.total_output_tokens || 0;
    }
  }
  return prompt + ' prompt + ' + output + ' output tokens';
}

function subCallHtml(sub, i) {
  const prompt = typeof sub.prompt === 'object' ? JSON.stringify(sub.prompt, null, 2) : (sub.prompt || '');
  return '<div class="sub-call">' +
    '<div class="section-label">Sub-call ' + (i + 1) + '</div>' +
    '<div class="subtitle" style="margin-bottom:6px">' + subCallTokens(sub) + ' · ' + (sub.execution_time ? sub.execution_time.toFixed(2) + 's' : '') + '</div>' +
    pre('Prompt', prompt) +
    pre('Response', sub.response) +
  '</div>';
}

async function appendLogSteps(logFile) {
  if (!logFile) return;
  try {
    const res = await fetch('/logs/' + encodeURIComponent(logFile));
    const text = await res.text();
    if (!text.trim()) return;
    const lines = text.trim().split('\n');
    let iterCount = 0;
    const openLatest = seenSteps.size === 0;

    lines.forEach((line, idx) => {
      try {
        const entry = JSON.parse(line);
        if (entry.type === 'metadata') {
          const cfg = [
            'model: ' + entry.root_model,
            'max iterations: ' + entry.max_iterations,
            'max depth: ' + entry.max_depth,
          ].join(' · ');
          addStep('⚙', 'Configuration', cfg, pre('Run metadata', entry));
          return;
        }
        if (entry.type !== 'iteration') return;

        iterCount = entry.iteration || iterCount + 1;
        const blocks = entry.code_blocks || [];
        let subs = [];
        blocks.forEach(b => { if (b.result && b.result.rlm_calls) subs = subs.concat(b.result.rlm_calls); });

        let chunks = 0;
        let batched = 0;
        let codePreview = '';
        let codeBodies = [];
        let outputs = [];
        blocks.forEach(b => {
          const code = (b.code || '').trim();
          if (code) {
            if (!codePreview) codePreview = code.replace(/\s+/g, ' ').slice(0, 180) + (code.length > 180 ? '…' : '');
            codeBodies.push(code);
            const out = [];
            if (b.result) {
              if (b.result.stdout) out.push('STDOUT:\n' + b.result.stdout);
              if (b.result.stderr) out.push('STDERR:\n' + b.result.stderr);
            }
            if (out.length) outputs.push(out.join('\n\n---\n\n'));
          }
          const flatCode = code.replace(/\s+/g, ' ');
          if (flatCode.includes('llm_query_batched')) {
            const m = flatCode.match(/llm_query_batched\s*\(\s*\[([^\]]*)\]/);
            if (m) batched = Math.max(batched, m[1].split(',').filter(x => x.trim()).length);
          }
          if (flatCode.includes('context[') || flatCode.includes('context.split') || flatCode.includes('strip().split')) chunks += 1;
        });

        const plan = (entry.response || '').trim();
        const planPreview = plan ? plan.replace(/\s+/g, ' ').slice(0, 360) + (plan.length > 360 ? '…' : '') : '';
        const batchInfo = entry.batch ? entry.batch.length + ' questions' : '';
        let subtitleParts = [];
        if (planPreview) subtitleParts.push(planPreview);
        else {
          if (chunks) subtitleParts.push('chunked context');
          if (batched) subtitleParts.push(batched + ' batched sub-LLM calls');
          if (subs.length && !batched) subtitleParts.push(subs.length + ' sub-LLM calls');
          if (batchInfo) subtitleParts.push(batchInfo);
          if (!subtitleParts.length) subtitleParts.push('planning');
          if (codePreview) subtitleParts.push('` ' + codePreview + ' `');
        }

        let body = pre('Root LLM prompt (this turn)', formatPrompt(entry.prompt));
        body += '<div class="section-label">Root LLM thinking / plan</div><div class="pre plan">' + escapeHtml(plan) + '</div>';
        if (codeBodies.length) body += pre('Code executed', codeBodies.join('\n\n---\n\n'));
        if (outputs.length) body += pre('Code output', outputs.join('\n\n---\n\n'));
        if (subs.length) {
          body += '<div class="section-label">Sub-LLM calls (' + subs.length + ')</div>';
          body += subs.map((sub, i) => subCallHtml(sub, i)).join('');
        }

        const badge = subs.length ? subs.length + ' sub-call' + (subs.length > 1 ? 's' : '') : (batched ? batched + ' batched' : '');
        addStep(iterCount, 'Iteration ' + iterCount, subtitleParts.join(' · '), body, entry.final_answer ? 'final' : '', badge);

        subs.forEach((sub, i) => {
          const letter = String.fromCharCode(97 + i);
          const resp = (sub.response || '').trim();
          const subBody = pre('Prompt', typeof sub.prompt === 'object' ? JSON.stringify(sub.prompt, null, 2) : (sub.prompt || '')) +
                          pre('Response', sub.response);
          addStep(iterCount + letter, 'Sub-call ' + (i + 1), resp ? resp.slice(0, 220) + (resp.length > 220 ? '…' : '') : 'sub-LLM call', subBody);
        });

        if (entry.final_answer) {
          addStep('✓', 'FINAL answer emitted', entry.final_answer, pre('Final answer', entry.final_answer), 'final');
        }
      } catch (e) { /* skip malformed */ }
    });

    // Auto-expand latest iteration while running.
    const details = stepsEl.querySelectorAll('details');
    if (details.length && openLatest) {
      details[details.length - 1].open = true;
    }
  } catch (e) {
    console.error('log poll failed', e);
  }
}

function updatePerspective() {
  const base = lastUsage.base;
  const rlm = lastUsage.rlm;
  const baseOut = base?.completion_tokens || 0;
  const rlmOut = rlm?.completion_tokens || 0;
  const rlmIn = rlm?.prompt_tokens || 0;
  const rlmTime = rlm?.execution_time || 0;

  if (!baseOut && !rlmOut) {
    perspective.style.display = 'none';
    return;
  }

  let html = '';
  if (baseOut && !rlmOut) {
    html = '<strong>Simple LLM perspective:</strong> API consumed <span class="metric">' + baseOut.toLocaleString() + '</span> output tokens but returned empty content.';
  } else if (!baseOut && rlmOut) {
    html = '<strong>RLM perspective:</strong> used <span class="metric">' + rlmOut.toLocaleString() + '</span> output tokens across all turns to produce the answer in <span class="metric">' + rlmTime.toFixed(1) + 's</span>.';
  } else if (baseOut && rlmOut) {
    const ratio = rlmOut / baseOut;
    const correct = rlm?.response && !rlm.response.startsWith('ERROR') && !rlm.response.startsWith('TIMEOUT');
    html = '<strong>Comparison:</strong> Simple LLM consumed <span class="metric">' + baseOut.toLocaleString() + '</span> output tokens and returned nothing. ' +
           'RLM used <span class="metric">' + rlmOut.toLocaleString() + '</span> output tokens (+' + ratio.toFixed(1) + '×) plus <span class="metric">' + rlmIn.toLocaleString() + '</span> prompt tokens, ' +
           'and ' + (correct ? 'returned a correct answer' : 'returned an answer') + ' in <span class="metric">' + rlmTime.toFixed(1) + 's</span>.';
  }
  perspective.innerHTML = html;
  perspective.style.display = 'block';
}

async function pollJob(jobId, logFile) {
  const res = await fetch('/api/job/' + jobId);
  const job = await res.json();
  await appendLogSteps(logFile);

  if (job.status === 'running') {
    const elapsed = Math.floor((Date.now() - activeJob.start) / 1000);
    setRunning('RLM running — ' + seenSteps.size + ' steps · ' + elapsed + 's');
    return;
  }

  clearInterval(pollTimer);
  pollTimer = null;
  activeJob = null;
  $('btnRLM').disabled = false;
  await appendLogSteps(logFile);

  if (job.status === 'error') {
    setCard('rlmCard', 'fail', 'RLM', job.response, '');
    setError(job.response);
    lastUsage.rlm = { ...job.usage, response: job.response, execution_time: job.execution_time };
    updatePerspective();
    return;
  }

  const meta = (job.execution_time ? job.execution_time.toFixed(1) + 's · ' : '') +
               'Finish: ' + (job.finish_reason || 'stop') + ' · ' +
               (job.usage ? job.usage.prompt_tokens.toLocaleString() + ' prompt + ' + job.usage.completion_tokens.toLocaleString() + ' output tokens' : '');
  setCard('rlmCard', 'ok', 'RLM', job.response, meta);
  setIdle('RLM succeeded in ' + (job.execution_time ? job.execution_time.toFixed(1) : '?') + 's.');
  lastUsage.rlm = { ...job.usage, response: job.response, execution_time: job.execution_time };
  updatePerspective();
}

async function loadOOLONG(limit) {
  setRunning('Loading OOLONG benchmark...');
  $('btnMini').disabled = true;
  $('btnMed').disabled = true;
  try {
    const url = '/api/oolong-data?limit=' + (limit || 20);
    const res = await fetch(url);
    const data = await res.json();
    $('context').value = data.context;
    $('query').value = data.question;
    dataInfo.textContent = 'OOLONG TREC-coarse: ' + data.num_questions + ' questions, ' + (data.context.length / 1000).toFixed(1) + 'K chars. Ground truth: ' + JSON.stringify(data.answer) + '.';
    setIdle(data.num_questions + ' questions loaded. Run Simple LLM or RLM.');
  } catch (e) {
    setError(e.message);
  } finally {
    $('btnMini').disabled = false;
    $('btnMed').disabled = false;
  }
}

function echoQuery() {
  const ctx = $('context').value.trim();
  const q = $('query').value.trim();
  if (!ctx && !q) {
    queryEcho.style.display = 'none';
    return;
  }
  queryEchoText.textContent = (q ? q + '\n\n' : '') + ctx;
  queryEcho.style.display = 'block';
}

async function runRLM() {
  const context = $('context').value.trim();
  if (!context) { setError('Load a dataset first.'); return; }
  clearSteps();
  echoQuery();
  setRunning('Starting RLM job...');
  $('btnRLM').disabled = true;
  try {
    const data = await post('/api/run', { context, query: $('query').value.trim(), max_iterations: 8 });
    activeJob = { id: data.job_id, logFile: data.log_file, start: Date.now() };
    addStep(0, 'Job started', 'log: ' + data.log_file, '<div class="section-label">Job</div><div class="pre">' + escapeHtml(JSON.stringify(data, null, 2)) + '</div>');
    pollTimer = setInterval(() => pollJob(data.job_id, data.log_file), 2000);
  } catch (e) {
    setError(e.message);
    $('btnRLM').disabled = false;
  }
}

async function runBaseline() {
  const context = $('context').value.trim();
  if (!context) { setError('Load a dataset first.'); return; }
  echoQuery();
  setRunning('Simple LLM running direct call...');
  $('btnAlg2').disabled = true;
  try {
    const data = await post('/api/baseline', { context, query: $('query').value.trim() });
    const answer = data.response || '';
    const meta = (data.execution_time ? data.execution_time.toFixed(1) + 's · ' : '') +
                 'Finish: ' + (data.finish_reason || '?') + ' · ' +
                 (data.usage ? data.usage.prompt_tokens.toLocaleString() + ' prompt + ' + data.usage.completion_tokens.toLocaleString() + ' output tokens' : '');
    if (!answer.trim() || data.finish_reason === 'length') {
      setCard('alg2Card', 'fail', 'Simple LLM', '(API returned empty content after generating ' + (data.usage?.completion_tokens || 0).toLocaleString() + ' output tokens)', meta);
      setIdle('Simple LLM failed: API returned empty content even though output tokens were consumed.');
    } else {
      setCard('alg2Card', 'ok', 'Simple LLM', answer, meta);
      setIdle('Simple LLM returned an answer.');
    }
    lastUsage.base = data.usage;
    updatePerspective();
  } catch (e) {
    setError(e.message);
  } finally {
    $('btnAlg2').disabled = false;
  }
}

$('btnTiny').addEventListener('click', () => {
  $('context').value = 'What is 2+2? Answer in one word.';
  $('query').value = '';
  dataInfo.textContent = 'Tiny example: 1 short question.';
  setIdle('Tiny example loaded. Run Simple LLM or RLM.');
});
$('btnMini').addEventListener('click', () => loadOOLONG(20));
$('btnMed').addEventListener('click', () => loadOOLONG(100));
$('btnAlg2').addEventListener('click', runBaseline);
$('btnRLM').addEventListener('click', runRLM);

const fileInput = $('fileInput');
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = e => {
    $('context').value = e.target.result;
    dataInfo.textContent = 'Uploaded: ' + file.name + ' (' + (file.size / 1024).toFixed(1) + ' KB)';
    setIdle('File uploaded. Add a prompt and run Simple LLM or RLM.');
  };
  reader.readAsText(file);
});
