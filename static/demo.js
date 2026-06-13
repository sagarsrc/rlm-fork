const $ = id => document.getElementById(id);
const status = $('status');
const dataInfo = $('dataInfo');
const stepsEl = $('steps');
const stepsWrap = $('stepsWrap');
let activeJob = null;
let pollTimer = null;
let seenSteps = new Set();

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
}

function addStep(number, title, detail, cls) {
  const key = number + ':' + title;
  if (seenSteps.has(key)) return;
  seenSteps.add(key);
  const div = document.createElement('div');
  div.className = 'step ' + (cls || '');
  div.innerHTML = '<span class="num">' + number + '</span><span class="title">' + title + '</span>' +
                  (detail ? '<div class="detail">' + escapeHtml(detail) + '</div>' : '');
  stepsEl.appendChild(div);
  stepsWrap.style.display = 'block';
  stepsWrap.scrollTop = stepsWrap.scrollHeight;
}

function escapeHtml(s) {
  return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function shortPlan(text) {
  const t = (text || '').trim();
  if (!t) return '';
  const oneline = t.replace(/\s+/g, ' ');
  return oneline.slice(0, 260) + (oneline.length > 260 ? '…' : '');
}

function shortCode(code) {
  const c = (code || '').trim();
  if (!c) return '';
  return c.replace(/\s+/g, ' ').slice(0, 120) + (c.length > 120 ? '…' : '');
}

async function appendLogSteps(logFile) {
  if (!logFile) return;
  try {
    const res = await fetch('/logs/' + encodeURIComponent(logFile));
    const text = await res.text();
    if (!text.trim()) return;
    const lines = text.trim().split('\n');
    let iterCount = 0;
    lines.forEach(line => {
      try {
        const entry = JSON.parse(line);
        if (entry.type !== 'iteration') return;
        iterCount = entry.iteration || iterCount + 1;
        const blocks = entry.code_blocks || [];
        let subs = [];
        blocks.forEach(b => { if (b.result && b.result.rlm_calls) subs = subs.concat(b.result.rlm_calls); });

        let chunks = 0;
        let batched = 0;
        let codePreview = '';
        blocks.forEach(b => {
          const code = (b.code || '').replace(/\s+/g, ' ');
          if (!codePreview && code) codePreview = shortCode(b.code);
          if (code.includes('llm_query_batched')) {
            const m = code.match(/llm_query_batched\s*\(\s*\[([^\]]*)\]/);
            if (m) batched = Math.max(batched, m[1].split(',').filter(x => x.trim()).length);
          }
          if (code.includes('context[') || code.includes('context.split') || code.includes('strip().split')) chunks += 1;
        });

        const plan = shortPlan(entry.response);
        const batchInfo = entry.batch ? entry.batch.length + ' questions' : '';
        let detail = [];
        if (chunks) detail.push('chunked context');
        if (batched) detail.push(batched + ' batched sub-LLM calls');
        if (subs.length && !batched) detail.push(subs.length + ' sub-LLM calls');
        if (batchInfo) detail.push(batchInfo);
        if (!detail.length) detail.push('planning');
        if (codePreview) detail.push('` ' + codePreview + ' `');
        if (plan) detail.push(plan);

        addStep(iterCount, 'Iteration ' + iterCount, detail.join('\n'), entry.final_answer ? 'final' : '');

        subs.forEach((sub, i) => {
          const letter = String.fromCharCode(97 + i);
          const resp = (sub.response || '').trim();
          addStep(iterCount + letter, 'Sub-call ' + (i + 1), resp ? resp.slice(0, 120) + (resp.length > 120 ? '…' : '') : 'sub-LLM call');
        });

        if (entry.final_answer) {
          addStep('✓', 'FINAL answer emitted', entry.final_answer, 'final');
        }
      } catch (e) { /* skip malformed */ }
    });
  } catch (e) {
    console.error('log poll failed', e);
  }
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
    setCard('rlmCard', 'fail', 'RLM — Recursive', job.response, '');
    setError(job.response);
    return;
  }

  const meta = (job.execution_time ? job.execution_time.toFixed(1) + 's · ' : '') + 'done';
  setCard('rlmCard', 'ok', 'RLM — Recursive', job.response, meta);
  setIdle('RLM succeeded in ' + (job.execution_time ? job.execution_time.toFixed(1) : '?') + 's.');
}

async function loadOOLONG(limit) {
  setRunning('Loading OOLONG benchmark...');
  $('btnOolong').disabled = true;
  $('btnMini').disabled = true;
  try {
    const url = limit ? '/api/oolong-data?limit=' + limit : '/api/oolong-data';
    const res = await fetch(url);
    const data = await res.json();
    $('context').value = data.context;
    $('query').value = data.question;
    dataInfo.textContent = 'OOLONG TREC-coarse: ' + data.num_questions + ' questions, ' + (data.context.length / 1000).toFixed(1) + 'K chars. Ground truth: ' + JSON.stringify(data.answer) + '.';
    setIdle(data.num_questions + ' questions loaded. Run Algorithm 2 or RLM.');
  } catch (e) {
    setError(e.message);
  } finally {
    $('btnOolong').disabled = false;
    $('btnMini').disabled = false;
  }
}

async function runRLM() {
  const context = $('context').value.trim();
  if (!context) { setError('Load a dataset first.'); return; }
  clearSteps();
  setRunning('Starting RLM job...');
  $('btnRLM').disabled = true;
  try {
    const data = await post('/api/run', { context, query: $('query').value.trim(), max_iterations: 8 });
    activeJob = { id: data.job_id, logFile: data.log_file, start: Date.now() };
    addStep(0, 'Job started', 'log: ' + data.log_file);
    pollTimer = setInterval(() => pollJob(data.job_id, data.log_file), 2000);
  } catch (e) {
    setError(e.message);
    $('btnRLM').disabled = false;
  }
}

async function runBaseline() {
  const context = $('context').value.trim();
  if (!context) { setError('Load a dataset first.'); return; }
  setRunning('Algorithm 2 running direct LLM call...');
  $('btnAlg2').disabled = true;
  try {
    const data = await post('/api/baseline', { context, query: $('query').value.trim() });
    const answer = data.response || '';
    const meta = 'Finish: ' + (data.finish_reason || '?') + ' · ' + (data.usage ? data.usage.prompt_tokens + ' + ' + data.usage.completion_tokens + ' tokens' : '');
    if (!answer.trim() || data.finish_reason === 'length') {
      setCard('alg2Card', 'fail', 'Algorithm 2 — Direct LLM', '(empty)', meta);
      setIdle('Algorithm 2 failed: context consumed all tokens.');
    } else {
      setCard('alg2Card', 'ok', 'Algorithm 2 — Direct LLM', answer, meta);
      setIdle('Algorithm 2 returned an answer.');
    }
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
  setIdle('Tiny example loaded. Run Algorithm 2 or RLM.');
});
$('btnOolong').addEventListener('click', () => loadOOLONG(null));
$('btnMini').addEventListener('click', () => loadOOLONG(20));
$('btnAlg2').addEventListener('click', runBaseline);
$('btnRLM').addEventListener('click', runRLM);
