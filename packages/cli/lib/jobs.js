'use strict';
/**
 * Stage definitions and execution.
 * Each stage runner prints ONE JSON summary object to stdout; logs go to stderr.
 * We never fabricate results: failures surface the process stderr tail.
 */
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const { EventEmitter } = require('events');
const {
  PKG_ROOT, ENGINE_DIR, RESULTS_DIR,
  venvPython, venvExists, ensureDataDir,
} = require('./setup');

const STAGES = ['ingest', 'nlp', 'models', 'stats', 'backtest', 'rag'];

const STAGE_META = {
  ingest: {
    title: 'Ingest', desc: 'GDELT news + yfinance prices into DuckDB',
    module: 'signal_lab.ingest.run', jsonFlag: true,
    args: ['--tickers', 'AAPL,MSFT,JPM', '--days', '30'],
  },
  nlp: {
    title: 'Entities', desc: 'Company mention to ticker extraction',
    module: 'signal_lab.nlp.run', jsonFlag: true, args: [],
  },
  models: {
    title: 'Sentiment model', desc: 'Classical ML on imbalanced data',
    module: 'signal_lab.models.run', jsonFlag: true, args: [],
  },
  stats: {
    title: 'Event study', desc: 'Do sentiment spikes move prices?',
    module: 'signal_lab.stats.run', jsonFlag: true, args: [],
  },
  backtest: {
    title: 'Backtest', desc: 'Walk-forward strategy comparison',
    module: 'signal_lab.backtest.run', jsonFlag: true, args: [],
  },
  rag: {
    title: 'RAG explainer', desc: 'Cited explanations over 10-K filings',
    module: 'signal_lab.rag.run', jsonFlag: true, args: [],
    needsEvent: true,
  },
};

/** Pick a default event for the RAG stage from cached stats results. */
function defaultRagEvent() {
  try {
    const p = JSON.parse(fs.readFileSync(path.join(RESULTS_DIR, 'stats.json'), 'utf8'));
    const evs = (p.result && p.result.events) || [];
    if (!evs.length) return null;
    const sig = evs.find((e) => e.p_value != null && e.p_value < 0.05);
    const e = sig || evs[0];
    return { ticker: e.ticker, date: e.date };
  } catch { return null; }
}

/** Extract the last JSON object from stdout text. Returns {ok, value} or {ok:false}. */
function extractJson(stdout) {
  const lines = String(stdout).split('\n').map((l) => l.trim()).filter(Boolean);
  for (let i = lines.length - 1; i >= 0; i--) {
    const line = lines[i];
    if (!line.startsWith('{')) continue;
    try {
      const v = JSON.parse(line);
      if (v && typeof v === 'object' && !Array.isArray(v)) return { ok: true, value: v };
    } catch { /* keep scanning */ }
  }
  // Fallback: try the whole blob (pretty-printed JSON).
  try {
    const v = JSON.parse(String(stdout).trim());
    if (v && typeof v === 'object' && !Array.isArray(v)) return { ok: true, value: v };
  } catch { /* no */ }
  return { ok: false };
}

/**
 * Run one stage. onLine({stream:'out'|'err', text}) receives output lines.
 * Resolves {ok:true, result} or {ok:false, error, stderrTail}.
 */
function runStage(stage, extraArgs, onLine) {
  return new Promise((resolve) => {
    const meta = STAGE_META[stage];
    if (!meta) return resolve({ ok: false, error: `unknown stage: ${stage}` });
    if (!venvExists()) {
      return resolve({ ok: false, error: 'engine not set up — run `signallab setup` first' });
    }
    ensureDataDir();
    let cmd, args;
    if (meta.module) {
      cmd = venvPython();
      let finalExtra = [...(extraArgs || [])];
      if (meta.needsEvent && !finalExtra.includes('--event')) {
        const ev = defaultRagEvent();
        if (!ev) {
          return resolve({ ok: false, error: 'no events to explain — run the stats stage first, or pass --event \'{"ticker":"AAPL","date":"2026-09-15"}\'' });
        }
        finalExtra = ['--event', JSON.stringify(ev), ...finalExtra];
      }
      args = ['-m', meta.module, ...(meta.jsonFlag ? ['--json'] : []), ...meta.args, ...finalExtra];
    } else {
      return resolve({ ok: false, error: `no runner configured for stage: ${stage}` });
    }
    const env = { ...process.env, PYTHONPATH: ENGINE_DIR };
    const child = spawn(cmd, args, { cwd: PKG_ROOT, env });
    let stdout = '', stderr = '';
    let outBuf = '', errBuf = '';
    const emit = (stream, data) => {
      let buf = stream === 'out' ? outBuf : errBuf;
      buf += data;
      const parts = buf.split('\n');
      buf = parts.pop();
      for (const line of parts) {
        const text = line.replace(/\r$/, '');
        if (text) { try { onLine && onLine({ stream, text }); } catch {} }
      }
      if (stream === 'out') outBuf = buf; else errBuf = buf;
    };
    child.stdout.on('data', (d) => { stdout += d; emit('out', String(d)); });
    child.stderr.on('data', (d) => { stderr += d; emit('err', String(d)); });
    child.on('error', (e) => resolve({ ok: false, error: `failed to start: ${e.message}` }));
    child.on('close', (code) => {
      const tail = (s) => String(s).split('\n').map((l) => l.trim()).filter(Boolean).slice(-30);
      const stderrTail = tail(stderr).join('\n');
      if (code !== 0) {
        return resolve({ ok: false, error: `exit code ${code}`, stderrTail });
      }
      const parsed = extractJson(stdout);
      if (!parsed.ok) {
        return resolve({ ok: false, error: 'runner produced no JSON summary', stderrTail });
      }
      try {
        const outPath = path.join(RESULTS_DIR, `${stage}.json`);
        const cacheObj = { stage, ran_at: new Date().toISOString(), result: parsed.value };
        const ei = args.indexOf('--event');
        if (ei >= 0 && args[ei + 1]) {
          try { cacheObj.event = JSON.parse(args[ei + 1]); } catch {}
        }
        fs.writeFileSync(outPath, JSON.stringify(cacheObj, null, 2));
      } catch (e) { /* caching is best-effort */ }
      resolve({ ok: true, result: parsed.value });
    });
  });
}

/** In-memory job registry for the server (SSE streaming + polling). */
function createJobManager() {
  const jobs = new Map();
  let seq = 0;

  function start(stage, extraArgs) {
    const id = `job-${Date.now()}-${++seq}`;
    const job = {
      id, stage, status: 'running', lines: [], result: null,
      error: null, startedAt: new Date().toISOString(), emitter: new EventEmitter(),
    };
    job.emitter.setMaxListeners(50);
    jobs.set(id, job);
    if (jobs.size > 50) {
      const oldest = [...jobs.keys()][0];
      if (oldest !== id) jobs.delete(oldest);
    }
    const push = (stream, text) => {
      const line = { t: new Date().toISOString(), s: stream, text };
      job.lines.push(line);
      if (job.lines.length > 2000) job.lines.splice(0, job.lines.length - 2000);
      job.emitter.emit('line', line);
    };
    runStage(stage, extraArgs, ({ stream, text }) => push(stream, text)).then((r) => {
      job.status = r.ok ? 'done' : 'failed';
      job.result = r.ok ? r.result : null;
      job.error = r.ok ? null : (r.error + (r.stderrTail ? '\n' + r.stderrTail : ''));
      job.finishedAt = new Date().toISOString();
      job.emitter.emit('end');
    });
    return job;
  }

  function get(id) { return jobs.get(id); }

  function summary(job) {
    return {
      job_id: job.id, stage: job.stage, status: job.status,
      startedAt: job.startedAt, finishedAt: job.finishedAt || null,
      error: job.error,
      has_result: !!job.result,
    };
  }

  return { start, get, summary, STAGES, STAGE_META };
}

module.exports = { STAGES, STAGE_META, runStage, extractJson, createJobManager };
