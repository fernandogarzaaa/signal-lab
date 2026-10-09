'use strict';
/**
 * signallab web server: serves the dashboard UI and the stage-runner API.
 *
 *   node server.js [--port N]
 *
 * API:
 *   GET  /api/status            engine + per-stage run state
 *   POST /api/run/:stage        start a stage job  {job_id, stage, status}
 *   GET  /api/jobs/:jobid       job status {status, error, has_result}
 *   GET  /api/logs/:jobid       Server-Sent Events stream of job output
 *   GET  /api/results/:stage    cached JSON or 404 {"error":"not run yet"}
 *   POST /api/seed              load the bundled demo snapshot
 */
const path = require('path');
const fs = require('fs');
const express = require('express');
const { spawn } = require('child_process');
const {
  PKG_ROOT, DATA_DIR, RESULTS_DIR, VENV_DIR, ENGINE_DIR,
  venvPython, venvExists, python3Version, ensureDataDir, seedDemo, doctor,
} = require('./lib/setup');
const { createJobManager, STAGES, STAGE_META } = require('./lib/jobs');

const PUBLIC_DIR = path.join(PKG_ROOT, 'public');

function stageState(stage) {
  const f = path.join(RESULTS_DIR, `${stage}.json`);
  try {
    const st = fs.statSync(f);
    return { ran: true, at: st.mtime.toISOString() };
  } catch {
    return { ran: false, at: null };
  }
}

function createApp() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));
  app.use(express.static(PUBLIC_DIR));

  const jobs = createJobManager();

  app.get('/api/status', (req, res) => {
    const stages = {};
    for (const s of STAGES) stages[s] = { ...STAGE_META[s] && { title: STAGE_META[s].title }, ...stageState(s) };
    res.json({
      engine: venvExists() ? 'ok' : 'missing',
      venv: VENV_DIR,
      python: python3Version(),
      node: process.version,
      stages,
    });
  });

  app.get('/api/doctor', (req, res) => res.json({ checks: doctor() }));

  app.post('/api/run/:stage', (req, res) => {
    const { stage } = req.params;
    if (!STAGES.includes(stage)) return res.status(400).json({ error: `unknown stage: ${stage}` });
    if (!venvExists()) {
      return res.status(400).json({ error: 'engine not set up — run `signallab setup` first' });
    }
    const extraArgs = Array.isArray(req.body && req.body.args) ? req.body.args.map(String) : [];
    const job = jobs.start(stage, extraArgs);
    res.json({ job_id: job.id, stage, status: job.status });
  });

  app.get('/api/jobs/:jobid', (req, res) => {
    const job = jobs.get(req.params.jobid);
    if (!job) return res.status(404).json({ error: 'unknown job' });
    res.json(jobs.summary(job));
  });

  // Server-Sent Events: replay buffered lines, then stream until job ends.
  app.get('/api/logs/:jobid', (req, res) => {
    const job = jobs.get(req.params.jobid);
    if (!job) return res.status(404).json({ error: 'unknown job' });
    res.writeHead(200, {
      'Content-Type': 'text/event-stream',
      'Cache-Control': 'no-cache',
      Connection: 'keep-alive',
    });
    const sendLine = (l) => res.write(`data: ${JSON.stringify({ t: l.t, s: l.s, text: l.text })}\n\n`);
    for (const l of job.lines) sendLine(l);
    const onLine = (l) => sendLine(l);
    const onEnd = () => {
      res.write(`event: done\ndata: ${JSON.stringify(jobs.summary(job))}\n\n`);
      cleanup();
    };
    const cleanup = () => {
      job.emitter.removeListener('line', onLine);
      job.emitter.removeListener('end', onEnd);
      try { res.end(); } catch {}
    };
    if (job.status !== 'running') { onEnd(); return; }
    job.emitter.on('line', onLine);
    job.emitter.on('end', onEnd);
    req.on('close', cleanup);
  });

  app.get('/api/results/:stage', (req, res) => {
    const { stage } = req.params;
    if (!STAGES.includes(stage)) return res.status(400).json({ error: `unknown stage: ${stage}` });
    const f = path.join(RESULTS_DIR, `${stage}.json`);
    try {
      res.json(JSON.parse(fs.readFileSync(f, 'utf8')));
    } catch {
      res.status(404).json({ error: 'not run yet' });
    }
  });

  // Model diagnostics (0.3.0): walk-forward + calibration artifacts.
  // Written by `python -m signal_lab.models.walk_forward` and
  // `python -m signal_lab.models.calibration` into DATA_DIR/artifacts.
  app.get('/api/diagnostics/:name', (req, res) => {
    const { name } = req.params;
    if (!/^[a-z_]+$/.test(name)) return res.status(400).json({ error: 'bad name' });
    const f = path.join(DATA_DIR, 'artifacts', `${name}.json`);
    try {
      res.json(JSON.parse(fs.readFileSync(f, 'utf8')));
    } catch {
      res.status(404).json({ error: 'not run yet' });
    }
  });

  app.post('/api/seed', (req, res) => {
    try {
      const copied = seedDemo();
      res.json({ ok: true, copied });
    } catch (e) {
      res.status(500).json({ ok: false, error: e.message });
    }
  });

  // Evidence endpoints (workstreams 1-2): validate ticker/date/k, then spawn
  // the venv engine module with --json (stdout is ONLY the JSON object;
  // logs go to stderr). Never fabricates: python failures surface the
  // stderr tail with a 502.
  function evidenceParams(req, res) {
    const ticker = String(req.query.ticker || '').toUpperCase().trim();
    const date = String(req.query.date || '').trim();
    const k = Math.min(20, Math.max(1, parseInt(req.query.k || '5', 10) || 5));
    if (!/^[A-Z]{1,6}$/.test(ticker)) {
      res.status(400).json({ error: 'bad ticker: 1-6 letters (e.g. MSFT)' });
      return null;
    }
    if (!/^\d{4}-\d{2}-\d{2}$/.test(date)) {
      res.status(400).json({ error: 'bad date: use YYYY-MM-DD' });
      return null;
    }
    if (!venvExists()) {
      res.status(400).json({ error: 'engine not set up — run `signallab setup` first' });
      return null;
    }
    return { ticker, date, k };
  }

  function runEngineModule(res, module, extraArgs) {
    const child = spawn(venvPython(), ['-m', module, ...extraArgs, '--json'],
      { cwd: PKG_ROOT, env: { ...process.env, PYTHONPATH: ENGINE_DIR } });
    let out = '', err = '';
    child.stdout.on('data', (d) => { out += d; });
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', (e) => res.status(502).json({ error: 'failed to start engine', detail: e.message }));
    child.on('close', (code) => {
      if (code !== 0) {
        const tail = String(err).split('\n').map((l) => l.trim())
          .filter(Boolean).slice(-12).join('\n').slice(0, 2000);
        return res.status(502).json({ error: `${module} exited with code ${code}`, detail: tail });
      }
      try {
        res.json(JSON.parse(out));
      } catch (e) {
        res.status(502).json({ error: `${module} returned invalid JSON`, detail: String(out).slice(0, 500) });
      }
    });
  }

  // "Why did it move?" (workstream 1): evidence for a ticker/date.
  app.get('/api/explain', (req, res) => {
    const p = evidenceParams(req, res);
    if (!p) return;
    runEngineModule(res, 'signal_lab.explain_move',
      ['--ticker', p.ticker, '--date', p.date, '--k', String(p.k)]);
  });

  // Historical analogues (workstream 2): similar past news events + forward returns.
  app.get('/api/analogues', (req, res) => {
    const p = evidenceParams(req, res);
    if (!p) return;
    runEngineModule(res, 'signal_lab.analogues',
      ['--ticker', p.ticker, '--date', p.date, '--k', String(p.k)]);
  });

  // Watchlist monitor (workstream 3): near-real-time GDELT polling.
  function watchlistTicker(req, res) {
    const ticker = String(req.body.ticker || req.params.ticker || '').toUpperCase().trim();
    if (!/^[A-Z0-9.\-]{1,8}$/.test(ticker)) {
      res.status(400).json({ error: 'bad ticker: 1-8 chars A-Z 0-9 . - (e.g. MSFT)' });
      return null;
    }
    if (!venvExists()) {
      res.status(400).json({ error: 'engine not set up — run `signallab setup` first' });
      return null;
    }
    return ticker;
  }

  // Fire-and-forget engine run: a poll cycle takes minutes (GDELT + bodies
  // + scoring), so the HTTP response returns immediately and the dashboard
  // reads the fresh state from /api/watchlist afterwards.
  function runMonitorPoll() {
    const child = spawn(venvPython(), ['-m', 'signal_lab.monitor', '--poll', '--json'],
      { cwd: PKG_ROOT, env: { ...process.env, PYTHONPATH: ENGINE_DIR } });
    let err = '';
    child.stderr.on('data', (d) => { err += d; });
    child.on('error', (e) => console.log(`[monitor] failed to start poll: ${e.message}`));
    child.on('close', (code) => {
      if (code !== 0) {
        console.log(`[monitor] poll exited with code ${code}: ${String(err).split('\n').filter(Boolean).slice(-5).join(' | ').slice(0, 500)}`);
      } else {
        console.log('[monitor] background poll finished');
      }
    });
  }

  app.get('/api/watchlist', (req, res) => {
    if (!venvExists()) {
      return res.status(400).json({ error: 'engine not set up — run `signallab setup` first' });
    }
    runEngineModule(res, 'signal_lab.monitor', ['--watchlist-json']);
  });

  app.post('/api/watchlist', (req, res) => {
    const ticker = watchlistTicker(req, res);
    if (!ticker) return;
    runEngineModule(res, 'signal_lab.monitor', ['--watchlist-add', ticker]);
  });

  app.delete('/api/watchlist/:ticker', (req, res) => {
    const ticker = watchlistTicker(req, res);
    if (!ticker) return;
    runEngineModule(res, 'signal_lab.monitor', ['--watchlist-remove', ticker]);
  });

  app.post('/api/monitor/poll', (req, res) => {
    if (!venvExists()) {
      return res.status(400).json({ error: 'engine not set up — run `signallab setup` first' });
    }
    runMonitorPoll();
    res.status(202).json({ ok: true, message: 'poll started in the background; refresh /api/watchlist for the fresh state' });
  });

  // Fallback to the dashboard for any other route.
  // Express 5 (path-to-regexp v8) rejects the bare '*' wildcard; '/{*splat}' is the supported form.
  app.get('/{*splat}', (req, res) => res.sendFile(path.join(PUBLIC_DIR, 'index.html')));

  // Workstream 3: background GDELT polling. GDELT refreshes roughly every
  // 15 minutes, so the default cadence matches it; polling faster buys
  // nothing. SIGNAL_LAB_MONITOR_MINUTES=0 disables the interval (manual
  // polls via POST /api/monitor/poll still work).
  const monitorMinutes = Number(process.env.SIGNAL_LAB_MONITOR_MINUTES || '15');
  if (monitorMinutes > 0 && venvExists()) {
    const ms = monitorMinutes * 60 * 1000;
    console.log(`[monitor] background polling every ${monitorMinutes} min (first run in 60s; SIGNAL_LAB_MONITOR_MINUTES=0 to disable)`);
    setTimeout(() => { console.log('[monitor] initial poll starting'); runMonitorPoll(); }, 60 * 1000);
    setInterval(() => { console.log('[monitor] scheduled poll starting'); runMonitorPoll(); }, ms);
  }

  return app;
}

function start(port, { openBrowser = true } = {}) {
  const app = createApp();
  const server = app.listen(port, () => {
    const url = `http://localhost:${port}`;
    console.log(`Signal Lab UI: ${url}`);
    if (openBrowser) {
      import('open').then((m) => m.default(url).catch((e) => {
        console.log(`(could not open a browser automatically: ${e.message})`);
        console.log(`Open this URL manually: ${url}`);
      })).catch(() => console.log(`Open this URL manually: ${url}`));
    }
  });
  return server;
}

if (require.main === module) {
  const args = process.argv.slice(2);
  let port = 3100;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--port' && args[i + 1]) port = Number(args[++i]);
  }
  ensureDataDir();
  start(port, { openBrowser: false });
  console.log('(server.js run directly; the signallab CLI opens the browser for you)');
}

module.exports = { createApp, start };
